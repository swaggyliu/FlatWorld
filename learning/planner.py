"""Latent-space CEM: sample actions, roll the world model, score a cost.

Cost is residual-corrected ``xy_head`` plus the task's geometric terms.
"""

import numpy as np
import torch

from learning.data.normalizer import Normalizer


class CEMPlanner:
    """GPU-batched receding-horizon CEM over imagined latent rollouts."""

    def __init__(self, models, norm: Normalizer, device: str = "cpu",
                 horizon: int = 32, population: int = 96, elites: int = 16,
                 iterations: int = 5, force_max: float = 6.0,
                 uncert_cost: float = 0.1, seed: int = 0,
                 tactile_residual: bool = False,
                 tactile_xy_tol: float = 0.03,
                 tactile_mode: str = None,
                 ensemble_reduce: str = "mean"):
        if not isinstance(models, (list, tuple)):
            models = [models]
        self.models = list(models)
        self.model = self.models[0]
        self.norm = norm
        self.device = device
        self.H = horizon
        self.P = population
        self.E = elites
        self.iters = iterations
        self.fmax = force_max
        self.uncert_cost = uncert_cost
        self.ensemble_reduce = str(ensemble_reduce)
        self.tactile_residual = bool(tactile_residual)
        self.tactile_xy_tol = float(tactile_xy_tol)
        if tactile_mode is None:
            tactile_mode = "gate" if self.tactile_residual else "on"
        if tactile_mode not in ("off", "gate", "on"):
            raise ValueError(f"tactile_mode={tactile_mode!r}")
        self.tactile_mode = tactile_mode
        self._pred_xy = None
        self.mean = torch.zeros(horizon, 2, device=device)
        span = 2.0 * force_max
        self.std = torch.full((horizon, 2), 0.25 * span, device=device)
        self.low = torch.tensor([-force_max, -force_max], device=device)
        self.high = torch.tensor([force_max, force_max], device=device)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        self._stats_cache = {}

    def _stats(self, model=None):
        """Per-model z-score tensors so mixed-dataset ensembles stay valid."""
        m = self.model if model is None else model
        key = id(m)
        cached = self._stats_cache.get(key)
        if cached is not None:
            return cached
        norm = getattr(m, "norm", None) or self.norm
        st = norm.to_tensors(self.device)
        mean = {k: v[0] for k, v in st.items()}
        std = {k: v[1] for k, v in st.items()}
        cached = (mean, std)
        self._stats_cache[key] = cached
        return cached

    def reset(self):
        self.mean = torch.zeros(self.H, 2, device=self.device)
        self.std = torch.full((self.H, 2), 0.25 * 2.0 * self.fmax, device=self.device)
        self._pred_xy = None

    def _normalize(self, key, x, model=None):
        mean, std = self._stats(model)
        return (x - mean[key]) / std[key]

    def _denorm_xy(self, xy_n, model=None):
        mean, std = self._stats(model)
        return xy_n * std["obj_states"][:2] + mean["obj_states"][:2]

    def _norm_geom(self, geom, model=None):
        g = torch.as_tensor(geom, device=self.device, dtype=torch.float32)
        mean, _ = self._stats(model)
        if "obj_geom" in mean:
            g = self._normalize("obj_geom", g, model)
        return g

    def encode_obs(self, obs: dict, use_tactile: bool = None):
        """Encode obs with every ensemble member. Returns list of (1, N, L).

        With ``tactile_mode='gate'``, the default is state-only. Tactile is
        added on the EE node only after the last imagined xy missed the
        real scene (slip / unexpected contact). ``off`` never uses tactile;
        ``on`` always does.
        """
        if use_tactile is None:
            if self.tactile_mode == "off":
                use_tactile = False
            elif self.tactile_mode == "on":
                use_tactile = True
            else:
                use_tactile = self._gate_tactile(obs)
        with torch.inference_mode():
            types = torch.as_tensor(obs["obj_types"], device=self.device)
            cmask = torch.as_tensor(obs["contact_mask"], device=self.device)
            zs = []
            for m in self.models:
                states = self._normalize(
                    "obj_states",
                    torch.as_tensor(obs["obj_states"], device=self.device), m)
                cfeat = self._normalize(
                    "contact_feat",
                    torch.as_tensor(obs["contact_feat"], device=self.device), m)
                tsum = self._normalize(
                    "tactile_summary",
                    torch.as_tensor(obs["tactile_summary"], device=self.device), m)
                geom = obs.get("obj_geom")
                geom_t = None
                if geom is not None:
                    g = torch.as_tensor(geom, device=self.device, dtype=torch.float32)
                    mean, _ = self._stats(m)
                    if "obj_geom" in mean:
                        g = self._normalize("obj_geom", g, m)
                    geom_t = g.unsqueeze(0)
                zs.append(m.encode(types.unsqueeze(0), states.unsqueeze(0),
                                   cfeat.unsqueeze(0), cmask.unsqueeze(0),
                                   tsum.unsqueeze(0), obj_geom=geom_t,
                                   use_tactile=use_tactile))
            return zs

    def _gate_tactile(self, obs: dict) -> bool:
        """Use tactile residual when the last open-loop xy disagrees with reality."""
        if self._pred_xy is None:
            return False
        real = np.asarray(obs["obj_states"])[:, :2]
        err = float(np.linalg.norm(real - self._pred_xy, axis=-1).mean())
        return err > self.tactile_xy_tol

    def remember_first_step(self, z0, action: np.ndarray, geom: np.ndarray):
        """Cache imagined xy after one model step of the executed force."""
        if self.tactile_mode != "gate":
            return
        z0s = [z0] if isinstance(z0, torch.Tensor) else list(z0)
        geom_n = self._norm_geom(geom, self.models[0])
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).view(1, 2)
        a = self._normalize("actions", a, self.models[0])
        with torch.inference_mode():
            m, z = self.models[0], z0s[0][:1]
            h = m.predictor.init_hidden(z)
            prev_c = m.pair_prob(z)
            z, _, _ = m.roll_step(z, a, h, geom_n, prev_c)
            self._pred_xy = self._denorm_xy(m.predictor.xy_head(z), m)[0].cpu().numpy()

    def _zero_xy(self, model, z0, geom_n, now_xy=None):
        """Zero-action imagined xy (H, N, 2) in raw units."""
        a0 = self._normalize("actions", torch.zeros(1, 2, device=self.device),
                             model)
        z, h = z0[:1], model.predictor.init_hidden(z0[:1])
        prev_c = model.pair_prob(z)
        frames = []
        for _ in range(self.H):
            z, h, prev_c = model.roll_step(z, a0, h, geom_n, prev_c)
            frames.append(self._denorm_xy(model.predictor.xy_head(z), model)[0])
        return torch.stack(frames)

    def _roll(self, model, z0, a, geom_n, cost, base, now_xy):
        """Imagine ``a`` and accumulate ``cost.step`` / ``cost.terminal``.

        Positions are residual-corrected against the zero-action baseline.
        """
        z = z0.expand(self.P, *z0.shape[1:])
        h = model.predictor.init_hidden(z)
        total = a.new_zeros(self.P)
        prev_c = model.pair_prob(z)
        pred = None
        prev = None
        for t in range(self.H):
            z, h, prev_c = model.roll_step(
                z, self._normalize("actions", a[:, t], model), h, geom_n, prev_c)
            pred = self._denorm_xy(model.predictor.xy_head(z), model)
            pred = pred - base[t] + now_xy
            total = total + cost.step(pred, a[:, t], prev)
            prev = pred
        return total + cost.terminal(pred)

    def plan(self, z0, geom: np.ndarray, states_now: np.ndarray, cost):
        """Optimize an action sequence. Returns best first action (2,) in N."""
        z0s = [z0] if isinstance(z0, torch.Tensor) else list(z0)
        now_xy = torch.as_tensor(states_now[:, :2], device=self.device,
                                 dtype=torch.float32)
        with torch.inference_mode():
            bases = [self._zero_xy(m, z, self._norm_geom(geom, m), now_xy)
                     for m, z in zip(self.models, z0s)]
            best_first = None
            for _ in range(self.iters):
                noise = torch.randn(self.P, self.H, 2, device=self.device)
                seqs = (self.mean + self.std * noise).clamp(self.low, self.high)
                costs = []
                for m, z, b in zip(self.models, z0s, bases):
                    costs.append(self._roll(
                        m, z, seqs, self._norm_geom(geom, m), cost,
                        base=b, now_xy=now_xy))
                C = torch.stack(costs, 0)
                if C.shape[0] == 1 or self.ensemble_reduce == "mean":
                    total = C.mean(0)
                else:
                    total = C.min(0).values
                if C.shape[0] > 1 and self.uncert_cost:
                    total = total + self.uncert_cost * C.std(0)
                elite = seqs[total.topk(self.E, largest=False).indices]
                self.mean = elite.mean(0).clone()
                self.std = elite.std(0).clamp_min(1e-3).clone()
                best_first = elite[0, 0].clone()
            rolled = torch.roll(self.mean, -1, 0)
            rolled[-1] = 0
            self.mean = rolled
            floor = (self.high - self.low) * 0.05
            self.std = torch.maximum(self.std * 0.9, floor).clone()
        return best_first.detach().float().cpu().numpy()
