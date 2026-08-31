"""Latent-space CEM: sample actions, roll the world model, score a cost.

The loop is task-agnostic. A cost object supplies ``step`` / ``terminal``
on residual-corrected ``xy_head`` trajectories. Geometric PushToGoal
terms live in ``learning.tasks.push_to_goal``.
"""

import numpy as np
import torch

from learning.data.normalizer import Normalizer


class CEMPlanner:
    """GPU-batched receding-horizon CEM over imagined latent rollouts."""

    def __init__(self, models, norm: Normalizer, device: str = "cpu",
                 horizon: int = 32, population: int = 96, elites: int = 16,
                 iterations: int = 5, force_max: float = 6.0,
                 uncert_cost: float = 0.1, seed: int = 0, **_ignore):
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
        self.mean = torch.zeros(horizon, 2, device=device)
        span = 2.0 * force_max
        self.std = torch.full((horizon, 2), 0.25 * span, device=device)
        self.low = torch.tensor([-force_max, -force_max], device=device)
        self.high = torch.tensor([force_max, force_max], device=device)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        st = norm.to_tensors(device)
        self.st_mean = {k: v[0] for k, v in st.items()}
        self.st_std = {k: v[1] for k, v in st.items()}
        self._xy_m = self.st_mean["obj_states"][:2]
        self._xy_s = self.st_std["obj_states"][:2]

    def reset(self):
        self.mean = torch.zeros(self.H, 2, device=self.device)
        self.std = torch.full((self.H, 2), 0.25 * 2.0 * self.fmax, device=self.device)

    def _normalize(self, key, x):
        return (x - self.st_mean[key]) / self.st_std[key]

    def _denorm_xy(self, xy_n):
        return xy_n * self._xy_s + self._xy_m

    def _norm_geom(self, geom):
        g = torch.as_tensor(geom, device=self.device, dtype=torch.float32)
        if "obj_geom" in self.st_mean:
            g = self._normalize("obj_geom", g)
        return g

    def encode_obs(self, obs: dict):
        """Encode obs with every ensemble member. Returns list of (1, N, L)."""
        with torch.inference_mode():
            states = self._normalize(
                "obj_states", torch.as_tensor(obs["obj_states"], device=self.device))
            cfeat = self._normalize(
                "contact_feat", torch.as_tensor(obs["contact_feat"], device=self.device))
            cmask = torch.as_tensor(obs["contact_mask"], device=self.device)
            tsum = self._normalize(
                "tactile_summary", torch.as_tensor(obs["tactile_summary"], device=self.device))
            types = torch.as_tensor(obs["obj_types"], device=self.device)
            geom = obs.get("obj_geom")
            geom_t = None
            if geom is not None:
                g = torch.as_tensor(geom, device=self.device, dtype=torch.float32)
                if "obj_geom" in self.st_mean:
                    g = self._normalize("obj_geom", g)
                geom_t = g.unsqueeze(0)
            return [m.encode(types.unsqueeze(0), states.unsqueeze(0),
                            cfeat.unsqueeze(0), cmask.unsqueeze(0),
                            tsum.unsqueeze(0), obj_geom=geom_t)
                    for m in self.models]

    def _zero_xy(self, model, z0, geom_n):
        """Zero-action imagined xy (H, N, 2) in raw units."""
        a0 = self._normalize("actions", torch.zeros(1, 2, device=self.device))
        z, h = z0[:1], model.predictor.init_hidden(z0[:1])
        prev_c = model.pair_prob(z)
        frames = []
        for _ in range(self.H):
            z, h, _ = model.predictor.step(z, a0, h, geom=geom_n,
                                           prev_contact=prev_c)
            prev_c = model.pair_prob(z)
            frames.append(self._denorm_xy(model.predictor.xy_head(z))[0])
        return torch.stack(frames)

    def _roll(self, model, z0, a, base, now_xy, geom_n, cost):
        """Imagine ``a`` and accumulate ``cost.step`` / ``cost.terminal``."""
        z = z0.expand(self.P, *z0.shape[1:])
        h = model.predictor.init_hidden(z)
        total = a.new_zeros(self.P)
        prev = None
        prev_c = model.pair_prob(z)
        xy = None
        for t in range(self.H):
            z, h, _ = model.predictor.step(
                z, self._normalize("actions", a[:, t]), h, geom=geom_n,
                prev_contact=prev_c)
            prev_c = model.pair_prob(z)
            xy = self._denorm_xy(model.predictor.xy_head(z))
            xy = xy - base[t] + now_xy
            total = total + cost.step(xy, a[:, t], prev)
            prev = xy
        return total + cost.terminal(xy)

    def plan(self, z0, geom: np.ndarray, states_now: np.ndarray, cost):
        """Optimize an action sequence. Returns best first action (2,) in N."""
        z0s = [z0] if isinstance(z0, torch.Tensor) else list(z0)
        now_xy = torch.as_tensor(states_now[:, :2], device=self.device,
                                 dtype=torch.float32)
        geom_n = self._norm_geom(geom)
        with torch.inference_mode():
            bases = [self._zero_xy(m, z, geom_n) for m, z in zip(self.models, z0s)]
            best_first = None
            for _ in range(self.iters):
                noise = torch.randn(self.P, self.H, 2, device=self.device)
                seqs = (self.mean + self.std * noise).clamp(self.low, self.high)
                costs = [self._roll(m, z, seqs, b, now_xy, geom_n, cost)
                         for m, z, b in zip(self.models, z0s, bases)]
                C = torch.stack(costs, 0)
                total = C.mean(0)
                if C.shape[0] > 1:
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
