"""PushToGoal task + CEM (cross-entropy method) latent-space planner.

Task: using ONLY the learned world model for planning, apply forces to
the end-effector so that a designated target object ends up within a
tolerance of a goal position. Execution and success measurement happen
in the FlatWorld simulator (the world model never sees future states).

Planner: receding-horizon CEM.
- encode the TRUE current observation -> z0
- sample P action sequences of horizon H, roll the latent dynamics,
  decode predicted states and score a cost
- keep elites, refit Gaussian mean/std, iterate; execute the first action
  of the best sequence, then replan (warm-start by shifting the mean)
"""

import glob
import os

import numpy as np
import torch

from learning.data.normalizer import Normalizer
from learning.env.flatworld_wrapper import PushSceneEnv, OBJ_TYPE_EE
from learning.models.lewm import StateLeWM


def _load_one(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = StateLeWM(n_obj=ckpt["n_obj"], latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm_name = ckpt.get("normalizer", "normalizer.json")
    # ens_0.pt / best.pt / last.pt all sit next to normalizer.json
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path)) or "."
    norm = Normalizer.load(os.path.join(ckpt_dir, os.path.basename(norm_name)))
    stride = int(ckpt.get("stride", 1))
    return model, norm, stride


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load one world model. Kept for debug scripts."""
    return _load_one(ckpt_path, device)


def load_ensemble(ckpt_path: str, device: str = "cpu"):
    """Load an ensemble.

    ``ckpt_path`` may be a file (single model) or a directory containing
    ``ens_*.pt`` members (excluding ``*_last.pt``).
    """
    if os.path.isdir(ckpt_path):
        files = sorted(
            f for f in glob.glob(os.path.join(ckpt_path, "ens_*.pt"))
            if not f.endswith("_last.pt")
        )
        if not files:
            fallback = os.path.join(ckpt_path, "last.pt")
            if not os.path.exists(fallback):
                fallback = os.path.join(ckpt_path, "best.pt")
            files = [fallback]
    else:
        files = [ckpt_path]
    models, norm, stride = [], None, 1
    for f in files:
        m, n, s = _load_one(f, device)
        models.append(m)
        norm, stride = n, s
    print(f"loaded {len(models)} world model(s) from {ckpt_path}")
    return models, norm, stride


class CEMPlanner:
    def __init__(self, models, norm: Normalizer, device: str = "cpu",
                 horizon: int = 8, population: int = 96, elites: int = 16,
                 iterations: int = 4, force_max: float = 6.0,
                 action_cost: float = 0.002, approach_cost: float = 0.5,
                 align_cost: float = 1.0, uncert_cost: float = 0.1,
                 contact_cost: float = 0.2, ride_cost: float = 2.5,
                 blocker_cost: float = 2.0, seed: int = 0):
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
        self.action_cost = action_cost
        self.approach_cost = approach_cost
        self.align_cost = align_cost
        self.uncert_cost = uncert_cost
        self.contact_cost = contact_cost
        self.ride_cost = ride_cost
        self.blocker_cost = blocker_cost
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((horizon, 2), dtype=np.float32)
        # +Fy is set per plan() from the gap to contact_y so tall targets
        # can be reached without mounting a neighbour.
        self.low = np.array([-force_max, -force_max], dtype=np.float32)
        self.high = np.array([force_max, force_max], dtype=np.float32)
        self.std = (self.high - self.low) * 0.25
        st = norm.to_tensors(device)
        self.st_mean = {k: v[0] for k, v in st.items()}
        self.st_std = {k: v[1] for k, v in st.items()}

    def reset(self):
        self.mean[:] = 0.0
        self.std[:] = (self.high - self.low) * 0.25

    def _normalize(self, key, x):
        return (x - self.st_mean[key]) / self.st_std[key]

    def _denormalize(self, key, x):
        return x * self.st_std[key] + self.st_mean[key]

    def encode_obs(self, obs: dict):
        """Encode obs with every ensemble member. Returns list of (1, L) tensors."""
        with torch.no_grad():
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
            zs = []
            for m in self.models:
                zs.append(m.encode(types.unsqueeze(0), states.unsqueeze(0),
                                   cfeat.unsqueeze(0), cmask.unsqueeze(0),
                                   tsum.unsqueeze(0), obj_geom=geom_t))
        return zs

    def _zero_baseline(self, model, z0):
        """Zero-action imagined states (H, N, 6) in raw units."""
        a0 = self._normalize("actions", torch.zeros(1, 2, device=self.device))
        zb = z0[:1]
        hb = model.predictor.init_hidden(zb)
        base = []
        for _ in range(self.H):
            zb, hb = model.predictor.step(zb, a0, hb)
            base.append(self._denormalize(
                "obj_states", model.decoder(zb)[0])[0])
        return torch.stack(base)

    def _ride_penalty(self, eff, ee_idx, target_idx, geom, ee_r):
        """Penalise sitting on a non-target top face (side contact lost)."""
        ee_x, ee_y = eff[:, ee_idx, 0], eff[:, ee_idx, 1]
        n = eff.shape[1]
        ride = torch.zeros(eff.shape[0], device=eff.device)
        for j in range(1, n):
            if j == target_idx:
                continue
            hw, hh = geom[j, 0], geom[j, 1]
            overlap = (ee_r + hw) - (ee_x - eff[:, j, 0]).abs()
            above = ee_y - (eff[:, j, 1] + hh + 0.02)
            ride = ride + torch.relu(above) * (overlap > 0).float()
        return ride

    def _block_penalty(self, eff, ee_idx, target_idx, geom):
        """Occupancy of the EE → target corridor: overlap on the target's
        left face plus leftover distance the blocker's left edge still
        needs to travel past that face."""
        ee_x = eff[:, ee_idx, 0]
        tx = eff[:, target_idx, 0]
        thw = geom[target_idx, 0]
        tgt_left = tx - thw
        n = eff.shape[1]
        pen = torch.zeros(eff.shape[0], device=eff.device)
        for j in range(1, n):
            if j == target_idx:
                continue
            xj = eff[:, j, 0]
            hw = geom[j, 0]
            left, right = xj - hw, xj + hw
            in_path = (right > ee_x) & (left < tx)
            overlap = torch.relu(right - tgt_left)
            remaining = torch.relu((tgt_left + 0.03) - left)
            pen = pen + (overlap + 0.4 * remaining) * in_path.float()
        return pen

    def _shove_penalty(self, eff, now_t, ee_idx, target_idx, shove_dx):
        """In-horizon cost: objects currently between EE and target should
        move right by ``shove_dx`` (gives CEM a gradient within H steps)."""
        if shove_dx <= 0.0:
            return torch.zeros(eff.shape[0], device=eff.device)
        ee_x0 = float(now_t[ee_idx, 0])
        tx0 = float(now_t[target_idx, 0])
        pen = torch.zeros(eff.shape[0], device=eff.device)
        for j in range(1, eff.shape[1]):
            if j == target_idx:
                continue
            x0 = float(now_t[j, 0])
            if x0 <= ee_x0 - 0.02 or x0 >= tx0 + 0.02:
                continue
            pen = pen + torch.relu((x0 + shove_dx) - eff[:, j, 0])
        return pen

    def _member_cost(self, model, z0, a, base, now_t, ee_idx, target_idx,
                     goal_x, tgt_y, contact_x, standoff, slip, coast_frac,
                     tgt_x0, geom, ee_r, coast_max, front_idx,
                     clear_dx, target_weight, shove_weight):
        """Per-member CEM cost (P,) plus EE-x / target-x trajectories for std."""
        z = z0.expand(self.P, -1)
        h = model.predictor.init_hidden(z)
        cost = torch.zeros(self.P, device=self.device)
        pushed = torch.full((self.P,), float(tgt_x0), device=self.device)
        clearing = front_idx != target_idx
        push_goal = (tgt_x0 + clear_dx) if clearing else goal_x
        ee_xs, tgt_xs = [], []
        p_c_last = torch.zeros(self.P, device=self.device)
        for t in range(self.H):
            a_n = self._normalize("actions", a[:, t])
            z, h = model.predictor.step(z, a_n, h)
            states = self._denormalize("obj_states", model.decoder(z)[0])
            eff = states - base[t].unsqueeze(0) + now_t.unsqueeze(0)
            ee_x, ee_y = eff[:, ee_idx, 0], eff[:, ee_idx, 1]
            p_c = torch.sigmoid(model.contact_logit(z))
            p_c_last = p_c
            d_app = torch.relu(contact_x - ee_x)
            pushed = torch.maximum(pushed, ee_x + standoff - slip)
            proj = pushed + torch.clamp(
                coast_frac * (pushed - tgt_x0), max=coast_max)
            d_goal = (proj - push_goal).abs()
            if clearing:
                d_goal = d_goal + target_weight * (
                    eff[:, target_idx, 0] - goal_x).abs()
            d_align = (ee_y - tgt_y).abs()
            d_ride = self._ride_penalty(eff, ee_idx, target_idx, geom, ee_r)
            d_block = self._block_penalty(eff, ee_idx, target_idx, geom)
            d_shove = self._shove_penalty(
                eff, now_t, ee_idx, target_idx, clear_dx)
            cost = cost + d_goal + self.approach_cost * d_app \
                + self.align_cost * d_align \
                + self.ride_cost * d_ride \
                + self.blocker_cost * d_block \
                + shove_weight * d_shove \
                + self.contact_cost * d_app * (1.0 - p_c) \
                + self.action_cost * (a[:, t] ** 2).mean(dim=-1)
            ee_xs.append(ee_x)
            tgt_xs.append(eff[:, target_idx, 0])
        proj = pushed + torch.clamp(
            coast_frac * (pushed - tgt_x0), max=coast_max)
        term = 2.0 * (proj - push_goal).abs()
        if clearing:
            term = term + 2.0 * target_weight * (
                eff[:, target_idx, 0] - goal_x).abs()
        cost = cost + term \
            + self.approach_cost * torch.relu(contact_x - ee_xs[-1]) \
            + self.contact_cost * torch.relu(contact_x - ee_xs[-1]) * (1.0 - p_c_last) \
            + self.ride_cost * self._ride_penalty(
                eff, ee_idx, target_idx, geom, ee_r) \
            + shove_weight * self._shove_penalty(
                eff, now_t, ee_idx, target_idx, clear_dx)
        return cost, torch.stack(ee_xs, 0), torch.stack(tgt_xs, 0)

    def plan(self, z0, ee_idx: int, target_idx: int,
             goal: np.ndarray, standoff: float = 0.2, contact_y: float = 0.105,
             slip: float = 0.0, coast_frac: float = 0.0,
             force_cap: float = None, states_now: np.ndarray = None,
             geom: np.ndarray = None, coast_max: float = 0.06,
             front_idx: int = None, clear_dx: float = 0.0,
             target_weight: float = 1.0, shove_weight: float = 0.0):
        """Optimize an action sequence. Returns best first action (2,) in N.

        Ensemble: each member is rolled out independently. CEM minimises
        mean cost + uncert_cost * std(cost), so sequences the models
        disagree on (hallucinated telekinesis / contact) are penalised.
        """
        if isinstance(z0, torch.Tensor):
            z0s = [z0]
        else:
            z0s = list(z0)
        if front_idx is None:
            front_idx = target_idx
        goal_x = float(goal[0])
        tgt_y = contact_y

        with torch.no_grad():
            bases = [self._zero_baseline(m, z) for m, z in zip(self.models, z0s)]
        if states_now is not None:
            now_t = torch.as_tensor(states_now, dtype=torch.float32,
                                    device=self.device)
        else:
            now_t = bases[0][0]
        n_obj = int(now_t.shape[0])
        if geom is None:
            geom_t = torch.ones(n_obj, 2, device=self.device) * 0.08
        else:
            geom_t = torch.as_tensor(geom, dtype=torch.float32, device=self.device)
        ee_r = float(geom_t[0, 0])
        tgt_x0 = float(now_t[front_idx, 0])
        contact_x = tgt_x0 - standoff
        best_seq = None

        ee_y_now = float(now_t[ee_idx, 1])
        fy_up = float(np.clip(0.8 + 15.0 * (contact_y - ee_y_now), 0.3, 2.5))
        fx_cap = self.fmax if force_cap is None else float(force_cap)
        low = np.array([-fx_cap, self.low[1]], dtype=np.float32)
        high = np.array([fx_cap, fy_up], dtype=np.float32)

        for _ in range(self.iters):
            seqs = self.rng.normal(size=(self.P, self.H, 2)).astype(np.float32)
            seqs = self.mean[None] + self.std[None] * seqs
            seqs = np.clip(seqs, low, high)
            a = torch.as_tensor(seqs, device=self.device)

            with torch.no_grad():
                member_costs = []
                for model, z, base in zip(self.models, z0s, bases):
                    c, _, _ = self._member_cost(
                        model, z, a, base, now_t, ee_idx, target_idx,
                        goal_x, tgt_y, contact_x, standoff, slip, coast_frac,
                        tgt_x0, geom_t, ee_r, coast_max, front_idx,
                        clear_dx, target_weight, shove_weight)
                    member_costs.append(c)
                C = torch.stack(member_costs, 0)          # (M, P)
                cost = C.mean(dim=0)
                if C.shape[0] > 1:
                    cost = cost + self.uncert_cost * C.std(dim=0)

            idx = torch.argsort(cost)[: self.E]
            elite = seqs[idx.cpu().numpy()]
            self.mean = elite.mean(axis=0)
            self.std = elite.std(axis=0) + 1e-3
            best_seq = elite[0]

        self.mean[:-1] = self.mean[1:]
        self.mean[-1] = 0.0
        self.std = np.maximum(self.std * 0.9, (self.high - self.low) * 0.05)
        return best_seq[0].astype(np.float32)


class PushToGoalTask:
    """Push a target object to a goal position with the learned world model."""

    def __init__(self, cfg, model, norm, device="cpu",
                 tol: float = 0.08, budget: int = 150, planner_kwargs: dict = None,
                 stride: int = 1, target_mode: str = "leftmost"):
        self.cfg = cfg
        self.env = PushSceneEnv(cfg)
        self.models = list(model) if isinstance(model, (list, tuple)) else [model]
        self.model = self.models[0]
        self.device = device
        self.tol = tol
        self.budget = budget              # in simulation frames
        self.stride = stride              # frames per planned action (action repeat)
        self.target_mode = target_mode    # "leftmost" | "random"
        self.planner = CEMPlanner(self.models, norm, device=device,
                                  force_max=cfg.collect.force_max,
                                  **(planner_kwargs or {}))
        self.target_idx = None
        self.ee_idx = 0

    def _geom(self) -> np.ndarray:
        if getattr(self.env, "obj_geom", None) is not None:
            return np.asarray(self.env.obj_geom, dtype=np.float32).copy()
        sc = self.cfg.scene
        g = np.zeros((self.env.n_obj, 2), dtype=np.float32)
        g[0] = (sc.ee_radius, sc.ee_radius)
        for i, t in enumerate(self.env.obj_types):
            if i == 0:
                continue
            if int(t) == 1:
                g[i] = sc.box_ext
            else:
                g[i] = (sc.ball_radius, sc.ball_radius)
        return g

    def _half_width(self, idx: int) -> float:
        return float(self._geom()[idx, 0])

    def _half_height(self, idx: int) -> float:
        return float(self._geom()[idx, 1])

    def _front_idx(self, states, target_idx: int) -> int:
        """Nearest object on the EE → target x-interval (inclusive)."""
        ee_x = float(states[0, 0])
        tx = float(states[target_idx, 0])
        lo, hi = min(ee_x, tx), max(ee_x, tx)
        best, best_x = target_idx, tx
        for j in range(1, len(states)):
            x = float(states[j, 0])
            if lo - 0.03 <= x <= hi + 0.03 and x < best_x - 1e-4:
                best, best_x = j, x
        return best

    def _is_riding(self, states, target_idx: int) -> bool:
        geom = self._geom()
        ee_r = float(geom[0, 0])
        ee_x, ee_y = float(states[0, 0]), float(states[0, 1])
        for j in range(1, len(states)):
            if j == target_idx:
                continue
            hw, hh = float(geom[j, 0]), float(geom[j, 1])
            top = float(states[j, 1]) + hh
            overlap = (ee_r + hw) - abs(ee_x - float(states[j, 0]))
            if ee_y > top + 0.02 and overlap > 0.0:
                return True
        return False

    def _free_run(self, obs, target_idx: int) -> float:
        """How far the target center can travel right before its surface
        touches the next object ahead (center gap minus both half widths)."""
        states = obs["obj_states"]
        tx = float(states[target_idx, 0])
        hw = self._half_width(target_idx)
        ahead = [(float(states[j, 0]), self._half_width(j))
                 for j in range(1, len(states))
                 if j != target_idx and states[j, 0] > tx + 0.05]
        if ahead:
            nx, nhw = min(ahead)
            return max(nx - tx - hw - nhw, 0.0)
        return 0.6

    def _pick_target(self, obs, rng: np.random.Generator = None) -> int:
        n = len(obs["obj_states"]) - 1
        if self.target_mode == "random" and rng is not None:
            return 1 + int(rng.integers(0, n))
        xs = obs["obj_states"][1:, 0]
        return 1 + int(np.argmin(xs))

    def _sample_goal(self, rng, obs, target_idx):
        """Goal to the RIGHT of the target."""
        states = obs["obj_states"]
        target_x = float(states[target_idx, 0])
        if self.target_mode == "random":
            dist = float(rng.uniform(0.12, 0.28))
        else:
            # floor 0.10 > tol 0.08 so the episode never starts successful
            dist = float(np.clip(self._free_run(obs, target_idx) - 0.02, 0.10, 0.30))
        goal_x = target_x + dist
        goal_y = float(states[target_idx, 1])
        return np.array([goal_x, goal_y], dtype=np.float32)

    def _hold(self, action, t, frames):
        obs = None
        for _ in range(self.stride):
            obs = self.env.step(action)
            t += 1
            if t >= self.budget:
                break
        if frames is not None:
            frames["obj_states"].append(obs["obj_states"])
            frames["contact_mask"].append(obs["contact_mask"])
            frames["actions"].append(np.asarray(action, dtype=np.float32))
        return obs, t

    def _slide_force(self, idx: int) -> float:
        """Approximate Coulomb slide force μ m g for object slot ``idx``."""
        m = float(self.env.obj_mass[idx])
        mu = float(self.env.obj_mu[idx])
        return mu * m * float(self.cfg.scene.gravity)

    def _path_slide_force(self, states, target_idx: int) -> float:
        """Sum of μmg for objects still on the EE → target x-interval."""
        ee_x = float(states[0, 0])
        tx = float(states[target_idx, 0])
        lo, hi = min(ee_x, tx) - 0.04, max(ee_x, tx) + 0.04
        total = 0.0
        for j in range(1, len(states)):
            if lo <= float(states[j, 0]) <= hi:
                total += self._slide_force(j)
        return total

    def _clear_action(self, cur, front: int) -> np.ndarray:
        """Drive into the object in front of the EE and shove it right.

        Used while ``front != target``: the world model is unreliable on
        multi-object contact, so clearing is a mass-aware open-loop push
        at the front object's mid-height rather than a CEM guess.
        """
        geom = self._geom()
        ee = cur["obj_states"][0]
        fr = cur["obj_states"][front]
        standoff = self._half_width(front) + float(geom[0, 0])
        contact_y = max(float(fr[1]), float(geom[0, 1]) + 0.005)
        need = self._path_slide_force(cur["obj_states"], self.target_idx)
        fcap = float(np.clip(1.2 * need, 4.0, 8.5))
        self.env.force_cap_override = fcap
        fx = fcap
        if float(ee[0]) < float(fr[0]) - standoff - 0.05:
            fx = min(fcap, 5.0)
        fy = float(np.clip(12.0 * (contact_y - float(ee[1])), -2.5, 2.2))
        return np.array([fx, fy], dtype=np.float32)

    def _plan_action(self, cur, goal):
        geom = self._geom()
        front = self._front_idx(cur["obj_states"], self.target_idx)
        blocked = front != self.target_idx
        standoff = self._half_width(front) + float(geom[0, 0])
        ee_floor = float(geom[0, 1]) + 0.005
        # Align to the object we are actually contacting (front), not the
        # (possibly different-height) designated target.
        contact_y = max(float(cur["obj_states"][front, 1]), ee_floor)
        need = self._slide_force(front)
        if blocked:
            slip, coast, coast_max = 0.02, 0.05, 0.04
            fcap = float(np.clip(1.25 * need, 2.8, 5.5))
            clear_dx, target_w, shove_w = 0.16, 0.12, 2.2
        elif int(self.env.obj_types[front]) == 1:
            slip, coast, coast_max = 0.02, 0.1, 0.06
            fcap = float(np.clip(1.05 * need, 2.2, 4.0))
            clear_dx, target_w, shove_w = 0.0, 1.0, 0.0
        else:
            slip, coast, coast_max = 0.0, 0.35, 0.06
            fcap = float(np.clip(1.05 * need, 2.4, 4.2))
            clear_dx, target_w, shove_w = 0.0, 1.0, 0.0
        z0 = self.planner.encode_obs(cur)
        return self.planner.plan(
            z0, self.ee_idx, self.target_idx, goal, standoff, contact_y,
            slip, coast, fcap, states_now=cur["obj_states"],
            geom=geom, coast_max=coast_max, front_idx=front,
            clear_dx=clear_dx, target_weight=target_w, shove_weight=shove_w)

    def run_episode(self, rng: np.random.Generator, record: bool = False):
        obs = self.env.reset(rng)
        self.target_idx = self._pick_target(obs, rng)
        goal = self._sample_goal(rng, obs, self.target_idx)
        target_pos = obs["obj_states"][self.target_idx, :2]
        self.planner.reset()

        frames = {"obj_states": [obs["obj_states"]],
                  "contact_mask": [obs["contact_mask"]],
                  "actions": []} if record else None
        init_dist = float(np.linalg.norm(target_pos - goal))
        final_dist = init_dist
        success = False
        settle_frame = self.budget
        stall_count = 0
        t = 0

        while t < self.budget:
            cur = self.env._observe()
            if "obj_geom" not in cur:
                cur = dict(cur)
                cur["obj_geom"] = self._geom()
            target_pos = cur["obj_states"][self.target_idx, :2]
            target_vel = float(np.linalg.norm(cur["obj_states"][self.target_idx, 3:5]))
            d = float(np.linalg.norm(target_pos - goal))
            final_dist = d
            if d < self.tol and target_vel < 0.08:
                success = True
                settle_frame = t
                self.env.set_force((0.0, 0.0))
                break
            if self._is_riding(cur["obj_states"], self.target_idx):
                obs, t = self._hold(np.array([1.0, -3.5], dtype=np.float32),
                                    t, frames)
                stall_count = 0
                continue
            front = self._front_idx(cur["obj_states"], self.target_idx)
            blocked = front != self.target_idx
            self.env.force_cap_override = None
            is_box = int(self.env.obj_types[self.target_idx]) == 1
            if not blocked:
                if is_box:
                    if target_vel > 0.25 or d < self.tol:
                        obs, t = self._hold(np.zeros(2, dtype=np.float32), t, frames)
                        stall_count = 0
                        continue
                elif d < 0.75 * self.tol:
                    obs, t = self._hold(np.zeros(2, dtype=np.float32), t, frames)
                    stall_count = 0
                    continue

            if blocked:
                if float(target_pos[0]) >= float(goal[0]) - 0.01:
                    obs, t = self._hold(np.zeros(2, dtype=np.float32), t, frames)
                    stall_count = 0
                    continue
                obs, t = self._hold(self._clear_action(cur, front), t, frames)
                stall_count = 0
                continue
            else:
                if d > self.tol and target_vel < 0.05:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= 2 and float(target_pos[0]) < float(goal[0]) - 0.015:
                    fx = float(self.planner.rng.uniform(1.5, 2.2))
                    obs, t = self._hold(np.array([fx, 0.0], dtype=np.float32),
                                        t, frames)
                    continue

            action = self._plan_action(cur, goal)
            obs, t = self._hold(action, t, frames)

        result = {
            "success": bool(success),
            "target_idx": int(self.target_idx),
            "init_dist": init_dist,
            "final_dist": final_dist,
            "settle_frame": settle_frame,
            "goal": goal.tolist(),
            "frames": (np.stack(frames["obj_states"]),
                       np.stack(frames["contact_mask"]),
                       np.stack(frames["actions"])) if record else None,
        }
        return result
