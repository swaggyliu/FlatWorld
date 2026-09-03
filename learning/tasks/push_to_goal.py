"""PushToGoal task and its imagined CEM cost.

Task: using ONLY the learned world model for planning, apply forces to
the end-effector so that a designated target object ends up within a
tolerance of a goal position. Execution and success measurement happen
in the FlatWorld simulator (the world model never sees future states).

The CEM loop lives in ``learning.planner``. Geometric helpers (contact
face) shape the *cost*; they do not emit the executed force.
Cost is scored on residual-corrected ``xy_head`` coordinates.
"""

import glob
import os

import numpy as np
import torch

from learning.data.normalizer import Normalizer
from learning.env.flatworld_wrapper import PushSceneEnv
from learning.models.lewm import StateLeWM
from learning.planner import CEMPlanner


def _load_one(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = StateLeWM(
        n_obj=ckpt["n_obj"], latent_dim=ckpt["latent_dim"],
        n_mp=int(ckpt.get("n_mp", 3)),
        tactile_drop_prob=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.tactile_residual = bool(ckpt.get("tactile_residual", True))
    model.eval()
    norm_name = ckpt.get("normalizer", "normalizer.json")
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path)) or "."
    norm = Normalizer.load(os.path.join(ckpt_dir, os.path.basename(norm_name)))
    stride = int(ckpt.get("stride", 1))
    return model, norm, stride


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load one world model. Kept for debug scripts."""
    return _load_one(ckpt_path, device)


def load_ensemble(ckpt_path, device: str = "cpu"):
    """Load an ensemble.

    ``ckpt_path`` may be a file, a directory containing ``ens_*.pt``
    (excluding ``*_last.pt``), or a list of files/directories. Each
    member keeps the normalizer from its own checkpoint directory so
    models trained on different datasets can be mixed.
    """
    paths = list(ckpt_path) if isinstance(ckpt_path, (list, tuple)) else [ckpt_path]
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(
                f for f in glob.glob(os.path.join(p, "ens_*.pt"))
                if not f.endswith("_last.pt")
            )
            if not found:
                fallback = os.path.join(p, "last.pt")
                if not os.path.exists(fallback):
                    fallback = os.path.join(p, "best.pt")
                found = [fallback]
            files.extend(found)
        else:
            files.append(p)
    models, norm, stride = [], None, 1
    for f in files:
        m, n, s = _load_one(f, device)
        m.norm = n
        models.append(m)
        if norm is None:
            norm, stride = n, s
    print(f"loaded {len(models)} world model(s) from {ckpt_path}")
    return models, norm, stride


_COST_KEYS = ("action_cost", "reach_cost", "settle_w")


class PushToGoalCost:
    """Decoded-xy cost for PushToGoal.

    Distance to goal plus a contact-face / catch-bumper term.
    Push face is the original behind-the-target side (sign frozen at
    episode start). After the target overshoots or is rolling toward
    the goal, the face switches to a world-fixed bumper at the goal
    instead of the far side of the ball (that side is usually jammed
    against the pile).
    """

    def __init__(self, action_cost: float = 0.0005, reach_cost: float = 0.35,
                 settle_w: float = 0.0, device: str = "cpu"):
        self.action_cost = action_cost
        self.reach_cost = reach_cost
        self.settle_w = settle_w
        self.device = device
        self.ee_idx = 0
        self.target_idx = 1
        self.goal_t = None
        self.face_t = None
        self.push_sign = 1.0
        self.standoff = 0.0
        self.tol = 0.08
        self.catch = False

    def bind(self, goal, states_now, geom, ee_idx: int, target_idx: int,
             min_gap: float = 0.02, push_sign: float = None, catch: bool = False):
        self.ee_idx = ee_idx
        self.target_idx = target_idx
        self.catch = bool(catch)
        g = torch.as_tensor(geom, device=self.device, dtype=torch.float32)
        hw = g[:, 0].clamp(min=1e-4)
        self.standoff = float(hw[target_idx] + hw[ee_idx])
        self.goal_t = torch.as_tensor(goal[:2], device=self.device, dtype=torch.float32)
        if push_sign is None:
            self.push_sign = 1.0 if float(goal[0]) >= float(states_now[target_idx, 0]) else -1.0
        else:
            self.push_sign = float(push_sign)
        st = np.asarray(states_now, dtype=np.float64)
        gm = np.asarray(geom, dtype=np.float64)
        gx, gy = float(goal[0]), float(goal[1])
        tx, ty = float(st[target_idx, 0]), float(st[target_idx, 1])
        r_ee = float(gm[ee_idx, 0])
        if self.catch:
            fx = gx + self.push_sign * (r_ee + 0.01)
            fy = max(float(gm[ee_idx, 1]) + 0.01, ty)
            for j in range(len(st)):
                if j == ee_idx:
                    continue
                gap_x = abs(float(st[j, 0]) - fx) - (float(gm[j, 0]) + r_ee)
                if gap_x < float(min_gap):
                    fy = max(fy, float(st[j, 1]) + float(gm[j, 1]) + r_ee + 0.02)
        else:
            fx = tx - self.push_sign * self.standoff
            fy = max(ty, float(gm[ee_idx, 1]) + 0.005)
        self.face_t = torch.tensor([fx, fy], device=self.device, dtype=torch.float32)

    def step(self, xy, a_t, prev=None):
        d_goal = (xy[:, self.target_idx] - self.goal_t).norm(dim=-1)
        d_face = (xy[:, self.ee_idx] - self.face_t).norm(dim=-1)
        cost = d_goal + self.reach_cost * d_face \
            + self.action_cost * (a_t ** 2).mean(dim=-1)
        if prev is not None and self.settle_w > 0:
            speed = (xy[:, self.target_idx] - prev[:, self.target_idx]).norm(dim=-1)
            in_tol = (self.tol - d_goal).clamp(min=0.0) / max(self.tol, 1e-6)
            cost = cost + self.settle_w * in_tol * speed
        return cost

    def terminal(self, xy):
        return 2.0 * (xy[:, self.target_idx] - self.goal_t).norm(dim=-1)


class PushToGoalTask:
    """Push a target object to a goal position with the learned world model."""

    def __init__(self, cfg, model, norm, device="cpu",
                 tol: float = None, vel_tol: float = None, budget: int = None,
                 planner_kwargs: dict = None,
                 stride: int = 1, target_mode: str = "leftmost",
                 ee_scale: float = 1.0, **_ignore):
        self.cfg = cfg
        self.env = PushSceneEnv(cfg)
        self.models = list(model) if isinstance(model, (list, tuple)) else [model]
        self.model = self.models[0] if self.models else None
        self.device = device
        tc = getattr(cfg, "task", None)
        self.tol = float(tc.tol if tol is None and tc is not None else (tol if tol is not None else 0.08))
        self.vel_tol = float(tc.vel_tol if vel_tol is None and tc is not None else (vel_tol if vel_tol is not None else 0.08))
        self.budget = int(tc.budget if budget is None and tc is not None else (budget if budget is not None else 400))
        self.stride = stride
        self.target_mode = target_mode
        self.ee_scale = float(ee_scale)
        pk = dict(planner_kwargs or {})
        cost_kw = {k: pk.pop(k) for k in _COST_KEYS if k in pk}
        if "tactile_residual" not in pk:
            pk["tactile_residual"] = bool(
                getattr(self.models[0], "tactile_residual", False))
        self.cost = PushToGoalCost(device=device, **cost_kw)
        self.cost.tol = self.tol
        self.planner = CEMPlanner(self.models, norm, device=device,
                                  force_max=cfg.collect.force_max, **pk)
        self.target_idx = None
        self.ee_idx = 0
        self.push_sign = 1.0

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

    def _workspace_x(self, idx: int):
        """Allowed center-x for object ``idx`` from scene / collect bounds."""
        sc, c = self.cfg.scene, self.cfg.collect
        hw = self._half_width(idx)
        margin = 0.05
        lo = max(float(sc.area_x[0]), float(c.x_lo)) + hw + margin
        hi = min(float(sc.area_x[1]), float(c.x_hi)) - hw - margin
        if hi < lo:
            lo, hi = float(c.x_lo) + hw, float(c.x_hi) - hw
        return lo, hi

    def _free_run(self, obs, target_idx: int, sign: float = 1.0) -> float:
        """How far the target center can travel in ``sign`` before contact
        or the workspace wall."""
        states = obs["obj_states"]
        tx = float(states[target_idx, 0])
        hw = self._half_width(target_idx)
        s = 1.0 if sign >= 0.0 else -1.0
        ahead = []
        for j in range(1, len(states)):
            if j == target_idx:
                continue
            xj = float(states[j, 0])
            if s * (xj - tx) > 0.5 * (hw + self._half_width(j)):
                ahead.append((xj, self._half_width(j)))
        if ahead:
            if s >= 0.0:
                nx, nhw = min(ahead)
                blocked = max(nx - tx - hw - nhw, 0.0)
            else:
                nx, nhw = max(ahead, key=lambda p: p[0])
                blocked = max(tx - nx - hw - nhw, 0.0)
        else:
            blocked = 1e6
        x_lo, x_hi = self._workspace_x(target_idx)
        wall = (x_hi - tx) if s >= 0.0 else (tx - x_lo)
        return float(min(blocked, max(wall, 0.0)))

    def _pick_target(self, obs, rng: np.random.Generator = None) -> int:
        n = len(obs["obj_states"]) - 1
        if self.target_mode == "random" and rng is not None:
            return 1 + int(rng.integers(0, n))
        xs = obs["obj_states"][1:, 0]
        if self.target_mode == "rightmost":
            return 1 + int(np.argmax(xs))
        return 1 + int(np.argmin(xs))

    def _goal_occupied(self, states, goal_x, target_idx) -> bool:
        gap = float(self.cfg.scene.min_gap)
        for j in range(1, len(states)):
            if j == target_idx:
                continue
            need = gap + self._half_width(j)
            if abs(float(states[j, 0]) - float(goal_x)) < need:
                return True
        return False

    def _sample_goal(self, rng, obs, target_idx):
        """Sample a left or right goal when both sides have room."""
        tc = getattr(self.cfg, "task", None)
        min_d = float(tc.goal_min) if tc is not None else (self.tol + 0.02)
        max_d = float(tc.goal_max) if tc is not None else 0.30
        states = obs["obj_states"]
        target_x = float(states[target_idx, 0])
        x_lo, x_hi = self._workspace_x(target_idx)
        candidates = []
        for sign in (1.0, -1.0):
            dist = float(np.clip(self._free_run(obs, target_idx, sign), min_d, max_d))
            gx = float(np.clip(target_x + sign * dist, x_lo, x_hi))
            if abs(gx - target_x) >= min_d and not self._goal_occupied(states, gx, target_idx):
                candidates.append(gx)
        if not candidates:
            sign = 1.0 if (x_hi - target_x) >= (target_x - x_lo) else -1.0
            gx = float(np.clip(target_x + sign * min_d, x_lo, x_hi))
            candidates.append(gx)
        goal_x = float(candidates[int(rng.integers(0, len(candidates)))])
        goal_y = float(states[target_idx, 1])
        return np.array([goal_x, goal_y], dtype=np.float32)

    def _succeeded(self, states, goal) -> bool:
        tgt = states[self.target_idx]
        d = float(np.linalg.norm(tgt[:2] - goal[:2]))
        vel = float(np.linalg.norm(tgt[3:5]))
        return d < self.tol and vel < self.vel_tol

    def _hold(self, action, t, frames):
        obs = None
        for _ in range(self.stride):
            obs = self.env.step(action)
            t += 1
            if frames is not None:
                frames["obj_states"].append(obs["obj_states"])
                frames["contact_mask"].append(obs["contact_mask"])
                frames["actions"].append(np.asarray(action, dtype=np.float32))
            if t >= self.budget:
                break
        return obs, t

    def _observe(self):
        cur = self.env._observe()
        if "obj_geom" not in cur:
            cur = dict(cur)
            cur["obj_geom"] = self._geom()
        return cur

    def plan_action(self, cur, goal) -> np.ndarray:
        """CEM first action for ``cur`` → ``goal``."""
        if "obj_geom" not in cur:
            cur = dict(cur)
            cur["obj_geom"] = self._geom()
        st = cur["obj_states"]
        if self._succeeded(st, goal):
            return np.zeros(2, dtype=np.float32)
        tgt = st[self.target_idx]
        d_pos = float(np.linalg.norm(tgt[:2] - goal[:2]))
        speed = float(np.linalg.norm(tgt[3:5]))
        along = float(tgt[3]) * float(self.push_sign)
        past = (float(tgt[0]) - float(goal[0])) * float(self.push_sign) > 0.0
        catch = past or (along > 0.12 and d_pos < 0.22)
        if d_pos < self.tol:
            return np.zeros(2, dtype=np.float32)
        z0 = self.planner.encode_obs(cur)
        geom = self._geom()
        old_settle = self.cost.settle_w
        if catch and self.cost.settle_w <= 0:
            self.cost.settle_w = 3.0
        self.cost.bind(
            goal, st, geom, self.ee_idx, self.target_idx,
            min_gap=float(self.cfg.scene.min_gap),
            push_sign=self.push_sign, catch=catch)
        action = self.planner.plan(
            z0, geom=geom, states_now=st, cost=self.cost)
        self.cost.settle_w = old_settle
        self.planner.remember_first_step(z0, action, geom)
        return action

    def _plan_action(self, cur, goal):
        return self.plan_action(cur, goal)

    def run_episode(self, rng: np.random.Generator, record: bool = False):
        obs = self.env.reset(rng)
        self.env.set_ee_radius(float(self.cfg.scene.ee_radius) * self.ee_scale)
        obs = self._observe()
        self.target_idx = self._pick_target(obs, rng)
        goal = self._sample_goal(rng, obs, self.target_idx)
        target_pos = obs["obj_states"][self.target_idx, :2]
        self.push_sign = 1.0 if float(goal[0]) >= float(target_pos[0]) else -1.0
        self.planner.reset()
        init_dist = float(np.linalg.norm(target_pos - goal))

        frames = {"obj_states": [obs["obj_states"]],
                  "contact_mask": [obs["contact_mask"]],
                  "actions": []} if record else None
        final_dist = init_dist
        success = False
        settle_frame = self.budget
        t = 0

        while t < self.budget:
            cur = self._observe()
            target_pos = cur["obj_states"][self.target_idx, :2]
            final_dist = float(np.linalg.norm(target_pos - goal))
            if self._succeeded(cur["obj_states"], goal):
                success = True
                settle_frame = t
                self.env.set_force((0.0, 0.0))
                break
            action = self.plan_action(cur, goal)
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
                       (np.stack(frames["actions"]) if frames["actions"]
                        else np.zeros((0, 2), dtype=np.float32))) if record else None,
        }
        return result
