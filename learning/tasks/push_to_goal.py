"""PushToGoal task and its imagined CEM cost.

Task: using ONLY the learned world model for planning, apply forces to
the end-effector so that a designated target object ends up within a
tolerance of a goal position. Execution and success measurement happen
in the FlatWorld simulator (the world model never sees future states).

The CEM loop lives in ``learning.planner``. Geometric helpers (contact
face, coast, brake) shape the *cost*; they do not emit the executed force.
"""

import glob
import os

import numpy as np
import torch

from learning.data.normalizer import Normalizer
from learning.env.flatworld_wrapper import PushSceneEnv
from learning.models.lewm import StateLeWM, rich_edges_from_checkpoint
from learning.planner import CEMPlanner


def _load_one(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = StateLeWM(
        n_obj=ckpt["n_obj"], latent_dim=ckpt["latent_dim"],
        n_mp=int(ckpt.get("n_mp", 2)),
        drop_tactile=bool(ckpt.get("drop_tactile", False)),
        drop_geom=bool(ckpt.get("drop_geom", False)),
        rich_edges=rich_edges_from_checkpoint(ckpt),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm_name = ckpt.get("normalizer", "normalizer.json")
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


_COST_KEYS = (
    "action_cost", "reach_cost", "brake_radius", "brake_w", "brake_sq",
    "coast_start", "coast_full", "coast_w", "reach_fade",
)


class PushToGoalCost:
    """Decoded-xy cost for PushToGoal.

    Geometric terms only (CEM still emits the force): distance to goal,
    reach the target contact face, coast, and brake.
    """

    def __init__(self, action_cost: float = 0.0005, reach_cost: float = 0.35,
                 brake_radius: float = 0.15, brake_w: float = 4.0,
                 brake_sq: float = 60.0, coast_start: float = 0.12,
                 coast_full: float = 0.0, coast_w: float = 6.0,
                 reach_fade: float = 0.8, device: str = "cpu"):
        self.action_cost = action_cost
        self.reach_cost = reach_cost
        self.brake_radius = brake_radius
        self.brake_w = brake_w
        self.brake_sq = brake_sq
        self.coast_start = coast_start
        self.coast_full = coast_full
        self.coast_w = coast_w
        self.reach_fade = reach_fade
        self.device = device
        self.ee_idx = 0
        self.target_idx = 1
        self.goal_t = None
        self.push_sign = 1.0
        self.standoff = 0.0

    def bind(self, goal, states_now, geom, ee_idx: int, target_idx: int,
             min_gap: float = 0.02):
        self.ee_idx = ee_idx
        self.target_idx = target_idx
        self.goal_t = torch.as_tensor(goal[:2], device=self.device, dtype=torch.float32)
        self.push_sign = 1.0 if float(goal[0]) >= float(states_now[target_idx, 0]) else -1.0
        g = torch.as_tensor(geom, device=self.device, dtype=torch.float32)
        hw = g[:, 0].clamp(min=1e-4)
        self.standoff = float(hw[target_idx] + hw[ee_idx])

    def step(self, xy, a_t, prev):
        d_goal = (xy[:, self.target_idx] - self.goal_t).norm(dim=-1)
        face = xy[:, self.target_idx].clone()
        face[:, 0] = face[:, 0] - self.push_sign * self.standoff
        d_face = (xy[:, self.ee_idx] - face).norm(dim=-1)
        coast = ((self.coast_start - d_goal)
                 / max(self.coast_start - self.coast_full, 1e-6)).clamp(0.0, 1.0)
        cost = d_goal \
            + (1.0 - self.reach_fade * coast) * self.reach_cost * d_face \
            + self.action_cost * (a_t ** 2).mean(dim=-1)
        sep = (xy[:, self.ee_idx] - xy[:, self.target_idx]).norm(dim=-1)
        too_close = (self.standoff + 0.01 - sep).clamp(min=0.0)
        cost = cost + self.coast_w * coast * too_close
        if prev is not None:
            speed = (xy[:, self.target_idx] - prev[:, self.target_idx]).norm(dim=-1)
            near = (self.brake_radius - d_goal).clamp(min=0.0)
            cost = cost + self.brake_w * near * speed \
                + self.brake_sq * near * speed * speed
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
        self.cost = PushToGoalCost(device=device, **cost_kw)
        self.planner = CEMPlanner(self.models, norm, device=device,
                                  force_max=cfg.collect.force_max, **pk)
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
            if t >= self.budget:
                break
        if frames is not None:
            frames["obj_states"].append(obs["obj_states"])
            frames["contact_mask"].append(obs["contact_mask"])
            frames["actions"].append(np.asarray(action, dtype=np.float32))
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
        if self._succeeded(cur["obj_states"], goal):
            return np.zeros(2, dtype=np.float32)
        tgt_s = cur["obj_states"][self.target_idx]
        d_pos = float(np.linalg.norm(tgt_s[:2] - goal[:2]))
        if d_pos < self.tol:
            return np.zeros(2, dtype=np.float32)
        z0 = self.planner.encode_obs(cur)
        geom = self._geom()
        self.cost.bind(
            goal, cur["obj_states"], geom, self.ee_idx, self.target_idx,
            min_gap=float(self.cfg.scene.min_gap))
        return self.planner.plan(
            z0, geom=geom, states_now=cur["obj_states"], cost=self.cost)

    def _plan_action(self, cur, goal):
        return self.plan_action(cur, goal)

    def _select_action(self, cur, goal):
        return self.plan_action(cur, goal)

    def run_episode(self, rng: np.random.Generator, record: bool = False):
        obs = self.env.reset(rng)
        self.env.set_ee_radius(float(self.cfg.scene.ee_radius) * self.ee_scale)
        obs = self._observe()
        self.target_idx = self._pick_target(obs, rng)
        goal = self._sample_goal(rng, obs, self.target_idx)
        target_pos = obs["obj_states"][self.target_idx, :2]
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
