"""Reacher: two-link revolute arm under gravity; action is PD joint angles."""

import numpy as np

from learning.configs.default import Config
from learning.env.lewm_scenes import ReacherEnv
from learning.tasks.cem_loop import CEMWorldTask


def reacher_config() -> Config:
    cfg = Config()
    cfg.scene.kind = "reacher"
    cfg.scene.use_pd = 2
    cfg.scene.gravity = 10.0
    cfg.scene.loop_damping = 0.05
    cfg.scene.use_ground = False
    cfg.scene.num_boxes = 2
    cfg.scene.num_balls = 0
    cfg.scene.dr_size_enabled = False
    cfg.scene.ee_mass = 0.35
    cfg.scene.link1_len = 0.28
    cfg.scene.link2_len = 0.24
    cfg.scene.link_thick = 0.06
    cfg.scene.link_mass = 0.35
    cfg.scene.pivot = (1.00, 0.52)
    cfg.scene.area_x = (0.12, 2.10)
    cfg.collect.out_dir = "learning/data/reacher_500"
    cfg.collect.seed = 43000
    cfg.collect.force_max = 2.8      # joint-angle clamp (rad) for PD targets
    cfg.collect.ou_sigma = 0.18
    cfg.collect.ou_theta = 0.92
    cfg.collect.x_lo = 0.08
    cfg.collect.x_hi = 2.12
    cfg.collect.y_lo = 0.04
    cfg.collect.y_hi = 0.90
    cfg.collect.mode_free = 1.0
    cfg.collect.mode_attract = 0.0
    cfg.collect.mode_push = 0.0
    cfg.collect.mode_release = 0.0
    cfg.collect.mode_hop = 0.0
    cfg.collect.mode_pile = 0.0
    cfg.collect.mode_door = 0.0
    cfg.collect.mode_spin = 0.0
    cfg.task.tol = 0.06
    cfg.task.vel_tol = 0.18
    cfg.task.budget = 400
    return cfg


class ReacherCost:
    """CEM scores pose_head xy of LINK2 (box center). No action regularizer:
    PD targets are absolute joint angles, so a² would push the arm home.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.l2_idx = ReacherEnv.LINK2
        self.goal_np = np.zeros(2, dtype=np.float32)

    def bind(self, goal):
        self.goal_np = np.asarray(goal[:2], dtype=np.float32)

    def _xy(self, pose):
        return pose[:, self.l2_idx, :2]

    def _goal(self, pose):
        if isinstance(pose, np.ndarray):
            return self.goal_np
        import torch
        return torch.as_tensor(self.goal_np, device=pose.device, dtype=pose.dtype)

    def step(self, pose, a_t, prev=None):
        xy, goal = self._xy(pose), self._goal(pose)
        if isinstance(pose, np.ndarray):
            return np.linalg.norm(xy - goal, axis=-1)
        return (xy - goal).norm(dim=-1)

    def terminal(self, pose):
        xy, goal = self._xy(pose), self._goal(pose)
        if isinstance(pose, np.ndarray):
            return 2.0 * np.linalg.norm(xy - goal, axis=-1)
        return 2.0 * (xy - goal).norm(dim=-1)


class ReacherTask(CEMWorldTask):
    def __init__(self, cfg=None, model=None, norm=None, device="cpu", **kwargs):
        cfg = cfg or reacher_config()
        env = ReacherEnv(cfg)
        cost = ReacherCost(device=device)
        super().__init__(cfg, env, model, norm, device=device, cost=cost, **kwargs)
        self.target_idx = ReacherEnv.LINK2
        self._l2_goal = None

    def sample_goal(self, rng, obs):
        """Sample a reachable link2-center goal (not the distal corner)."""
        now = np.asarray(obs["obj_states"][ReacherEnv.LINK2, :2], dtype=np.float64)
        best = None
        best_d = -1.0
        for _ in range(80):
            th1 = float(rng.uniform(-2.8, 2.8))
            th2 = float(rng.uniform(-2.6, 2.6))
            _, l2c, _ = self.env.link_goals(th1, th2)
            g = np.asarray(l2c, dtype=np.float64)
            if not self.env.goal_reachable(g, scored="center"):
                continue
            d = float(np.linalg.norm(now - g))
            if d > best_d:
                best_d, best = d, g
            if d >= 0.18:
                self._l2_goal = g
                return g.astype(np.float32)
        if best is None:
            _, l2c, _ = self.env.link_goals(1.0, 0.6)
            best = np.asarray(l2c, dtype=np.float64)
        self._l2_goal = best
        return np.asarray(self._l2_goal, dtype=np.float32)

    def bind_cost(self, obs, goal):
        self.cost.bind(goal)

    def metric(self, obs, goal) -> float:
        p = np.asarray(obs["obj_states"][ReacherEnv.LINK2, :2], dtype=np.float64)
        return float(np.linalg.norm(p - np.asarray(goal)[:2]))

    def plan_action(self, cur, goal) -> np.ndarray:
        if self.metric(cur, goal) < self.tol:
            return self._idle_action()
        return super().plan_action(cur, goal)

    def succeeded(self, obs, goal) -> bool:
        st = obs["obj_states"]
        d = float(np.linalg.norm(st[ReacherEnv.LINK2, :2] - np.asarray(goal)[:2]))
        vel = float(np.linalg.norm(st[ReacherEnv.LINK2, 3:5]))
        return d < self.tol and vel < self.vel_tol
