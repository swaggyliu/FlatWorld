"""Two-room: EE navigates through a y-aligned doorway into the other room."""

import numpy as np

from learning.configs.default import Config
from learning.env.lewm_scenes import TwoRoomEnv
from learning.tasks.cem_loop import CEMWorldTask


def two_room_config() -> Config:
    cfg = Config()
    cfg.scene.kind = "two_room"
    cfg.scene.gravity = 0.0
    cfg.scene.loop_damping = 3.0
    cfg.scene.use_ground = False
    cfg.scene.num_boxes = 2
    cfg.scene.num_balls = 0
    cfg.scene.dr_size_enabled = True
    cfg.scene.dr_ee_r_scale = (0.6, 1.3)
    cfg.scene.area_x = (0.12, 2.10)
    cfg.scene.wall_hw = 0.04
    cfg.scene.door_gap = 0.40
    cfg.scene.door_x = 1.10
    cfg.scene.door_y = 0.55
    cfg.scene.room_y0 = -1.40
    cfg.scene.room_h = 2.40
    cfg.collect.out_dir = "learning/data/two_room_500"
    cfg.collect.seed = 41000
    cfg.collect.x_lo = 0.08
    cfg.collect.x_hi = 2.12
    cfg.collect.y_lo = 0.08
    cfg.collect.y_hi = 1.00
    cfg.collect.hop_y = 0.55
    cfg.collect.hop_y_hi = 0.95
    cfg.collect.mode_free = 0.30
    cfg.collect.mode_attract = 0.0
    cfg.collect.mode_push = 0.0
    cfg.collect.mode_release = 0.10
    cfg.collect.mode_hop = 0.15
    cfg.collect.mode_pile = 0.0
    cfg.collect.mode_door = 0.45
    cfg.collect.mode_spin = 0.0
    cfg.task.tol = 0.10
    cfg.task.vel_tol = 0.15
    cfg.task.budget = 400
    return cfg


class TwoRoomCost:
    """Native-LeWM style: CEM cost is EE-to-goal distance only.

    Goal is sampled in the other room, so minimizing this *and* a world
    model that cannot walk through walls is what produces a doorway path.
    """

    def __init__(self, action_cost: float = 0.0005, device: str = "cpu"):
        self.action_cost = action_cost
        self.device = device
        self.ee_idx = 0
        self.goal_np = np.zeros(2, dtype=np.float32)

    def bind(self, goal):
        self.goal_np = np.asarray(goal[:2], dtype=np.float32)

    def _goal(self, pose):
        if isinstance(pose, np.ndarray):
            return self.goal_np
        import torch
        return torch.as_tensor(self.goal_np, device=pose.device, dtype=pose.dtype)

    def step(self, pose, a_t, prev=None):
        ee = pose[:, self.ee_idx, :2]
        goal = self._goal(pose)
        if isinstance(pose, np.ndarray):
            d = np.linalg.norm(ee - goal, axis=-1)
            return d + self.action_cost * (np.asarray(a_t) ** 2).mean(axis=-1)
        d = (ee - goal).norm(dim=-1)
        return d + self.action_cost * (a_t ** 2).mean(dim=-1)

    def terminal(self, pose):
        ee = pose[:, self.ee_idx, :2]
        goal = self._goal(pose)
        if isinstance(pose, np.ndarray):
            return 2.0 * np.linalg.norm(ee - goal, axis=-1)
        return 2.0 * (ee - goal).norm(dim=-1)


class TwoRoomTask(CEMWorldTask):
    def __init__(self, cfg=None, model=None, norm=None, device="cpu", **kwargs):
        cfg = cfg or two_room_config()
        env = TwoRoomEnv(cfg)
        cost = TwoRoomCost(device=device)
        super().__init__(cfg, env, model, norm, device=device, cost=cost, **kwargs)
        self.target_idx = 0

    def sample_goal(self, rng, obs):
        door = float(self.env.door_x)
        dy = float(self.env.door_y)
        spawn_left = bool(getattr(self.env, "spawn_left", True))
        if spawn_left:
            gx = float(rng.uniform(door + 0.28, 1.95))
        else:
            gx = float(rng.uniform(0.22, door - 0.28))
        gy = float(np.clip(rng.uniform(dy - 0.12, dy + 0.12), 0.16, 0.94))
        return np.array([gx, gy], dtype=np.float32)

    def bind_cost(self, obs, goal):
        self.cost.bind(goal)

    def metric(self, obs, goal) -> float:
        p = obs["obj_states"][0, :2]
        return float(np.linalg.norm(p - np.asarray(goal)[:2]))

    def succeeded(self, obs, goal) -> bool:
        st = obs["obj_states"][0]
        d = float(np.linalg.norm(st[:2] - np.asarray(goal)[:2]))
        vel = float(np.linalg.norm(st[3:5]))
        return d < self.tol and vel < self.vel_tol
