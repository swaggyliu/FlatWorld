"""Game session: FlatWorld physics + CEM policy + authored levels."""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from learning.configs.default import Config
from learning.tasks.push_to_goal import PushToGoalTask, load_ensemble

from .levels import spec_for


def _hit_object(states, types, geom, wx, wy) -> int | None:
    best, best_d = None, 1e9
    for i in range(1, len(states)):
        dx = wx - float(states[i, 0])
        dy = wy - float(states[i, 1])
        hw, hh = float(geom[i, 0]), float(geom[i, 1])
        rad = max(hw, hh) + 0.02
        d = (dx * dx + dy * dy) ** 0.5
        if d < rad and d < best_d:
            best, best_d = i, d
    return best


class GameSession:
    def __init__(self, checkpoint: str, device: str, full_ensemble: bool = False):
        cfg = Config()
        cfg.collect.force_max = 6.0
        cfg.scene.noise_enabled = False
        cfg.scene.dr_enabled = False
        cfg.scene.dr_size_enabled = False
        if not full_ensemble:
            import os
            if os.path.isdir(checkpoint):
                one = os.path.join(checkpoint, "ens_0.pt")
                if not os.path.exists(one):
                    one = os.path.join(checkpoint, "best.pt")
                checkpoint = one
        models, norm, stride = load_ensemble(checkpoint, device)
        self.task = PushToGoalTask(
            cfg, models, norm, device=device, tol=0.08, budget=10_000,
            stride=max(int(stride), 1), target_mode="random",
            planner_kwargs=dict(horizon=6, population=48, iterations=2,
                                seed=0))
        self.level = 0
        self.spec = spec_for(1)
        self.level_seed_base = int(time.time()) % 1_000_000
        self.win_hold = 0
        self.stride_left = 0
        self.action = np.zeros(2, dtype=np.float32)
        self.stall_count = 0
        self.hit_flash = 0.0
        self.level_goal = None
        self.steer_goal = None
        self.level_target_idx = 1
        self.push_idx = None
        self.hazard_idx = 1
        self.running = False
        self.status = "CLICK THE GOLD ONE"
        self.trail = deque(maxlen=48)
        self.obs = None
        self.restitution = 0.15
        self.friction = 0.50
        self.scale = 1.0
        self.elapsed = 0.0
        self.clock_on = False
        self.level_score = 0
        self.total_score = 0
        self.pushes = 0
        self.last_stars = 0
        self.best_time: dict[str, float] = {}
        self.next_level()

    def _level_seed(self) -> int:
        return int(self.level_seed_base + self.level * 1_000_003)

    def _boot_level(self, apply_loadout: bool = True):
        """Rebuild the current authored layout. Retry keeps the current loadout."""
        spec = spec_for(self.level)
        self.spec = spec
        rng = np.random.default_rng(self._level_seed())
        self.obs = self.task.env.reset(rng)
        if apply_loadout:
            self.restitution = float(spec.bounce)
            self.friction = float(spec.grip)
            self.scale = float(spec.scale)
        self.apply_equipment()
        self.task.env.place_lineup(spec.xs)
        self.obs = self.task.env._observe()
        self.level_target_idx = int(spec.target)
        self.hazard_idx = int(spec.hazard)
        self.task.target_idx = self.level_target_idx
        gy = float(self.obs["obj_states"][self.level_target_idx, 1])
        lo, hi = self.task._workspace_x(self.level_target_idx)
        gx = float(np.clip(spec.goal_x, lo, hi))
        self.level_goal = np.array([gx, gy], dtype=np.float32)
        self.push_idx = None
        self.steer_goal = None
        self.task.planner.reset()
        self.hit_flash = 0.0
        self.running = False
        self.action[:] = 0
        self.stride_left = 0
        self.stall_count = 0
        self.win_hold = 0
        self.intro_hold = 90
        self.trail.clear()
        self.elapsed = 0.0
        self.clock_on = False
        self.level_score = 0
        self.pushes = 0
        self.last_stars = 0
        self.task.env.set_force((0.0, 0.0))
        self.status = f"LEVEL {self.level}  ·  {spec.name}"

    def apply_equipment(self):
        """Write the current spring / tread / scale sliders onto the spirit."""
        self.restitution = float(np.clip(self.restitution, 0.0, 1.0))
        self.friction = float(np.clip(self.friction, 0.0, 1.0))
        self.scale = float(np.clip(self.scale, 0.5, 2.0))
        env = self.task.env
        env.set_ee_contact(self.friction, self.restitution)
        env.set_ee_radius(float(env.cfg.scene.ee_radius) * self.scale)

    @staticmethod
    def score_for(elapsed: float) -> int:
        """Shorter time → higher score. 0s = 10000, 8s = 5000, 24s = 2500."""
        t = max(0.0, float(elapsed))
        return int(round(10_000.0 * 8.0 / (8.0 + t)))

    def _grade(self) -> int:
        stars = 1
        if self.elapsed <= float(self.spec.par):
            stars = 2
        if (self.elapsed <= float(self.spec.par) * 0.75
                and self.pushes <= int(self.spec.push_cap)):
            stars = 3
        return stars

    def reset(self):
        """Retry this level. Same layout and star; keeps current loadout."""
        self._boot_level(apply_loadout=False)

    def next_level(self):
        """Advance to the next authored stage. Applies that stage's suggested loadout."""
        if self.status.startswith("WIN"):
            self.total_score += int(self.level_score)
        self.level += 1
        self._boot_level(apply_loadout=True)

    def pace_score(self) -> int:
        if self.status.startswith("WIN"):
            return int(self.level_score)
        return self.score_for(self.elapsed)

    def display_score(self) -> int:
        if self.status.startswith("WIN"):
            return int(self.total_score) + int(self.level_score)
        return int(self.total_score)

    def best_for_level(self) -> float | None:
        return self.best_time.get(self.spec.name)

    def handle_click(self, wx: float, wy: float):
        """Click object to choose what to push; click empty to set a step dest."""
        if self.status.startswith("LOSE") or self.status.startswith("WIN"):
            return
        if not self.clock_on:
            self.clock_on = True
        hit = _hit_object(self.obs["obj_states"], self.obs["obj_types"],
                          self.obs["obj_geom"], wx, wy)
        if hit is not None:
            self.push_idx = int(hit)
            self.task.target_idx = self.push_idx
            self.steer_goal = None
            self.running = False
            self.stride_left = 0
            self.stall_count = 0
            self.action[:] = 0.0
            self.task.planner.reset()
            self.task.env.set_force((0.0, 0.0))
            self.status = "CLICK A SPOT TO PUSH"
            return
        if self.push_idx is None:
            self.status = "CLICK THE GOLD ONE FIRST"
            return
        ty = float(self.obs["obj_states"][self.push_idx, 1])
        x_lo, x_hi = self.task._workspace_x(self.push_idx)
        wx = float(np.clip(wx, x_lo, x_hi))
        self.steer_goal = np.array([wx, ty], dtype=np.float32)
        self.task.target_idx = self.push_idx
        self.running = True
        self.pushes += 1
        self.status = "PUSHING"
        self.task.planner.reset()
        self.stall_count = 0
        self.stride_left = 0
        self.action[:] = 0.0
        self.task.env.set_force((0.0, 0.0))

    def _level_won(self, cur) -> bool:
        pos = cur["obj_states"][self.level_target_idx, :2]
        vel = float(np.linalg.norm(cur["obj_states"][self.level_target_idx, 3:5]))
        d = float(np.linalg.norm(pos - self.level_goal))
        return d < self.task.tol and vel < self.task.vel_tol

    def _ee_hazard_force(self) -> float:
        env = self.task.env
        gid = env.rigid_ids[self.hazard_idx]
        ee = env.ee_idx
        n = int(env.rm.num_contacts.numpy()[0])
        if n <= 0:
            return 0.0
        n = min(n, env.rm.MAX_CONTACTS)
        ca = env.rm.contact_rigid_a.numpy()
        cb = env.rm.contact_rigid_b.numpy()
        cf = env.rm.contact_force.numpy()
        peak = 0.0
        for i in range(n):
            a, b = int(ca[i]), int(cb[i])
            if (a == ee and b == gid) or (b == ee and a == gid):
                peak = max(peak, float(np.hypot(cf[i][0], cf[i][1])))
        return peak

    def _policy(self, cur, careful: bool) -> np.ndarray:
        task = self.task
        task.target_idx = int(self.push_idx)
        goal = self.steer_goal
        target_pos = cur["obj_states"][task.target_idx, :2]
        target_vel = float(np.linalg.norm(cur["obj_states"][task.target_idx, 3:5]))
        d = float(np.linalg.norm(target_pos - goal))
        if d < task.tol and target_vel < task.vel_tol:
            self.running = False
            self.status = "READY  (click next object or spot)"
            return np.zeros(2, dtype=np.float32)
        a = task.plan_action(cur, goal)
        if careful:
            a = a * 0.55
        return np.asarray(a, dtype=np.float32)

    def tick(self, careful: bool = False):
        if self.obs is None:
            return
        self.trail.append((float(self.obs["obj_states"][0, 0]),
                           float(self.obs["obj_states"][0, 1])))
        if self.hit_flash > 0:
            self.hit_flash = max(0.0, self.hit_flash - 0.08)

        ended = self.status.startswith("WIN") or self.status.startswith("LOSE")
        if self.clock_on and not ended:
            self.elapsed += 1.0 / 60.0

        if self.status.startswith("WIN"):
            self.win_hold += 1
            self.action[:] = 0
            self.task.env.set_force((0.0, 0.0))
            self.obs = self.task.env._observe()
            if self.win_hold >= 150:
                self.next_level()
            return

        if self.status.startswith("LOSE"):
            self.action[:] = 0
            self.task.env.set_force((0.0, 0.0))
            self.obs = self.task.env._observe()
            return

        if self.intro_hold > 0:
            self.intro_hold -= 1
            if self.intro_hold == 0 and self.status.startswith("LEVEL"):
                self.status = "CLICK THE GOLD ONE"

        cur = dict(self.obs)
        cur["obj_geom"] = self.task._geom()
        if self.level_goal is not None and self._level_won(cur):
            self.clock_on = False
            self.level_score = self.score_for(self.elapsed)
            self.last_stars = self._grade()
            prev = self.best_time.get(self.spec.name)
            if prev is None or self.elapsed < prev:
                self.best_time[self.spec.name] = float(self.elapsed)
            star_s = "*" * self.last_stars
            self.status = f"WIN  {star_s}  +{self.level_score}"
            self.running = False
            self.action[:] = 0
            self.task.env.set_force((0.0, 0.0))
            return

        if not self.running or self.steer_goal is None or self.push_idx is None:
            self.action[:] = 0
            self.task.env.set_force((0.0, 0.0))
            self.obs = self.task.env._observe()
            return

        if self.stride_left <= 0:
            self.action = self._policy(cur, careful)
            self.stride_left = self.task.stride
        self.stride_left -= 1
        self.obs = self.task.env.step(self.action)

        F = self._ee_hazard_force()
        if F > 0.6:
            self.clock_on = False
            self.level_score = 0
            self.last_stars = 0
            self.status = "LOSE  ·  hit red   R retry"
            self.running = False
            self.hit_flash = 1.0
            self.action[:] = 0
            self.task.env.set_force((0.0, 0.0))
