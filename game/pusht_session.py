"""Spirit Push-T: split-screen You vs AI.

Left  — player stick force on the spirit, push T to the goal.
Right — frozen WM-obs PPO.
First side to succeed wins. Each round starts with a 3s countdown.

  python -m game --mode pusht
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from tensordict import TensorDict


@dataclass
class SharedPushTAssets:
    """WM + PPO loaded once and reused across concurrent web sessions."""

    wm: Any
    policy: Any
    hold: int
    task_features: Callable
    device: str
    inference_lock: threading.Lock
    _train_env: Any = None

    def close(self):
        env = self._train_env
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
            self._train_env = None


def load_shared_pusht_assets(
    wm_path: str,
    policy_path: str,
    device: str | None = None,
    hold: int = 5,
) -> SharedPushTAssets:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wm, policy, env, hold_i, task_features = _load_policy(
        wm_path, policy_path, device, hold)
    return SharedPushTAssets(
        wm=wm,
        policy=policy,
        hold=int(hold_i),
        task_features=task_features,
        device=device,
        inference_lock=threading.Lock(),
        _train_env=env,
    )


def _load_policy(wm_path: str, policy_path: str, device: str, hold: int):
    from learning.tasks.push_to_goal import load_ensemble
    from learning.rl.imagine import WMBundle, task_features
    from learning.train_rsl_wm_push_t import WMObsVecEnv
    from learning.rl.push_t_vec_env import PushTVecEnv
    from learning.train_rsl_push_t import make_train_cfg
    from rsl_rl.runners import OnPolicyRunner
    import argparse

    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Push-T policy not found: {policy_path}\n"
            "Train with: python -m learning.train_rsl_wm_push_t "
            "--wm learning/checkpoints/push_t "
            "--out learning/checkpoints/push_t_wm_rsl --target-rate 0.9")

    models, norm, stride = load_ensemble(wm_path, device)
    wm = WMBundle(models[0], norm, device, stride=stride)
    for p in wm.model.parameters():
        p.requires_grad_(False)
    wm.model.eval()
    hold = max(int(hold), int(stride), 1)

    base = PushTVecEnv(
        num_envs=1, hold=hold, device="cpu", seed=0,
        num_workers=0, max_ee_t_dist=None)
    env = WMObsVecEnv(base, wm, device=device)
    args = argparse.Namespace(
        num_steps=32, max_iterations=1, save_interval=100,
        entropy_coef=0.01, lr=1e-3)
    cfg = make_train_cfg(args)
    cfg["run_name"] = "pusht_versus"
    log_dir = os.path.join(os.path.dirname(policy_path), "_game_logs")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)
    # Checkpoints may be CUDA-saved; map onto the requested device (CPU node OK).
    runner.load(policy_path, map_location=device)
    policy = runner.get_inference_policy(device=device)
    return wm, policy, env, hold, task_features


class _Side:
    __slots__ = ("task", "trail", "action", "hold_left", "ok", "done",
                 "t", "label")

    def __init__(self, task, label: str):
        self.task = task
        self.label = label
        self.trail = deque(maxlen=80)
        self.action = np.zeros(2, dtype=np.float32)
        self.hold_left = 0
        self.ok = False
        self.done = False
        self.t = 0


class PushTVersusSession:
    """You (stick) vs AI — first to seat the T wins."""

    COUNTDOWN_S = 3.0
    DEFAULT_ROUND_SECONDS = 30
    MIN_ROUND_SECONDS = 10
    MAX_ROUND_SECONDS = 180
    SUCCESS_TOL = 0.12  # Match the saved WM/PPO evaluation's position tolerance.
    PLAYER_FORCE = 5.5  # N, within typical force_max=6
    # Physics substeps per web/desktop tick. Player uses latest stick each step;
    # With the default five-step action hold, AI inference runs at most once
    # per tick. Easier modes advance only the AI simulation more slowly.
    SUBSTEPS = 3

    def __init__(
        self,
        wm_path: str = "learning/checkpoints/push_t",
        policy_path: str = "learning/checkpoints/push_t_wm_rsl/best.pt",
        device: str | None = None,
        hold: int = 5,
        seed: int = 1000,
        shared: SharedPushTAssets | None = None,
        tick_hz: float = 30.0,
    ):
        self.seed = int(seed)
        self.ep = 0
        self.paused = False
        self.player_wins = 0
        self.ai_wins = 0
        self.draws = 0
        self.winner = None  # "player" | "ai" | "draw" | None
        self.phase = "countdown"  # countdown | race | over
        self.countdown_left = self.COUNTDOWN_S
        self._last_wall = time.perf_counter()
        self.status = "LOADING"
        # Continuous stick in [-1, 1]^2 (virtual joystick / 360° drag).
        self.player_stick = np.zeros(2, dtype=np.float32)
        self._shared = shared
        self._owns_assets = shared is None

        if shared is not None:
            self.device = shared.device
            self.wm = shared.wm
            self.policy = shared.policy
            self.hold = int(shared.hold)
            self._task_features = shared.task_features
            self._train_env = None
            self._inference_lock = shared.inference_lock
        else:
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self.wm, self.policy, self._train_env, self.hold, self._task_features = (
                _load_policy(wm_path, policy_path, device, hold))
            self._inference_lock = threading.Lock()

        # Keep the trained control cadence and force at every difficulty.
        # Only the AI's simulation progress changes with difficulty.
        self._base_hold = max(1, int(self.hold))
        self.difficulty = "medium"
        self.ai_speed = 1.0
        self._ai_step_credit = 0.0

        from learning.tasks.push_t import PushTTask, push_t_config
        cfg = push_t_config()
        self.cfg = cfg
        self.fmax = float(cfg.collect.force_max)
        # Two independent plants, same config.
        self.left = _Side(
            PushTTask(cfg, tol=self.SUCCESS_TOL, planner_kwargs=dict(use_servo=True),
                      ee_spawn="support"),
            "YOU")
        self.right = _Side(
            PushTTask(push_t_config(), tol=self.SUCCESS_TOL, planner_kwargs=dict(use_servo=True),
                      ee_spawn="support"),
            "AI")
        # A wall clock keeps the HUD and timeout consistent under CPU load.
        self.selected_round_duration = self.DEFAULT_ROUND_SECONDS
        self.round_duration = self.selected_round_duration
        self.round_elapsed = 0.0
        self.sg = self.left.task.stem_goal
        self.bg = self.left.task.bar_goal
        self.gth = self.left.task.goal_theta
        self._set_difficulty("medium")
        self.reset()
    def reset(self):
        for side in (self.left, self.right):
            # Original reset: random starting pose per round, identical on both
            # sides. T geometry and goal stay fixed in push_t_config().
            rng = np.random.default_rng(self.seed + self.ep)
            side.task.env.reset(rng)
            side.task.env.set_ee_radius(
                float(side.task.cfg.scene.ee_radius) * side.task.ee_scale)
            side.task.goal = side.task.sample_goal(rng, side.task.observe())
            side.trail.clear()
            side.action[:] = 0
            side.hold_left = 0
            side.ok = False
            side.done = False
            side.t = 0
            side.task.env.set_force((0.0, 0.0))

        self.sg = self.left.task.stem_goal
        self.bg = self.left.task.bar_goal
        self.gth = self.left.task.goal_theta

        self.phase = "countdown"
        self.paused = False
        self._ai_step_credit = 0.0
        self.round_duration = self.selected_round_duration
        self.round_elapsed = 0.0
        self.countdown_left = self.COUNTDOWN_S
        self._last_wall = time.perf_counter()
        self.winner = None
        self.status = "Get ready…"
        self.player_stick[:] = 0.0

    def next_episode(self):
        self.ep += 1
        self.reset()

    def set_round_duration(self, seconds: int):
        """Apply to the next round, or the opening countdown on connection."""
        if (type(seconds) is not int
                or not self.MIN_ROUND_SECONDS <= seconds <= self.MAX_ROUND_SECONDS):
            raise ValueError("Round duration must be an integer from 10 to 180 seconds.")
        self.selected_round_duration = seconds
        if self.phase == "countdown":
            self.round_duration = seconds

    def set_stick(self, x: float, y: float):
        """Virtual joystick: x right, y up, each clamped to [-1, 1].

        Safe to call concurrently with tick (float stores); web reader applies
        stick without waiting for the AI/physics session lock.
        """
        sx = float(np.clip(x, -1.0, 1.0))
        sy = float(np.clip(y, -1.0, 1.0))
        n = float(np.hypot(sx, sy))
        if n > 1.0:
            sx /= n
            sy /= n
        if n < 0.05:
            self.player_stick[0] = 0.0
            self.player_stick[1] = 0.0
        else:
            self.player_stick[0] = sx
            self.player_stick[1] = sy

    def _player_force(self) -> np.ndarray:
        fcap = min(self.PLAYER_FORCE, self.fmax)
        sx = float(self.player_stick[0])
        sy = float(self.player_stick[1])
        v = np.array([sx, sy], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            # Analog magnitude: deflection in unit disk scales force.
            v = v / n * (fcap * min(n, 1.0))
        else:
            v[:] = 0.0
        return v

    @torch.no_grad()
    def _ai_action(self, cur: dict) -> np.ndarray:
        # Serialize WM/policy forwards when many web sessions share weights.
        with self._inference_lock:
            z, _, _, _ = self.wm.encode_one(cur, use_tactile=False)
            pose = torch.as_tensor(
                cur["obj_states"][:, :3], dtype=torch.float32,
                device=self.wm.device).unsqueeze(0)
            feat = self._task_features(pose, self.sg, self.bg, self.gth)
            x = torch.cat([z.reshape(1, -1), feat], dim=-1)
            pol_dev = next(self.policy.parameters()).device
            td = TensorDict({"policy": x.to(pol_dev)}, batch_size=[1])
            a = self.policy(td, stochastic_output=False)
            if isinstance(a, tuple):
                a = a[0]
            a = a.detach().float().cpu().numpy().reshape(-1)
        return (np.clip(a, -1.0, 1.0) * self.fmax).astype(np.float32)

    # Legacy protocol keys remain compatible with existing browser clients.
    # Slow the complete AI trajectory, preserving its trained force/dynamics
    # and one decision per base_hold AI physics steps (normally five).
    _DIFFICULTY = {
        "low": {"speed": 0.10},
        "medium": {"speed": 0.40},
        "high": {"speed": 1.0},
    }

    def _set_difficulty(self, level: str) -> str:
        key = str(level or "medium").strip().lower()
        if key not in self._DIFFICULTY:
            key = "medium"
        cfg = self._DIFFICULTY[key]
        self.difficulty = key
        self.ai_speed = float(cfg["speed"])
        self.hold = self._base_hold
        return key
    def _step_side(self, side: _Side, action: np.ndarray, hold_every: bool):
        if side.done:
            return
        cur = side.task.observe()
        st = side.task.env.raw_states()
        side.trail.append((float(st[0, 0]), float(st[0, 1])))
        if side.task.succeeded(cur, side.task.goal):
            side.ok = True
            side.done = True
            side.action[:] = 0
            side.task.env.set_force((0.0, 0.0))
            return
        if hold_every:
            if side.hold_left <= 0:
                side.action = action
                side.hold_left = self.hold
            side.task.env.step(side.action)
            side.hold_left -= 1
        else:
            side.action = action
            side.task.env.step(action)
        side.t += 1

    def tick(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_wall)
        self._last_wall = now
        if self.paused:
            return

        if self.phase == "countdown":
            self.countdown_left = max(0.0, self.countdown_left - dt)
            n = int(np.ceil(self.countdown_left))
            self.status = f"{n}" if n > 0 else "GO!"
            if self.countdown_left <= 0.0:
                self.phase = "race"
                self.status = "RACE!"
            return

        if self.phase == "over":
            return

        self.round_elapsed = min(self.round_duration, self.round_elapsed + dt)
        if self.round_elapsed >= self.round_duration:
            self._finish_round("draw")
            return

        # Accumulate fractional AI steps so difficulty changes playback speed,
        # rather than stretching actions or weakening their contact forces.
        for _ in range(self.SUBSTEPS):
            if self.phase != "race":
                break
            # Latest stick each substep (may be updated mid-tick by WS reader).
            self._step_side(self.left, self._player_force(), hold_every=False)
            if not self.right.done:
                self._ai_step_credit += self.ai_speed
                if self._ai_step_credit >= 1.0 - 1e-9:
                    self._ai_step_credit = max(0.0, self._ai_step_credit - 1.0)
                    step_a = (self._ai_action(self.right.task.observe())
                              if self.right.hold_left <= 0 else self.right.action)
                    self._step_side(self.right, step_a, hold_every=True)

            if self.left.ok and not self.right.ok:
                self.winner = "player"
            elif self.right.ok and not self.left.ok:
                self.winner = "ai"
            elif self.left.ok and self.right.ok:
                # Both completed on this race substep. AI physics-step counts
                # differ by speed and cannot be used as elapsed race time.
                self.winner = "draw"
            elif self.left.done and self.right.done:
                self.winner = "draw"

            if self.winner is not None:
                self._finish_round(self.winner)
                break

    def _finish_round(self, winner: str):
        self.winner = winner
        self.phase = "over"
        self.player_stick[:] = 0.0
        for side in (self.left, self.right):
            side.action[:] = 0.0
            side.task.env.set_force((0.0, 0.0))
        if winner == "player":
            self.player_wins += 1
            self.status = "YOU WIN!"
        elif winner == "ai":
            self.ai_wins += 1
            self.status = "AI WINS"
        else:
            self.draws += 1
            self.status = "TIME'S UP — DRAW" if self.round_elapsed >= self.round_duration else "DRAW"

    def close(self):
        # Shared web assets are owned by the server lifespan, not each session.
        if not self._owns_assets:
            return
        if hasattr(self, "_train_env") and self._train_env is not None:
            if hasattr(self._train_env, "close"):
                self._train_env.close()
            self._train_env = None

    def snapshot(self) -> dict:
        """JSON-serializable state for the web client."""
        def _side(side: _Side) -> dict:
            # Sensor noise belongs to the policy's observations, never the HUD.
            # Reading raw state also avoids consuming the policy's noise RNG.
            st = side.task.env.raw_states()
            geom = side.task.env.obj_geom
            return {
                "states": st.tolist(),
                "geom": geom.tolist(),
                "action": [float(side.action[0]), float(side.action[1])],
                "trail": [[float(x), float(y)] for x, y in side.trail],
                "ok": bool(side.ok),
                "done": bool(side.done),
                "t": int(side.t),
            }

        return {
            "round_id": self.ep,
            "phase": self.phase,
            "status": self.status,
            "countdown": float(self.countdown_left),
            "round_duration": float(self.round_duration),
            "time_remaining": max(0.0, float(self.round_duration - self.round_elapsed)),
            "winner": self.winner,
            "paused": bool(self.paused),
            "difficulty": self.difficulty,
            "shape": {
                "symmetric": True,
                "bar_offset": 0.0,
            },
            "score": {
                "you": int(self.player_wins),
                "ai": int(self.ai_wins),
                "draw": int(self.draws),
            },
            "goal": {
                "stem": [float(self.sg[0]), float(self.sg[1])],
                "bar": [float(self.bg[0]), float(self.bg[1])],
                "theta": float(self.gth),
            },
            "left": _side(self.left),
            "right": _side(self.right),
        }


# Back-compat alias for older imports.
PushTSession = PushTVersusSession
