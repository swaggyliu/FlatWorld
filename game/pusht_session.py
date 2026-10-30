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
from typing import Any

import numpy as np


@dataclass
class SharedPushTAssets:
    """Policy backend loaded once and reused across concurrent web sessions.

    ``policy`` exposes ``.act(cur, stem_goal, bar_goal, goal_theta) -> [-1,1]^2``
    and ``.hold``. Prefer ONNX Runtime on CPU game hosts when
    ``pusht_infer.onnx`` sits next to the PPO checkpoint.

    ``active_sessions`` is updated by the web server so ticks can shed AI
    work under Warp's effectively-serialized CPU physics.
    """

    policy: Any
    hold: int
    device: str
    backend: str
    inference_lock: threading.Lock
    # Warp module/kernel load is not safe across concurrent env constructors.
    env_init_lock: threading.Lock = None  # type: ignore[assignment]
    active_sessions: int = 0

    def __post_init__(self):
        if self.env_init_lock is None:
            self.env_init_lock = threading.Lock()

    def close(self):
        return


def load_shared_pusht_assets(
    wm_path: str,
    policy_path: str,
    device: str | None = None,
    hold: int = 5,
) -> SharedPushTAssets:
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Push-T policy not found: {policy_path}\n"
            "Train with: python -m learning.train_rsl_wm_push_t "
            "--wm learning/checkpoints/push_t "
            "--out learning/checkpoints/push_t_wm_rsl --target-rate 0.9\n"
            "Optional ONNX: python -m learning.export_pusht_onnx "
            f"--wm {wm_path} --policy {policy_path}")

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    from learning.onnx_pusht import load_pusht_policy

    policy, backend = load_pusht_policy(
        wm_path, policy_path, device=device, hold=hold, prefer_onnx=True)
    print(f"[pusht] inference backend={backend} hold={policy.hold}", flush=True)
    return SharedPushTAssets(
        policy=policy,
        hold=int(policy.hold),
        device=device,
        backend=backend,
        inference_lock=threading.Lock(),
    )


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
    # One physics frame per net tick. Warp CPU advances serialize across
    # sessions; old SUBSTEPS=3 blew the 30 Hz budget past ~2 concurrent players.
    SUBSTEPS = 1

    def __init__(
        self,
        wm_path: str = "learning/checkpoints/push_t",
        policy_path: str = "learning/checkpoints/push_t_wm_rsl/best.pt",
        device: str | None = None,
        hold: int = 5,
        seed: int = 1000,
        shared: SharedPushTAssets | None = None,
        tick_hz: float = 20.0,
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
            self.policy = shared.policy
            self.hold = int(shared.hold)
            self.backend = shared.backend
            self._inference_lock = shared.inference_lock
        else:
            if device is None:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            assets = load_shared_pusht_assets(
                wm_path, policy_path, device=device, hold=hold)
            self.device = assets.device
            self.policy = assets.policy
            self.hold = int(assets.hold)
            self.backend = assets.backend
            self._inference_lock = assets.inference_lock
            self._shared = assets
            self._owns_assets = True

        # Difficulty = policy decision rate (Hz), not a playback-speed fraction.
        self._base_hold = max(1, int(self.hold))
        self.difficulty = "medium"
        self.decision_hz = 10.0
        self._ai_decision_accum = 0.0
        self.tick_hz = float(max(tick_hz, 5.0))

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
        self._ai_decision_accum = 0.0
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

    def _ai_action(self, cur: dict) -> np.ndarray:
        # Serialize encode+act when many web sessions share one ORT/torch backend.
        with self._inference_lock:
            a = self.policy.act(cur, self.sg, self.bg, self.gth)
        return (np.asarray(a, dtype=np.float32).reshape(-1) * self.fmax).astype(
            np.float32)

    def _policy_obs(self, side: _Side) -> dict:
        """Minimal obs for ONNX/torch act — skip tactile packing."""
        env = side.task.env
        st = env.raw_states()
        if hasattr(env, "rel_type_mat"):
            rel = env.rel_type_mat()
        else:
            from learning.models.relations import empty_rel
            rel = empty_rel(st.shape[0])
        return {
            "obj_states": st,
            "obj_types": env.obj_types_np,
            "obj_geom": env.obj_geom,
            "rel_type": rel,
        }

    def _succeeded_raw(self, side: _Side, st: np.ndarray) -> bool:
        task = side.task
        d_s = float(np.linalg.norm(st[1, :2] - task.stem_goal))
        d_b = float(np.linalg.norm(st[2, :2] - task.bar_goal))
        th = float(st[1, 2])
        ang = abs(float(np.arctan2(
            np.sin(th - task.goal_theta), np.cos(th - task.goal_theta))))
        vel = float(np.linalg.norm(st[1, 3:5]))
        return (d_s < task.tol and d_b < task.tol
                and ang < task.ang_tol and vel < task.vel_tol)

    # Decision rate targets (policy queries / wall-clock second). Capped by
    # tick_hz at runtime so the label stays meaningful as the server rate moves.
    # Protocol levels stay low|medium|high for existing clients.
    _DIFFICULTY = {
        "low": {"decision_hz": 5},
        "medium": {"decision_hz": 10},
        "high": {"decision_hz": 20},
    }

    def _set_difficulty(self, level: str) -> str:
        key = str(level or "medium").strip().lower()
        if key not in self._DIFFICULTY:
            key = "medium"
        cfg = self._DIFFICULTY[key]
        self.difficulty = key
        self.decision_hz = float(cfg["decision_hz"])
        self.hold = self._base_hold
        self._ai_decision_accum = 0.0
        return key

    def _decision_period(self) -> float:
        hz = min(float(self.decision_hz), float(self.tick_hz))
        return 1.0 / max(hz, 1.0)

    def _step_side(self, side: _Side, action: np.ndarray, hold_every: bool = False):
        if side.done:
            return
        # raw_states + advance_action: avoid observe()/tactile twice per frame.
        st = side.task.env.raw_states()
        side.trail.append((float(st[0, 0]), float(st[0, 1])))
        if self._succeeded_raw(side, st):
            side.ok = True
            side.done = True
            side.action[:] = 0
            side.task.env.set_force((0.0, 0.0))
            return
        side.action = np.asarray(action, dtype=np.float32).reshape(-1)
        side.task.env.advance_action(side.action)
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

        # Soft per-tick budget: always advance the player; AI yields first under load.
        budget = 0.55 / self.tick_hz
        t_tick = time.perf_counter()
        period = self._decision_period()

        for _ in range(self.SUBSTEPS):
            if self.phase != "race":
                break
            self._step_side(self.left, self._player_force())
            if not self.right.done:
                # Wall-clock decision cadence (not a sim-speed percentage).
                self._ai_decision_accum += dt / max(self.SUBSTEPS, 1)
                if self._ai_decision_accum > period * 1.25:
                    self._ai_decision_accum = period * 1.25
                under_budget = (time.perf_counter() - t_tick) < budget
                if under_budget and self._ai_decision_accum >= period:
                    self._ai_decision_accum -= period
                    self.right.action = self._ai_action(self._policy_obs(self.right))
                if under_budget:
                    self._step_side(self.right, self.right.action)

            if self.left.ok and not self.right.ok:
                self.winner = "player"
            elif self.right.ok and not self.left.ok:
                self.winner = "ai"
            elif self.left.ok and self.right.ok:
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
        if self._owns_assets and self._shared is not None:
            self._shared.close()
            self._shared = None

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
            "decision_hz": float(min(self.decision_hz, self.tick_hz)),
            "online": int(getattr(self._shared, "active_sessions", 1) or 1),
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
