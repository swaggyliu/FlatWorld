"""rsl-rl VecEnv wrapper around FlatWorld Push-T (physics only, no WM).

Uses a process pool so many independent Warp scenes step in parallel.
Threads do not help (GIL); CUDA is slower for these tiny scenes.
"""

from __future__ import annotations

import contextlib
import io
import os
from multiprocessing import Pipe, Process
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


def _obs_row(obj_states: np.ndarray, sg, bg, gth: float) -> np.ndarray:
    """Flatten states + task features → (obs_dim,) float32."""
    from learning.rl.imagine import task_features

    st = torch.as_tensor(obj_states, dtype=torch.float32).reshape(-1)
    pose = torch.as_tensor(obj_states[:, :3], dtype=torch.float32).unsqueeze(0)
    feat = task_features(pose, sg, bg, gth).reshape(-1)
    return torch.cat([st, feat], dim=0).numpy().astype(np.float32)


def _succeeded(st: np.ndarray, sg, bg, *, tol, ang_tol, vel_tol) -> bool:
    d_s = float(np.linalg.norm(st[1, :2] - sg))
    d_b = float(np.linalg.norm(st[2, :2] - bg))
    ang = abs(float(np.arctan2(np.sin(st[1, 2]), np.cos(st[1, 2]))))
    vel = float(np.linalg.norm(st[1, 3:5]))
    return d_s < tol and d_b < tol and ang < ang_tol and vel < vel_tol


class _LocalPushTBatch:
    """Owns ``n`` PushTEnv instances inside one worker process."""

    def __init__(self, spec: dict):
        from learning.env.lewm_scenes import PushTEnv
        from learning.tasks.push_t import push_t_config
        from learning.rl.imagine import FEAT_DIM

        self.n = int(spec["n_local"])
        self.hold = int(spec["hold"])
        self.max_episode_length = int(spec["max_episode_length"])
        self.max_ee_t_dist = float(spec["max_ee_t_dist"])
        self.workspace_xy = tuple(spec["workspace_xy"])
        self.fail_penalty = float(spec["fail_penalty"])
        # Default False: OOB is a soft penalty only (see step()).
        self.terminate_on_oob = bool(spec.get("terminate_on_oob", False))
        self.fmax = float(spec["fmax"])
        self.ee_spawn = str(spec["ee_spawn"])
        self.ee_scale = float(spec["ee_scale"])
        self.tol = float(spec["tol"])
        self.ang_tol = float(spec["ang_tol"])
        self.vel_tol = float(spec["vel_tol"])
        self.obs_dim = 18 + FEAT_DIM

        seed0 = int(spec["seed"])
        quiet = bool(spec.get("quiet", True))

        self.envs = []
        self.rngs = []
        cfg = push_t_config()
        gx = float(getattr(cfg.task, "goal_x", 1.50))
        gy = float(getattr(cfg.task, "goal_y", 0.52))
        gth = float(getattr(cfg.task, "goal_theta", 0.0))
        self.ee_radius = float(cfg.scene.ee_radius) * self.ee_scale

        ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        with ctx:
            for i in range(self.n):
                # Separate RigidManager per env → already isolated. Keep
                # env_id=-1 so we don't pay per-env BVH rebuild cost.
                # Cross-env filtering (when sharing one manager) is handled
                # in BVH / RigidManager via domain.env_id.
                env = PushTEnv(push_t_config())
                env.ee_spawn = self.ee_spawn
                self.envs.append(env)
                self.rngs.append(np.random.default_rng(seed0 + i))
        self.sg, self.bg = self.envs[0].t_goal_parts(gx, gy, gth)
        self.gth = gth
        self.sg_np = np.asarray(self.sg, dtype=np.float64)
        self.bg_np = np.asarray(self.bg, dtype=np.float64)

        self.ep_len = np.zeros(self.n, dtype=np.int64)
        # Potential Φ = d_goal + reach_w * d_reach  →  r = Φ_prev - Φ
        # Absolute −d rewards + early fail terminations teach OOB suicide.
        self._phi = np.zeros(self.n, dtype=np.float64)
        self._obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        for i in range(self.n):
            self._obs[i] = self._reset_one(i)

    def _costs(self, st: np.ndarray) -> tuple[float, float, float]:
        """Return (d_goal, d_reach, phi)."""
        stem = st[1, :2]
        bar = st[2, :2]
        ee = st[0, :2]
        th = float(st[1, 2])
        d_s = float(np.linalg.norm(stem - self.sg_np))
        d_b = float(np.linalg.norm(bar - self.bg_np))
        ang = abs(float(np.arctan2(np.sin(th - self.gth),
                                   np.cos(th - self.gth))))
        d_goal = d_s + d_b + 0.5 * ang
        d_reach = float(min(
            np.linalg.norm(ee - stem), np.linalg.norm(ee - bar)))
        phi = d_goal + 0.5 * d_reach
        return d_goal, d_reach, phi

    def _reset_one(self, i: int) -> np.ndarray:
        env = self.envs[i]
        rng = self.rngs[i]
        env.reset(rng)
        env.set_ee_radius(self.ee_radius)
        obs = env._observe()
        self.ep_len[i] = 0
        _, _, self._phi[i] = self._costs(obs["obj_states"])
        return _obs_row(obs["obj_states"], self.sg, self.bg, self.gth)

    def get_obs(self) -> np.ndarray:
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        """``actions`` in Newtons, shape (n, 2)."""
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.bool_)
        time_outs = np.zeros(self.n, dtype=np.bool_)
        successes = np.zeros(self.n, dtype=np.float32)
        fails = np.zeros(self.n, dtype=np.float32)
        far = np.zeros(self.n, dtype=np.float32)
        oob = np.zeros(self.n, dtype=np.float32)
        xmin, xmax, ymin, ymax = self.workspace_xy

        for i, env in enumerate(self.envs):
            a = actions[i].astype(np.float32)
            for _ in range(self.hold):
                env.step(a)

            cur = env._observe()
            st = cur["obj_states"]
            d_goal, d_reach, phi = self._costs(st)
            real_ok = _succeeded(
                st, self.sg, self.bg,
                tol=self.tol, ang_tol=self.ang_tol, vel_tol=self.vel_tol)
            if real_ok:
                successes[i] = 1.0

            ee = st[0, :2]
            stem = st[1, :2]
            d_ee_t = d_reach
            out_of_bounds = (
                float(ee[0]) < xmin or float(ee[0]) > xmax
                or float(ee[1]) < ymin or float(ee[1]) > ymax
                or float(stem[0]) < xmin or float(stem[0]) > xmax
                or float(stem[1]) < ymin or float(stem[1]) > ymax
            )
            ee_too_far = (
                self.max_ee_t_dist < float("inf")
                and d_ee_t > self.max_ee_t_dist
            )
            # Hard fails only if explicitly enabled. Default: no fail-dones —
            # absolute −d + early terminate makes "suicide to OOB" optimal.
            failed = (not real_ok) and (
                (self.max_ee_t_dist < float("inf") and ee_too_far)
                or (self.fail_penalty > 0 and out_of_bounds
                    and bool(getattr(self, "terminate_on_oob", False)))
            )

            self.ep_len[i] += 1
            timed_out = bool(self.ep_len[i] >= self.max_episode_length)
            done = real_ok or failed or timed_out

            # Potential-based progress + sparse success.
            rew = float(self._phi[i] - phi)
            self._phi[i] = phi
            if real_ok:
                rew += 1.0
            if out_of_bounds:
                rew -= 0.05  # soft nudge only; do not truncate the episode
            if failed:
                rew -= abs(self.fail_penalty)
            rewards[i] = rew
            dones[i] = done
            time_outs[i] = timed_out and (not real_ok) and (not failed)
            if failed:
                fails[i] = 1.0
            if ee_too_far:
                far[i] = 1.0
            if out_of_bounds:
                oob[i] = 1.0

            if done:
                self._obs[i] = self._reset_one(i)
            else:
                self._obs[i] = _obs_row(st, self.sg, self.bg, self.gth)

        return {
            "obs": self._obs.copy(),
            "rewards": rewards,
            "dones": dones,
            "time_outs": time_outs,
            "successes": successes,
            "fails": fails,
            "far": far,
            "oob": oob,
            "ep_len": self.ep_len.copy(),
        }


def _worker_main(remote, parent_remote, spec: dict):
    parent_remote.close()
    # Avoid Warp/CPU oversubscription across many processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        batch = _LocalPushTBatch(spec)
        remote.send(("ready", {
            "n_local": batch.n,
            "obs_dim": batch.obs_dim,
            "sg": np.asarray(batch.sg, dtype=np.float32),
            "bg": np.asarray(batch.bg, dtype=np.float32),
            "gth": float(batch.gth),
            "budget": int(spec["budget"]),
        }))
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(("ok", batch.step(data)))
            elif cmd == "get_obs":
                remote.send(("ok", batch.get_obs()))
            elif cmd == "close":
                remote.send(("ok", None))
                break
            else:
                remote.send(("err", f"unknown cmd {cmd}"))
    except Exception as exc:  # noqa: BLE001 — surface to parent
        try:
            remote.send(("err", repr(exc)))
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        remote.close()


class PushTVecEnv(VecEnv):
    """Vectorized Push-T for rsl-rl OnPolicyRunner (multiprocess).

    Observation group ``policy`` = flatten(obj_states[3,6]) + task_features(13).
    Actions are interpreted in [-1, 1] and scaled by ``force_max``.
    Each env step holds the force for ``hold`` physics substeps (default 5).
    """

    def __init__(
        self,
        num_envs: int = 128,
        hold: int = 5,
        max_episode_length: int | None = None,
        device: str = "cpu",
        seed: int = 0,
        ee_spawn: str = "support",
        num_workers: int | None = None,
        max_ee_t_dist: float | None = None,
        workspace_xy: tuple = (-0.2, 2.4, -0.2, 1.6),
        fail_penalty: float = 2.0,
        ee_scale: float = 1.0,
        tol: float | None = None,
        ang_tol: float | None = None,
        vel_tol: float | None = None,
    ):
        from learning.tasks.push_t import push_t_config
        from learning.rl.imagine import FEAT_DIM

        self.num_envs = int(num_envs)
        self.num_actions = 2
        self.hold = int(hold)
        self.device = torch.device(device)
        # None / <=0 → disable EE-far termination (keep OOB + success).
        self.max_ee_t_dist = (
            float("inf") if max_ee_t_dist is None or float(max_ee_t_dist) <= 0
            else float(max_ee_t_dist)
        )
        self.workspace_xy = tuple(float(x) for x in workspace_xy)
        self.fail_penalty = float(fail_penalty)
        self._FEAT_DIM = FEAT_DIM
        self._obs_dim = 18 + FEAT_DIM

        cfg = push_t_config()
        if tol is not None:
            cfg.task.tol = float(tol)
        if ang_tol is not None:
            cfg.task.ang_tol = float(ang_tol)
        if vel_tol is not None:
            cfg.task.vel_tol = float(vel_tol)
        self.tol = float(cfg.task.tol)
        self.ang_tol = float(getattr(cfg.task, "ang_tol", 0.25))
        self.vel_tol = float(cfg.task.vel_tol)
        self.fmax = float(cfg.collect.force_max)
        budget = int(cfg.task.budget)
        self.max_episode_length = int(
            max_episode_length
            if max_episode_length is not None
            else max(budget // max(self.hold, 1), 64)
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        cpu = os.cpu_count() or 8
        if num_workers is None:
            # Windows often fails loading many Warp DLLs at once (WinError 8).
            # Cap workers and pack many compact envs into each process.
            num_workers = min(self.num_envs, max(1, min(6, cpu // 2)))
        self.num_workers = max(1, int(num_workers))
        self.num_workers = min(self.num_workers, self.num_envs)

        # Split envs across workers as evenly as possible.
        base = self.num_envs // self.num_workers
        rem = self.num_envs % self.num_workers
        self._counts = [
            base + (1 if w < rem else 0) for w in range(self.num_workers)]
        self._offsets = np.cumsum([0] + self._counts[:-1]).tolist()

        self.cfg = {
            "num_envs": self.num_envs,
            "hold": self.hold,
            "ee_spawn": ee_spawn,
            "max_ee_t_dist": self.max_ee_t_dist,
            "workspace_xy": self.workspace_xy,
            "num_workers": self.num_workers,
        }

        self._remotes: list[Any] = []
        self._procs: list[Process] = []
        metas = []
        # Start workers one-by-one so Warp DLL loads do not OOM (WinError 8).
        for w in range(self.num_workers):
            parent, child = Pipe()
            spec = {
                "n_local": self._counts[w],
                "hold": self.hold,
                "max_episode_length": self.max_episode_length,
                "max_ee_t_dist": self.max_ee_t_dist,
                "workspace_xy": self.workspace_xy,
                "fail_penalty": self.fail_penalty,
                "fmax": self.fmax,
                "ee_spawn": ee_spawn,
                "ee_scale": float(ee_scale),
                "tol": self.tol,
                "ang_tol": self.ang_tol,
                "vel_tol": self.vel_tol,
                "budget": budget,
                "seed": int(seed) + 10007 * w,
                "quiet": True,
            }
            proc = Process(
                target=_worker_main, args=(child, parent, spec), daemon=True)
            proc.start()
            child.close()
            self._procs.append(proc)
            self._remotes.append(parent)
            tag, payload = parent.recv()
            if tag != "ready":
                self.close()
                raise RuntimeError(f"worker {w} init failed: {payload}")
            metas.append(payload)
            print(
                f"  worker {w + 1}/{self.num_workers} ready "
                f"({self._counts[w]} envs)",
                flush=True)
        self.sg = metas[0]["sg"]
        self.bg = metas[0]["bg"]
        self.gth = float(metas[0]["gth"])

        obs = self._gather_obs()
        policy = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self._obs = TensorDict({"policy": policy}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------ API
    def get_observations(self) -> TensorDict:
        return self._obs.clone()

    def step(self, actions: torch.Tensor):
        a_cmd = actions.detach().float().cpu().numpy()
        a_cmd = np.clip(a_cmd, -1.0, 1.0) * self.fmax

        for w, remote in enumerate(self._remotes):
            lo = self._offsets[w]
            hi = lo + self._counts[w]
            remote.send(("step", a_cmd[lo:hi]))

        obs = np.empty((self.num_envs, self._obs_dim), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.empty(self.num_envs, dtype=np.bool_)
        time_outs = np.empty(self.num_envs, dtype=np.bool_)
        successes = np.empty(self.num_envs, dtype=np.float32)
        fails = np.empty(self.num_envs, dtype=np.float32)
        far = np.empty(self.num_envs, dtype=np.float32)
        oob = np.empty(self.num_envs, dtype=np.float32)
        ep_len = np.empty(self.num_envs, dtype=np.int64)

        for w, remote in enumerate(self._remotes):
            tag, payload = remote.recv()
            if tag != "ok":
                raise RuntimeError(f"worker step failed: {payload}")
            lo = self._offsets[w]
            hi = lo + self._counts[w]
            obs[lo:hi] = payload["obs"]
            rewards[lo:hi] = payload["rewards"]
            dones[lo:hi] = payload["dones"]
            time_outs[lo:hi] = payload["time_outs"]
            successes[lo:hi] = payload["successes"]
            fails[lo:hi] = payload["fails"]
            far[lo:hi] = payload["far"]
            oob[lo:hi] = payload["oob"]
            ep_len[lo:hi] = payload["ep_len"]

        self.episode_length_buf = torch.as_tensor(
            ep_len, device=self.device, dtype=torch.long)
        policy = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self._obs = TensorDict({"policy": policy}, batch_size=[self.num_envs])

        extras = {
            "time_outs": torch.as_tensor(
                time_outs, device=self.device, dtype=torch.bool),
            "log": {
                "/episode/success": float(successes.mean()),
                "/episode/fail": float(fails.mean()),
                "/episode/fail_ee_far": float(far.mean()),
                "/episode/fail_oob": float(oob.mean()),
                "/episode/reward_mean": float(rewards.mean()),
            },
        }
        return (
            self._obs.clone(),
            torch.as_tensor(rewards, device=self.device),
            torch.as_tensor(dones, device=self.device, dtype=torch.bool),
            extras,
        )

    def close(self):
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:  # noqa: BLE001
                pass
        for remote in self._remotes:
            try:
                remote.recv()
            except Exception:  # noqa: BLE001
                pass
            try:
                remote.close()
            except Exception:  # noqa: BLE001
                pass
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
        self._remotes.clear()
        self._procs.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------- helpers
    def _gather_obs(self) -> np.ndarray:
        for remote in self._remotes:
            remote.send(("get_obs", None))
        obs = np.empty((self.num_envs, self._obs_dim), dtype=np.float32)
        for w, remote in enumerate(self._remotes):
            tag, payload = remote.recv()
            if tag != "ok":
                raise RuntimeError(f"worker get_obs failed: {payload}")
            lo = self._offsets[w]
            hi = lo + self._counts[w]
            obs[lo:hi] = payload
        return obs
