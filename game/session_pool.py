"""Multi-process Push-T session workers.

Each worker process owns its own Warp envs + ONNX policy. Sessions on different
workers truly run in parallel (unlike threads, which serialize on Warp/CPU).

The asyncio web server only does WebSocket I/O and RPC into these workers.
"""

from __future__ import annotations

import os
import traceback
import itertools
import threading
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any


def _worker_main(
    req_q,
    rep_q,
    ready_q,
    worker_id: int,
    wm_path: str,
    policy_path: str,
    hold: int,
    device: str,
) -> None:
    """Entry point inside a spawned process."""
    try:
        # Limit BLAS/ORT threads so N workers do not oversubscribe a small box.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        from game.pusht_session import load_shared_pusht_assets, PushTVersusSession

        assets = load_shared_pusht_assets(
            wm_path=wm_path, policy_path=policy_path, device=device, hold=hold)
        sessions: dict[str, PushTVersusSession] = {}
        # Stick arrives ~60Hz; keep only the latest vector per session so a
        # backlog cannot delay physics ticks on the same FIFO.
        latest_stick: dict[str, tuple[float, float]] = {}
        deferred: list[dict] = []
        print(
            f"[pusht-worker {worker_id}] ready backend={assets.backend}",
            flush=True,
        )
        ready_q.put({"worker_id": worker_id, "ok": True, "backend": assets.backend})
    except Exception as exc:
        traceback.print_exc()
        ready_q.put({
            "worker_id": worker_id, "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        return

    def _reply(msg: dict, payload: dict) -> None:
        if msg.get("noreply"):
            return
        rep_q.put({"rid": msg.get("rid"), **payload})

    def _apply_latest_sticks() -> None:
        if not latest_stick:
            return
        pending = latest_stick.copy()
        latest_stick.clear()
        for sid, (x, y) in pending.items():
            sess = sessions.get(sid)
            if sess is not None:
                sess.set_stick(x, y)

    def _drain_sticks() -> None:
        """Fold queued stick msgs into latest_stick; defer everything else."""
        while True:
            try:
                nxt = req_q.get_nowait()
            except Empty:
                break
            if not isinstance(nxt, dict):
                continue
            if nxt.get("cmd") == "stick":
                sid = str(nxt.get("sid", ""))
                try:
                    latest_stick[sid] = (float(nxt["x"]), float(nxt["y"]))
                except (KeyError, TypeError, ValueError):
                    pass
                _reply(nxt, {"ok": True})
            else:
                deferred.append(nxt)

    while True:
        msg = deferred.pop(0) if deferred else req_q.get()
        if not isinstance(msg, dict):
            continue
        # Always collapse stick spam before any heavier command.
        if msg.get("cmd") == "stick":
            sid = str(msg.get("sid", ""))
            try:
                latest_stick[sid] = (float(msg["x"]), float(msg["y"]))
            except (KeyError, TypeError, ValueError):
                pass
            _reply(msg, {"ok": True})
            _drain_sticks()
            _apply_latest_sticks()
            continue

        _drain_sticks()
        _apply_latest_sticks()
        cmd = msg.get("cmd")
        try:
            if cmd == "shutdown":
                for sess in list(sessions.values()):
                    try:
                        sess.close()
                    except Exception:
                        pass
                sessions.clear()
                latest_stick.clear()
                try:
                    assets.close()
                except Exception:
                    pass
                _reply(msg, {"ok": True})
                break

            if cmd == "create":
                sid = str(msg["sid"])
                with assets.env_init_lock:
                    sess = PushTVersusSession(
                        shared=assets,
                        seed=int(msg["seed"]),
                        tick_hz=float(msg.get("tick_hz", 20.0)),
                    )
                sessions[sid] = sess
                assets.active_sessions = int(msg.get("active_sessions", len(sessions)))
                snap = sess.snapshot()
                _reply(msg, {"ok": True, "snap": snap})
                continue

            if cmd == "tick":
                sid = str(msg["sid"])
                sess = sessions[sid]
                sess.tick_hz = float(msg.get("tick_hz", sess.tick_hz))
                assets.active_sessions = int(
                    msg.get("active_sessions", assets.active_sessions or 1))
                sess.tick()
                snap = sess.snapshot()
                if "online" in msg:
                    snap["online"] = int(msg["online"])
                _reply(msg, {
                    "ok": True,
                    "snap": snap,
                    "paused": bool(sess.paused),
                    "phase": sess.phase,
                })
                continue

            if cmd == "control":
                sid = str(msg["sid"])
                sess = sessions[sid]
                op = msg.get("op")
                if op == "pause":
                    sess.paused = not sess.paused
                elif op == "difficulty":
                    sess._set_difficulty(msg.get("level", "medium"))
                elif op == "round_duration":
                    sess.set_round_duration(int(msg["seconds"]))
                elif op == "rematch":
                    if "seconds" in msg:
                        sess.set_round_duration(int(msg["seconds"]))
                    sess.next_episode()
                else:
                    raise ValueError(f"unknown control op {op!r}")
                snap = sess.snapshot()
                _reply(msg, {
                    "ok": True,
                    "snap": snap,
                    "paused": bool(sess.paused),
                    "phase": sess.phase,
                })
                continue

            if cmd == "close":
                sid = str(msg["sid"])
                sess = sessions.pop(sid, None)
                latest_stick.pop(sid, None)
                if sess is not None:
                    sess.close()
                assets.active_sessions = max(0, len(sessions))
                _reply(msg, {"ok": True})
                continue

            raise ValueError(f"unknown cmd {cmd!r}")
        except Exception as exc:
            traceback.print_exc()
            _reply(msg, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@dataclass
class _Worker:
    worker_id: int
    proc: Any
    req_q: Any
    rep_q: Any
    lock: threading.Lock
    _ids: Any
    n_sessions: int = 0

    def send_noreply(self, msg: dict) -> None:
        self.req_q.put({**msg, "noreply": True})

    def call(self, msg: dict, timeout: float = 60.0) -> dict:
        with self.lock:
            rid = next(self._ids)
            self.req_q.put({**msg, "rid": rid, "noreply": False})
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"worker {self.worker_id} RPC timeout on {msg.get('cmd')}")
                rep = self.rep_q.get(timeout=remaining)
                if rep.get("rid") == rid:
                    if not rep.get("ok", False):
                        raise RuntimeError(rep.get("error") or "worker RPC failed")
                    return rep


class RemoteSession:
    """Proxy for a PushTVersusSession living in a worker process."""

    def __init__(self, worker: _Worker, sid: str, pool: "SessionProcessPool"):
        self._worker = worker
        self.sid = sid
        self._pool = pool
        self.tick_hz = 20.0
        self.paused = False
        self.phase = "countdown"
        self._closed = False

    def set_stick(self, x: float, y: float) -> None:
        if self._closed:
            return
        self._worker.send_noreply({
            "cmd": "stick", "sid": self.sid, "x": float(x), "y": float(y),
        })

    def set_round_duration(self, seconds: int) -> None:
        self._call_control({"op": "round_duration", "seconds": int(seconds)})

    def next_episode(self) -> None:
        self._call_control({"op": "rematch"})

    def rematch(self, seconds: int | None = None) -> dict:
        payload = {"op": "rematch"}
        if seconds is not None:
            payload["seconds"] = int(seconds)
        return self._call_control(payload)

    def toggle_pause(self) -> dict:
        return self._call_control({"op": "pause"})

    def set_difficulty(self, level: str) -> dict:
        return self._call_control({"op": "difficulty", "level": level})

    def _call_control(self, payload: dict) -> dict:
        rep = self._worker.call({"cmd": "control", "sid": self.sid, **payload})
        self.paused = bool(rep.get("paused", self.paused))
        self.phase = str(rep.get("phase", self.phase))
        return rep.get("snap") or {}

    def tick_snapshot(self, *, online: int, active_sessions: int) -> dict:
        rep = self._worker.call({
            "cmd": "tick",
            "sid": self.sid,
            "tick_hz": float(self.tick_hz),
            "online": int(online),
            "active_sessions": int(active_sessions),
        })
        self.paused = bool(rep.get("paused", self.paused))
        self.phase = str(rep.get("phase", self.phase))
        return rep["snap"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._worker.call({"cmd": "close", "sid": self.sid}, timeout=30.0)
        finally:
            self._pool._release(self._worker)


class SessionProcessPool:
    """Pool of spawned workers; new sessions go to the least-loaded worker."""

    def __init__(
        self,
        *,
        wm_path: str,
        policy_path: str,
        hold: int = 5,
        device: str = "cpu",
        n_workers: int | None = None,
    ):
        import multiprocessing as mp

        self.wm_path = wm_path
        self.policy_path = policy_path
        self.hold = hold
        self.device = device
        # Prefer physical-core count when hyperthreads inflate cpu_count (e.g. 2C/4T).
        cpu = os.cpu_count() or 2
        default_workers = max(1, min(cpu // 2 if cpu >= 4 else cpu, 4))
        self.n_workers = max(1, int(n_workers if n_workers is not None else default_workers))
        self.ctx = mp.get_context("spawn")
        self.workers: list[_Worker] = []
        self._sid = itertools.count(1)
        self._lock = threading.Lock()
        self.backend = "unknown"
        self._started = False

    def start(self, ready_timeout: float = 180.0) -> None:
        if self._started:
            return
        ready_q = self.ctx.Queue()
        for i in range(self.n_workers):
            req_q = self.ctx.Queue()
            rep_q = self.ctx.Queue()
            proc = self.ctx.Process(
                target=_worker_main,
                name=f"pusht-worker-{i}",
                args=(
                    req_q, rep_q, ready_q, i,
                    self.wm_path, self.policy_path, self.hold, self.device,
                ),
                daemon=True,
            )
            proc.start()
            self.workers.append(_Worker(
                worker_id=i,
                proc=proc,
                req_q=req_q,
                rep_q=rep_q,
                lock=threading.Lock(),
                _ids=itertools.count(1),
            ))

        deadline = time.monotonic() + ready_timeout
        backends = []
        for _ in range(self.n_workers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.shutdown()
                raise TimeoutError("session workers failed to become ready")
            msg = ready_q.get(timeout=remaining)
            if not msg.get("ok"):
                self.shutdown()
                raise RuntimeError(
                    f"worker {msg.get('worker_id')} failed: {msg.get('error')}")
            backends.append(msg.get("backend"))
        self.backend = backends[0] if backends else "unknown"
        self._started = True
        print(
            f"[pusht-web] process pool ready workers={self.n_workers} "
            f"backend={self.backend}",
            flush=True,
        )

    def create(self, *, seed: int, tick_hz: float, active_sessions: int) -> RemoteSession:
        with self._lock:
            worker = min(self.workers, key=lambda w: (w.n_sessions, w.worker_id))
            worker.n_sessions += 1
            sid = f"w{worker.worker_id}-s{next(self._sid)}"
        try:
            rep = worker.call({
                "cmd": "create",
                "sid": sid,
                "seed": int(seed),
                "tick_hz": float(tick_hz),
                "active_sessions": int(active_sessions),
            }, timeout=120.0)
        except Exception:
            self._release(worker)
            raise
        sess = RemoteSession(worker, sid, self)
        sess.tick_hz = float(tick_hz)
        snap = rep.get("snap") or {}
        sess.paused = bool(snap.get("paused", False))
        sess.phase = str(snap.get("phase", "countdown"))
        print(
            f"[pusht-web] session {sid} on worker {worker.worker_id} "
            f"(load={worker.n_sessions})",
            flush=True,
        )
        return sess

    def _release(self, worker: _Worker) -> None:
        with self._lock:
            worker.n_sessions = max(0, worker.n_sessions - 1)

    @property
    def alive(self) -> bool:
        return self._started and all(w.proc.is_alive() for w in self.workers)

    def shutdown(self) -> None:
        for w in self.workers:
            try:
                if w.proc.is_alive():
                    w.req_q.put({"cmd": "shutdown", "rid": f"stop-{w.worker_id}"})
            except Exception:
                pass
        deadline = time.monotonic() + 10.0
        for w in self.workers:
            remaining = max(0.1, deadline - time.monotonic())
            w.proc.join(timeout=remaining)
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(timeout=2.0)
        self.workers.clear()
        self._started = False
