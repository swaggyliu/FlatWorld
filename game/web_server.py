"""WebSocket backend for Push-T You vs AI (browser client).

Run on a GPU/CPU host that has FlatWorld + checkpoints::

    pip install fastapi uvicorn[standard]
    python -m game.web_server --host 127.0.0.1 --port 8765

Each WebSocket connection gets an isolated game session (shared WM/PPO weights).

Protocol (JSON text frames)::

    client -> {"type":"hello","round_duration":30}
    client -> {"type":"stick","x":0.5,"y":-0.2}  # virtual joystick [-1,1]
    client -> {"type":"keys","down":["w","a"]}    # legacy WASD/D-pad (mapped to stick)
    client -> {"type":"rematch"}
    client -> {"type":"pause"}
    client -> {"type":"difficulty","level":"low"|"medium"|"high"}
    server -> snapshot dict from PushTVersusSession.snapshot()
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
import traceback
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import count

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


DEFAULT_ORIGINS = frozenset({
    "https://swaggyliu.github.io",
    "https://www.flexphysics.com",
    "https://flexphysics.com",
    "http://localhost:8765",
    "http://127.0.0.1:8765",
})


@dataclass(frozen=True)
class WebLimits:
    per_ip: int = 2
    idle_seconds: float = 90.0
    session_seconds: float = 900.0
    paused_seconds: float = 60.0
    finished_seconds: float = 30.0
    message_bytes: int = 1024
    message_rate: float = 60.0
    message_burst: int = 90
    rematch_seconds: float = 2.0
    control_seconds: float = 0.25
    connect_burst: int = 16
    connect_window: float = 60.0
    tracked_ips: int = 4096
    check_seconds: float = 1.0


class _TokenBucket:
    def __init__(self, rate: float, capacity: int, now: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = now

    def take(self, now: float) -> bool:
        self.tokens = min(
            self.capacity, self.tokens + max(0.0, now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class _Admission:
    """Event-loop-only accounting; no await between reservation and counting."""

    def __init__(self, maximum: int, limits: WebLimits):
        self.maximum = maximum
        self.limits = limits
        self.active = 0
        self.per_ip: dict[str, int] = {}
        self.attempts: OrderedDict[str, _TokenBucket] = OrderedDict()

    def reserve(self, ip: str, now: float) -> bool:
        # Keep recent attempt counters across disconnects, with bounded memory.
        while self.attempts:
            oldest = next(iter(self.attempts))
            if now - self.attempts[oldest].updated < self.limits.connect_window:
                break
            self.attempts.popitem(last=False)
        bucket = self.attempts.get(ip)
        if bucket is None:
            if len(self.attempts) >= self.limits.tracked_ips:
                return False
            bucket = _TokenBucket(
                self.limits.connect_burst / self.limits.connect_window,
                self.limits.connect_burst, now)
            self.attempts[ip] = bucket
        self.attempts.move_to_end(ip)
        if not bucket.take(now):
            return False
        if self.active >= self.maximum or self.per_ip.get(ip, 0) >= self.limits.per_ip:
            return False
        self.active += 1
        self.per_ip[ip] = self.per_ip.get(ip, 0) + 1
        return True

    def release(self, ip: str) -> None:
        self.active -= 1
        remaining = self.per_ip[ip] - 1
        if remaining:
            self.per_ip[ip] = remaining
        else:
            del self.per_ip[ip]


async def _in_thread(fn, *args):
    """Wait for in-flight physics work before its lock/resources are released."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def _tune_cpu_threads() -> None:
    """Avoid torch/OMP oversubscription on small multi-session CPU boxes."""
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _build_app(
    wm: str,
    policy: str,
    hold: int,
    seed: int,
    hz: float,
    max_sessions: int,
    *,
    n_workers: int | None = None,
    limits: WebLimits | None = None,
    allowed_origins: set[str] | frozenset[str] | None = None,
    allow_no_origin: bool = False,
) -> FastAPI:
    limits = limits or WebLimits()
    origins = DEFAULT_ORIGINS if allowed_origins is None else frozenset(allowed_origins)
    admission = _Admission(max_sessions, limits)
    state = {
        "pool": None,
        "error": "",
        "seed_seq": count(seed),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        def _start_pool():
            _tune_cpu_threads()
            from game.session_pool import SessionProcessPool
            # One Warp+ONNX process per worker so multiplayer uses multiple cores.
            pool = SessionProcessPool(
                wm_path=wm, policy_path=policy, hold=hold, device="cpu",
                n_workers=n_workers)
            pool.start()
            return pool

        try:
            state["pool"] = await asyncio.to_thread(_start_pool)
            state["error"] = ""
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            print(f"[pusht-web] pool start failed: {state['error']}", flush=True)
        yield
        pool = state["pool"]
        if pool is not None:
            try:
                await asyncio.to_thread(pool.shutdown)
            except Exception:
                pass

    app = FastAPI(title="FlatWorld Push-T You vs AI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_methods=["GET"],
    )

    @app.get("/health")
    async def health():
        pool = state["pool"]
        ok = pool is not None and pool.alive
        return JSONResponse(
            {
                "ok": ok,
                "error": state["error"] or None,
                "active_sessions": admission.active,
                "max_sessions": int(max_sessions),
                "workers": None if pool is None else int(pool.n_workers),
                "backend": None if pool is None else pool.backend,
            },
            status_code=200 if ok else 503,
        )

    @app.websocket("/ws")
    async def ws_game(ws: WebSocket):
        # CORS middleware only handles HTTP. Enforce WS origins before upgrading.
        # WeChat mini program clients (wx.connectSocket) send no Origin header;
        # --allow-no-origin / PUSHT_ALLOW_NO_ORIGIN=1 admits them. Browser
        # clients still must present a whitelisted exact Origin.
        origin_headers = ws.headers.getlist("origin")
        if origin_headers:
            if len(origin_headers) != 1 or origin_headers[0] not in origins:
                await ws.close(code=1008)
                return
        elif not allow_no_origin:
            await ws.close(code=1008)
            return
        pool = state["pool"]
        if pool is None or not pool.alive:
            await ws.close(code=1013)
            return
        # Only Uvicorn's explicitly trusted loopback proxy may rewrite client IP.
        # Never read X-Forwarded-For directly from an untrusted websocket header.
        ip = ws.client.host if ws.client else "unknown"
        if not admission.reserve(ip, time.monotonic()):
            await ws.close(code=1013)
            return
        session_seed = next(state["seed_seq"])

        session = None
        # Protects rematch/pause/difficulty vs tick/snapshot. Stick updates are
        # fire-and-forget IPC so AI ticks never delay player input.
        session_lock = asyncio.Lock()
        stop = asyncio.Event()
        dt = 1.0 / max(float(hz), 5.0)
        tasks = []
        started = last_activity = time.monotonic()
        last_rematch = last_control = float("-inf")
        messages = _TokenBucket(limits.message_rate, limits.message_burst, started)
        close_code, close_reason = 1000, "Session ended. Press Rematch to reconnect."

        try:
            await ws.accept()

            def _make():
                return pool.create(
                    seed=session_seed, tick_hz=hz,
                    active_sessions=admission.active)

            try:
                session = await _in_thread(_make)
            except Exception:
                traceback.print_exc()
                await ws.send_json({
                    "type": "error",
                    "message": "Unable to start game. Please try again later.",
                })
                await ws.close()
                return

            print(
                f"[pusht-web] session start seed={session_seed} "
                f"active={admission.active}",
                flush=True,
            )

            async def reader():
                nonlocal last_activity, last_rematch, last_control
                nonlocal close_code, close_reason
                try:
                    while not stop.is_set():
                        event = await ws.receive()
                        if event["type"] == "websocket.disconnect":
                            return
                        raw = event.get("text")
                        if raw is None or len(raw.encode("utf-8")) > limits.message_bytes:
                            close_code, close_reason = 1009, "Invalid message size or format."
                            return
                        now = time.monotonic()
                        if not messages.take(now):
                            close_code, close_reason = 1008, "Too many messages. Please wait before reconnecting."
                            return
                        try:
                            def reject_constant(value):
                                raise ValueError(value)
                            msg = json.loads(raw, parse_constant=reject_constant)
                            if not isinstance(msg, dict):
                                raise ValueError("Expected an object")
                        except (ValueError, RecursionError):
                            close_code, close_reason = 1008, "Invalid game message."
                            return
                        typ = msg.get("type")
                        if not isinstance(typ, str):
                            close_code, close_reason = 1008, "Invalid message type."
                            return
                        obj = session
                        # Stick / legacy keys: apply immediately — never wait
                        # for tick/AI lock (that was the lag source).
                        if typ == "stick":
                            try:
                                sx = float(msg.get("x", 0.0))
                                sy = float(msg.get("y", 0.0))
                                if not math.isfinite(sx) or not math.isfinite(sy):
                                    raise ValueError("Non-finite stick")
                            except (TypeError, ValueError, OverflowError):
                                close_code, close_reason = 1008, "Invalid stick input."
                                return
                            obj.set_stick(sx, sy)
                            if sx or sy:
                                last_activity = now
                            continue
                        if typ == "keys":
                            # Pages cache / older clients still send WASD.
                            keys = msg.get("down", [])
                            if not isinstance(keys, list) or len(keys) > 4 or any(
                                not isinstance(k, str) or k.lower() not in {"w", "a", "s", "d"}
                                for k in keys
                            ):
                                close_code, close_reason = 1008, "Invalid keys input."
                                return
                            down = {k.lower() for k in keys}
                            fx = fy = 0.0
                            if "a" in down:
                                fx -= 1.0
                            if "d" in down:
                                fx += 1.0
                            if "w" in down:
                                fy += 1.0
                            if "s" in down:
                                fy -= 1.0
                            obj.set_stick(fx, fy)
                            if fx or fy:
                                last_activity = now
                            continue
                        # Optional on hello/rematch for cached clients. Reject
                        # booleans, floats and unbounded values before mutation.
                        if typ in {"hello", "rematch"} and "round_duration" in msg:
                            seconds = msg["round_duration"]
                            if type(seconds) is not int or not 10 <= seconds <= 180:
                                close_code, close_reason = 1008, "Invalid round duration."
                                return
                        if typ == "hello":
                            if "round_duration" in msg:
                                async with session_lock:
                                    await _in_thread(
                                        obj.set_round_duration, msg["round_duration"])
                            continue
                        if typ not in {"rematch", "pause", "difficulty"}:
                            close_code, close_reason = 1008, "Unknown game message."
                            return
                        if typ == "rematch":
                            if now - last_rematch < limits.rematch_seconds:
                                continue
                            last_rematch = now
                        else:
                            if now - last_control < limits.control_seconds:
                                continue
                            if typ == "difficulty" and (
                                not isinstance(msg.get("level"), str)
                                or msg["level"] not in {"low", "medium", "high"}
                            ):
                                close_code, close_reason = 1008, "Invalid difficulty."
                                return
                            last_control = now
                        async with session_lock:
                            if typ == "rematch":
                                secs = msg.get("round_duration")
                                await _in_thread(obj.rematch, secs)
                            elif typ == "pause":
                                await _in_thread(obj.toggle_pause)
                            elif typ == "difficulty":
                                await _in_thread(
                                    obj.set_difficulty, msg.get("level", "medium"))
                        last_activity = now
                except WebSocketDisconnect:
                    stop.set()
                except Exception:
                    stop.set()

            async def writer():
                try:
                    while not stop.is_set():
                        t0 = asyncio.get_event_loop().time()
                        # With process workers, 4 sessions can run physics in parallel.
                        # Still shed a bit above 4 so one box stays responsive.
                        n_active = int(admission.active)
                        if n_active <= 4:
                            tick_dt = 1.0 / 30.0
                        elif n_active <= 6:
                            tick_dt = 1.0 / 20.0
                        else:
                            tick_dt = 1.0 / 15.0
                        session.tick_hz = 1.0 / max(tick_dt, 1e-3)
                        async with session_lock:
                            def _advance(n=n_active):
                                return session.tick_snapshot(
                                    online=n, active_sessions=n)
                            snap = await _in_thread(_advance)
                        snap["type"] = "state"
                        snap["online"] = n_active
                        await ws.send_json(snap)
                        elapsed = asyncio.get_event_loop().time() - t0
                        await asyncio.sleep(max(0.0, tick_dt - elapsed))
                except WebSocketDisconnect:
                    stop.set()
                except Exception:
                    traceback.print_exc()
                    stop.set()

            async def watchdog():
                nonlocal close_reason
                paused_since = finished_since = None
                while not stop.is_set():
                    now = time.monotonic()
                    paused_since = (paused_since or now) if session.paused else None
                    finished_since = (finished_since or now) if session.phase == "over" else None
                    if now - started >= limits.session_seconds:
                        close_reason = "Session time limit. Press Rematch to reconnect."
                        return
                    if now - last_activity >= limits.idle_seconds:
                        close_reason = "Session idle. Press Rematch to reconnect."
                        return
                    if paused_since is not None and now - paused_since >= limits.paused_seconds:
                        close_reason = "Pause time limit. Press Rematch to reconnect."
                        return
                    if finished_since is not None and now - finished_since >= limits.finished_seconds:
                        close_reason = "Game finished. Press Rematch to reconnect."
                        return
                    await asyncio.sleep(limits.check_seconds)

            tasks = [asyncio.create_task(fn()) for fn in (reader, writer, watchdog)]
            await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(ws.close(code=close_code, reason=close_reason), 2.0)
            except Exception:
                pass
            if session is not None:
                try:
                    await _in_thread(session.close)
                except Exception:
                    pass
            admission.release(ip)
            print(
                f"[pusht-web] session end seed={session_seed} "
                f"active={admission.active}",
                flush=True,
            )

    # Serve Canvas frontend from the same origin as /ws (avoids mixed-content).
    docs_pusht = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "docs", "pusht")
    docs_pusht = os.path.abspath(docs_pusht)
    if os.path.isdir(docs_pusht):
        @app.get("/")
        async def index():
            return FileResponse(os.path.join(docs_pusht, "index.html"))

        app.mount("/", StaticFiles(directory=docs_pusht, html=True), name="pusht")

    return app


def main():
    _tune_cpu_threads()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--wm", default="learning/checkpoints/push_t")
    ap.add_argument("--policy",
                    default="learning/checkpoints/push_t_wm_rsl/best.pt")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000,
                    help="Seed for the original randomized starting poses.")
    # Default tick hint; live rate adapts with concurrent load (see writer).
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="Physics worker processes (0 = min(cpu_count, 4)).")
    ap.add_argument("--max-sessions", type=int, default=8,
                    help="Max concurrent players (each gets an isolated game).")
    ap.add_argument("--max-sessions-per-ip", type=int, default=8)
    ap.add_argument("--allowed-origin", action="append", default=[],
                    help="Additional exact browser origin (repeatable).")
    ap.add_argument("--allow-no-origin", action="store_true", default=None,
                    help="Accept WebSocket clients without an Origin header "
                         "(WeChat mini program). Also enabled via "
                         "PUSHT_ALLOW_NO_ORIGIN=1.")
    args = ap.parse_args()

    # Ensure repo root is on sys.path when launched as module.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    import uvicorn
    allow_no_origin = args.allow_no_origin or os.environ.get(
        "PUSHT_ALLOW_NO_ORIGIN", "").strip().lower() in {"1", "true", "yes"}
    app = _build_app(
        args.wm, args.policy, args.hold, args.seed, args.hz, args.max_sessions,
        n_workers=(None if args.workers <= 0 else args.workers),
        limits=WebLimits(per_ip=args.max_sessions_per_ip),
        allowed_origins=DEFAULT_ORIGINS | set(args.allowed_origin) | {
            value.strip() for value in os.environ.get("PUSHT_ALLOWED_ORIGINS", "").split(",")
            if value.strip()
        },
        allow_no_origin=allow_no_origin)
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="info",
        proxy_headers=True, forwarded_allow_ips="127.0.0.1,::1",
        ws="websockets", ws_max_size=WebLimits.message_bytes, ws_max_queue=8,
        ws_per_message_deflate=False)


if __name__ == "__main__":
    main()
