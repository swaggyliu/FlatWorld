"""WebSocket backend for Push-T You vs AI (browser client).

Run on a GPU/CPU host that has FlatWorld + checkpoints::

    pip install fastapi uvicorn[standard]
    python -m game.web_server --host 0.0.0.0 --port 8765

Each WebSocket connection gets an isolated game session (shared WM/PPO weights).

Protocol (JSON text frames)::

    client -> {"type":"hello"}
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
import os
import traceback
from contextlib import asynccontextmanager
from itertools import count

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


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
) -> FastAPI:
    state = {
        "assets": None,
        "error": "",
        "active": 0,
        "lock": asyncio.Lock(),
        "seed_seq": count(seed),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        def _load():
            _tune_cpu_threads()
            from game.pusht_session import load_shared_pusht_assets
            # Aliyun game host has no GPU; map_location=cpu is intentional.
            return load_shared_pusht_assets(
                wm_path=wm, policy_path=policy, device="cpu", hold=hold)

        try:
            state["assets"] = await asyncio.to_thread(_load)
            state["error"] = ""
            print("[pusht-web] shared assets ready", flush=True)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            print(f"[pusht-web] load failed: {state['error']}", flush=True)
        yield
        assets = state["assets"]
        if assets is not None:
            try:
                assets.close()
            except Exception:
                pass

    app = FastAPI(title="FlatWorld Push-T You vs AI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        ok = state["assets"] is not None
        return JSONResponse(
            {
                "ok": ok,
                "error": state["error"] or None,
                "active_sessions": int(state["active"]),
                "max_sessions": int(max_sessions),
            },
            status_code=200 if ok else 503,
        )

    @app.websocket("/ws")
    async def ws_game(ws: WebSocket):
        await ws.accept()
        if state["assets"] is None:
            await ws.send_json({
                "type": "error",
                "message": state["error"] or "session not ready",
            })
            await ws.close()
            return

        async with state["lock"]:
            if state["active"] >= max_sessions:
                await ws.send_json({
                    "type": "error",
                    "message": f"server full ({max_sessions} players) — try again soon",
                })
                await ws.close()
                return
            state["active"] += 1
            session_seed = next(state["seed_seq"])

        session = None
        # Protects rematch/pause/difficulty vs tick/snapshot. Stick updates are
        # intentionally lock-free so AI inference never delays player input.
        session_lock = asyncio.Lock()
        stop = asyncio.Event()
        dt = 1.0 / max(float(hz), 5.0)

        try:
            from game.pusht_session import PushTVersusSession

            def _make():
                return PushTVersusSession(
                    shared=state["assets"],
                    seed=session_seed,
                )

            try:
                session = await asyncio.to_thread(_make)
            except Exception as exc:
                traceback.print_exc()
                await ws.send_json({
                    "type": "error",
                    "message": f"failed to start game: {type(exc).__name__}: {exc}",
                })
                await ws.close()
                return

            print(
                f"[pusht-web] session start seed={session_seed} "
                f"active={state['active']}",
                flush=True,
            )

            async def reader():
                try:
                    while not stop.is_set():
                        raw = await ws.receive_text()
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        typ = msg.get("type")
                        obj = session
                        # Stick / legacy keys: apply immediately — never wait
                        # for tick/AI lock (that was the lag source).
                        if typ == "stick":
                            try:
                                sx = float(msg.get("x", 0.0))
                                sy = float(msg.get("y", 0.0))
                            except (TypeError, ValueError):
                                sx = sy = 0.0
                            obj.set_stick(sx, sy)
                            continue
                        if typ == "keys":
                            # Pages cache / older clients still send WASD.
                            down = {
                                str(k).lower()
                                for k in (msg.get("down") or [])
                            }
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
                            continue
                        async with session_lock:
                            if typ == "rematch":
                                obj.next_episode()
                            elif typ == "pause":
                                obj.paused = not obj.paused
                            elif typ == "difficulty":
                                obj._set_difficulty(msg.get("level", "medium"))
                            elif typ == "hello":
                                pass
                except WebSocketDisconnect:
                    stop.set()
                except Exception:
                    stop.set()

            async def writer():
                try:
                    while not stop.is_set():
                        t0 = asyncio.get_event_loop().time()
                        async with session_lock:
                            await asyncio.to_thread(session.tick)
                            snap = session.snapshot()
                        snap["type"] = "state"
                        await ws.send_json(snap)
                        # Sleep the remainder of the tick period so slow AI
                        # ticks do not stack extra delay on top of overruns.
                        elapsed = asyncio.get_event_loop().time() - t0
                        await asyncio.sleep(max(0.0, dt - elapsed))
                except WebSocketDisconnect:
                    stop.set()
                except Exception:
                    traceback.print_exc()
                    stop.set()

            reader_t = asyncio.create_task(reader())
            writer_t = asyncio.create_task(writer())
            await asyncio.wait(
                {reader_t, writer_t}, return_when=asyncio.FIRST_COMPLETED)
            stop.set()
            for t in (reader_t, writer_t):
                t.cancel()
            try:
                await ws.close()
            except Exception:
                pass
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            async with state["lock"]:
                state["active"] = max(0, int(state["active"]) - 1)
            print(
                f"[pusht-web] session end seed={session_seed} "
                f"active={state['active']}",
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
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--wm", default="learning/checkpoints/push_t")
    ap.add_argument("--policy",
                    default="learning/checkpoints/push_t_wm_rsl/best.pt")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    # 30Hz broadcast; tick sleep accounts for AI overrun (see writer).
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--max-sessions", type=int, default=8,
                    help="Max concurrent players (each gets an isolated game).")
    args = ap.parse_args()

    # Ensure repo root is on sys.path when launched as module.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    import uvicorn
    app = _build_app(
        args.wm, args.policy, args.hold, args.seed, args.hz, args.max_sessions)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
