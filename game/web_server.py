"""WebSocket backend for Push-T You vs AI (browser client).

Run on a GPU/CPU host that has FlatWorld + checkpoints::

    pip install fastapi uvicorn[standard]
    python -m game.web_server --host 0.0.0.0 --port 8765

Protocol (JSON text frames)::

    client -> {"type":"hello"}
    client -> {"type":"keys","down":["w","a"]}   # currently held keys
    client -> {"type":"rematch"}
    client -> {"type":"pause"}
    server -> snapshot dict from PushTVersusSession.snapshot()
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


def _build_app(wm: str, policy: str, hold: int, seed: int, hz: float) -> FastAPI:
    session = {"obj": None, "lock": asyncio.Lock(), "error": ""}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Lazy-load in a worker thread so startup stays responsive.
        def _load():
            from game.pusht_session import PushTVersusSession
            return PushTVersusSession(
                wm_path=wm, policy_path=policy, device="cpu",
                hold=hold, seed=seed)

        try:
            session["obj"] = await asyncio.to_thread(_load)
            session["error"] = ""
            print("[pusht-web] session ready", flush=True)
        except Exception as exc:
            session["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            print(f"[pusht-web] load failed: {session['error']}", flush=True)
        yield
        obj = session["obj"]
        if obj is not None:
            try:
                obj.close()
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
        ok = session["obj"] is not None
        return JSONResponse(
            {"ok": ok, "error": session["error"] or None},
            status_code=200 if ok else 503,
        )

    @app.websocket("/ws")
    async def ws_game(ws: WebSocket):
        await ws.accept()
        if session["obj"] is None:
            await ws.send_json({
                "type": "error",
                "message": session["error"] or "session not ready",
            })
            await ws.close()
            return

        keys: set[str] = set()
        stop = asyncio.Event()
        dt = 1.0 / max(float(hz), 5.0)

        async def reader():
            nonlocal keys
            try:
                while not stop.is_set():
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    typ = msg.get("type")
                    async with session["lock"]:
                        obj = session["obj"]
                        if typ == "keys":
                            down = {str(k).lower() for k in msg.get("down", [])}
                            keys = down & {"w", "a", "s", "d"}
                            for k in ("w", "a", "s", "d"):
                                obj.set_key(k, k in keys)
                        elif typ == "rematch":
                            obj.next_episode()
                        elif typ == "pause":
                            obj.paused = not obj.paused
                        elif typ == "hello":
                            pass
            except WebSocketDisconnect:
                stop.set()
            except Exception:
                stop.set()

        async def writer():
            try:
                while not stop.is_set():
                    async with session["lock"]:
                        obj = session["obj"]
                        await asyncio.to_thread(obj.tick)
                        snap = obj.snapshot()
                    snap["type"] = "state"
                    await ws.send_json(snap)
                    await asyncio.sleep(dt)
            except WebSocketDisconnect:
                stop.set()
            except Exception:
                traceback.print_exc()
                stop.set()

        reader_t = asyncio.create_task(reader())
        writer_t = asyncio.create_task(writer())
        done, pending = await asyncio.wait(
            {reader_t, writer_t}, return_when=asyncio.FIRST_COMPLETED)
        stop.set()
        for t in pending:
            t.cancel()
        try:
            await ws.close()
        except Exception:
            pass

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--wm", default="learning/checkpoints/push_t")
    ap.add_argument("--policy",
                    default="learning/checkpoints/push_t_wm_rsl/best.pt")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    # Ensure repo root is on sys.path when launched as module.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    import uvicorn
    app = _build_app(args.wm, args.policy, args.hold, args.seed, args.hz)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
