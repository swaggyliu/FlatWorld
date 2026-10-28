"""ASGI helpers: expose Push-T behind www.flexphysics.com via existing /cal WS nginx.

Mounted on the node2 compute Starlette app (see solverServer patch) so gateway
nginx does not need a new location. Paths:

  wss://www.flexphysics.com/cal/task-data/ws/_fw
  https://www.flexphysics.com/cal/task-data/ws/_fw/health
"""

from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocket, WebSocketDisconnect

UPSTREAM_HTTP = "http://127.0.0.1:8765"
UPSTREAM_WS = "ws://127.0.0.1:8765/ws"


async def flatworld_health(_request: Request) -> Response:
    """Proxy GET /health; used under the auth-free /cal/task-data/ws/ location."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 8765), timeout=3.0
        )
    except Exception as exc:
        return Response(
            f'{{"ok":false,"error":"upstream_unreachable:{type(exc).__name__}"}}',
            status_code=503,
            media_type="application/json",
        )
    try:
        writer.write(
            b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(65536), timeout=3.0)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    if b"\r\n\r\n" not in raw:
        return Response(
            '{"ok":false,"error":"bad_upstream_response"}',
            status_code=502,
            media_type="application/json",
        )
    header_blob, body = raw.split(b"\r\n\r\n", 1)
    status_line = header_blob.split(b"\r\n", 1)[0].decode("latin1", "replace")
    try:
        status = int(status_line.split(" ", 2)[1])
    except Exception:
        status = 502
    return Response(body, status_code=status, media_type="application/json")


async def flatworld_ws(websocket: WebSocket) -> None:
    """Bidirectional WebSocket bridge to local Push-T (127.0.0.1:8765/ws)."""
    import websockets
    from websockets.exceptions import ConnectionClosed

    origin = websocket.headers.get("origin")
    # Prefer nginx X-Real-IP (gateway sets $remote_addr). Fall back to peer.
    x_real = (websocket.headers.get("x-real-ip") or "").strip()
    xff = (websocket.headers.get("x-forwarded-for") or "").split(",", 1)[0].strip()
    peer = websocket.client.host if websocket.client else "0.0.0.0"
    client_ip = x_real or xff or peer
    extra_headers: list[tuple[str, str]] = [
        ("X-Forwarded-For", client_ip),
        ("X-Real-IP", client_ip),
        ("X-Forwarded-Proto", "https"),
    ]
    if origin:
        extra_headers.append(("Origin", origin))

    try:
        upstream = await websockets.connect(
            UPSTREAM_WS,
            additional_headers=extra_headers,
            max_size=2**20,
            open_timeout=5,
        )
    except Exception:
        # Reject before accept so the browser sees a failed handshake.
        await websocket.close(code=1013)
        return

    await websocket.accept()

    async def client_to_upstream() -> None:
        try:
            while True:
                message: Any = await websocket.receive()
                mtype = message.get("type")
                if mtype == "websocket.disconnect":
                    break
                if mtype != "websocket.receive":
                    continue
                if "text" in message and message["text"] is not None:
                    await upstream.send(message["text"])
                elif "bytes" in message and message["bytes"] is not None:
                    await upstream.send(message["bytes"])
        except WebSocketDisconnect:
            pass
        except ConnectionClosed:
            pass
        finally:
            await upstream.close()

    async def upstream_to_client() -> None:
        try:
            async for message in upstream:
                if isinstance(message, bytes):
                    await websocket.send_bytes(message)
                else:
                    await websocket.send_text(message)
        except ConnectionClosed:
            pass
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    task_c = asyncio.create_task(client_to_upstream())
    task_u = asyncio.create_task(upstream_to_client())
    _done, pending = await asyncio.wait(
        {task_c, task_u}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(task_c, task_u, return_exceptions=True)
    try:
        await upstream.close()
    except Exception:
        pass
