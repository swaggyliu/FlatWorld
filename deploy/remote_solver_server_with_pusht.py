"""Patched solverServer entry — Push-T bridge on flexphysics.com WS location."""

from .api import api as fastapi_app
from .lifecycle import compute_pool_lifespan
from .flexphysics_pusht_proxy import flatworld_health, flatworld_ws
from starlette.applications import Starlette
from starlette.routing import Mount, Route, WebSocketRoute

# Unified ASGI app on node2 :9527 (proxied by gateway /cal/*).
# FlatWorld Push-T is bridged under the existing auth-free WS location
# /cal/task-data/ws/ so www.flexphysics.com needs no nginx edit.
app = Starlette(
    lifespan=compute_pool_lifespan,
    routes=[
        WebSocketRoute("/cal/task-data/ws/_fw", flatworld_ws),
        Route("/cal/task-data/ws/_fw/health", flatworld_health, methods=["GET"]),
        Mount("/cal", app=fastapi_app),
    ],
)
