"""Exercise the real ASGI routes with cheap sessions, without loading AI weights."""

import asyncio
import json
import sys
import threading
import time
import types
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from game import web_server


ORIGIN = "https://swaggyliu.github.io"


class Session:
    instances = []
    reset_delay = 0
    construction_error = False

    def __init__(self, **kwargs):
        if self.construction_error:
            raise RuntimeError("private server details")
        self.paused = False
        self.phase = "race"
        self.difficulty = "medium"
        self.rematches = 0
        self.stick = (0, 0)
        self.closed = False
        self.reset_running = False
        self.close_during_reset = False
        self.reset_thread = None
        self.instances.append(self)

    def tick(self):
        pass

    def snapshot(self):
        return {"phase": self.phase, "paused": self.paused,
                "difficulty": self.difficulty, "rematches": self.rematches,
                "stick": self.stick}

    def next_episode(self):
        self.reset_running = True
        self.reset_thread = threading.get_ident()
        try:
            time.sleep(self.reset_delay)
            self.rematches += 1
            self.phase = "countdown"
        finally:
            self.reset_running = False

    def set_stick(self, x, y):
        self.stick = (x, y)

    def _set_difficulty(self, level):
        self.difficulty = level

    def close(self):
        self.close_during_reset = self.reset_running
        self.closed = True


@pytest.fixture
def sessions(monkeypatch):
    Session.instances = []
    monkeypatch.setattr(Session, "reset_delay", 0)
    monkeypatch.setattr(Session, "construction_error", False)
    monkeypatch.setattr(web_server, "_tune_cpu_threads", lambda: None)
    monkeypatch.setitem(sys.modules, "game.pusht_session",
                        types.SimpleNamespace(
                            PushTVersusSession=Session,
                            load_shared_pusht_assets=lambda **kw: types.SimpleNamespace(close=lambda: None)))
    return Session


@asynccontextmanager
async def server(**overrides):
    maximum = overrides.pop("maximum", 8)
    limits = replace(web_server.WebLimits(), check_seconds=0.005, **overrides)
    app = web_server._build_app("unused", "unused", 5, 1000, 100, maximum, limits=limits)
    async with app.router.lifespan_context(app):
        yield app


class Connection:
    def __init__(self, app, ip="192.0.2.1", origin=ORIGIN, extra_headers=()):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.incoming.put_nowait({"type": "websocket.connect"})
        headers = list(extra_headers)
        if origin is not None:
            headers.append((b"origin", origin.encode()))
        scope = {"type": "websocket", "asgi": {"version": "3.0"},
                 "path": "/ws", "raw_path": b"/ws", "root_path": "",
                 "query_string": b"", "headers": headers,
                 "client": (ip, 12345), "server": ("localhost", 8765),
                 "scheme": "ws", "subprotocols": []}
        self.task = asyncio.create_task(app(scope, self.incoming.get, self.outgoing.put))

    async def event(self):
        return await asyncio.wait_for(self.outgoing.get(), 3)

    async def accepted(self):
        assert (await self.event())["type"] == "websocket.accept"
        assert (await self.event())["type"] == "websocket.send"

    def send(self, message):
        self.raw(json.dumps(message))

    def raw(self, text):
        self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    async def closed(self):
        while True:
            event = await self.event()
            if event["type"] == "websocket.close":
                await asyncio.wait_for(self.task, 3)
                return event

    async def disconnect(self):
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self.task, 3)


async def eventually(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.002)
    await asyncio.wait_for(poll(), 2)


@pytest.mark.parametrize("origin,extra", [
    (None, ()), ("null", ()), ("https://untrusted.example", ()),
    (ORIGIN, ((b"origin", b"https://untrusted.example"),)),
])
def test_disallowed_origins_allocate_no_session(sessions, origin, extra):
    async def scenario():
        async with server() as app:
            ws = Connection(app, origin=origin, extra_headers=extra)
            assert (await ws.closed())["code"] == 1008
            assert not sessions.instances
    asyncio.run(scenario())


def test_ip_and_global_quotas_and_disconnect_reclaim(sessions):
    async def scenario():
        async with server(maximum=3) as app:
            first, second = Connection(app), Connection(app)
            await first.accepted()
            await second.accepted()
            spoofed = Connection(app, extra_headers=((b"x-forwarded-for", b"192.0.2.99"),))
            assert (await spoofed.closed())["code"] == 1013
            other = Connection(app, ip="192.0.2.2")
            await other.accepted()
            full = Connection(app, ip="192.0.2.3")
            assert (await full.closed())["code"] == 1013
            assert len(sessions.instances) == 3
            await first.disconnect()
            replacement = Connection(app, ip="192.0.2.3")
            await replacement.accepted()
            await asyncio.gather(second.disconnect(), other.disconnect(), replacement.disconnect())
            assert all(s.closed for s in sessions.instances)
    asyncio.run(scenario())


def test_connection_rate_survives_reconnect(sessions):
    async def scenario():
        async with server(connect_burst=2) as app:
            for _ in range(2):
                ws = Connection(app)
                await ws.accepted()
                await ws.disconnect()
            ws = Connection(app)
            assert (await ws.closed())["code"] == 1013
            other = Connection(app, ip="192.0.2.2")
            await other.accepted()
            await other.disconnect()
    asyncio.run(scenario())


def test_proxy_client_ips_are_separate_and_untrusted_headers_cannot_spoof(sessions):
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    async def scenario():
        async with server(per_ip=1) as app:
            proxy = ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1,::1")
            first = Connection(proxy, ip="127.0.0.1", extra_headers=(
                (b"x-forwarded-for", b"192.0.2.1"),))
            second = Connection(proxy, ip="127.0.0.1", extra_headers=(
                (b"x-forwarded-for", b"192.0.2.2"),))
            await first.accepted()
            await second.accepted()
            spoofed = Connection(proxy, ip="127.0.0.1", extra_headers=(
                (b"x-forwarded-for", b"198.51.100.99, 192.0.2.1"),))
            assert (await spoofed.closed())["code"] == 1013
            direct = Connection(proxy, ip="192.0.2.1", extra_headers=(
                (b"x-forwarded-for", b"192.0.2.99"),))
            assert (await direct.closed())["code"] == 1013
            await asyncio.gather(first.disconnect(), second.disconnect())
    asyncio.run(scenario())


@pytest.mark.parametrize("limit,change", [
    ("idle_seconds", None), ("paused_seconds", "paused"),
    ("finished_seconds", "phase"), ("session_seconds", None),
])
def test_session_expiry_reclaims_slot(sessions, limit, change):
    async def scenario():
        async with server(maximum=1, **{limit: 0.06}) as app:
            ws = Connection(app)
            await ws.accepted()
            if change:
                setattr(sessions.instances[-1], change, True if change == "paused" else "over")
            # Non-input hello frames do not keep an idle session alive.
            ws.send({"type": "hello"})
            event = await ws.closed()
            assert event["code"] == 1000
            assert "limit" in event["reason"] or "idle" in event["reason"] or "finished" in event["reason"]
            replacement = Connection(app)
            await replacement.accepted()
            await replacement.disconnect()
    asyncio.run(scenario())


@pytest.mark.parametrize("raw,code", [
    ("x" * 1025, 1009), ('{"type":"hello","data":"' + "汉" * 400 + '"}', 1009),
    ("[]", 1008), ("null", 1008), ("{", 1008),
    ('{"type":[]}', 1008), ('{"type":"stick","x":NaN}', 1008),
    ('{"type":"stick","x":"Infinity"}', 1008),
    ('{"type":"keys","down":{}}', 1008),
    ('{"type":"difficulty","level":[]}', 1008),
])
def test_invalid_messages_close_only_sender(sessions, raw, code):
    async def scenario():
        async with server() as app:
            bad, good = Connection(app), Connection(app, ip="192.0.2.2")
            await bad.accepted()
            await good.accepted()
            bad.raw(raw)
            assert (await bad.closed())["code"] == code
            good.send({"type": "stick", "x": 0.5, "y": -0.2})
            await eventually(lambda: sessions.instances[1].stick == (0.5, -0.2))
            await good.disconnect()
    asyncio.run(scenario())


def test_binary_message_rejected(sessions):
    async def scenario():
        async with server() as app:
            ws = Connection(app)
            await ws.accepted()
            ws.incoming.put_nowait({"type": "websocket.receive", "bytes": b"{}"})
            assert (await ws.closed())["code"] == 1009
    asyncio.run(scenario())


def test_message_flood_and_rematch_cooldown(sessions):
    async def scenario():
        async with server(message_burst=10, message_rate=1) as app:
            ws = Connection(app)
            await ws.accepted()
            for _ in range(20):
                ws.send({"type": "rematch"})
            assert (await ws.closed())["code"] == 1008
            assert sessions.instances[0].rematches == 1
    asyncio.run(scenario())


def test_normal_joystick_rate_is_allowed(sessions):
    async def scenario():
        async with server() as app:
            ws = Connection(app)
            await ws.accepted()
            for i in range(100):
                ws.send({"type": "stick", "x": 0.5, "y": i / 100})
                await asyncio.sleep(0.025)
            await eventually(lambda: sessions.instances[0].stick == (0.5, 0.99))
            assert not ws.task.done()
            await ws.disconnect()
    asyncio.run(scenario())


def test_reset_does_not_block_other_player_and_cleanup_waits(sessions, monkeypatch):
    monkeypatch.setattr(sessions, "reset_delay", 0.15)
    async def scenario():
        async with server() as app:
            first, other = Connection(app), Connection(app, ip="192.0.2.2")
            await first.accepted()
            await other.accepted()
            first.send({"type": "rematch"})
            await eventually(lambda: sessions.instances[0].reset_running)
            other.send({"type": "stick", "x": 0.6, "y": 0.1})
            await eventually(lambda: sessions.instances[1].stick == (0.6, 0.1))
            assert sessions.instances[0].reset_running
            assert sessions.instances[0].reset_thread != threading.get_ident()
            # Cancel the app while the reset worker is in flight.
            first.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first.task
            assert sessions.instances[0].closed
            assert not sessions.instances[0].close_during_reset
            await other.disconnect()
    asyncio.run(scenario())


def test_failed_creation_releases_quota_and_hides_details(sessions, monkeypatch):
    async def scenario():
        async with server(maximum=1) as app:
            monkeypatch.setattr(sessions, "construction_error", True)
            ws = Connection(app)
            assert (await ws.event())["type"] == "websocket.accept"
            error = json.loads((await ws.event())["text"])
            assert "private" not in error["message"]
            await ws.closed()
            monkeypatch.setattr(sessions, "construction_error", False)
            retry = Connection(app)
            await retry.accepted()
            await retry.disconnect()
    asyncio.run(scenario())
