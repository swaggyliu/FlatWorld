"""Concurrent Push-T WebSocket load test (run on the game host vs localhost).

Example:
  python deploy/loadtest_pusht.py --url ws://127.0.0.1:8765/ws --clients 8 --seconds 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time


async def one_client(idx: int, url: str, seconds: float, hz_stick: float) -> dict:
    import websockets

    intervals: list[float] = []
    tick_over: int = 0
    frames = 0
    err = ""
    t_end = time.perf_counter() + seconds
    stick_dt = 1.0 / max(hz_stick, 1.0)
    next_stick = 0.0
    last = None
    try:
        async with websockets.connect(
            url, max_size=2**20, open_timeout=10, close_timeout=2,
            additional_headers={"Origin": "https://www.flexphysics.com"},
        ) as ws:
            await ws.send(json.dumps({"type": "hello", "round_duration": 60}))
            await ws.send(json.dumps({"type": "difficulty", "level": "high"}))
            while time.perf_counter() < t_end:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                now = time.perf_counter()
                if last is not None:
                    dt = now - last
                    intervals.append(dt)
                    if dt > 0.055:  # >55ms ≈ dropped below ~18 Hz
                        tick_over += 1
                last = now
                frames += 1
                if now >= next_stick:
                    # Circle the stick so AI + player physics stay busy.
                    ang = now * 2.2 + idx
                    await ws.send(json.dumps({
                        "type": "stick",
                        "x": float(0.7 * __import__("math").cos(ang)),
                        "y": float(0.7 * __import__("math").sin(ang)),
                    }))
                    next_stick = now + stick_dt
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    if not intervals:
        return {"idx": idx, "ok": False, "error": err or "no frames"}
    return {
        "idx": idx,
        "ok": True,
        "error": err,
        "frames": frames,
        "hz_mean": 1.0 / statistics.mean(intervals),
        "hz_p50": 1.0 / statistics.median(intervals),
        "dt_p95_ms": 1000.0 * statistics.quantiles(intervals, n=20)[18],
        "dt_max_ms": 1000.0 * max(intervals),
        "slow_frames": tick_over,
        "slow_pct": 100.0 * tick_over / len(intervals),
    }


async def staggered_client(idx: int, url: str, seconds: float, hz_stick: float, stagger: float):
    if stagger > 0:
        await asyncio.sleep(idx * stagger)
    return await one_client(idx, url, seconds, hz_stick)


async def main_async(args):
    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        staggered_client(i, args.url, args.seconds, args.stick_hz, args.stagger)
        for i in range(args.clients)
    ])
    wall = time.perf_counter() - t0
    ok = [r for r in results if r.get("ok")]
    print(f"clients={args.clients} wall={wall:.1f}s ok={len(ok)}/{args.clients}")
    for r in results:
        if not r.get("ok"):
            print(f"  #{r['idx']} FAIL {r.get('error')}")
            continue
        print(
            f"  #{r['idx']} hz_mean={r['hz_mean']:.1f} p50={r['hz_p50']:.1f} "
            f"dt_p95={r['dt_p95_ms']:.1f}ms max={r['dt_max_ms']:.1f}ms "
            f"slow={r['slow_pct']:.1f}% frames={r['frames']}"
        )
    if ok:
        print(
            f"SUMMARY mean_hz={statistics.mean(r['hz_mean'] for r in ok):.1f} "
            f"worst_p95_ms={max(r['dt_p95_ms'] for r in ok):.1f} "
            f"worst_max_ms={max(r['dt_max_ms'] for r in ok):.1f} "
            f"avg_slow%={statistics.mean(r['slow_pct'] for r in ok):.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--stick-hz", type=float, default=30.0)
    ap.add_argument("--stagger", type=float, default=0.15,
                    help="Delay between client connects (seconds).")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
