#!/usr/bin/env python3
"""Repeatable fixed-step wall-clock benchmark for the 10k spring-mass example.

Each sample launches a fresh process and runs a fixed number of headless
simulation frames. The result includes process startup, initialization,
simulation, final state readback, and assertions; it is not kernel-only.
Compare only on the same GitHub-hosted runner class.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "test2D" / "test_2Dspring_10000.py"


def run_once(steps: int, frame_dt: float) -> float:
    env = os.environ.copy()
    env.update({"HEADLESS": "1", "CI": "true", "FLATWORLD_DEVICE": "cpu"})
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, str(CASE), "--headless", "--steps", str(steps), "--frame-dt", str(frame_dt)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--steps", type=int, default=120, help="Fixed simulation frames per sample")
    parser.add_argument(
        "--frame-dt", type=float, default=1.0 / 1200.0,
        help="Simulated time advanced per frame; use a small stable interval for long runs",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1 or args.steps < 1 or args.frame_dt <= 0:
        parser.error("--warmup must be >= 0, --repeat and --steps must be >= 1, --frame-dt must be > 0")

    for _ in range(args.warmup):
        run_once(args.steps, args.frame_dt)

    samples = [run_once(args.steps, args.frame_dt) for _ in range(args.repeat)]
    result = {
        "case": "test2D/test_2Dspring_10000.py",
        "device": "cpu",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "simulation_steps": args.steps,
        "frame_dt_seconds": args.frame_dt,
        "simulated_time_seconds": round(args.steps * args.frame_dt, 6),
        "seconds": [round(x, 6) for x in samples],
        "median_seconds": round(statistics.median(samples), 6),
        "min_seconds": round(min(samples), 6),
        "max_seconds": round(max(samples), 6),
    }
    print("FLATWORLD_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
