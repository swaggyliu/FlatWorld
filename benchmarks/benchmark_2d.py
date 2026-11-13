#!/usr/bin/env python3
"""Repeatable fixed-step benchmark for the 10k spring-mass example.

Each sample reports both fresh-process wall time and in-process simulation
elapsed time. The latter starts after scene initialization and ends after the
final device-to-host position readback. Neither metric is kernel-only.
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


def run_once(steps: int, frame_dt: float) -> tuple[float, float]:
    env = os.environ.copy()
    env.update({"HEADLESS": "1", "CI": "true", "FLATWORLD_DEVICE": "cpu"})
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(CASE), "--headless", "--steps", str(steps), "--frame-dt", str(frame_dt)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - start
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, completed.args, output=completed.stdout)

    simulation_seconds = None
    for line in completed.stdout.splitlines():
        if line.startswith("FLATWORLD_SIMULATION_SECONDS="):
            simulation_seconds = float(line.split("=", 1)[1])
    if simulation_seconds is None:
        print(completed.stdout, file=sys.stderr)
        raise RuntimeError("Benchmark child did not report FLATWORLD_SIMULATION_SECONDS")
    return wall_seconds, simulation_seconds


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
    wall_samples = [sample[0] for sample in samples]
    simulation_samples = [sample[1] for sample in samples]
    result = {
        "case": "test2D/test_2Dspring_10000.py",
        "device": "cpu",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "simulation_steps": args.steps,
        "frame_dt_seconds": args.frame_dt,
        "simulated_time_seconds": round(args.steps * args.frame_dt, 6),
        "seconds": [round(x, 6) for x in wall_samples],
        "median_seconds": round(statistics.median(wall_samples), 6),
        "min_seconds": round(min(wall_samples), 6),
        "max_seconds": round(max(wall_samples), 6),
        "simulation_seconds": [round(x, 6) for x in simulation_samples],
        "median_simulation_seconds": round(statistics.median(simulation_samples), 6),
        "min_simulation_seconds": round(min(simulation_samples), 6),
        "max_simulation_seconds": round(max(simulation_samples), 6),
    }
    print("FLATWORLD_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
