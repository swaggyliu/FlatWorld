#!/usr/bin/env python3
"""Repeatable wall-clock smoke benchmark for the 10k spring-mass example.

This measures the complete headless example (initialization + simulation +
assertions), not an isolated kernel. Use it to track regressions on the same
GitHub-hosted runner; do not compare numbers across different runner classes.
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


def run_once() -> float:
    env = os.environ.copy()
    env.update({"HEADLESS": "1", "CI": "true", "FLATWORLD_DEVICE": "cpu"})
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, str(CASE), "--headless"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")

    for _ in range(args.warmup):
        run_once()

    samples = [run_once() for _ in range(args.repeat)]
    result = {
        "case": "test2D/test_2Dspring_10000.py",
        "device": "cpu",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seconds": [round(x, 6) for x in samples],
        "median_seconds": round(statistics.median(samples), 6),
        "min_seconds": round(min(samples), 6),
        "max_seconds": round(max(samples), 6),
    }
    print("FLATWORLD_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
