#!/usr/bin/env python3
"""Compare serial PGS with deterministic contact-island PGS on CPU Warp.

Two graph shapes are measured: many independent two-body islands and one
connected chain. The latter is a control showing the island solver's setup
cost when there is little parallelism.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import warp as wp

import flatworld  # noqa: F401  (adds the package directory to sys.path)
from rigidmanager import (
    _build_pgs_islands_kernel,
    _solve_pgs_islands_kernel,
    _solve_pgs_kernel,
)
from wp_init import ensure_warp


def make_problem(num_islands: int, connected: bool, device: str):
    rows = 2 * num_islands
    bodies = num_islands + 1 if connected else 2 * num_islands
    bodypair = np.zeros((rows, 2), dtype=np.int32)
    jac_a = np.zeros((rows, 3), dtype=np.float32)
    jac_b = np.zeros((rows, 3), dtype=np.float32)
    rhs = np.zeros(rows, dtype=np.float32)
    limits = np.zeros((rows, 2), dtype=np.float32)
    parent_row = np.full(rows, -1, dtype=np.int32)
    velocity = np.zeros((bodies, 2), dtype=np.float32)

    for island in range(num_islands):
        a = island if connected else 2 * island
        b = island + 1 if connected else 2 * island + 1
        normal = 2 * island
        tangent = normal + 1
        bodypair[normal] = [a, b]
        bodypair[tangent] = [a, b]
        jac_a[normal] = [0.0, 1.0, 0.0]
        jac_b[normal] = [0.0, 1.0, 0.0]
        jac_a[tangent] = [1.0, 0.0, 0.0]
        jac_b[tangent] = [1.0, 0.0, 0.0]
        rhs[normal] = 0.1
        limits[normal] = [0.0, 1.0e6]
        limits[tangent] = [-0.5, 0.5]
        parent_row[tangent] = normal
        velocity[a, 1] = -1.0

    return {
        "num_constraints": wp.array(np.array([rows], dtype=np.int32), dtype=int, device=device),
        "bodypair": wp.array(bodypair, dtype=wp.vec2i, device=device),
        "jac_a": wp.array(jac_a, dtype=wp.vec3, device=device),
        "jac_b": wp.array(jac_b, dtype=wp.vec3, device=device),
        "rhs": wp.array(rhs, dtype=float, device=device),
        "limits": wp.array(limits, dtype=wp.vec2, device=device),
        "parent_row": wp.array(parent_row, dtype=int, device=device),
        "velocity": wp.array(velocity, dtype=wp.vec2, device=device),
        "rot_velocity": wp.zeros(bodies, dtype=float, device=device),
        "mass": wp.array(np.ones(bodies, dtype=np.float32), dtype=float, device=device),
        "inertia": wp.array(np.ones(bodies, dtype=np.float32), dtype=float, device=device),
        "island_parent": wp.zeros(bodies, dtype=int, device=device),
        "island_rank": wp.zeros(bodies, dtype=int, device=device),
        "island_offsets": wp.zeros(bodies, dtype=int, device=device),
        "island_counts": wp.zeros(bodies, dtype=int, device=device),
        "island_cursor": wp.zeros(bodies, dtype=int, device=device),
        "island_rows": wp.zeros(rows, dtype=int, device=device),
        "rows": rows,
        "bodies": bodies,
    }


def run_once(problem, solver: str, iterations: int) -> tuple[float, tuple[np.ndarray, ...]]:
    start = time.perf_counter()
    if solver == "serial":
        wp.launch(
            _solve_pgs_kernel,
            dim=1,
            inputs=[
                iterations, 1.0e-8, problem["num_constraints"], problem["rows"],
                problem["velocity"], problem["rot_velocity"], problem["mass"], problem["inertia"],
                problem["bodypair"], problem["jac_a"], problem["jac_b"], problem["rhs"],
                problem["limits"], problem["lambda"], problem["parent_row"],
            ],
            device="cpu",
        )
    else:
        wp.launch(
            _build_pgs_islands_kernel,
            dim=1,
            inputs=[
                problem["num_constraints"], problem["rows"], problem["bodies"], problem["bodypair"],
                problem["island_parent"], problem["island_rank"], problem["island_offsets"],
                problem["island_counts"], problem["island_cursor"], problem["island_rows"],
            ],
            device="cpu",
        )
        wp.launch(
            _solve_pgs_islands_kernel,
            dim=problem["bodies"],
            inputs=[
                iterations, 1.0e-8, problem["bodies"], problem["island_parent"],
                problem["island_offsets"], problem["island_counts"], problem["island_rows"],
                problem["velocity"], problem["rot_velocity"], problem["mass"], problem["inertia"],
                problem["bodypair"], problem["jac_a"], problem["jac_b"], problem["rhs"],
                problem["limits"], problem["lambda"], problem["parent_row"],
            ],
            device="cpu",
        )
    state = (
        problem["velocity"].numpy(),
        problem["rot_velocity"].numpy(),
        problem["lambda"].numpy(),
    )
    return time.perf_counter() - start, state


def fresh_problem(num_islands: int, connected: bool, device: str):
    problem = make_problem(num_islands, connected, device)
    problem["lambda"] = wp.zeros(problem["rows"], dtype=float, device=device)
    return problem


def benchmark_scenario(num_islands: int, connected: bool, warmup: int, repeat: int):
    device = "cpu"
    iterations = 50
    # Check numerical equivalence before collecting timings.
    serial_problem = fresh_problem(num_islands, connected, device)
    island_problem = fresh_problem(num_islands, connected, device)
    _, serial_state = run_once(serial_problem, "serial", iterations)
    _, island_state = run_once(island_problem, "islands", iterations)
    for actual, expected in zip(island_state, serial_state):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    for _ in range(warmup):
        run_once(fresh_problem(num_islands, connected, device), "serial", iterations)
        run_once(fresh_problem(num_islands, connected, device), "islands", iterations)

    serial_samples = [
        run_once(fresh_problem(num_islands, connected, device), "serial", iterations)[0]
        for _ in range(repeat)
    ]
    island_samples = [
        run_once(fresh_problem(num_islands, connected, device), "islands", iterations)[0]
        for _ in range(repeat)
    ]
    serial_median = statistics.median(serial_samples)
    island_median = statistics.median(island_samples)
    return {
        "shape": "one_connected_chain" if connected else "independent_pairs",
        "num_islands_expected": 1 if connected else num_islands,
        "num_constraints": 2 * num_islands,
        "iterations": iterations,
        "serial_median_seconds": round(serial_median, 6),
        "island_median_seconds": round(island_median, 6),
        "island_speedup": round(serial_median / island_median, 4) if island_median else None,
        "serial_samples": [round(x, 6) for x in serial_samples],
        "island_samples": [round(x, 6) for x in island_samples],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--islands", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if args.islands < 2 or args.warmup < 0 or args.repeat < 1:
        parser.error("--islands must be >= 2, --warmup >= 0, --repeat >= 1")

    ensure_warp()
    result = {
        "device": "cpu",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "scenarios": [
            benchmark_scenario(args.islands, False, args.warmup, args.repeat),
            benchmark_scenario(args.islands, True, args.warmup, args.repeat),
        ],
    }
    print("FLATWORLD_PGS_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
