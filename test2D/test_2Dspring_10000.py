import numpy as np
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import (
    Elastic,
    EnforceAcc,
    EnforceVel,
    ExplicitLoop,
    Fixed,
    Force,
    Gravity,
    Mesh,
    SolidProp,
    SpringMassDomain,
)
from test_utils import create_gui_if_available, init_sim


def test_2Dspring(headless=False, max_steps=None, frame_dt=1.0 / 60.0):
    init_sim()

    numEl = 100
    dx = 0.6 / numEl
    conn = np.zeros((4 * numEl * numEl, 2), dtype=np.int32)
    coords = np.zeros(((numEl + 1) * (numEl + 1), 2), dtype=np.float32)
    ndCounter = 0
    elCount = 0
    for i in range(numEl + 1):
        for j in range(numEl + 1):
            coords[ndCounter] = np.array([dx * i, dx * j])
            ndCounter += 1

    for i in range(numEl):
        for j in range(numEl):
            conn[elCount] = [i * (numEl + 1) + j, i * (numEl + 1) + j + 1]
            conn[elCount + 1] = [i * (numEl + 1) + j + 1, (i + 1) * (numEl + 1) + j]
            conn[elCount + 2] = [(i + 1) * (numEl + 1) + j, i * (numEl + 1) + j]
            conn[elCount + 3] = [i * (numEl + 1) + j, i * (numEl + 1) + j + 1]

            elCount += 4

    domain1 = SpringMassDomain(2, coords + 0.25, conn, [10.0, 1.0, 1.0], [Gravity([0, -100.0])], False)

    femdomains = [domain1]

    looper = ExplicitLoop(0.0, femdomains, useAdapativeDT=True)
    colors = [0xFF0033, 0xFF00FF, 0xFFFF00]

    gui = create_gui_if_available("SPRING2D", res=(720, 720)) if not headless else None
    t = 0.0
    steps = 0
    simulation_start = time.perf_counter()
    while (gui is None or gui.running) and (
        steps < max_steps if max_steps is not None else t < 0.1
    ):
        # Advance one visual frame using adaptive substeps.
        looper.advanceWithTime(frame_dt)
        t += frame_dt
        steps += 1

        # Only read simulation state for rendering. In headless mode, defer
        # device-to-host copies until the simulation has finished.
        if gui is not None:
            pos_frame = looper.femSpringManager.coords.numpy()[: looper.femSpringManager.totalNodes.numpy()[0]]
            e2n = looper.femSpringManager.connectivity.numpy()[: looper.femSpringManager.totalElements.numpy()[0]]
            a, b = pos_frame[e2n[:, 0]], pos_frame[e2n[:, 1]]
            gui.circles(pos_frame, radius=2, color=0xFFAA33)
            gui.lines(a, b, color=colors[0])
            gui.show()

    pos = looper.femSpringManager.coords.numpy()[: looper.femSpringManager.totalNodes.numpy()[0]]
    simulation_elapsed = time.perf_counter() - simulation_start
    if max_steps is not None:
        print(f"FLATWORLD_SIMULATION_SECONDS={simulation_elapsed:.9f}")
    min = np.min(pos, axis=0)
    max = np.max(pos, axis=0)
    print("Final position bounds:", min, max)
    if max_steps is None:
        assert np.allclose(min, [0.25, -0.5278], atol=1e-2), "Min position out of expected range."
        assert np.allclose(max, [0.85, 0.072], atol=1e-2), "Max position out of expected range."
    else:
        assert np.all(np.isfinite(pos)), "Simulation produced non-finite positions."
        # Uniform gravity should translate the undeformed spring lattice by
        # -0.5 * g * T^2. The small benchmark dt makes this a useful physical
        # sanity check in addition to the finite-state guard.
        simulated_time = max_steps * frame_dt
        expected_shift = -50.0 * simulated_time * simulated_time
        assert np.allclose(min, [0.25, 0.25 + expected_shift], atol=1e-2), "Benchmark minimum position drifted."
        assert np.allclose(max, [0.85, 0.85 + expected_shift], atol=1e-2), "Benchmark maximum position drifted."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dspring")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Run a fixed number of visual frames (benchmark mode)"
    )
    parser.add_argument(
        "--frame-dt", type=float, default=1.0 / 60.0, help="Simulated time advanced per frame"
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be >= 1")
    if not np.isfinite(args.frame_dt) or args.frame_dt <= 0.0:
        parser.error("--frame-dt must be a finite positive number")
    test_2Dspring(headless=args.headless, max_steps=args.steps, frame_dt=args.frame_dt)
