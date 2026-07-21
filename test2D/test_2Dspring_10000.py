import numpy as np
import os
import sys

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


def test_2Dspring(headless=False):
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

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, femdomains, useAdapativeDT=True)
    colors = [0xFF0033, 0xFF00FF, 0xFFFF00]

    gui = create_gui_if_available("SPRING2D", res=(720, 720)) if not headless else None
    t = 0.0
    while (gui is None or gui.running) and t < 0.1:
        # advance one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)
        t += frame_dt

        # render
        pos = looper.femSpringManager.coords.numpy()[: looper.femSpringManager.totalNodes.numpy()[0]]
        e2n = looper.femSpringManager.connectivity.numpy()[: looper.femSpringManager.totalElements.numpy()[0]]
        a, b = pos[e2n[:, 0]], pos[e2n[:, 1]]
        if gui is not None:
            gui.circles(pos, radius=2, color=0xFFAA33)
            gui.lines(a, b, color=colors[0])
            gui.show()

    min = np.min(pos, axis=0)
    max = np.max(pos, axis=0)
    print("Final position bounds:", min, max)
    assert np.allclose(min, [0.25, -0.5278], atol=1e-2), "Min position out of expected range."
    assert np.allclose(max, [0.85, 0.072], atol=1e-2), "Max position out of expected range."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dspring")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dspring(headless=args.headless)
