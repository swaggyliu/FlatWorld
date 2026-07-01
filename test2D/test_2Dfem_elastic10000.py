import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, ExplicitLoop, FemDomain, Gravity, Mesh, SolidProp
from test_utils import create_gui_if_available


def test_2Dfem_elastic(headless=False):
    ti.init(offline_cache=True, arch=ti.cpu)

    numEl = 100
    dx = 0.6 / numEl
    conn = np.zeros((2 * numEl * numEl, 3), dtype=np.int32)
    coords = np.zeros(((numEl + 1) * (numEl + 1), 2), dtype=np.float32)
    ndCounter = 0
    elCount = 0
    for i in range(numEl + 1):
        for j in range(numEl + 1):
            coords[ndCounter] = np.array([dx * i, dx * j])
            ndCounter += 1

    for i in range(numEl):
        for j in range(numEl):
            conn[elCount] = [i * (numEl + 1) + j, i * (numEl + 1) + j + 1, (i + 1) * (numEl + 1) + j]
            conn[elCount + 1] = [i * (numEl + 1) + j + 1, (i + 1) * (numEl + 1) + j + 1, (i + 1) * (numEl + 1) + j]
            elCount += 2

    mesh = Mesh(2, conn, coords + 0.3)
    bcs = [Gravity([0, -1.0])]
    mat = Elastic(E=2e4, nu=0.2, rho=40.0)
    prop = SolidProp(mat)

    domain = FemDomain(mesh, prop, bcs)

    # Adaptive dt loop for large mesh with fixed 60 FPS stepping
    looper = ExplicitLoop(0.0, [domain], useAdapativeDT=True)

    gui = create_gui_if_available("FEM2D", res=(720, 720)) if not headless else None
    t = 0.0
    frame_dt = 1.0 / 60.0
    while gui is None or gui.running:
        looper.advanceWithTime(frame_dt)
        t += frame_dt
        pos = looper.femSpringManager.coords.to_numpy()
        if pos[0, 1] < 0.0:
            print("The time is: ", t)
            assert np.isclose(t, np.sqrt(0.6), atol=frame_dt)
            break

        # gui.circles(pos, radius=2, color=0xFFAA33)
        # e2n = mesh.connectivity
        # a, b, c = pos[e2n[:, 0]], pos[e2n[:, 1]], pos[e2n[:, 2]]
        # gui.triangles(a, b, c, color=0xFF0033)
        # gui.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dfem_elastic")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dfem_elastic(headless=args.headless)
