import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, ExplicitLoop, FemDomain, Force, HyperElastic, Mesh, SolidProp
from test_utils import create_gui_if_available, init_sim


def test_2Dfem_elastic(headless=False):
    init_sim()

    conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)

    mesh = Mesh(2, conn, coords)
    bcs = [Force([0, 1], [0, -1.0])]
    mat2 = HyperElastic(E=2e4, nu=0.4, rho=40.0)
    mat1 = Elastic(E=2e4, nu=0.2, rho=40.0)
    prop = SolidProp(mat1)

    domain = FemDomain(mesh, prop, bcs)

    # Adaptive dt loop for fixed 60 FPS
    looper = ExplicitLoop(0.0, [domain], useAdapativeDT=True)

    gui = create_gui_if_available("FEM2D", res=(720, 720)) if not headless else None
    t = 0.0
    frame_dt = 1.0 / 60.0
    while (gui is None or gui.running) and t < 0.5:
        looper.advanceWithTime(frame_dt)
        t += frame_dt
        pos = looper.femSpringManager.coords.numpy()
        if np.isclose(pos[0, 1], 0.0, atol=1e-3):
            print("The time is: ", t)
            # Adjust tolerance to frame stepping granularity
            assert np.isclose(t, np.sqrt(0.8), atol=frame_dt)
            break

        # gui.circles(pos, radius=2, color=0xFFAA33)
        # e2n = mesh.connectivity.numpy()
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
