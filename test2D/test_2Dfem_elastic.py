import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, ExplicitLoop, FemDomain, Gravity, Mesh, SolidProp
from test_utils import create_gui_if_available, init_sim


def test_2Dfem_elastic(headless=False):
    init_sim()

    conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)

    mesh = Mesh(2, conn, coords)
    bcs = [Gravity([0, -1.0])]
    mat = Elastic(E=2e4, nu=0.2, rho=40.0)
    prop = SolidProp(mat)

    domain = FemDomain(mesh, prop, bcs)

    # Use adaptive dt with fixed 60 FPS stepping
    looper = ExplicitLoop(0.0, [domain], useAdapativeDT=True)

    gui = create_gui_if_available("FEM2D", res=(720, 720)) if not headless else None
    t = 0.0
    frame_dt = 1.0 / 60.0
    while gui is None or gui.running:
        looper.advanceWithTime(frame_dt)
        t += frame_dt
        pos = looper.femSpringManager.coords.numpy()
        if pos[0, 1] < 0.0:
            print("The time is: ", t)
            # Original assertion used dt accumulation; tolerance adjusted for frame stepping
            assert np.isclose(t, 1.0, atol=frame_dt)
            break

        if gui is not None:
            looper.femSpringManager.drawMesh(gui, 0xFF0033)
            gui.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dfem_elastic")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dfem_elastic(headless=args.headless)
