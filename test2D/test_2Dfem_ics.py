import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, ExplicitLoop, FemDomain, Fixed, Force, InitialVel, Mesh, SolidProp
from test_utils import create_gui_if_available, init_sim


def test_2Dfem_elastic(headless=False):
    init_sim()

    conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)

    mesh = Mesh(2, conn, coords)
    bcs = [Force([0, 1], [0, -100.0]), Fixed([2, 3])]
    mat = Elastic(E=2e4, nu=0.2, rho=40.0)
    prop = SolidProp(mat)

    domain = FemDomain(mesh, prop, bcs)

    mesh1 = Mesh(2, conn, coords + 0.25)
    bcs1 = [InitialVel([0, 1], [0, -1.0])]

    domain = FemDomain(mesh, prop, bcs, False)
    domain1 = FemDomain(mesh1, prop, [], False, bcs1)
    femdomains = [domain, domain1]

    # Adaptive dt loop fixed to 60 FPS
    looper = ExplicitLoop(0.0, femdomains, useAdapativeDT=True)
    colors = [0xFF0033, 0xFF00FF, 0xFFFF00]

    gui = create_gui_if_available("FEM2D", res=(720, 720)) if not headless else None
    frame_dt = 1.0 / 60.0
    t = 0.0
    while (gui is None or gui.running) and t < 1.0:
        looper.advanceWithTime(frame_dt)

        if gui is not None:
            looper.femSpringManager.drawMesh(gui, 0xFF0033)
            gui.show()
        t += frame_dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dfem_elastic")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dfem_elastic(headless=args.headless)
