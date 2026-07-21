import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, EnforceAcc, EnforceVel, ExplicitLoop, FemDomain, Fixed, Force, Mesh, SolidProp
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
    bcs1 = [EnforceVel([0, 1], [0, -1.0]), Fixed([2, 3])]

    mesh2 = Mesh(2, conn, coords - 0.25)
    bcs2 = [EnforceAcc([0, 1], [0, -10.0]), Fixed([2, 3])]

    domain = FemDomain(mesh, prop, bcs, False)
    domain1 = FemDomain(mesh1, prop, bcs1, False)
    domain2 = FemDomain(mesh2, prop, bcs2, False)
    femdomains = [domain, domain1, domain2]

    # Adaptive dt loop for fixed 60 FPS
    looper = ExplicitLoop(0.0, femdomains, useAdapativeDT=True)
    colors = [0xFF0033, 0xFF00FF, 0xFFFF00]

    gui = create_gui_if_available("FEM2D", res=(720, 720)) if not headless else None
    t = 0.0
    frame_dt = 1.0 / 60.0
    while gui is None or gui.running:
        looper.advanceWithTime(frame_dt)
        t += frame_dt

        if t > 1:
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
