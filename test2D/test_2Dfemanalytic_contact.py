import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import GroundDomain, Elastic, ExplicitLoop, FemDomain, FEMesher, Gravity, Mesh, SolidProp
from test_utils import create_gui_if_available


def test_2Dfem_contact(headless=False):
    msh = FEMesher(2)

    ti.init(offline_cache=True, arch=ti.cpu)
    offset = 0.1
    mesh = msh.createCircle([0.5, 0.5], offset)
    bcs = [Gravity([0, -10.0])]
    mat = Elastic(E=2e4, nu=0.2, rho=40.0)
    prop = SolidProp(mat)

    domains = [GroundDomain(2, [0.0, 0.3], [0.0, 1.0], [])]
    domains += [FemDomain(mesh, prop, bcs)]

    # Adaptive dt loop with fixed 60 FPS
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("FEM2D", res=(720, 720)) if not headless else None
    frame_dt = 1.0 / 60.0
    t = 0.0
    while (gui is None or gui.running) and t < 1.0:
        looper.advanceWithTime(frame_dt)

        nnd = 0
        color = 0xAAFFFF
        # print(mesh.coords.to_numpy())
        if gui is not None:
            looper.femSpringManager.drawMesh(gui, 0xFF0033)
            gui.line(np.array([0, 0.3]), np.array([1, 0.3]), 1, 0xFF00FF)
            gui.show()
        t += frame_dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dfem_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dfem_contact(headless=args.headless)
