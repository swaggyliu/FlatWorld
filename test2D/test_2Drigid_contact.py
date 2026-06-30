import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import BallRigid, Elastic, ExplicitLoop, FemDomain, Gravity, Mesh, RigidBodyDomain, SolidProp
from test_utils import create_gui_if_available


def test_2Drigid_contact(headless=False):

    ti.init(offline_cache=True, arch=ti.cpu)
    rigid0 = BallRigid(2, [0.5, 0.8], 0.1, 1.0)
    bcs = [Gravity([0, -10.0])]
    rigid1 = BallRigid(2, [0.5, 0.3], 0.1, 1.0)
    bcs = [Gravity([0, 10.0])]
    do1 = RigidBodyDomain(rigid1, bcs)
    domains = [RigidBodyDomain(rigid0, bcs), do1]

    frame_dt = 1.0 / 60.0

    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("FEM2D", res=(1080, 1080)) if not headless else None
    t = 0.0
    while (gui is None or gui.running) and t < 1.0:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        nnd = 0
        colors = [0xAAFFFF, 0xFF00FF]
        for i, rigiddomain in enumerate(looper.domains):
            pos = rigiddomain.getCurrentRefPoint()
            if gui is not None:
                gui.circle(pos, colors[i], 108)

            if gui is not None:
                gui.show()
    print("Final positions:", pos)
    assert pos[1] >= 0.4, "Some nodes penetrated the rigid surface."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Drigid_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Drigid_contact(headless=args.headless)
