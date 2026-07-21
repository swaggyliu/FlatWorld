import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import GroundDomain, BallRigid, EnforceVel, ExplicitLoop, Gravity, RigidBodyDomain
from test_utils import create_gui_if_available, init_sim


def test_2Drigidanal_contact(headless=False):
    init_sim()
    rigid1 = BallRigid(2, [0.5, 0.8], 0.1, 1.0)
    bcs = [Gravity([0, -10.0])]

    anl = GroundDomain(2, [0, 0.3], [0.0, 1.0], [EnforceVel([0], [1, 1])])
    domains = [RigidBodyDomain(rigid1, bcs), anl]

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("RigidAna2D", res=(720, 720)) if not headless else None
    t = 0.0
    while (gui is None or gui.running) and t < 0.3:
        looper.advanceWithTime(frame_dt)
        t += frame_dt
        pos = looper.domains[0].getCurrentRefPoint()
        if gui is not None:
            gui.circle(pos, radius=51, color=0xFFAA33)
        point = anl.getCurrentRefPoint()
        if gui is not None:
            gui.line(point - np.array([1.0, 0.0]), point + np.array([1.0, 0.0]), 1, 0xFF00FF)
            gui.show()
    print("Final position:", pos)
    assert pos[1] > 0.92, "Some nodes penetrated the rigid surface."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Drigidanal_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Drigidanal_contact(headless=args.headless)
