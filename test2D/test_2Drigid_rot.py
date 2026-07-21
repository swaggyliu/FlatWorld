import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import (
    GroundDomain,
    BallRigid,
    BoxRigid,
    EnforceRotAcc,
    EnforceRotVel,
    ExplicitLoop,
    Fixed,
    Gravity,
    InitialVel,
    RigidBodyDomain,
)
from test_utils import create_gui_if_available, init_sim


def test_2Drigidrot(headless=False):
    init_sim()

    rigid1 = BoxRigid(2, [0.5, 0.4], [0.1, 0.1], [0.0], 100.0)
    rigid2 = BoxRigid(2, [0.4, 0.6], [0.1, 0.1], [0.0], 100.0)
    bcs1 = [EnforceRotVel([0], [10.0])]
    bcs2 = [EnforceRotAcc([0], [10.0])]
    rigiddomain1 = RigidBodyDomain(rigid1, bcs1, True)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs2, True)

    domains = [rigiddomain1, rigiddomain2]
    domains += [GroundDomain(2, [0.1, 0.1], [0.0, 1.0], [])]
    domains += [GroundDomain(2, [0.1, 0.1], [1.0, 0.0], [])]
    domains += [GroundDomain(2, [0.9, 0.9], [0.0, -1.0], [])]
    domains += [GroundDomain(2, [0.9, 0.9], [-1.0, 0.0], [])]

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)
    length = 720
    height = 720
    gui = create_gui_if_available("rigid2D", res=(720, 720)) if not headless else None
    t = 0.0
    while (gui is None or gui.running) and t < 1.0:
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        if gui is not None:
            rigiddomain1.draw(gui, 0xAAFFFF, length)
            rigiddomain2.draw(gui, 0xFFB6C1, length)
            gui.show()

    angle1 = rigiddomain1.getCurrentRefAngles()[0]
    angle2 = rigiddomain2.getCurrentRefAngles()[0]
    print("Final angles (rad):", angle1, angle2)
    assert np.isclose(angle1, 10.0 - 4 * np.pi, atol=1e-2), "Rigid1 did not reach the expected rotation."
    assert np.isclose(angle2, 5.0 - 2 * np.pi, atol=1e-2), "Rigid2 did not reach the expected rotation."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Drigidrot")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Drigidrot(headless=args.headless)
