import math
import numpy as np
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from flatworld import (
    GroundDomain,
    BallRigid,
    Elastic,
    EnforceRotVel,
    ExplicitLoop,
    FemDomain,
    Fixed,
    FixedAll,
    Force,
    Gravity,
    InitialVel,
    RigidBodyDomain,
    SolidProp,
)
from flatworld.joints import RevoluteJoint
from flatworld.rigidmanager import RigidManager
from test_utils import create_gui_if_available, init_sim


def test_revolute_joint(headless=False, kernel_profile=True):
    init_sim()

    radius = 0.05
    rigid1 = BallRigid(2, [0.5, 0.1], radius, 1.0)
    rigid2 = BallRigid(2, [0.9, 0.1], radius, 1.0)
    rigiddomain1 = RigidBodyDomain(rigid1, [Fixed([0])], True)
    rigiddomain2 = RigidBodyDomain(rigid2, [Force([0], [0, 1.355])], True)

    domains = [rigiddomain1, rigiddomain2]

    frame_dt = 1.0 / 60.0
    joint = RevoluteJoint(0, 1, [0.5, 0.1], [0, 0])
    looper = ExplicitLoop(0.0, domains, joints=[joint], useAdapativeDT=True, damping=0)
    length = 720
    height = 720
    gui = create_gui_if_available("JointDemo", res=(720, 720)) if not headless else None
    t = 0.0
    frame = 0
    mgr = looper.rigidManager

    while frame < 120:

        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        # draw
        frame += 1
        if gui is not None:
            gui.clear(0x112F41)
            for i, domain in enumerate(domains):
                color = 0xFF0000 if i == 0 else 0x0000FF
                domain.draw(gui, color=color, resolution=length)

            joint.draw(gui, 0xFFFFFF, resolution=length)

            gui.text(f"Frame {frame}", (0.02, 0.02), color=0x0)
            gui.show()

    pos0 = rigiddomain1.getCurrentRefPoint()
    pos1 = rigiddomain2.getCurrentRefPoint()

    link_length = np.sqrt(np.sum((pos0 - pos1) ** 2))

    init_link1_length = 0.4

    print(f"Final positions of joints:")
    print(f"  Fix_joint: {pos0}")
    print(f"  Release_point: {pos1}")
    print(f"  Link length: {link_length}")
    print(f"  Init link length: {init_link1_length}")

    assert np.allclose(pos0, [0.5, 0.1], atol=1e-2), "Fix joint did not settle at expected position."
    assert np.allclose(pos1, [0.1, 0.1], atol=1e-2), "Release point did not settle at expected position."
    print("Simulation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_revolute_joint")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_revolute_joint(headless=args.headless)
