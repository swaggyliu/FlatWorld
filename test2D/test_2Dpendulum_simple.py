"""
Simple two rigid bodiesRevoluteJointConnection test
useRevoluteJointConnect the upper and lower rigid bodies
Date: 2025-11-08
"""

import math
import numpy as np
import os
import sys
import taichi as ti
import time

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import ExplicitLoop, FixedAll, Gravity, RigidBodyDomain
from flatworld.joints import RevoluteJoint
from flatworld.rigid import BallRigid
from test_utils import create_gui_if_available


def test_two_body_joint(headless=False):
    """Two rigid bodies pass throughrevolute jointTest of connection"""
    print("=" * 60)
    print("Two rigid bodiesRevoluteJointConnection test")
    print("=" * 60)

    # initialization Taichi
    try:
        ti.init(offline_cache=True, arch=ti.gpu)
        print("use GPU accelerate")
    except:
        ti.init(offline_cache=True, arch=ti.cpu)
        print("use CPU calculate")

    # Rigid body parameters
    upper_pos = [0.5, 0.9]
    lower_pos = [0.6, 0.7]
    lower2_pos = [0.4, 0.5]

    ball_radius = 0.04
    ball_mass = 1.0

    r1 = BallRigid(d=2, origin=upper_pos, radius=ball_radius, mass=ball_mass)

    r2 = BallRigid(d=2, origin=lower_pos, radius=ball_radius, mass=ball_mass)

    r3 = BallRigid(d=2, origin=lower2_pos, radius=ball_radius, mass=ball_mass)

    # Create a rigid body domain
    upper_domain = RigidBodyDomain(rigid=r1, bcs=[Gravity([0, -9.8]), FixedAll([0])], considerContact=True)

    lower_domain = RigidBodyDomain(rigid=r2, bcs=[Gravity([0, -9.8])], considerContact=True)

    lower2 = RigidBodyDomain(rigid=r3, bcs=[Gravity([0, -9.8])], considerContact=True)

    domains = [upper_domain, lower_domain, lower2]

    # createrevolute jointConnect two rigid bodies
    joint1 = upper_pos
    joint2 = lower_pos
    revolute_joint = RevoluteJoint(
        0,
        1,
        anchor=joint1,
        axis=[0, 0],
    )
    revolute_joint2 = RevoluteJoint(1, 2, anchor=joint2, axis=[0, 0])
    joints = [revolute_joint, revolute_joint2]

    # Create a simulation loop with joints
    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, joints=joints, useAdapativeDT=True)

    # createGUI
    gui = create_gui_if_available("RevoluteJoint pendulum", res=(800, 800)) if not headless else None

    frame = 0
    max_frames = 60

    print("\nStart simulation...")
    start_time = time.time()

    while (gui is None or gui.running) and frame < max_frames:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        # clear screen
        if gui is not None:
            gui.clear(0x112F41)

            upper_domain.draw(gui, color=0xFF0000, resolution=800)
            lower_domain.draw(gui, color=0x0000FF, resolution=800)
            lower2.draw(gui, color=0x00FF00, resolution=800)

            revolute_joint.draw(gui, color=0x000000, resolution=800)
            revolute_joint2.draw(gui, color=0x000000, resolution=800)

            gui.show()
        frame += 1

    end_time = time.time()

    pos0 = upper_domain.getCurrentRefPoint()
    pos1 = lower_domain.getCurrentRefPoint()
    pos2 = lower2.getCurrentRefPoint()
    link1_length = np.sqrt(np.sum((pos0 - pos1) ** 2))
    link2_length = np.sqrt(np.sum((pos1 - pos2) ** 2))
    init_link1_length = np.sqrt(np.sum((np.array(upper_pos) - np.array(lower_pos)) ** 2))
    init_link2_length = np.sqrt(np.sum((np.array(lower_pos) - np.array(lower2_pos)) ** 2))

    print(f"Final positions of joints:")
    print(f"  Fix_joint: {pos0}")
    print(f"  Revolute_joint: {pos1}")
    print(f"  Release_point: {pos2}")
    print(f"  Link1 length: {link1_length}")
    print(f"  Link2 length: {link2_length}")
    print(f"  Init link1 length: {init_link1_length}")
    print(f"  Init link2 length: {init_link2_length}")

    assert np.allclose(pos0, [0.5, 0.9], atol=1e-2), "Fix joint did not settle at expected position."
    assert np.allclose(pos1, [0.401, 0.7], atol=1e-2), "Revolute joint did not settle at expected position."
    assert np.allclose(pos2, [0.52, 0.443], atol=1e-2), "Release point did not settle at expected position."
    assert (
        link1_length < init_link1_length + 0.001
    ), f"Link1 did not settle at expected distance. Final length: {link1_length}, Initial length: {init_link1_length}"
    assert (
        link2_length < init_link2_length + 0.001
    ), f"Link2 did not settle at expected distance. Final length: {link2_length}, Initial length: {init_link2_length}"
    print("Simulation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_two_body_joint")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_two_body_joint(headless=args.headless)
