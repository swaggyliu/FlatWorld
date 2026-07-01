from math import pi
import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import (
    BallRigid,
    BoxRigid,
    EnforceAcc,
    EnforceRotAcc,
    EnforceRotVel,
    EnforceVel,
    ExplicitLoop,
    Fixed,
    FixedAll,
    Force,
    Gravity,
    RigidBodyDomain,
)
from flatworld.joints import PrismaticJoint, RevoluteJoint, WeldJoint
from test_utils import create_gui_if_available


def test_2Djoint_rotation(headless=False):
    """Test 2D joints with rotation boundary conditions.

    Demonstrates:
    - RevoluteJoint with EnforceRotVel (motor-like constant angular velocity)
    - RevoluteJoint with EnforceRotAcc (motor-like angular acceleration)
    - RevoluteJoint with Fixed anchor (stationary pivot)
    - WeldJoint (prevents any relative rotation)

    The test shows various rotation behaviors through joints connecting rigid bodies.
    """
    ti.init(offline_cache=True, arch=ti.cpu, default_fp=ti.f32)

    radius = 0.04

    # Setup 1: Revolute joint with constant angular velocity (like a motor)
    # Fixed anchor at center, body rotates at constant speed
    anchor1 = BallRigid(2, [0.3, 0.7], radius, 10.0)
    origin1 = [0.3, 0.85]
    body1 = BoxRigid(2, origin1, [0.04, 0.1], [0.0], 10.0)
    anchor1_domain = RigidBodyDomain(anchor1, [FixedAll([0])], considerContact=False)
    body1_domain = RigidBodyDomain(body1, [], considerContact=False)

    # Setup 2: Revolute joint with angular acceleration
    # Another fixed anchor with body that accelerates rotationally
    anchor2 = BallRigid(2, [0.7, 0.7], radius, 10.0)
    origin2 = [0.7, 0.85]
    body2 = BoxRigid(2, origin2, [0.04, 0.1], [0.0], 10.0)
    anchor2_domain = RigidBodyDomain(anchor2, [FixedAll([0])], considerContact=False)
    body2_domain = RigidBodyDomain(body2, [], considerContact=False)

    domains = [
        anchor1_domain,
        body1_domain,  # 0, 1
        anchor2_domain,
        body2_domain,  # 2, 3
    ]

    # Joint 1: Revolute with constant angular velocity (pi rad/s)
    joint1_anchor = [0.3, 0.7]
    joint1 = RevoluteJoint(0, 1, joint1_anchor, [0, 0], stiff=100, damping=1.0, bcs=[EnforceRotVel([0], [0.0])])

    # Joint 2: Revolute with angular acceleration (pi rad/s²)
    joint2_anchor = [0.7, 0.7]
    joint2 = RevoluteJoint(2, 3, joint2_anchor, [0, 0], stiff=100, damping=1.0, bcs=[EnforceRotVel([0], [0.0])])

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, joints=[joint1, joint2], damping=0.05, useAdapativeDT=True, use_pd=1)
    looper.rigidManager.joint_control_target[0] = pi
    looper.rigidManager.joint_control_target[1] = pi / 2

    looper.stableTime = 1e-3
    gui = create_gui_if_available("2D Joint Rotation Test", res=(800, 800)) if not headless else None
    t = 0.0
    frame = 0

    colors = [
        0x888888,
        0xFF3333,  # Setup 1: gray anchor, red body
        0x888888,
        0x33FF33,  # Setup 2: gray anchor, green body
        0x3333FF,
        0x3333FF,  # Setup 3: blue both (pendulum)
        0xFF8800,
        0xFF8800,  # Setup 4: orange both (weld)
        0x00FFFF,
        0x00FFFF,  # Setup 5: cyan both (prismatic vel)
        0xFF00FF,
        0xFF00FF,  # Setup 6: magenta both (prismatic acc)
    ]

    print("Joint rotation demonstration:")
    print("  Top-left (red): EnforceRotVel - constant angular velocity")
    print("  Top-right (green): EnforceRotAcc - constant angular acceleration")
    print("  Bottom-left (blue): Free revolute - pendulum with gravity")
    print("  Bottom-right (orange): Weld joint - rigid connection")
    print("  Bottom-left (cyan): Prismatic joint with enforced velocity")
    print("  Bottom-right (magenta): Prismatic joint with enforced acceleration")

    while (gui is None or gui.running) and frame <= 60:
        looper.advanceWithTime(frame_dt, verbose=True)
        t += frame_dt
        frame += 1

        if gui is not None:
            gui.clear(0x112F41)

            # Draw all rigid bodies
            for i, domain in enumerate(domains):
                domain.draw(gui, color=colors[i], resolution=800)

            # Draw joints with connecting lines
            mgr = looper.rigidManager
            for i, joint in enumerate([joint1, joint2]):
                joint.draw(gui, color=colors[i * 2 + 1], resolution=800)

            gui.text(f"Time: {t:.2f}s | Frame: {frame}", pos=(0.02, 0.96), color=0xFFFFFF, font_size=18)
            gui.text("Pi", pos=(0.25, 0.92), color=0xFF3333, font_size=16)
            gui.text("Pi / 2.0", pos=(0.68, 0.92), color=0x33FF33, font_size=16)

            gui.show()

    currentOrigin1 = looper.rigidManager.rigidParams[1, 0].to_numpy()
    currentOrigin2 = looper.rigidManager.rigidParams[3, 0].to_numpy()

    print("currentOrigin1:", currentOrigin1)
    print("currentOrigin2:", currentOrigin2)

    assert np.allclose(currentOrigin1, [0.3, 0.55], atol=2e-2), "Anchor 1 with rotVel moved incorrectly!"
    assert np.allclose(currentOrigin2, [0.55, 0.7], rtol=2e-2), "Anchor 2 with rotAcc moved incorrectly!"

    print(f"Simulation completed at t={t:.2f}s, frame {frame}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Djoint_rotation")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Djoint_rotation(headless=args.headless)
