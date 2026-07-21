from math import pi
import numpy as np
import os
import sys

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
from test_utils import create_gui_if_available, init_sim


def test_2Djoint_rotation(headless=False):
    """Test 2D joints with rotation boundary conditions.

    Demonstrates:
    - RevoluteJoint with EnforceRotVel (motor-like constant angular velocity)
    - RevoluteJoint with EnforceRotAcc (motor-like angular acceleration)
    - RevoluteJoint with Fixed anchor (stationary pivot)
    - WeldJoint (prevents any relative rotation)

    The test shows various rotation behaviors through joints connecting rigid bodies.
    """
    init_sim()

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

    # Setup 3: Two bodies connected by revolute joint, both free to move
    # Shows pendulum-like behavior with external force
    body3a = BallRigid(2, [0.3, 0.4], radius, 10.0)
    origin3 = [0.3, 0.25]
    body3b = BoxRigid(2, origin3, [0.04, 0.1], [0.0], 0.5)
    body3a_domain = RigidBodyDomain(body3a, [], considerContact=False)
    body3b_domain = RigidBodyDomain(body3b, [Gravity([0.0, -20.0])], considerContact=False)

    # Setup 4: Weld joint (no rotation allowed) - for comparison
    body4a = BallRigid(2, [0.7, 0.4], radius, 10.0)
    origin4 = [0.8, 0.25]
    body4b = BoxRigid(2, origin4, [0.04, 0.1], [0.0], 10.0)
    body4a_domain = RigidBodyDomain(body4a, [FixedAll([0])], considerContact=False)
    body4b_domain = RigidBodyDomain(body4b, [Gravity([0.0, -20.0])], considerContact=False)

    # Setup 5, prismatic joint, enforced velocity
    body5a = BallRigid(2, [0.3, 0.1], radius, 10.0)
    body5b = BallRigid(2, [0.3, 0.1], radius, 10.0)

    body5a_domain = RigidBodyDomain(body5a, [FixedAll([0])], considerContact=False)
    body5b_domain = RigidBodyDomain(body5b, [], considerContact=False)

    # Setup 6, prismatic joint, enforced acceleration
    body6a = BallRigid(2, [0.7, 0.1], radius, 10.0)
    body6b = BallRigid(2, [0.7, 0.1], radius, 10.0)
    body6a_domain = RigidBodyDomain(body6a, [FixedAll([0])], considerContact=False)
    body6b_domain = RigidBodyDomain(body6b, [], considerContact=False)

    domains = [
        anchor1_domain,
        body1_domain,  # 0, 1
        anchor2_domain,
        body2_domain,  # 2, 3
        body3a_domain,
        body3b_domain,  # 4, 5
        body4a_domain,
        body4b_domain,  # 6, 7
        body5a_domain,
        body5b_domain,  # 8, 9
        body6a_domain,
        body6b_domain,  # 10, 11
    ]

    # Joint 1: Revolute with constant angular velocity (pi rad/s)
    joint1_anchor = [0.3, 0.7]
    joint1 = RevoluteJoint(0, 1, joint1_anchor, [0, 0], bcs=[EnforceRotVel([0], [pi], origin=joint1_anchor)])

    # Joint 2: Revolute with angular acceleration (pi rad/s²)
    joint2_anchor = [0.7, 0.7]
    joint2 = RevoluteJoint(2, 3, joint2_anchor, [0, 0], bcs=[EnforceRotAcc([0], [pi], origin=joint2_anchor)])

    # Joint 3: Free revolute (pendulum behavior)
    joint3_anchor = [0.3, 0.4]
    joint3 = RevoluteJoint(4, 5, joint3_anchor, [0, 0])

    # Joint 4: Weld joint (rigid connection, no rotation)
    joint4_anchor = [0.7, 0.4]
    joint4 = WeldJoint(6, 7, joint4_anchor, bcs=[])

    # joint 5: prismatic joint with enforced velocity
    joint5_anchor = [0.3, 0.1]
    joint5 = PrismaticJoint(8, 9, joint5_anchor, [1, 0], [EnforceVel([0], [0.2, 0.0])])

    joint6_anchor = [0.7, 0.1]
    joint6 = PrismaticJoint(10, 11, joint6_anchor, [1, 0], [EnforceAcc([0], [0.2, 0.0])])

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, joints=[joint1, joint2, joint3, joint4, joint5, joint6], useAdapativeDT=True)
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
            for i, joint in enumerate([joint1, joint2, joint3, joint4, joint5, joint6]):
                joint.draw(gui, color=colors[i * 2 + 1], resolution=800)

            gui.text(f"Time: {t:.2f}s | Frame: {frame}", pos=(0.02, 0.96), color=0xFFFFFF, font_size=18)
            gui.text("RotVel", pos=(0.25, 0.92), color=0xFF3333, font_size=16)
            gui.text("RotAcc", pos=(0.68, 0.92), color=0x33FF33, font_size=16)
            gui.text("Free", pos=(0.26, 0.42), color=0x3333FF, font_size=16)
            gui.text("Weld", pos=(0.68, 0.42), color=0xFF8800, font_size=16)

            gui.show()

    currentOrigin1 = looper.rigidManager.rigidParams.numpy()[1, 0]
    currentOrigin2 = looper.rigidManager.rigidParams.numpy()[3, 0]
    currentOrigin3 = looper.rigidManager.rigidParams.numpy()[5, 0]
    currentOrigin4 = looper.rigidManager.rigidParams.numpy()[7, 0]
    currentOrigin5 = looper.rigidManager.rigidParams.numpy()[9, 0]
    currentOrigin6 = looper.rigidManager.rigidParams.numpy()[11, 0]

    print("currentOrigin1:", currentOrigin1)
    print("currentOrigin2:", currentOrigin2)
    print("currentOrigin3:", currentOrigin3)
    print("currentOrigin4:", currentOrigin4)
    print("currentOrigin5:", currentOrigin5)
    print("currentOrigin6:", currentOrigin6)

    assert np.allclose(currentOrigin1, [0.3, 0.55], atol=2e-2), "Anchor 1 with rotVel moved incorrectly!"
    assert np.allclose(currentOrigin2, [0.55, 0.7], rtol=2e-2), "Anchor 2 with rotAcc moved incorrectly!"
    assert np.allclose(
        currentOrigin3, [0.3, -0.25], atol=1e-2
    ), "Anchor 3 with free revolute and gravity moved inincorrectly!"
    assert np.allclose(currentOrigin4, [0.8, 0.25], rtol=2e-2), "Anchor 4 with weld joint moved incorrectly!"
    assert np.allclose(currentOrigin5, [0.5, 0.1], rtol=2e-2), "Anchor 5 with vel moved incorrectly!"
    assert np.allclose(currentOrigin6, [0.8, 0.1], rtol=2e-2), "Anchor 6 with acc moved incorrectly!"

    print(f"Simulation completed at t={t:.2f}s, frame {frame}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Djoint_rotation")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Djoint_rotation(headless=args.headless)
