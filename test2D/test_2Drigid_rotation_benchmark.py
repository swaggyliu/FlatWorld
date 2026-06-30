import math
import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import (
    GroundDomain,
    BallRigid,
    BoxRigid,
    CapsuleRigid,
    EnforceRotAcc,
    EnforceRotVel,
    ExplicitLoop,
    RigidBodyDomain,
)
from test_utils import create_gui_if_available


def test_2Drigid_rotation_benchmark(headless=False):

    ti.init(offline_cache=True, arch=ti.cpu, debug=False)
    pi = math.pi
    pivot1 = [0.2, 0.7]
    rigid1 = BallRigid(2, pivot1, 0.04, 0.01)
    rigiddomain1 = RigidBodyDomain(
        rigid1, bcs=[EnforceRotVel([0], [pi], origin=pivot1)], considerContact=False, initials=[]
    )
    pivot2 = [0.2, 0.5]
    rigid2 = BallRigid(2, pivot2, 0.04, 0.01)
    rigiddomain2 = RigidBodyDomain(
        rigid2, bcs=[EnforceRotAcc([0], [2 * pi], origin=pivot2)], considerContact=False, initials=[]
    )

    pivot4 = [0.4, 0.7]
    rigid4 = BoxRigid(2, pivot4, [0.1, 0.04], [0.0], 0.01)
    rigiddomain4 = RigidBodyDomain(
        rigid4, bcs=[EnforceRotVel([0], [pi], origin=pivot4)], considerContact=False, initials=[]
    )
    pivot5 = [0.4, 0.5]
    rigid5 = BoxRigid(2, pivot5, [0.1, 0.04], [0.0], 0.01)
    rigiddomain5 = RigidBodyDomain(
        rigid5, bcs=[EnforceRotAcc([0], [2 * pi], origin=pivot5)], considerContact=False, initials=[]
    )

    pivot6 = [0.3, 0.3]
    rigid6 = BoxRigid(2, pivot6, [0.4, 0.3], [0], 0.04, 2.0)
    rigiddomain6 = RigidBodyDomain(
        rigid6, bcs=[EnforceRotVel([0], [pi], origin=pivot6)], considerContact=False, initials=[]
    )
    pivot7 = [0.5, 0.3]
    rigid7 = BoxRigid(2, pivot7, [0.6, 0.3], [0], 0.04, 2.0)
    rigiddomain7 = RigidBodyDomain(
        rigid7, bcs=[EnforceRotAcc([0], [2 * pi], origin=pivot7)], considerContact=False, initials=[]
    )

    pivot8 = [0.8, 0.5]
    rigid8 = CapsuleRigid(2, pivot8, [0.9, 0.5], [0], 0.04, 2.0)
    rigiddomain8 = RigidBodyDomain(
        rigid8, bcs=[EnforceRotVel([0], [pi], origin=pivot8)], considerContact=False, initials=[]
    )
    pivot9 = [0.8, 0.3]
    rigid9 = CapsuleRigid(2, pivot9, [0.9, 0.3], [0], 0.04, 2.0)
    rigiddomain9 = RigidBodyDomain(
        rigid9, bcs=[EnforceRotAcc([0], [2 * pi], origin=pivot9)], considerContact=False, initials=[]
    )

    domains = [
        rigiddomain1,
        rigiddomain2,
        rigiddomain4,
        rigiddomain5,
        rigiddomain6,
        rigiddomain7,
        rigiddomain8,
        rigiddomain9,
    ]

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("2D Rigid Rotation BCs", res=(720, 720)) if not headless else None
    t = 0.0

    color_box = 0x3333FF

    print("Rotating rigid demonstration:")
    print("  Red: EnforceRotVel (constant angular velocity)")
    print("  Green: EnforceRotAcc (constant angular acceleration)")
    print("  Blue: EnforceRotVel (negative - clockwise)")
    print("  White: Static boundary walls")

    while (gui is None or gui.running) and t < 10.0:  # Run for 10 seconds
        looper.advanceWithTime(frame_dt)
        t += frame_dt

        if gui is not None:
            gui.clear(0x222222)

            rigiddomain1.draw(gui, color=0xFF3333, resolution=720)
            rigiddomain2.draw(gui, color=0x33FF33, resolution=720)
            rigiddomain4.draw(gui, color=0xFF3333, resolution=720)
            rigiddomain5.draw(gui, color=0x33FF33, resolution=720)
            rigiddomain6.draw(gui, color=0xFF3333, resolution=720)
            rigiddomain7.draw(gui, color=0x33FF33, resolution=720)
            rigiddomain8.draw(gui, color=0xFF3333, resolution=720)
            rigiddomain9.draw(gui, color=0x33FF33, resolution=720)

            # Draw pivot points
            pivots = np.array([pivot4, pivot5])
            gui.circles(pivots, radius=3, color=0xFFFF00)

            gui.text(f"Time: {t:.2f}s", pos=(0.02, 0.95), color=0xFFFFFF, font_size=20)

            gui.show()

        if abs(t - 1) < 1e-3:
            # RigidBodyDomain doesn't expose runtime angle directly. Query the RigidManager
            # orientation (2D stored as scalar in quat[index][0]) and convert to a unit normal.
            mgr = looper.rigidManager
            quat_arr = mgr.quat.to_numpy()
            idx1 = int(rigiddomain1.ndOffset)
            idx2 = int(rigiddomain2.ndOffset)
            idx4 = int(rigiddomain4.ndOffset)
            idx5 = int(rigiddomain5.ndOffset)
            idx6 = int(rigiddomain6.ndOffset)
            idx7 = int(rigiddomain7.ndOffset)
            idx8 = int(rigiddomain8.ndOffset)
            idx9 = int(rigiddomain9.ndOffset)
            angle1 = float(quat_arr[idx1][0])
            angle2 = float(quat_arr[idx2][0])
            angle4 = float(quat_arr[idx4][0])
            angle5 = float(quat_arr[idx5][0])
            angle6 = float(quat_arr[idx6][0])
            angle7 = float(quat_arr[idx7][0])
            angle8 = float(quat_arr[idx8][0])
            angle9 = float(quat_arr[idx9][0])
            pos1 = np.array([math.cos(angle1), math.sin(angle1)])
            pos2 = np.array([math.cos(angle2), math.sin(angle2)])
            pos4 = np.array([math.cos(angle4), math.sin(angle4)])
            pos5 = np.array([math.cos(angle5), math.sin(angle5)])
            pos6 = np.array([math.cos(angle6), math.sin(angle6)])
            pos7 = np.array([math.cos(angle7), math.sin(angle7)])
            pos8 = np.array([math.cos(angle8), math.sin(angle8)])
            pos9 = np.array([math.cos(angle9), math.sin(angle9)])
            # print(f"At t={t}s, Ball1 angle={angle1:.6f}, Ball2 angle={angle2:.6f}, Box1 angle={angle4:.6f}, Box2 angle={angle5:.6f}")
            # print(f"At t={t}s, Ball1 Pos: {pos1}, Ball2 Pos: {pos2}, Box1 Pos: {pos4}, Box2 Pos: {pos5}")
            checks = {
                "1_ball_EnforceRotVel": {"value": pos1, "expected": np.array([-1.0, 0.0])},
                "2_ball_EnforceRotAcc": {"value": pos2, "expected": np.array([-1.0, 0.0])},
                "4_box_EnforceRotVel": {"value": pos4, "expected": np.array([-1.0, 0.0])},
                "5_box_EnforceRotAcc": {"value": pos5, "expected": np.array([-1.0, 0.0])},
                "6_box_EnforceRotVel": {"value": pos6, "expected": np.array([-1.0, 0.0])},
                "7_box_EnforceRotAcc": {"value": pos7, "expected": np.array([-1.0, 0.0])},
                "8_capsule_EnforceRotVel": {"value": pos8, "expected": np.array([-1.0, 0.0])},
                "9_capsule_EnforceRotAcc": {"value": pos9, "expected": np.array([-1.0, 0.0])},
            }
            allclose = []
            for name, data in checks.items():
                # Ensure we're comparing XY components consistently
                actual = np.asarray(data["value"])[:2]
                expected = np.asarray(data["expected"])[:2]
                is_close = np.allclose(actual, expected, atol=1e-2, equal_nan=False)
                allclose.append(is_close)

                print(f"{name} Test:")
                print(f"  Expected: {data['expected']}")
                print(f"  Actual:   {data['value']}")

                if is_close:
                    print("  Result:   Success\n")
                else:
                    print("  Result:   Failure\n")

            assert np.allclose(allclose, [True for i in range(8)])
            break
    print(f"Simulation completed at t={t:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Drigid_rotation_benchmark")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Drigid_rotation_benchmark(headless=args.headless)
