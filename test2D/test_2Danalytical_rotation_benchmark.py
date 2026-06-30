import math
import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import GroundDomain, BoxRigid, EnforceRotAcc, EnforceRotVel, ExplicitLoop, RigidBodyDomain
from test_utils import create_gui_if_available


def test_2Danalytical_rotation(headless=False):
    """Test 2D analytical planes with rotation boundary conditions.

    Demonstrates:
    - EnforceRotVel: prescribes constant rotational velocity
    - EnforceRotAcc: prescribes constant rotational acceleration

    Both cause the analytical planes to rotate around a pivot point.
    """
    ti.init(offline_cache=True, arch=ti.cpu, debug=False)
    pi = math.pi
    # Create rotating analytical barriers (walls) with rotation BCs
    # Wall 1: Vertical wall rotating with constant angular velocity (10 rad/s CCW)
    pivot1 = [0.3, 0.5]  # Center of domain
    wall1 = GroundDomain(
        2, pivot1, [1.0, 0.0], bcs=[EnforceRotVel([0], [pi], origin=pivot1)], considerContact=False
    )

    # Wall 2: Horizontal wall rotating with constant angular acceleration (5 rad/s^2 CCW)
    pivot2 = [0.3, 0.7]
    wall2 = GroundDomain(
        2, pivot2, [0.0, 1.0], bcs=[EnforceRotAcc([0], [2 * pi], origin=pivot2)], considerContact=False
    )

    # Wall 3: Diagonal wall rotating with negative angular velocity (CW rotation)
    pivot3 = [0.3, 0.3]
    wall3 = GroundDomain(
        2, pivot3, [0.707, 0.707], bcs=[EnforceRotVel([0], [-pi], origin=pivot3)], considerContact=False
    )

    # Static boundary walls (no rotation)
    domains = [wall1, wall2, wall3]

    domains += [GroundDomain(2, [0.05, 0.5], [1.0, 0.0], [])]  # Left wall
    domains += [GroundDomain(2, [0.95, 0.5], [-1.0, 0.0], [])]  # Right wall
    domains += [GroundDomain(2, [0.5, 0.05], [0.0, 1.0], [])]  # Bottom wall
    domains += [GroundDomain(2, [0.5, 0.95], [0.0, -1.0], [])]  # Top wall

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("2D Analytical Rotation BCs", res=(720, 720)) if not headless else None
    t = 0.0

    color_box = 0x3333FF

    print("Rotating analytical walls demonstration:")
    print("  Red: EnforceRotVel (constant angular velocity)")
    print("  Green: EnforceRotAcc (constant angular acceleration)")
    print("  Blue: EnforceRotVel (negative - clockwise)")
    print("  White: Static boundary walls")

    while (gui is None or gui.running) and t < 1.0:  # Run for 10 seconds
        looper.advanceWithTime(frame_dt)
        t += frame_dt

        if gui is not None:
            gui.clear(0x222222)

            # Draw rotating walls with different colors
            wall1.draw(gui, color=0xFF3333, resolution=720)  # Red
            wall2.draw(gui, color=0x33FF33, resolution=720)  # Green
            wall3.draw(gui, color=0x3333FF, resolution=720)  # Blue

            # Draw static boundary walls
            for i in range(3, 7):
                domains[i].draw(gui, color=0xAAAAAA)

            # Draw pivot points
            pivots = np.array([pivot1, pivot2, pivot3])
            gui.circles(pivots, radius=3, color=0xFFFF00)

            gui.text(f"Time: {t:.2f}s", pos=(0.02, 0.95), color=0xFFFFFF, font_size=20)

            gui.show()

        if abs(t - 1) < 1e-3:
            pos1 = wall1.getCurrentNormal()
            pos2 = wall2.getCurrentNormal()
            pos3 = wall3.getCurrentNormal()
            print(f"At t={t}s, Wall1 Pos: {pos1}, Wall2 Pos: {pos2}, Wall3 Pos: {pos3}")
            checks = {
                "1_Wall_EnforceRotVel": {"value": pos1, "expected": np.array([-1.0, 0.0])},
                "2_Wall_EnforceRotAcc": {"value": pos2, "expected": np.array([0, -1.0])},
                "3_Wall_EnforceRotVel": {"value": pos3, "expected": np.array([-0.707, -0.707])},
            }
            allclose = []
            for name, data in checks.items():
                actual = data["value"]
                expected = data["expected"]
                is_close = np.allclose(actual, expected, atol=3e-2, rtol=1e-2, equal_nan=False)
                allclose.append(is_close)

                print(f"{name} Test:")
                print(f"  Expected: {expected}")
                print(f"  Actual:   {actual}")

                if is_close:
                    print("  Result:   Success\n")
                else:
                    print("  Result:   Failure\n")

            assert np.allclose(allclose, [True for i in range(3)])
            return 1
    print(f"Simulation completed at t={t:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Danalytical_rotation")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Danalytical_rotation(headless=args.headless)
