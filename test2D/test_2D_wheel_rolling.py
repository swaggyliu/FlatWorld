import numpy as np
import os
import sys
import taichi as ti
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
    Gravity,
    Mesh,
    RigidBodyDomain,
    SolidProp,
)
from test_utils import create_gui_if_available


def test_cylinder_wheel_rolling(headless=False):

    ti.init(offline_cache=True, arch=ti.cpu, kernel_profiler=False)

    domains = []

    # Create a cylinder wheel
    # Parameters: center at (0.5, 0.2), radius 0.1, height 0.05 (for 2D representation)
    wheel_radius = 0.2
    wheel_mass = 1.0

    wheel = BallRigid(2, [0.5, 0.2], wheel_radius, wheel_mass)

    # Apply constant angular velocity to make it roll
    angular_velocity = -10.0  # radians per second (clockwise)
    bcs = [EnforceRotVel([0], [angular_velocity]), Gravity([0.0, -9.81])]

    wheel_domain = RigidBodyDomain(wheel, bcs, considerContact=True, friction=1.0)
    domains.append(wheel_domain)

    # Create analytical ground plane (horizontal at y=0)
    ground = GroundDomain(2, [0.0, 0.0], [0.0, 1.0])  # point on plane, normal vector
    domains.append(ground)

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)
    looper.stableTime = 1e-3

    gui = create_gui_if_available("Cylinder Wheel Rolling", res=(800, 800)) if not headless else None
    t = 0.0
    frame = 0
    tt0 = time.time()

    while (gui is None or gui.running) and t <= 1.0:  # Run for 1 second
        tt = time.time()
        looper.advanceWithTime(frame_dt)

        # Clear GUI
        if gui is not None:
            gui.clear(0x112F41)

        # Draw ground
        if gui is not None:
            gui.line((0.0, 0.0), (1.0, 0.0), color=0xFFFFFF, radius=2)

        # Draw wheel
        pos = wheel_domain.getCurrentRefPoint()
        if gui is not None:
            gui.circle(pos, radius=wheel_radius * 800, color=0xFFAA00)

        # Display info
        if gui is not None:
            gui.text(f"Time: {t:.2f}s", (0.02, 0.95), font_size=20, color=0xFFFFFF)
            gui.text(f"Angular Velocity: {angular_velocity:.1f} rad/s", (0.02, 0.90), font_size=20, color=0xFFFFFF)

            gui.show()
        t += frame_dt
        frame += 1

        if frame % 60 == 0:  # Print every second
            print(f"Time: {t:.2f}s, Frame: {frame}")
    print("The final position of the wheel center is:", pos)
    assert pos[0] > 2.25, "Wheel did not roll forward as expected."

    print(f"Simulation completed. Total frames: {frame}, Total time: {t:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_cylinder_wheel_rolling")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_cylinder_wheel_rolling(headless=args.headless)
