import math
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
    EnforceRotAcc,
    EnforceRotVel,
    ExplicitLoop,
    Fixed,
    FixedAll,
    Force,
    Gravity,
    RigidBodyDomain,
)
from flatworld.joints import RevoluteJoint, WeldJoint
from test_utils import create_gui_if_available, init_sim


def test_2Dsolar_system(headless=False):

    init_sim()
    sun_radius_vis = 0.06

    earth_orbit_radius = 0.25
    earth_radius_vis = 0.02  # sun_radius_vis/100
    # orbital period in seconds (visual time)
    earth_period = 10.0

    moon_orbit_radius = 0.06
    moon_radius_vis = 0.008  # earth_radius_vis/3.5
    moon_period = 2.5

    pi = math.pi
    pivot1 = [0.5, 0.5]
    Sun = BallRigid(2, pivot1, sun_radius_vis, 1.0)
    rigiddomain1 = RigidBodyDomain(Sun, bcs=[FixedAll([0])], considerContact=False, initials=[])
    pivot2 = [0.75, 0.5]
    Earth = BallRigid(2, pivot2, earth_radius_vis, 1.0)
    rigiddomain2 = RigidBodyDomain(Earth, bcs=[], considerContact=False, initials=[])

    pivot3 = [0.81, 0.5]
    Moon = BallRigid(2, pivot3, moon_radius_vis, 1.0)
    rigiddomain3 = RigidBodyDomain(Moon, bcs=[], considerContact=False, initials=[])

    domains = [rigiddomain1, rigiddomain2, rigiddomain3]

    joint1 = RevoluteJoint(0, 1, pivot1, [0, 0], bcs=[EnforceRotVel([0], [10 * pi / earth_period], origin=pivot1)])
    joint2 = RevoluteJoint(1, 2, pivot2, [0, 0], bcs=[EnforceRotVel([0], [10 * pi / moon_period], origin=pivot2)])

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, joints=[joint1, joint2], useAdapativeDT=True)

    gui = create_gui_if_available("2D Rigid Rotation BCs", res=(720, 720)) if not headless else None
    t = 0.0

    color_box = 0x3333FF

    print("Rotating rigid demonstration:")
    print("  Sun (yellow): fixed at center")
    print("  Earth (blue): orbits Sun")
    print("  Moon (gray): orbits Earth")

    # Initialize trajectory tracking
    trail_length = 600  # Store up to 600 points (10 seconds at 60 fps)
    trail_sun = []
    trail_earth = []
    trail_moon = []

    while (gui is None or gui.running) and t < 2.0:  # Run for 2 seconds

        if t + frame_dt > 2.0:
            frame_dt = 2.0 - t  # Adjust last step to end exactly at 2.0 seconds
        looper.advanceWithTime(frame_dt)
        t += frame_dt

        # Get current positions from rigidManager
        # mgr = looper.rigidManager
        # params = mgr.rigidParams.numpy()
        # pos_sun = params[0, 0, :2]      # rigiddomain1 (index 0)
        # pos_earth = params[1, 0, :2]    # rigiddomain2 (index 1)
        # pos_moon = params[2, 0, :2]     # rigiddomain3 (index 2)

        # # Update trails (keep last trail_length points)
        # trail_sun.append(pos_sun.copy())
        # if len(trail_sun) > trail_length:
        #     trail_sun.pop(0)

        # trail_earth.append(pos_earth.copy())
        # if len(trail_earth) > trail_length:
        #     trail_earth.pop(0)

        # trail_moon.append(pos_moon.copy())
        # if len(trail_moon) > trail_length:
        #     trail_moon.pop(0)

        # gui.clear(0x222222)

        # # Draw trails as small circles
        # if len(trail_sun) > 1:
        #     gui.circles(np.array(trail_sun), radius=1, color=0xFFCC33)
        # if len(trail_earth) > 1:
        #     gui.circles(np.array(trail_earth), radius=1, color=0x3399FF)
        # if len(trail_moon) > 1:
        #     gui.circles(np.array(trail_moon), radius=1, color=0xDDDDDD)

        # Display time
        if gui is not None:
            # Draw current bodies
            rigiddomain1.draw(gui, color=0xFFCC33, resolution=720)
            rigiddomain2.draw(gui, color=0x3399FF, resolution=720)
            rigiddomain3.draw(gui, color=0xDDDDDD, resolution=720)
            gui.text(f"Time: {t:.2f}s", pos=(0.02, 0.95), color=0xFFFFFF, font_size=20)

            gui.show()
    earth_pos = rigiddomain2.getCurrentRefPoint()
    moon_pos = rigiddomain3.getCurrentRefPoint()
    print("Final Earth position: ", earth_pos)
    print("Final Moon position: ", moon_pos)
    assert np.allclose(earth_pos, [0.75, 0.5], atol=1e-2), "Earth final position incorrect"
    assert np.allclose(moon_pos, [0.81, 0.5], atol=1e-2), "Moon final position incorrect"
    print(f"Simulation completed at t={t:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dsolar_system")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dsolar_system(headless=args.headless)
