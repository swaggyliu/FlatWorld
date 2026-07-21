"""
Dominoes test case
use Box Rigid as dominoes，Ball Rigid as a trigger ball
"""

import numpy as np
import os

# Add parent directory to path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from flatworld import Gravity, RigidBodyDomain
from flatworld import GroundDomain
from flatworld.definitions import *
from flatworld.explicitloop import ExplicitLoop
from flatworld import BallRigid, BoxRigid
from test_utils import create_gui_if_available, init_sim


def test_2drigid_domino(headless=False):

    init_sim()

    # scene parameters
    d = 2  # 2D simulation
    dt = 0.003

    # domino parameters
    domino_width = 0.1
    domino_height = 0.3
    domino_thickness = 0.05
    domino_spacing = 0.15  # spacing between dominoes
    num_dominoes = 15  # number of dominoes
    domino_start_x = 0.5
    domino_y = domino_height / 2  # The bottom of the domino is on the ground

    # Trigger ball parameters
    ball_radius = 0.1
    ball_start_x = 0.3
    ball_start_y = 0.3
    ball_initial_velocity = [3.0, 0.0]  # Roll right to hit the first domino

    # Create domain list
    domains = []

    # 1. create ground（parsing domain - horizontal plane）
    ground = GroundDomain(d=d, point=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]), bcs=[])
    domains.append(ground)

    # 2. Create dominoes（vertical slender square）
    for i in range(num_dominoes):
        x_pos = domino_start_x + i * domino_spacing

        # Create an upright domino
        domino_box = BoxRigid(
            d=d,
            ext=np.array([domino_thickness, domino_height]),  # Thin and tall rectangle
            mass=1.0,
            angle=[0.0],  # The initial angle is0（vertical）
            origin=np.array([x_pos, domino_y]),
        )

        domino_domain = RigidBodyDomain(rigid=domino_box, bcs=[Gravity(np.array([0.0, -9.8]))])
        domains.append(domino_domain)

    # 3. Create a trigger ball
    trigger_ball = BallRigid(d=d, radius=ball_radius, mass=1.0, origin=np.array([ball_start_x, ball_start_y]))

    ball_domain = RigidBodyDomain(rigid=trigger_ball, bcs=[Gravity(np.array([0.0, -9.8]))])
    domains.append(ball_domain)

    # Create an explicit time-stepping loop（Use adaptivedt + Fixed frame assist）
    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    # Set the initial speed for the trigger ball
    ball_rigid_idx = ball_domain.ndOffset
    v_np = looper.rigidManager.V.numpy()
    v_np[ball_rigid_idx] = ball_initial_velocity
    looper.rigidManager.V.assign(v_np)

    # createGUI
    window_width = 800
    window_height = 800
    gui = (
        create_gui_if_available("domino test (Domino Test)", res=(window_width, window_height))
        if not headless
        else None
    )

    # Camera parameters（2DView range）
    view_left = -0.5
    view_right = 4.0
    view_bottom = -0.2
    view_top = 1.5

    # main loop
    frame = 0
    paused = False
    exportFrame = 0
    frame_dt = 1.0 / 60.0

    print("control:")
    print("  space bar: pause/continue")
    print("  Rkey: reset scene")
    print("  ESC: quit")

    while (gui is None or gui.running) and frame < 60:
        # handle events

        if not paused:
            # advance exactly one visual frame using adaptive substeps
            looper.advanceWithTime(frame_dt)
            frame += 1

        # Draw the ground
        if gui is not None:
            gui.line([0, 0], [1, 0], radius=3, color=0x666666)

            # Draw all rigid bodies
            colors = []
            for i, domain in enumerate(domains):
                if domain.type == DomainType.RIGID:
                    rigid_type = domain.rigid.rtype
                    if rigid_type == RigidType.BOX:
                        colors.append(0x4A90E2)  # blue dominoes
                    elif rigid_type == RigidType.BALL:
                        colors.append(0xE74C3C)  # red trigger ball
                    else:
                        colors.append(0xFFFFFF)

            # Batch drawing（userigidManagerofdrawAllmethod）
            looper.rigidManager.drawAll(gui, domains, colors=colors, resolution=800)
            gui.text(f"Frame: {frame}", pos=(0.02, 0.98), color=0xFFFFFF, font_size=20)
            gui.text(f"Time: {frame * frame_dt:.2f}s", pos=(0.02, 0.94), color=0xFFFFFF, font_size=20)
            gui.text(f"Dominoes: {num_dominoes}", pos=(0.02, 0.90), color=0xFFFFFF, font_size=20)

            gui.show()

    if gui is not None:
        gui.close()
    print("Simulation ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2drigid_domino")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2drigid_domino(headless=args.headless)
