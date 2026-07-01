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
    CapsuleRigid,
    Elastic,
    EnforceVel,
    ExplicitLoop,
    FemDomain,
    Gravity,
    Mesh,
    RigidBodyDomain,
    SolidProp,
)
from test_utils import create_gui_if_available


def test_2Drigid_contact(headless=False):

    ti.init(offline_cache=True, arch=ti.cpu)
    numberRigid = 25
    domains = []
    colors = []
    nx = int(np.sqrt(numberRigid))
    ny = nx
    dx = 0.8 / nx
    radius = dx / 3
    length = dx * 0.3
    for i in range(nx):
        for j in range(ny):
            cx = 0.1 + dx * i
            cy = 0.1 + dx * j
            # make capsule vertical: endpoints above and below center
            p1 = [cx, cy - length * 0.5]
            p2 = [cx, cy + length * 0.5]
            rigid1 = CapsuleRigid(2, p1, p2, [0], radius, 1.0)
            bcs = [Gravity([np.random.rand() * 10.0 - 5.0, np.random.rand() * 10.0 - 5.0])]

            domains.append(RigidBodyDomain(rigid1, bcs, considerContact=True))
            colors.append(int(np.random.rand() * 0xFFFFFF))

    anl1 = GroundDomain(2, [0, 0.0], [0.0, 1.0])
    anl2 = GroundDomain(2, [0, 1.0], [0.0, -1.0])
    anl3 = GroundDomain(2, [0, 0.0], [1.0, 0.0])
    anl4 = GroundDomain(2, [1.0, 0.0], [-1.0, 0.0])
    domains += [anl1, anl2, anl3, anl4]

    colors = np.array(colors, dtype=np.int32)

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = (
        create_gui_if_available("RIGID2D - Capsules", res=(1080, 1080), background_color=0x112F41)
        if not headless
        else None
    )
    t = 0.0
    while (gui is None or gui.running) and t < 0.5:

        tt = time.time()
        looper.advanceWithTime(frame_dt)

        t += frame_dt
        timeend = time.time()
        dur = timeend - tt
        print("It takes {} s.".format(dur))
        # # draw each rigid
        if gui is not None:
            looper.rigidManager.drawAll(gui, domains, colors, 1080)
            gui.show()

    print("\nPerforming quantitative penetration test...")
    # Get center positions and radii from rigid manager
    mgr = looper.rigidManager
    params_np = mgr.rigidParams.to_numpy()
    radius_np = mgr.radius.to_numpy()
    num_rigids = mgr.numRigids

    overlap_count = 0

    # Check all pairs of capsules (center-to-center distance vs sum of radii)
    for i in range(numberRigid):
        p_i = params_np[i, 0]
        r_i = radius_np[i]

        for j in range(i + 1, numberRigid):
            p_j = params_np[j, 0]
            r_j = radius_np[j]

            # Distance between center points
            dist = np.linalg.norm(p_i - p_j)

            # Check if distance is less than sum of radii (with small tolerance)
            # This ensures that the spheres at the center of each capsule do not penetrate.
            if dist < (r_i + r_j) - 0.005:
                overlap_count += 1

    if overlap_count > 0:
        print(f"Warning: Found {overlap_count} pairs with significant center point overlap.")
        assert False, "Quantitative penetration test failed: significant center-to-center penetration detected."
    else:
        print("Success: All capsule centers are well-separated (distance > sum of radii).")

    print("\nQuantitative distance check passed (within experimental tolerance).")
    print("Simulation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Drigid_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Drigid_contact(headless=args.headless)
