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

    ti.init(offline_cache=True, arch=ti.cpu, kernel_profiler=False)
    numberRigid = 25
    domains = []
    colors = []
    nx = np.sqrt(numberRigid).astype(int)
    ny = nx
    dx = 0.8 / nx
    radius = dx / 4
    for i in range(nx):
        for j in range(ny):
            rigid1 = BallRigid(2, [0.1 + dx * i, 0.1 + dx * j], radius, 1.0)
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

    gui = create_gui_if_available("FEM2D", res=(1080, 1080)) if not headless else None
    t = 0.0
    frame = 0
    tt0 = time.time()
    while (gui is None or gui.running) and t < 1.0:
        tt = time.time()
        looper.advanceWithTime(frame_dt)

        t += frame_dt
        frame += 1
        timeend = time.time()
        dur = timeend - tt
        print("It takes {} s.".format(dur))
        pos = looper.rigidManager.rigidParams.to_numpy()[:numberRigid, 0, :]
        if gui is not None:
            gui.circles(pos, color=colors, radius=radius * 1080)
            gui.show()

    tt1 = time.time()
    print("Total time:", tt1 - tt0)
    print("FPS", 1000 / (tt1 - tt0))
    # ti.profiler.print_scoped_profiler_info()
    # ti.profiler.print_kernel_profiler_info()
    # ti.profiler.clear_kernel_profiler_info()

    print("\nPerforming quantitative penetration test...")
    # Get center positions and radii from rigid manager
    mgr = looper.rigidManager
    params_np = mgr.rigidParams.to_numpy()
    radius_np = mgr.radius.to_numpy()
    num_rigids = mgr.numRigids

    overlap_count = 0

    # Check all pairs of balls (center-to-center distance vs sum of radii)
    for i in range(numberRigid):
        p_i = params_np[i, 0]
        r_i = radius_np[i]

        for j in range(i + 1, numberRigid):
            p_j = params_np[j, 0]
            r_j = radius_np[j]

            # Distance between center points
            dist = np.linalg.norm(p_i - p_j)

            # Check if distance is less than sum of radii (with small tolerance)
            # This ensures that the spheres do not penetrate.
            if dist < (r_i + r_j) - 0.005:
                overlap_count += 1

    if overlap_count > 0:
        print(f"Warning: Found {overlap_count} pairs with significant center point overlap.")
        assert False, "Quantitative penetration test failed: significant center-to-center penetration detected."
    else:
        print("Success: All balls are well-separated (distance > sum of radii).")

    print("\nQuantitative distance check passed (within experimental tolerance).")
    print("Simulation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Drigid_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Drigid_contact(headless=args.headless)
