import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from flatworld import (
    GroundDomain,
    EnforceRotVel,
    EnforceVel,
    ExplicitLoop,
    Fixed,
    FixedAll,
    Gravity,
    InitialVel,
    RigidBodyDomain,
)
from flatworld.joints import RevoluteJoint, WeldJoint
from flatworld.rigid import BallRigid, BoxRigid
from flatworld.rigidmanager import RigidManager
from math import pi
import numpy as np
from test_utils import create_gui_if_available, init_sim


def test_linkage(headless=False, kernel_profile=False):
    init_sim()
    gv = Gravity([0, -9.8])
    fixall = FixedAll([0])
    radius = 0.05
    omega = 4.0 * pi  # Angular velocity: 1 change/Second
    enforce_rot = EnforceRotVel([0], [omega])

    # Performance tracking
    perf_times = {"joints": 0.0, "rigid_update": 0.0, "bvh": 0.0, "total": 0.0}

    rigid1 = BoxRigid(2, [0.2, 0.2], [0.1, 0.1], [0.0], 1.0)
    rigid2 = BoxRigid(2, [0.3, 0.125], [0.6, 0.05], [0.0], 1.0)
    rigid3 = BoxRigid(2, [0.3, 0.275], [0.6, 0.05], [0.0], 1.0)
    rigid4 = BallRigid(2, [0.75, 0.6], radius / 2, 1.0)
    rigid5 = BallRigid(2, [0.75, 0.5], radius * 2, 1.0)  # drive disc
    rigid6 = BallRigid(2, [0.2, 0.2], 0.03, 1.0)

    rigiddomain1 = RigidBodyDomain(rigid1, [], True)
    rigiddomain2 = RigidBodyDomain(rigid2, [fixall], True)  # fixed，No contact inspection required�?
    rigiddomain3 = RigidBodyDomain(rigid3, [fixall], True)  # fixed，No contact inspection required�?
    rigiddomain4 = RigidBodyDomain(rigid4, [], False)  # No consideration for contact
    rigiddomain5 = RigidBodyDomain(rigid5, [Fixed([0]), enforce_rot], False)  # drive disc，No consideration for contact
    rigiddomain6 = RigidBodyDomain(rigid6, [], False)  # gravity disc，No consideration for contact

    domains = [rigiddomain1, rigiddomain2, rigiddomain3, rigiddomain4, rigiddomain5, rigiddomain6]
    colors = [0xFF0000, 0x000000, 0x000000, 0xFFFF00, 0xFF00FF, 0x00FFFF]
    anchor1 = [0.7, 0.5]
    anchor2 = [0.2, 0.2]

    j1 = RevoluteJoint(3, 5, anchor1, bcs=[], axis=[0, 0])
    j2 = WeldJoint(4, 3, [0.75, 0.6], bcs=[])  # rigid4 and rigid5 welding
    j3 = RevoluteJoint(5, 0, anchor2, bcs=[], axis=[0, 0])
    joints = [j1, j2, j3]
    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, joints, useAdapativeDT=True)
    length = 1280
    height = 1280
    gui = create_gui_if_available("RobotDemo", res=(length, height)) if not headless else None
    t = 0.0
    frame = 0
    last_print = 0.0
    while frame < 120:

        # advance one visual frame using adaptive substeps
        t_start = time.time()
        looper.advanceWithTime(frame_dt)
        t_end = time.time()

        perf_times["total"] += t_end - t_start

        t += frame_dt
        frame += 1

        # Print performance stats every 5 seconds
        if t - last_print >= 5.0:
            avg_time = perf_times["total"] / frame if frame > 0 else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Frame {frame}, Time: {t:.2f}s, Avg frame time: {avg_time*1000:.2f}ms, FPS: {fps:.1f}")
            last_print = t

        node1 = rigiddomain1.getCurrentRefPoint()
        nodeanchor = j3.getCurrentAnchorPoint()
        if gui is not None:
            gui.clear(0x112F41)
            looper.rigidManager.drawAll(gui, domains, colors, resolution=length)

            anchors = []
            for i in range(3):
                joints[i].draw(gui, color=0x000000, resolution=length)
                anchors.append(joints[i].getCurrentAnchorPoint())

        # Show performance information
        # avg_time = perf_times['total'] / frame if frame > 0 else 0
        # fps = 1.0 / avg_time if avg_time > 0 else 0
        # gui.text(f'Time: {t:.4f} s', pos=(0.02, 0.95), color=0x000000, font_size=24)
        # gui.text(f'FPS: {fps:.1f} ({avg_time*1000:.2f}ms/frame)', pos=(0.02, 0.90), color=0x000000, font_size=20)
        # gui.text(f'Substeps: {looper.counter.numpy()[0]}', pos=(0.02, 0.85), color=0x000000, font_size=20)

        if gui is not None:
            gui.show()

    print(f"Rigid1 pos: {node1}, Anchor pos: {nodeanchor}")
    assert np.allclose(node1, [0.2, 0.2], atol=1e-2), "End effector did not reach the expected position."
    assert np.allclose(nodeanchor, [0.2, 0.2], atol=1e-2), "Anchor did not stay in the expected position."

    # ti.profiler.print_kernel_profiler_info()
    # ti.profiler.clear_kernel_profiler_info()


if __name__ == "__main__":
    test_linkage()
