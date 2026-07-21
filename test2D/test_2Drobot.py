import os
import sys
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
    Fixed,
    Force,
    Gravity,
    InitialVel,
    RigidBodyDomain,
    SolidProp,
)
from flatworld.joints import RevoluteJoint, WeldJoint
from flatworld.rigid import BoxRigid
from flatworld.rigidmanager import RigidManager
from math import pi
from test_utils import create_gui_if_available, init_sim


def test_robot(headless=False, kernel_profile=False):
    init_sim()
    gv = Gravity([0, -9.8])
    radius = 0.05
    center1 = [0.5, 0.85]
    rigid1 = BallRigid(2, center1, radius, 1.0)
    center2 = [0.5, 0.6]
    rigid2 = BoxRigid(2, center2, [0.2, 0.3], [0.0], 1.0)
    center3 = [0.3, 0.7]
    rigid3 = BoxRigid(2, center3, [0.2, 0.05], [pi / 3], 1.0)
    center4 = [0.25, 0.5]
    rigid4 = BoxRigid(2, center4, [0.2, 0.05], [2 * pi / 3], 1.0)
    center5 = [0.7, 0.7]
    rigid5 = BoxRigid(2, center5, [0.2, 0.05], [2 * pi / 3], 1.0)
    center6 = [0.75, 0.5]
    rigid6 = BoxRigid(2, center6, [0.2, 0.05], [pi / 3], 1.0)
    center7 = [0.5, 0.25]
    rigid7 = BoxRigid(2, center7, [0.15, 0.30], [0.0], 1.0)

    rigiddomain1 = RigidBodyDomain(rigid1, [gv], True)
    rigiddomain2 = RigidBodyDomain(rigid2, [gv], True)
    rigiddomain3 = RigidBodyDomain(rigid3, [gv], True)
    rigiddomain4 = RigidBodyDomain(rigid4, [gv], True)
    rigiddomain5 = RigidBodyDomain(rigid5, [gv], True)
    rigiddomain6 = RigidBodyDomain(rigid6, [gv], True)
    rigiddomain7 = RigidBodyDomain(rigid7, [gv], True)

    ground = GroundDomain(2, [0, 0.05], [0.0, 1.0], [], True)

    domains = [rigiddomain1, rigiddomain2, rigiddomain3, rigiddomain4, rigiddomain5, rigiddomain6, rigiddomain7, ground]
    colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0x00FFFF, 0xFF00FF, 0xC0C0C0]
    anchor1 = [0.5, 0.77]
    anchor2 = [0.5, 0.42]
    anchor3 = [0.35, 0.7]
    anchor4 = [0.2, 0.6]
    anchor5 = [0.65, 0.7]
    anchor6 = [0.8, 0.6]

    j1 = WeldJoint(0, 1, anchor1, bcs=[])
    j2 = WeldJoint(2, 1, anchor3, bcs=[])
    j3 = RevoluteJoint(2, 3, anchor4, bcs=[], axis=[0, 0])
    j4 = WeldJoint(1, 4, anchor5, bcs=[])
    j5 = RevoluteJoint(4, 5, anchor6, bcs=[], axis=[0, 0])
    j6 = WeldJoint(1, 6, anchor2, bcs=[])
    joints = [j1, j2, j3, j4, j5, j6]
    frame_dt = 1.0 / 60.0

    looper = ExplicitLoop(0.0, domains, joints, useAdapativeDT=True)
    length = 1280
    height = 1280
    gui = create_gui_if_available("RobotDemo", res=(720, 720)) if not headless else None
    t = 0.0
    frame = 0

    # Performance timing
    frame_times = []

    while frame < 120:

        frame_start = time.time()

        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        frame_times.append(time.time() - frame_start)

        frame += 1
        if gui is not None:
            gui.clear(0x112F41)
            for i in range(7):
                color = colors[i]
                domains[i].draw(gui, color=color, resolution=length)

            anchors = []
            for i in range(6):
                joints[i].draw(gui, color=0x000000, resolution=length)

        # gui.line(anchors[0], domains[0].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[0], domains[1].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[1], domains[1].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[1], domains[2].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[2], domains[2].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[2], domains[3].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[3], domains[1].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[3], domains[4].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[4], domains[4].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[4], domains[5].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[5], domains[0].getCurrentRefPoint(), 5, 0x000000)
        # gui.line(anchors[5], domains[6].getCurrentRefPoint(), 5, 0x000000)

        if gui is not None:
            gui.line([0, 0.05], [1.0, 0.05], 5, 0xFFF000)
            gui.text(f"Time: {t:.4f} s", pos=(0.02, 0.95), color=0x000000, font_size=24)
            gui.text(f"Hello World!", pos=(0.9, 0.95), color=0x000000, font_size=24)

            gui.show()

    # Performance statistics
    import numpy as np

    frame_times = np.array(frame_times)
    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Total frames: {len(frame_times)}")
    print(f"Average frame time: {np.mean(frame_times)*1000:.2f} ms")
    print(f"Min frame time: {np.min(frame_times)*1000:.2f} ms")
    print(f"Max frame time: {np.max(frame_times)*1000:.2f} ms")
    print(f"Std deviation: {np.std(frame_times)*1000:.2f} ms")
    print(f"Average FPS: {1.0/np.mean(frame_times):.1f}")
    print("=" * 60)

    if kernel_profile:
        print("\nKERNEL PROFILER INFO:")
        print("=" * 60)
        # Warp profiler placeholder (was ti.profiler)
        pass
        # wp.synchronize()  # optional

    # ---------------------------------------------------------
    # Final Result Verification
    # ---------------------------------------------------------
    # Convert lists to numpy for calculation
    c1 = np.array(center1)
    c2 = np.array(center2)
    c3 = np.array(center3)
    c4 = np.array(center4)
    c5 = np.array(center5)
    c6 = np.array(center6)
    c7 = np.array(center7)

    a1 = np.array(anchor1)
    a2 = np.array(anchor2)
    a3 = np.array(anchor3)
    a4 = np.array(anchor4)
    a5 = np.array(anchor5)
    a6 = np.array(anchor6)

    pos1 = rigiddomain1.getCurrentRefPoint()  # Head
    pos2 = rigiddomain2.getCurrentRefPoint()  # Torso
    pos3 = rigiddomain3.getCurrentRefPoint()  # L.Upper
    pos4 = rigiddomain4.getCurrentRefPoint()  # L.Lower
    pos5 = rigiddomain5.getCurrentRefPoint()  # R.Upper
    pos6 = rigiddomain6.getCurrentRefPoint()  # R.Lower
    pos7 = rigiddomain7.getCurrentRefPoint()  # Legs

    joint1_pos = j1.getCurrentAnchorPoint()  # J1: Head(0)-Torso(1)
    joint2_pos = j2.getCurrentAnchorPoint()  # J2: L.Upper(2)-Torso(1)
    joint3_pos = j3.getCurrentAnchorPoint()  # J3: L.Upper(2)-L.Lower(3)
    joint4_pos = j4.getCurrentAnchorPoint()  # J4: Torso(1)-R.Upper(4)
    joint5_pos = j5.getCurrentAnchorPoint()  # J5: R.Upper(4)-R.Lower(5)
    joint6_pos = j6.getCurrentAnchorPoint()  # J6: Torso(1)-Legs(6)

    print(f"\nFinal positions evaluation:")
    test_results = []

    # J1: Head - Torso at Anchor1. Reference Frame: Head (1)
    if not np.allclose(np.sqrt(np.sum((pos1 - joint1_pos) ** 2)), np.sqrt(np.sum((c1 - a1) ** 2)), atol=1e-2):
        test_results.append(
            f"❌ Joint1 (Head-Torso) length error wrt Head. Expected {np.sqrt(np.sum((c1 - a1) ** 2)):.4f}, got {np.sqrt(np.sum((pos1 - joint1_pos) ** 2)):.4f}"
        )
    else:
        test_results.append(f"✓ Joint1 (Head-Torso) verified")

    # J2: L.Upper - Torso at Anchor3. Reference Frame: L.Upper (3)
    if not np.allclose(np.sqrt(np.sum((pos3 - joint2_pos) ** 2)), np.sqrt(np.sum((c3 - a3) ** 2)), atol=1e-2):
        test_results.append(
            f"❌ Joint2 (L.Upper-Torso) length error wrt L.Upper. Expected {np.sqrt(np.sum((c3 - a3) ** 2)):.4f}, got {np.sqrt(np.sum((pos3 - joint2_pos) ** 2)):.4f}"
        )
    else:
        test_results.append(f"✓ Joint2 (L.Upper-Torso) verified")

    # J3: L.Upper - L.Lower at Anchor4. Reference Frame: L.Upper (3)
    if not np.allclose(np.sqrt(np.sum((pos3 - joint3_pos) ** 2)), np.sqrt(np.sum((c3 - a4) ** 2)), atol=1e-2):
        test_results.append(
            f"❌ Joint3 (L.Elbow) length error wrt L.Upper. Expected {np.sqrt(np.sum((c3 - a4) ** 2)):.4f}, got {np.sqrt(np.sum((pos3 - joint3_pos) ** 2)):.4f}"
        )
    else:
        test_results.append(f"✓ Joint3 (L.Elbow) verified")

    # J4: R.Upper - Torso at Anchor5. Reference Frame: R.Upper (5/index 4 in loop) which is rigiddomain5 (pos5).
    if not np.allclose(np.sqrt(np.sum((pos5 - joint4_pos) ** 2)), np.sqrt(np.sum((c5 - a5) ** 2)), atol=1e-2):
        test_results.append(
            f"❌ Joint4 (R.Upper-Torso) length error wrt R.Upper. Expected {np.sqrt(np.sum((c5 - a5) ** 2)):.4f}, got {np.sqrt(np.sum((pos5 - joint4_pos) ** 2)):.4f}"
        )
    else:
        test_results.append(f"✓ Joint4 (R.Upper-Torso) verified")

    # J5: R.Upper - R.Lower at Anchor6. Reference Frame: R.Upper (5).
    if not np.allclose(np.sqrt(np.sum((pos5 - joint5_pos) ** 2)), np.sqrt(np.sum((c5 - a6) ** 2)), atol=1e-2):
        test_results.append(
            f"❌ Joint5 (R.Elbow) length error wrt R.Upper. Expected {np.sqrt(np.sum((c5 - a6) ** 2)):.4f}, got {np.sqrt(np.sum((pos5 - joint5_pos) ** 2)):.4f}"
        )
    else:
        test_results.append(f"✓ Joint5 (R.Elbow) verified")

    # J6: Torso - Legs at Anchor2. Reference Frame: Legs (7/index 6) which is rigiddomain7 (pos7).
    if not np.allclose(np.sqrt(np.sum((pos7 - joint6_pos) ** 2)), np.sqrt(np.sum((c7 - a2) ** 2)), atol=1e-2):
        test_results.append(
            f"❌ Joint6 (Torso-Legs) length error wrt Legs. Expected {np.sqrt(np.sum((c7 - a2) ** 2)):.4f}, got {np.sqrt(np.sum((pos7 - joint6_pos) ** 2)):.4f}"
        )
    else:
        test_results.append(f"✓ Joint6 (Torso-Legs) verified")

    print("\nTest Summary:")
    for res in test_results:
        print(res)

    failed_count = sum(1 for r in test_results if r.startswith("❌"))
    passed_count = sum(1 for r in test_results if r.startswith("✓"))

    print("=" * 60)
    print(f"total：pass {passed_count} item，fail {failed_count} item")
    if failed_count == 0:
        print("✅ 3DRobot simulation test completed(2Dversion) - All checks passed")
    else:
        assert False, f"⚠️  3DRobot simulation test completed（2D） - have {failed_count} Check failed"


if __name__ == "__main__":
    # Set to True to enable kernel profiling
    test_robot(kernel_profile=True)
