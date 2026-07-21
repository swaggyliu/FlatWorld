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
    Elastic,
    ExplicitLoop,
    FemDomain,
    FEMesher,
    Gravity,
    Mesh,
    MeshRigid,
    RigidBodyDomain,
    SolidProp,
)
from test_utils import create_gui_if_available, init_sim
import time


def test_2Dfemrigid_contact(headless=False):
    """Test contact between FEM body and three types of rigid bodies:
    - BallRigid (primitive sphere)
    - BoxRigid (primitive box with rotation)
    - MeshRigid (triangle mesh rigid body)
    """

    init_sim()

    # Create FEM deformable body (circle in the middle-bottom)
    msh = FEMesher(2)
    fem_mesh = msh.createCircle([0.5, 0.25], 0.12)
    fem_bcs = [Gravity([0, -10.0])]  # Gravity pulling down
    mat = Elastic(E=5e6, nu=0.3, rho=1000.0)
    prop = SolidProp(mat)
    fem_domain = FemDomain(fem_mesh, prop, fem_bcs, considerContact=True)

    # Rigid Body 1: BallRigid (left side, falling)
    ball_rigid = BallRigid(2, [0.25, 0.7], 0.08, 10.0)
    ball_bcs = [Gravity([0, -10.0])]
    ball_domain = RigidBodyDomain(ball_rigid, ball_bcs, considerContact=True)

    # Rigid Body 2: BoxRigid (center, falling with rotation)
    box_rigid = BoxRigid(2, [0.5, 0.75], [0.15, 0.08], [pi / 6], 10.0)  # Rotated 30 degrees
    box_bcs = [Gravity([0, -10.0])]
    box_domain = RigidBodyDomain(box_rigid, box_bcs, considerContact=True)

    # Rigid Body 3: MeshRigid (right side, triangle mesh)
    rigid_mesh = msh.createCircle([0.55, 0.5], 0.08)
    mesh_rigid = MeshRigid(2, rigid_mesh, [0.0], 10.0)
    mesh_bcs = [Gravity([0, -10.0])]
    mesh_domain = RigidBodyDomain(mesh_rigid, mesh_bcs, considerContact=True)

    # Ground (analytical plane at bottom)
    from flatworld import GroundDomain

    ground = GroundDomain(2, [0.5, 0.0], [0.0, 1.0])  # Horizontal plane at y=0

    domains = [fem_domain, ball_domain, box_domain, mesh_domain, ground]

    # Adaptive dt loop with fixed 60 FPS stepping
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = (
        create_gui_if_available("FEM-Rigid Contact Test (Ball, Box, Mesh)", res=(1080, 1080)) if not headless else None
    )
    frame_dt = 1.0 / 60.0
    t = 0.0

    # Colors for visualization
    color_fem = 0xFF3333  # Red for FEM
    color_ball = 0x33FF33  # Green for ball
    color_box = 0x3333FF  # Blue for box
    color_mesh = 0xFFFF33  # Yellow for mesh rigid
    color_ground = 0x888888  # Gray for ground

    print("=== 2D FEM-Rigid Contact Test ===")
    print("Objects:")
    print("  1. FEM body (red circle) - deformable")
    print("  2. Ball rigid (green) - primitive sphere")
    print("  3. Box rigid (blue) - primitive box with rotation")
    print("  4. Mesh rigid (yellow) - triangle mesh rigid body")
    print("  5. Ground (gray line) - analytical plane")
    print("\nAll rigid bodies will fall and collide with FEM body and ground")
    print("Press ESC to exit\n")

    frame = 0
    while t < 1.0:
        tt = time.time()
        looper.advanceWithTime(frame_dt)
        t += frame_dt
        frame += 1
        tt1 = time.time()
        print(f"Frame {frame} advance time: {tt1 - tt:.4f}s")

        # # Display info
        # if frame % 50 == 0:
        print(f"Time: {t:.4f}s, Frame: {frame}")
        if gui is not None:
            # # Draw FEM domain (deformable body)
            looper.femSpringManager.drawMesh(gui, color=color_fem)
            looper.rigidManager.drawAll(
                gui, domains, colors=[color_ball, color_box, color_mesh, color_ground], resolution=1080
            )
            gui.show()

    # Print profiling info
    # print("\n=== Performance Profiling ===")
    # ti.profiler.print_kernel_profiler_info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dfemrigid_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dfemrigid_contact(headless=args.headless)
