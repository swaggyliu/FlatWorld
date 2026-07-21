import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, ExplicitLoop, FemDomain, FEMesher, Gravity, Mesh, SolidProp, VoxelGridDomain
from test_utils import create_gui_if_available, init_sim


def build_voxel_pit_2d(nx=160, ny=100, lb=(0.0, 0.0), ub=(1.0, 1.0)):
    """Build a concave voxel pit: filled solid up to a varying height forming a bowl-like depression."""
    occ = np.zeros((nx, ny), dtype=np.int32)
    cx = nx * 0.5
    r = nx * 0.45
    max_h = int(ny * 0.55)
    for i in range(nx):
        dx = i - cx
        inside = r * r - dx * dx
        if inside > 0:
            depth = inside / (r * r)  # 0..1
            h = int(max_h * (1.3 - depth))  # lower center
        else:
            h = max_h
        if h > 0:
            occ[i, :h] = 1
    vox = VoxelGridDomain(d=2, nx=nx, ny=ny, lb=list(lb), ub=list(ub), considerContact=True, occupancy_np=occ)
    return vox


def test_2Dfem_voxel_contact(headless=False):
    init_sim()

    reader = FEMesher(2)
    radius = 0.07
    mesh = reader.createCircle([0.5, 0.75], radius)

    mat = Elastic(E=2e4, nu=0.25, rho=50.0)
    prop = SolidProp(mat)
    bcs = [Gravity([0.0, -9.8])]
    fem = (
        FemDomain(mesh, prop, bcs, considerContact=True, initials=[])
        if hasattr(FemDomain, "__init__")
        else FemDomain(mesh, prop, bcs)
    )

    vox = build_voxel_pit_2d(12, 8)

    loop = ExplicitLoop(0.0, [vox, fem], useAdapativeDT=True)

    gui = create_gui_if_available("2D FEM vs Voxel Pit", res=(720, 720)) if not headless else None
    frame_dt = 1.0 / 60.0
    steps = 0
    while (gui is None or gui.running) and steps < 60:
        loop.advanceWithTime(frame_dt)
        if gui is not None:
            gui.clear(0x112F41)
            # Draw FEM mesh
            loop.femSpringManager.drawMesh(gui, color=0xFF0033)
            # Draw voxel edges
            vox.draw(gui, color=0x333333)
            gui.show()
        steps += 1

    coords = loop.femSpringManager.coords.numpy()[: mesh.numNodes]
    # Check not fallen far below lower bound y (0.0)
    assert coords[:, 1].min() > 0.02, "FEM object fell through voxel pit"


if __name__ == "__main__":
    test_2Dfem_voxel_contact()
