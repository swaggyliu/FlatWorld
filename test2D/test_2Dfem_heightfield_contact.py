import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, ExplicitLoop, FemDomain, FEMesher, Gravity, HeightFieldDomain, Mesh, SolidProp
from test_utils import create_gui_if_available


def build_heightfield_bowl(nx=513, cx=0.5, r=0.5, zc=0.7):
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    h = np.empty(nx, dtype=np.float32)
    for i, x in enumerate(xs):
        dx = x - cx
        inside = r * r - dx * dx
        if inside >= 0.0:
            h[i] = zc - np.sqrt(inside)
        else:
            h[i] = zc  # rim
    return h


def test_2Dfem_heightfield_contact(headless=False):
    ti.init(offline_cache=True, arch=ti.cpu)

    # FEM disk placed above bowl heightfield
    reader = FEMesher(2)
    radius = 0.1
    mesh = reader.createCircle([0.5, 0.65], radius)

    mat = Elastic(E=2e4, nu=0.2, rho=40.0)
    prop = SolidProp(mat)
    bcs = [Gravity([0.0, -10.0])]
    fem = (
        FemDomain(mesh, prop, bcs, considerContact=True, initials=[])
        if hasattr(FemDomain, "__init__")
        else FemDomain(mesh, prop, bcs)
    )

    # Heightfield bowl
    h = build_heightfield_bowl()
    hf = HeightFieldDomain(2, h, lb=[0.0, 0.0], ub=[1.0, 1.0], considerContact=True)

    loop = ExplicitLoop(0.0, [hf, fem], useAdapativeDT=True)

    gui = create_gui_if_available("2D FEM vs HeightField", res=(720, 720)) if not headless else None
    frame_dt = 1.0 / 60.0
    steps = 0
    while (gui is None or gui.running) and steps < 60:
        loop.advanceWithTime(frame_dt)
        # gui.clear(0xFFFFFF)
        # Draw FEM
        # Draw heightfield curve
        if gui is not None:
            loop.femSpringManager.drawMesh(gui, color=0xFF0033)
            hf.draw(gui, color=0x444444)
            gui.show()
        steps += 1

    # Simple post-check: ensure no node is far below bowl rim (penetration resolved)
    coords = loop.femSpringManager.coords.to_numpy()[: mesh.numNodes]
    assert coords[:, 1].min() > 0.05, "FEM object fell through heightfield"


if __name__ == "__main__":
    test_2Dfem_heightfield_contact()
