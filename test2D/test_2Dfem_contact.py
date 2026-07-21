import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, ExplicitLoop, FemDomain, FEMesher, Gravity, Mesh, SolidProp
from multiprocessing import Queue
from test_utils import create_gui_if_available, init_sim


def get_domains():
    msh = FEMesher(2)
    offset = 0.1
    mesh1 = msh.createCircle([0.5, 0.5], offset)
    bcs1 = [Gravity([0, 10.0])]

    mesh2 = msh.createCircle([0.5, 0.8], offset)
    bcs2 = [Gravity([0, -10.0])]
    mat = Elastic(E=2e4, nu=0.2, rho=40.0)
    prop = SolidProp(mat)

    domains = [FemDomain(mesh1, prop, bcs1), FemDomain(mesh2, prop, bcs2)]
    return domains


def test_2Dfem_contact(headless=False):
    init_sim()

    domains = get_domains()

    # Adaptive dt loop with fixed 60 FPS frame stepping
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("FEM2D", res=(720, 720)) if not headless else None
    t = 0.0
    frame_dt = 1.0 / 60.0
    while gui is None or gui.running:
        looper.advanceWithTime(frame_dt)
        t += frame_dt

        if t > 1.5:
            break

        if gui is not None:
            looper.femSpringManager.drawMesh(gui, 0xFF0033)
            gui.show()


def generate_disp_format_data():
    init_sim()
    domains = get_domains()

    t = 0.0
    looper = ExplicitLoop(0.0, domains, damping=0.0001, useAdapativeDT=True)
    step = 1
    data = list()
    while t <= 0.1:
        looper.advanceWithTime(1.0 / 60.0)
        t += 1.0 / 60.0
        meshes = [domain.mesh for domain in domains]
        data.append({"step": step, "time": t, "position": export_2d_meshes_nodes(meshes)})
    dataFormat = {"result": data}
    return dataFormat


def get_mesh_format_data():
    init_sim()
    return export_2d_meshes_as_json([domain.mesh for domain in get_domains()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dfem_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dfem_contact(headless=args.headless)
