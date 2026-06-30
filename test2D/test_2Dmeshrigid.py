import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import (
    GroundDomain,
    BallRigid,
    Elastic,
    ExplicitLoop,
    FemDomain,
    FEMesher,
    Fixed,
    Gravity,
    Mesh,
    MeshRigid,
    RigidBodyDomain,
    SolidProp,
)
from test_utils import create_gui_if_available


def test_2Drigid_contact(headless=False):

    ti.init(offline_cache=True, arch=ti.cpu, debug=False)
    conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)
    coords1 = np.array([[0.3, 0.3], [0.5, 0.3], [0.3, 0.45], [0.5, 0.45]], dtype=np.float32)
    mesh0 = Mesh(2, conn, coords)
    mesh1 = Mesh(2, conn, coords1)

    rigid0 = MeshRigid(2, mesh0, [0.0], 1.0)
    rigid1 = MeshRigid(2, mesh1, [0.0], 1.0)
    bcs0 = [Gravity([0, -1000.0])]
    bcs1 = [Fixed([0])]
    r1 = RigidBodyDomain(rigid0, bcs0)
    r2 = RigidBodyDomain(rigid1, bcs1)
    domains = [r1, r2]

    anl1 = GroundDomain(2, [0, 0.0], [0.0, 1.0])
    anl2 = GroundDomain(2, [0, 1.0], [0.0, -1.0])
    anl3 = GroundDomain(2, [0, 0.0], [1.0, 0.0])
    anl4 = GroundDomain(2, [1.0, 0.0], [-1.0, 0.0])
    domains += [anl1, anl2, anl3, anl4]

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("Rigid2D", res=(1080, 1080)) if not headless else None
    t = 0.0
    while (gui is None or gui.running) and t < 1.0:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        if gui is not None:
            color = 0xAAFFFF
            r1.draw(gui, color, 1080)
            r2.draw(gui, color, 1080)
            gui.show()

    pos1 = r1.getCurrentRefPoint()
    pos2 = r2.getCurrentRefPoint()
    distance = np.linalg.norm(pos1 - pos2)
    print("Final distance between rigid bodies:", distance)
    assert distance > 0.22, "Rigid bodies are too close, contact handling may have failed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Drigid_contact")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Drigid_contact(headless=args.headless)
