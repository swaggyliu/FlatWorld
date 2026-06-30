import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import Elastic, EnforceAcc, EnforceVel, ExplicitLoop, Fixed, Force, SolidProp, SpringMassDomain
from test_utils import create_gui_if_available


def test_2Dspring(headless=False):
    ti.init(offline_cache=True, arch=ti.cpu)

    conn = np.array([[0, 1], [1, 2], [0, 2], [2, 3], [3, 1], [0, 3]], dtype=np.int32)
    coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)
    bcs0 = [Force([0, 1], [0, -1.0]), Fixed([2, 3])]
    domain0 = SpringMassDomain(2, coords, conn, [100.0, 1.0, 1.0], bcs0, False)

    bcs1 = [EnforceVel([0, 1], [0, -1.0]), Fixed([2, 3])]
    bcs2 = [EnforceAcc([0, 1], [0, -10.0]), Fixed([2, 3])]

    domain1 = SpringMassDomain(2, coords - 0.25, conn, [100.0, 1.0, 1.0], bcs1, False)
    domain2 = SpringMassDomain(2, coords - 0.5, conn, [100.0, 1.0, 1.0], bcs2, False)
    femdomains = [domain0, domain1, domain2]

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, femdomains, useAdapativeDT=True)
    colors = [0xFF0033, 0xFF00FF, 0xFFFF00]

    gui = create_gui_if_available("SPRING2D", res=(720, 720)) if not headless else None
    t = 0.0
    while (gui is None or gui.running) and t < 0.2:
        # advance one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)
        t += frame_dt

        # rendering
        totalEls = looper.femSpringManager.totalElements[None]
        totalNds = looper.femSpringManager.totalNodes[None]
        pos = looper.femSpringManager.coords.to_numpy()[:totalNds]
        e2n = looper.femSpringManager.connectivity.to_numpy()[:totalEls]
        if gui is not None:
            gui.circles(pos, radius=2, color=0xFFAA33)
            a, b = pos[e2n[:, 0]], pos[e2n[:, 1]]
            gui.lines(a, b, color=0xFFAA33)
            gui.show()

    min = np.min(pos, axis=0)
    max = np.max(pos, axis=0)
    print("Final node positions:", min, max)
    assert np.allclose(min, [0.0, -0.252], atol=1e-3), "Min position incorrect"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2Dspring")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2Dspring(headless=args.headless)
