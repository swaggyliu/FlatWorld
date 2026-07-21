import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import (
    GroundDomain,
    Elastic,
    EnforceAcc,
    EnforceVel,
    ExplicitLoop,
    FemDomain,
    Fixed,
    Force,
    InitialVel,
    SolidProp,
)
from test_utils import create_gui_if_available, init_sim


def test_2DAnalytical_bcs(headless=False):
    init_sim()

    bc0 = EnforceVel([0], [0, 5.0])
    bc1 = EnforceAcc([0], [0, -50.0])
    ic3 = InitialVel([0], [-1.0, 0.0])
    anal0 = GroundDomain(2, [0.0, 0.3], [0, 1.0], bcs=[bc0])
    anal1 = GroundDomain(2, [0.0, 0.8], [0, -1.0], bcs=[bc1])
    anal3 = GroundDomain(2, [1.0, 0.0], [-1.0, 0.0], initials=[ic3])
    domains = [anal0, anal1, anal3]

    frame_dt = 1.0 / 50.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available("analytical2D", res=(720, 720)) if not headless else None
    t = 0.0
    while gui is None or gui.running:
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        pos0 = anal0.getCurrentRefPoint()
        pos1 = anal1.getCurrentRefPoint()
        pos3 = anal3.getCurrentRefPoint()

        if t >= 0.04:
            print("The points: {}, {}, {}".format(pos0, pos1, pos3))
            assert (
                np.allclose(pos0, [0.0, 0.5], atol=3e-3)
                and np.allclose(pos1, [0.0, 0.76], atol=2e-3)
                and np.allclose(pos3, [0.96, 0.0], atol=1e-3)
            )
            break

        if gui is not None:
            gui.line(pos0, pos0 + np.array([1.0, 0.0]), 3, 0xFF00FF)
            gui.line(pos1, pos1 + np.array([1.0, 0.0]), 3, 0xFF00FF)
            gui.line(pos3, pos3 + np.array([0.0, 1.0]), 3, 0xFF00FF)
            gui.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_2DAnalytical_bcs")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    args = parser.parse_args()
    test_2DAnalytical_bcs(headless=args.headless)
