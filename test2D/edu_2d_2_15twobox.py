import math
import numpy as np
import os
import sys
from test_utils import init_sim, create_gui_if_available

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import (
    GroundDomain,
    BallRigid,
    BoxRigid,
    Elastic,
    ExplicitLoop,
    FemDomain,
    Fixed,
    FixedAll,
    Force,
    Gravity,
    Mesh,
    RigidBodyDomain,
    SolidProp,
)


def edu_2d_2_15():

    init_sim()
    rigid1 = BoxRigid(2, [0.5, 0.5], [0.3, 0.2], [0, 0], 1.0)
    rigid2 = BoxRigid(2, [0.5, 0.3], [1, 0.2], [0, 0], 1.0)
    bcs = [Gravity([0, -10.0])]
    rigiddomain1 = RigidBodyDomain(rigid1, bcs=[Gravity([0, -10.0]), Force([0], [2.0, 0.0])], friction=0.5)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs=[FixedAll([0])], friction=0.5)

    analytical1 = GroundDomain(2, (0.3, 0.4), (0, 1), bcs=[FixedAll([0])])
    domains = [rigiddomain1, rigiddomain2]

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = create_gui_if_available('EDU2D', res=(720, 720), background_color=0x112F41)
    if gui is None:
        print('No display; skipping GUI loop')
        return
    t = 0.0
    while gui.running and t < 10.0:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        rigiddomain1.draw(gui, 0x347EA8, 720)
        rigiddomain2.draw(gui, 0x000000, 720)

        gui.show()


if __name__ == "__main__":
    edu_2d_2_15()
