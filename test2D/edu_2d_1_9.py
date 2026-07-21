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
    Elastic,
    ExplicitLoop,
    FemDomain,
    Fixed,
    Gravity,
    Mesh,
    RevoluteJoint,
    RigidBodyDomain,
    SolidProp,
)


def edu_2d_1_10():

    init_sim()
    Fix = [0.5, 0.9]
    rigid2 = BallRigid(2, [0.5, 0.575], 0.1, 1.0)
    bcs = [Gravity([0, -10.0])]
    rigid1 = BallRigid(2, [0.5, 1.1], 0.1, 1.0)
    rigiddomain1 = RigidBodyDomain(rigid1, bcs=[Fixed([0])], considerContact=False)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs)

    analytical1 = GroundDomain(2, (0.3, 0.3), (0, 1), bcs=[Fixed([0])])
    # analytical2 = GroundDomain(2, (0.38,0.36), (4,3), bcs=[Fixed([0])])
    analytical3 = GroundDomain(2, (0.3, 0.3), (-3, 4), bcs=[Fixed([0])])
    domains = [rigiddomain1, rigiddomain2, analytical1, analytical3]

    joint1 = RevoluteJoint(0, 1, Fix, [0, 0], bcs=[])

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True, joints=[joint1])

    gui = create_gui_if_available('EDU2D', res=(720, 720), background_color=0x112F41)
    if gui is None:
        print('No display; skipping GUI loop')
        return
    t = 0.0
    while gui.running and t < 10.0:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        # rigiddomain1.draw(gui, 0x347ea8, 720)
        rigiddomain2.draw(gui, 0xD92B6B, 720)
        analytical1.draw(gui, 0x000000, leftlength=0, rightlength=0.6, linewidth=2)
        # analytical2.draw(gui, 0x000000, leftlength=0.4, rightlength=0, linewidth=2)
        analytical3.draw(gui, 0x000000, leftlength=0, rightlength=0.7, linewidth=2)

        pos = rigiddomain2.getCurrentRefPoint()
        gui.circle(pos, 0x000000, 720 * 0.01)
        gui.line([0.4, 0.9], [0.6, 0.9], color=0x000000, radius=2)
        gui.line(Fix, pos, color=0x000000, radius=2)

        gui.show()


if __name__ == "__main__":
    edu_2d_1_10()
