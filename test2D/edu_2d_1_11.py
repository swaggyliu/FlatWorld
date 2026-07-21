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
    ExplicitLoop,
    Fixed,
    Gravity,
    HeightFieldDomain,
    RevoluteJoint,
    RigidBodyDomain,
)


def edu_2d_1_11():

    init_sim()
    Fix = [0.5, 0.9]
    rigid1 = BallRigid(2, Fix, 1.0, 1.0)
    rigid2 = BallRigid(2, [0.25, 0.45], 0.1, 1.0)
    rigid3 = BallRigid(2, [0.75, 0.45], 0.1, 1.0)
    bcs = [Gravity([0, -10.0])]
    rigiddomain1 = RigidBodyDomain(rigid1, bcs=[Fixed([0])], considerContact=False)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs)
    rigiddomain3 = RigidBodyDomain(rigid3, bcs)

    domains = [rigiddomain1, rigiddomain2, rigiddomain3]

    joint1 = RevoluteJoint(0, 1, Fix, [0, 0], bcs=[])
    joint2 = RevoluteJoint(0, 2, Fix, [0, 0], bcs=[])
    # joint1 = RevoluteJoint(0, 1, Fix, [0, 0], bcs=[Fixed([0])], stiffness=1e2)

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, joints=[joint1, joint2], useAdapativeDT=True)

    gui = create_gui_if_available('EDU2D', res=(720, 720), background_color=0x112F41)
    if gui is None:
        print('No display; skipping GUI loop')
        return
    t = 0.0
    while gui.running and t < 10.0:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        rigiddomain2.draw(gui, 0x347EA8, 720)
        rigiddomain3.draw(gui, 0xD92B6B, 720)

        pos2 = rigiddomain2.getCurrentRefPoint()
        gui.circle(pos2, 0x000000, 720 * 0.01)
        pos3 = rigiddomain3.getCurrentRefPoint()
        gui.circle(pos3, 0x000000, 720 * 0.01)
        gui.line(Fix, pos2, color=0x000000, radius=2)
        gui.line(Fix, pos3, color=0x000000, radius=2)
        gui.line([0.4, 0.9], [0.6, 0.9], color=0x000000, radius=2)
        gui.show()


if __name__ == "__main__":
    edu_2d_1_11()
