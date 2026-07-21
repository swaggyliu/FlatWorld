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
    ExplicitLoop,
    Fixed,
    FixedAll,
    Gravity,
    HeightFieldDomain,
    RevoluteJoint,
    RigidBodyDomain,
)


def edu_2d_3_1():

    init_sim()
    Fix1 = [0.7, 0.9]
    Fix2 = [0.1, 0.5]
    rigid1 = BallRigid(2, Fix1, 1.0, 1.0)
    rigid2 = BallRigid(2, Fix2, 1.0, 1.0)
    rigid3 = BoxRigid(2, [0.5, 0.5], [0.2, 0.1], [0, 0], 1.0)

    bcs = [Gravity([0, -10.0])]
    rigiddomain1 = RigidBodyDomain(rigid1, bcs=[Fixed([0])], considerContact=False)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs=[Fixed([0])])
    rigiddomain3 = RigidBodyDomain(rigid3, bcs)

    domains = [rigiddomain1, rigiddomain2, rigiddomain3]

    joint1 = RevoluteJoint(0, 2, Fix1, [0, 0], bcs=[])
    joint2 = RevoluteJoint(1, 2, Fix2, [0, 0], bcs=[])

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

        rigiddomain3.draw(gui, 0xD92B6B, 720)

        pos3 = rigiddomain3.getCurrentRefPoint()
        gui.circle(pos3, 0x000000, 720 * 0.01)
        gui.line(Fix1, pos3, color=0x000000, radius=2)
        gui.line(Fix2, pos3, color=0x000000, radius=2)
        gui.line([0.6, 0.9], [0.8, 0.9], color=0x000000, radius=2)
        gui.line([0.1, 0.4], [0.1, 0.6], color=0x000000, radius=2)
        gui.show()


if __name__ == "__main__":
    edu_2d_3_1()
