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
    HeightFieldDomain,
    Mesh,
    RigidBodyDomain,
    SolidProp,
)


def edu_2d_1_3():

    init_sim()
    rigid1 = BallRigid(2, [0.4, 0.3], 0.1, 1.0)
    rigid2 = BallRigid(2, [0.6, 0.3], 0.1, 1.0)
    bcs = [Gravity([0, -10.0])]
    rigiddomain1 = RigidBodyDomain(rigid1, bcs, restitution=0.2)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs, restitution=0.2)
    nx = 513
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    r = 0.5
    cx = 0.5
    zc = 0.5
    h = np.empty(nx, dtype=np.float32)
    for i, x in enumerate(xs):
        dx = x - cx
        inside = r * r - dx * dx
        if inside >= 0.0:
            h[i] = zc - np.sqrt(inside)
        else:
            # Outside the bowl footprint: stay at rim height
            h[i] = zc

    # Create height field domain (2D: z = h(x)) over [0,1]
    hf = HeightFieldDomain(2, h, lb=[0.0, 0.0], ub=[1.0, 1.0], considerContact=True)
    domains = [rigiddomain1, rigiddomain2, hf]

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
        rigiddomain2.draw(gui, 0xD92B6B, 720)
        hf.draw(gui, 0x000000, linewidth=3)
        pos = rigiddomain1.getCurrentRefPoint()
        gui.circle(pos, 0x000000, 720 * 0.01)
        pos = rigiddomain2.getCurrentRefPoint()
        gui.circle(pos, 0x000000, 720 * 0.01)
        gui.show()


if __name__ == "__main__":
    edu_2d_1_3()
