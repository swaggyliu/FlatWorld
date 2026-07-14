import math
import numpy as np
import os
import sys
import taichi as ti

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


def edu_2d_1_8():

    ti.init(offline_cache=True, arch=ti.cpu)
    Fix = [0.5, 0.9]
    rigid1 = BallRigid(2, Fix, 1.0, 1.0)
    rigid2 = BallRigid(2, [0.15, 0.65], 0.1, 1.0)
    bcs = [Gravity([0, -10.0])]
    rigiddomain1 = RigidBodyDomain(rigid1, bcs=[Fixed([0])], considerContact=False)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs)
    nx = 513
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    r = 0.4
    cx = 0.5
    zc = 0.1
    h = np.empty(nx, dtype=np.float32)
    for i, x in enumerate(xs):
        dx = x - cx
        inside = r * r - dx * dx
        if inside >= 0.0:
            h[i] = zc + np.sqrt(inside)
        else:
            # Outside the bowl footprint: stay at rim height
            h[i] = zc

    # Create height field domain (2D: z = h(x)) over [0,1]
    hf = HeightFieldDomain(2, h, lb=[0.0, 0.0], ub=[1.0, 1.0], considerContact=True)
    domains = [rigiddomain1, rigiddomain2, hf]

    joint1 = RevoluteJoint(0, 1, Fix, [0, 0], bcs=[])
    # joint1 = RevoluteJoint(0, 1, Fix, [0, 0], bcs=[Fixed([0])], stiffness=1e2)

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, joints=[joint1], useAdapativeDT=True)

    gui = ti.GUI("EDU2D", res=(720, 720), background_color=0xFFFFFF)
    t = 0.0
    while gui.running and t < 10.0:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        rigiddomain2.draw(gui, 0x347EA8, 720)
        hf.draw(gui, 0x000000, linewidth=3)

        pos = rigiddomain2.getCurrentRefPoint()
        gui.circle(pos, 0x000000, 720 * 0.01)
        gui.line(Fix, pos, color=0x000000, radius=2)
        gui.line([0.4, 0.9], [0.6, 0.9], color=0x000000, radius=2)
        gui.show()


if __name__ == "__main__":
    edu_2d_1_8()
