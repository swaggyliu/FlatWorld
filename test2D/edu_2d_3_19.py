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
    BoxRigid,
    ExplicitLoop,
    Fixed,
    FixedAll,
    Gravity,
    HeightFieldDomain,
    RevoluteJoint,
    RigidBodyDomain,
    SpringMassDomain,
)


def edu_2d_3_19():

    ti.init(offline_cache=True, arch=ti.cpu)
    Fix1 = [0.9, 0.1 + 0.1 * math.sqrt(3)]
    box = [0.5, 0.1 + 0.1 * math.sqrt(3) + 0.4 / 3 * math.sqrt(3)]
    rigid1 = BallRigid(2, Fix1, 1.0, 1.0)
    rigid2 = BoxRigid(2, box, [0.4, 0.4], [-math.pi / 6], 1.0)

    bcs = [Gravity([0, -10.0])]
    rigiddomain1 = RigidBodyDomain(rigid1, bcs=[FixedAll([0])], considerContact=False)
    rigiddomain2 = RigidBodyDomain(rigid2, bcs)
    analytical1 = GroundDomain(2, (0.8, 0.1), (1, math.sqrt(3)), bcs=[FixedAll([0])])

    domains = [rigiddomain1, rigiddomain2, analytical1]
    # Create spring-mass system: 2 nodes with coordinates, 1 connection between them
    coords = np.array([Fix1, box], dtype=np.float32)  # 2 nodes in 2D
    conns = np.array([[0, 1]], dtype=np.int32)  # 1 connection between node 0 and 1
    domain0 = SpringMassDomain(2, coords, conns, prop=[100.0, 1.0, 1.0])

    domains += [domain0]

    # joint1 = RevoluteJoint(0, 1, Fix, [0, 0], bcs=[Fixed([0])], stiffness=1e2)

    frame_dt = 1.0 / 60.0
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

    gui = ti.GUI("EDU2D", res=(720, 720), background_color=0xFFFFFF)
    t = 0.0
    while gui.running and t < 10.0:
        # advance exactly one visual frame using adaptive substeps
        looper.advanceWithTime(frame_dt)

        t += frame_dt

        rigiddomain2.draw(gui, 0xD92B6B, 720)
        analytical1.draw(gui, 0x000000, leftlength=0.7, linewidth=2)

        pos2 = rigiddomain2.getCurrentRefPoint()
        gui.circle(pos2, 0x000000, 720 * 0.01)
        gui.line(Fix1, pos2, color=0x000000, radius=2)
        # gui.line([0.6,0.9], [0.8,0.9], color=0x000000, radius=2)

        gui.show()


if __name__ == "__main__":
    edu_2d_3_19()
