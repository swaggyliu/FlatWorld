"""2D box collision response tests."""

import os
import sys

import numpy as np
import pytest
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import BoxRigid, ExplicitLoop, InitialVel, RigidBodyDomain


if not ti.lang.impl.get_runtime().prog:
    ti.init(arch=ti.cpu, debug=False)


def test_box_collision_2d():
    box1 = BoxRigid(d=2, origin=[-2, 0], ext=[1, 1], angle=[0], mass=1.0)
    box2 = BoxRigid(d=2, origin=[2, 0], ext=[1, 1], angle=[0], mass=1.0)

    domain1 = RigidBodyDomain(box1, initials=[InitialVel([0], [2.0, 0])])
    domain2 = RigidBodyDomain(box2, initials=[InitialVel([0], [-2.0, 0])])

    loop = ExplicitLoop(0.01, [domain1, domain2], joints=[], damping=0, useAdapativeDT=False)

    for i in range(100):
        loop.advance()
        pos1 = domain1.getCurrentRefPoint()
        pos2 = domain2.getCurrentRefPoint()
        dist = abs(pos2[0] - pos1[0])
        if i > 50 and dist > 1.1:
            break
    else:
        raise AssertionError("2D box collision response was not detected")


def test_rotated_box_collision_2d():
    box1 = BoxRigid(d=2, origin=[-2, 0], ext=[2, 0.5], angle=[np.pi / 4], mass=1.0)
    box2 = BoxRigid(d=2, origin=[2, 0], ext=[2, 0.5], angle=[-np.pi / 4], mass=1.0)

    domain1 = RigidBodyDomain(box1, initials=[InitialVel([0], [1.5, 0])])
    domain2 = RigidBodyDomain(box2, initials=[InitialVel([0], [-1.5, 0])])

    loop = ExplicitLoop(0.01, [domain1, domain2], joints=[], damping=0, useAdapativeDT=False)

    for i in range(150):
        loop.advance()
        pos1 = domain1.getCurrentRefPoint()
        pos2 = domain2.getCurrentRefPoint()
        dist = abs(pos2[0] - pos1[0])
        if i > 80 and dist > 1.5:
            break
    else:
        raise AssertionError("2D rotated box collision response may be incorrect")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])