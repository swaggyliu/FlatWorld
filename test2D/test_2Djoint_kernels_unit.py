"""2D joint-kernel integration tests."""

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import BallRigid, ExplicitLoop, FixedAll, Gravity, RigidBodyDomain
from flatworld import PrismaticJoint, RevoluteJoint, WeldJoint
from test_utils import init_sim

init_sim()


def test_revolute_joint_link_preservation_2d():
    anchor_pos = [0.5, 0.9]
    lower_pos = [0.5, 0.7]
    radius = 0.02

    d1 = RigidBodyDomain(BallRigid(2, anchor_pos, radius, 1.0), [Gravity([0, -9.8]), FixedAll([0])], considerContact=True)
    d2 = RigidBodyDomain(BallRigid(2, lower_pos, radius, 1.0), [Gravity([0, -9.8])], considerContact=True)

    joint = RevoluteJoint(0, 1, anchor=anchor_pos, axis=[0, 0])
    looper = ExplicitLoop(0.0, [d1, d2], joints=[joint], useAdapativeDT=True)

    init_len = np.linalg.norm(np.array(anchor_pos) - np.array(lower_pos))
    for _ in range(30):
        looper.advanceWithTime(1.0 / 60.0)

    final_len = np.linalg.norm(d1.getCurrentRefPoint() - d2.getCurrentRefPoint())
    assert abs(final_len - init_len) < 0.01


def test_weld_joint_2d():
    pos_a = [0.5, 0.9]
    pos_b = [0.5, 0.7]

    d1 = RigidBodyDomain(BallRigid(2, pos_a, 0.02, 1.0), [Gravity([0, -9.8]), FixedAll([0])], considerContact=True)
    d2 = RigidBodyDomain(BallRigid(2, pos_b, 0.02, 1.0), [Gravity([0, -9.8])], considerContact=True)

    joint = WeldJoint(0, 1, anchor=pos_a)
    looper = ExplicitLoop(0.0, [d1, d2], joints=[joint], useAdapativeDT=True)
    for _ in range(30):
        looper.advanceWithTime(1.0 / 60.0)

    diff = np.linalg.norm((d2.getCurrentRefPoint() - d1.getCurrentRefPoint()) - (np.array(pos_b) - np.array(pos_a)))
    assert diff < 0.05


def test_prismatic_joint_2d():
    pos_a = [0.5, 0.9]
    pos_b = [0.5, 0.7]

    d1 = RigidBodyDomain(BallRigid(2, pos_a, 0.02, 1.0), [FixedAll([0])], considerContact=True)
    d2 = RigidBodyDomain(BallRigid(2, pos_b, 0.02, 1.0), [Gravity([0, -9.8])], considerContact=True)

    joint = PrismaticJoint(0, 1, anchor=pos_a, axis=[0, 1])
    looper = ExplicitLoop(0.0, [d1, d2], joints=[joint], useAdapativeDT=True)
    for _ in range(30):
        looper.advanceWithTime(1.0 / 60.0)

    p0 = d1.getCurrentRefPoint()
    p1 = d2.getCurrentRefPoint()
    assert abs(p1[0] - p0[0]) < 0.02
    assert p1[1] < pos_b[1] - 0.01


def test_double_pendulum_2d():
    p0 = [0.5, 0.9]
    p1 = [0.6, 0.7]
    p2 = [0.7, 0.5]
    radius = 0.02

    d1 = RigidBodyDomain(BallRigid(2, p0, radius, 1.0), [Gravity([0, -9.8]), FixedAll([0])], considerContact=True)
    d2 = RigidBodyDomain(BallRigid(2, p1, radius, 1.0), [Gravity([0, -9.8])], considerContact=True)
    d3 = RigidBodyDomain(BallRigid(2, p2, radius, 1.0), [Gravity([0, -9.8])], considerContact=True)

    looper = ExplicitLoop(
        0.0,
        [d1, d2, d3],
        joints=[RevoluteJoint(0, 1, anchor=p0, axis=[0, 0]), RevoluteJoint(1, 2, anchor=p1, axis=[0, 0])],
        useAdapativeDT=True,
    )
    for _ in range(60):
        looper.advanceWithTime(1.0 / 60.0)

    pos0 = d1.getCurrentRefPoint()
    pos1 = d2.getCurrentRefPoint()
    pos2 = d3.getCurrentRefPoint()
    assert abs(np.linalg.norm(pos0 - pos1) - np.linalg.norm(np.array(p0) - np.array(p1))) < 0.02
    assert abs(np.linalg.norm(pos1 - pos2) - np.linalg.norm(np.array(p1) - np.array(p2))) < 0.02
    assert np.allclose(pos0, p0, atol=0.02)


if __name__ == "__main__":
    test_revolute_joint_link_preservation_2d()
    test_weld_joint_2d()
    test_prismatic_joint_2d()
    test_double_pendulum_2d()