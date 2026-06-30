"""2D cylinder and capsule bounding-box tests."""

import os
import sys

import numpy as np
import pytest
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import CapsuleRigid, ExplicitLoop, Gravity, RigidBodyDomain


def _expected_bbox_capsule(lc, uc, radius):
    lc = np.array(lc, dtype=float)
    uc = np.array(uc, dtype=float)
    center = 0.5 * (lc + uc)
    axis_vec_half = lc - center
    axis_len = np.linalg.norm(axis_vec_half)
    if axis_len < 1e-12:
        raise ValueError("Axis length too small for capsule test")
    direction = axis_vec_half / axis_len
    p0 = center - direction * axis_len
    p1 = center + direction * axis_len
    lb = []
    ub = []
    for i in range(len(lc)):
        perp = radius * np.sqrt(max(0.0, 1.0 - direction[i] * direction[i]))
        axial = radius * abs(direction[i])
        coord_min = min(p0[i], p1[i])
        coord_max = max(p0[i], p1[i])
        lb.append(coord_min - perp - axial)
        ub.append(coord_max + perp + axial)
    return np.array(lb), np.array(ub)


def test_capsule_bbox_2d():
    ti.init(arch=ti.cpu)

    rigid1 = CapsuleRigid(2, [0.5, 0.3], [0.5, 0.7], [0], 0.1, 1.0)
    domain1 = RigidBodyDomain(rigid1, [Gravity([0, -10])], considerContact=True)

    rigid2 = CapsuleRigid(2, [0.3, 0.5], [0.7, 0.5], [0], 0.1, 1.0)
    domain2 = RigidBodyDomain(rigid2, [Gravity([0, -10])], considerContact=True)

    ExplicitLoop(0.001, [domain1, domain2], [])

    lb1, ub1 = domain1.getBBox()
    assert abs(lb1[0] - 0.4) < 0.01
    assert abs(ub1[0] - 0.6) < 0.01
    assert abs(lb1[1] - 0.2) < 0.01
    assert abs(ub1[1] - 0.8) < 0.01

    lb2, ub2 = domain2.getBBox()
    assert abs(lb2[0] - 0.2) < 0.01
    assert abs(ub2[0] - 0.8) < 0.01
    assert abs(lb2[1] - 0.4) < 0.01
    assert abs(ub2[1] - 0.6) < 0.01


def test_capsule_bbox_inclined_2d():
    ti.init(arch=ti.cpu)

    lc = [0.4, 0.3]
    uc = [0.7, 0.55]
    radius = 0.05
    rigid = CapsuleRigid(2, lc, uc, [0], radius, 1.0)
    domain = RigidBodyDomain(rigid, [], considerContact=True)
    ExplicitLoop(0.001, [domain], [])

    lb, ub = domain.getBBox()
    exp_lb, exp_ub = _expected_bbox_capsule(lc, uc, radius)
    assert np.allclose(lb.to_numpy(), exp_lb, atol=1e-2)
    assert np.allclose(ub.to_numpy(), exp_ub, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])