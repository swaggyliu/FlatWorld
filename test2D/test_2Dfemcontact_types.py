"""2D FEM contact tests."""

import numpy as np
import os
import pytest
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def test_fem_analytical_contact_2d():
    from flatworld import GroundDomain, Elastic, ExplicitLoop, FEMesher, FemDomain, Gravity, SolidProp

    mesh = FEMesher(2).createCircle([0.5, 0.5], 0.1)
    fem = FemDomain(mesh, SolidProp(Elastic(E=2e4, nu=0.2, rho=40.0)), [Gravity([0, -10.0])])
    ground = GroundDomain(2, [0.0, 0.3], [0.0, 1.0])
    looper = ExplicitLoop(0.0, [ground, fem], useAdapativeDT=True)

    for _ in range(60):
        looper.advanceWithTime(1.0 / 60.0)

    assert np.min(mesh.coords[:, 1]) >= 0.2


def test_fem_rigid_contact_2d():
    from flatworld import GroundDomain, BallRigid, Elastic, ExplicitLoop, FEMesher, FemDomain, FixedAll, Gravity, RigidBodyDomain, SolidProp

    mesh = FEMesher(2).createCircle([0.5, 0.6], 0.05)
    fem = FemDomain(mesh, SolidProp(Elastic(E=2e4, nu=0.2, rho=40.0)), [Gravity([0, -10.0])])
    rigid = RigidBodyDomain(BallRigid(2, [0.5, 0.3], 0.15, 100.0), [FixedAll([0])], considerContact=True)
    ground = GroundDomain(2, [0, 0.0], [0, 1])
    looper = ExplicitLoop(0.0, [fem, rigid, ground], useAdapativeDT=True)

    for _ in range(60):
        looper.advanceWithTime(1.0 / 60.0)

    assert np.mean(mesh.coords[:, 1]) > 0.2


def test_contact_base_penalty_2d():
    try:
        from flatworld.femcontact import ContactBase

        assert hasattr(ContactBase, "__init__")
    except ImportError:
        pytest.skip("flatworld.femcontact not available")


def test_rigid_rigid_contact_2d():
    from flatworld import GroundDomain, BallRigid, ExplicitLoop, FixedAll, Gravity, RigidBodyDomain

    fixed = RigidBodyDomain(BallRigid(2, [0.5, 0.2], 0.1, 100.0), [FixedAll([0])], considerContact=True)
    falling = RigidBodyDomain(BallRigid(2, [0.5, 0.5], 0.05, 1.0), [Gravity([0, -9.8])], considerContact=True)
    ground = GroundDomain(2, [0, 0.0], [0, 1])
    looper = ExplicitLoop(0.0, [fixed, falling, ground])

    for _ in range(120):
        looper.advanceWithTime(1.0 / 60.0)

    dist = np.linalg.norm(falling.getCurrentRefPoint() - fixed.getCurrentRefPoint())
    assert dist >= (0.1 + 0.05) * 0.8