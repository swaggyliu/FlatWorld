"""Multi-env contact isolation via RigidBodyDomain.env_id."""

import os
import sys

import numpy as np
import pytest
from test_utils import init_sim

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import BoxRigid, ExplicitLoop, InitialVel, RigidBodyDomain

init_sim()


def _run_pair(env_a: int, env_b: int, steps: int = 120):
    box1 = BoxRigid(d=2, origin=[-1.5, 0.0], ext=[0.5, 0.5], angle=[0], mass=1.0)
    box2 = BoxRigid(d=2, origin=[1.5, 0.0], ext=[0.5, 0.5], angle=[0], mass=1.0)
    d1 = RigidBodyDomain(
        box1, initials=[InitialVel([0], [3.0, 0.0])], env_id=env_a)
    d2 = RigidBodyDomain(
        box2, initials=[InitialVel([0], [-3.0, 0.0])], env_id=env_b)
    loop = ExplicitLoop(0.01, [d1, d2], joints=[], damping=0, useAdapativeDT=False)
    xs = []
    for _ in range(steps):
        loop.advance()
        xs.append((float(d1.getCurrentRefPoint()[0]),
                   float(d2.getCurrentRefPoint()[0])))
    return xs


def test_same_env_id_collides():
    """Shared env_id → boxes bounce; centers stay separated by ~extent sum."""
    xs = _run_pair(0, 0)
    # After meeting, they should not cross deeply (gap stays near 1.0).
    late = xs[80:]
    gaps = [abs(b - a) for a, b in late]
    assert min(gaps) > 0.85, f"expected collision bounce, min gap={min(gaps):.3f}"


def test_different_env_id_no_contact():
    """Distinct env_ids → boxes pass through (ghost) despite overlapping AABBs."""
    xs = _run_pair(0, 1)
    # Without contact they keep moving toward/through each other.
    a0, b0 = xs[0]
    a1, b1 = xs[-1]
    # Left box should have moved right past the origin; right box left past it.
    assert a1 > 0.5, f"left box stuck/blocked a={a1:.3f}"
    assert b1 < -0.5, f"right box stuck/blocked b={b1:.3f}"
    # And they should have crossed (a > b late).
    assert a1 > b1, "boxes did not pass through each other"


def test_global_env_id_still_collides():
    """env_id < 0 is global and collides with tagged bodies."""
    xs = _run_pair(-1, 0)
    late = xs[80:]
    gaps = [abs(b - a) for a, b in late]
    assert min(gaps) > 0.85, f"global vs tagged should collide, gap={min(gaps):.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
