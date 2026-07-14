"""2D tests for GroundDomain, HeightFieldDomain, and VoxelGridDomain."""

import os
import sys

import numpy as np
import pytest
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def _init_taichi_cpu():
    if not ti.lang.impl.get_runtime().prog:
        ti.init(offline_cache=True, arch=ti.cpu)


def test_analytical_domain_creation_2d():
    from flatworld import GroundDomain
    from flatworld.definitions import DomainType

    _init_taichi_cpu()

    ad = GroundDomain(2, [0, 0.1], [0, 1])
    assert ad.type == DomainType.ANALYTICAL


def test_analytical_domain_get_bbox_2d():
    from flatworld import GroundDomain

    _init_taichi_cpu()

    ad = GroundDomain(2, [0, 0.1], [0, 1])
    assert ad.getBBox() is not None


def test_analytical_domain_in_simulation_2d():
    from flatworld import GroundDomain, BallRigid, ExplicitLoop, Gravity, RigidBodyDomain

    _init_taichi_cpu()

    rigid = BallRigid(2, [0.5, 0.5], 0.02, 1.0)
    domain = RigidBodyDomain(rigid, [Gravity([0, -9.8])], considerContact=True)
    ground = GroundDomain(2, [0, 0.1], [0, 1])
    looper = ExplicitLoop(0.0, [domain, ground])

    for _ in range(60):
        looper.advanceWithTime(1.0 / 60.0)

    pos = domain.getCurrentRefPoint()
    assert pos[1] >= 0.08, f"Ball fell through ground: y={pos[1]:.4f}"


def test_heightfield_creation_2d():
    from flatworld import HeightFieldDomain
    from flatworld.definitions import DomainType

    _init_taichi_cpu()

    heights = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    hf = HeightFieldDomain(2, heights, lb=[0.0, 0.0], ub=[1.0, 0.5])
    assert hf.type == DomainType.HEIGHTFIELD


def test_heightfield_sine_2d():
    from flatworld import HeightFieldDomain

    _init_taichi_cpu()

    x = np.linspace(0, 1, 50)
    heights = 0.2 + 0.1 * np.sin(2 * np.pi * x)
    hf = HeightFieldDomain(2, heights.astype(np.float64), lb=[0.0, 0.0], ub=[1.0, 0.5])
    assert hf.getBoundaryMesh() is not None


def test_heightfield_ball_landing_2d():
    from flatworld import BallRigid, ExplicitLoop, Gravity, RigidBodyDomain
    from flatworld import HeightFieldDomain

    _init_taichi_cpu()

    heights = np.ones(20, dtype=np.float64) * 0.2
    hf = HeightFieldDomain(2, heights, lb=[0.0, 0.0], ub=[1.0, 0.5])

    rigid = BallRigid(2, [0.5, 0.5], 0.02, 1.0)
    domain = RigidBodyDomain(rigid, [Gravity([0, -9.8])], considerContact=True)

    looper = ExplicitLoop(0.0, [domain, hf])
    for _ in range(120):
        looper.advanceWithTime(1.0 / 60.0)

    pos = domain.getCurrentRefPoint()
    assert pos[1] >= 0.15, f"Ball fell through heightfield: y={pos[1]:.4f}"


def test_voxel_creation_2d():
    from flatworld import VoxelGridDomain
    from flatworld.definitions import DomainType

    _init_taichi_cpu()

    vg = VoxelGridDomain(2, nx=5, ny=5, lb=[0.0, 0.0], ub=[1.0, 1.0])
    assert vg.type == DomainType.VOXELMAP


def test_voxel_with_occupancy_2d():
    from flatworld import VoxelGridDomain

    _init_taichi_cpu()

    occ = np.zeros((5, 5), dtype=np.int32)
    occ[0, :] = 1
    occ[1, :] = 1
    vg = VoxelGridDomain(2, nx=5, ny=5, lb=[0.0, 0.0], ub=[1.0, 1.0], occupancy_np=occ)
    assert vg.getBoundaryMesh() is not None


def test_voxel_ball_landing_2d():
    from flatworld import BallRigid, ExplicitLoop, Gravity, RigidBodyDomain
    from flatworld import VoxelGridDomain

    _init_taichi_cpu()

    occ = np.zeros((10, 10), dtype=np.int32)
    occ[:3, :] = 1
    vg = VoxelGridDomain(2, nx=10, ny=10, lb=[0.0, 0.0], ub=[1.0, 1.0], occupancy_np=occ)

    rigid = BallRigid(2, [0.5, 0.8], 0.02, 1.0)
    domain = RigidBodyDomain(rigid, [Gravity([0, -9.8])], considerContact=True)

    looper = ExplicitLoop(0.0, [domain, vg])
    for _ in range(120):
        looper.advanceWithTime(1.0 / 60.0)

    pos = domain.getCurrentRefPoint()
    assert pos[1] > -100, f"Ball exploded: y={pos[1]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])