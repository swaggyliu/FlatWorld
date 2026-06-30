"""2D BVH CollisionDetector tests."""

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


_AABBS_2D = None


def _ensure_aabbs_2d(capacity):
    global _AABBS_2D
    _init_taichi_cpu()
    if _AABBS_2D is None:
        _AABBS_2D = ti.Vector.field(2, dtype=ti.f32, shape=(capacity, 2))
    return _AABBS_2D


def test_bvh_no_collision_2d():
    from flatworld.bvh import CollisionDetector

    _init_taichi_cpu()
    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    aabbs[0, 0] = [0.0, 0.0]
    aabbs[0, 1] = [0.1, 0.1]
    aabbs[1, 0] = [0.5, 0.5]
    aabbs[1, 1] = [0.6, 0.6]

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([0, 1], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) == 0


def test_bvh_overlapping_2d():
    from flatworld.bvh import CollisionDetector

    _init_taichi_cpu()
    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    aabbs[0, 0] = [0.0, 0.0]
    aabbs[0, 1] = [0.3, 0.3]
    aabbs[1, 0] = [0.2, 0.2]
    aabbs[1, 1] = [0.5, 0.5]

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([-1, -1], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) >= 1


def test_bvh_same_env_no_collision_2d():
    from flatworld.bvh import CollisionDetector

    _init_taichi_cpu()
    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    aabbs[0, 0] = [0.0, 0.0]
    aabbs[0, 1] = [0.5, 0.5]
    aabbs[1, 0] = [0.1, 0.1]
    aabbs[1, 1] = [0.6, 0.6]

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([0, 0], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) == 0


def test_bvh_many_objects_2d():
    from flatworld.bvh import CollisionDetector

    _init_taichi_cpu()
    n = 50
    cd = CollisionDetector(2, max_nodes=n * 4)
    cd.reset()

    aabbs = _ensure_aabbs_2d(256)
    np.random.seed(42)
    positions = np.random.rand(n, 2).astype(np.float32)
    for i in range(n):
        aabbs[i, 0] = positions[i].tolist()
        aabbs[i, 1] = (positions[i] + 0.05).tolist()

    cd.addObjects_batch(aabbs, np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert isinstance(cd.get_collision_pairs(), np.ndarray)


def test_bvh_add_one_object_2d():
    from flatworld.bvh import CollisionDetector

    _init_taichi_cpu()
    cd = CollisionDetector(2, max_nodes=16)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    aabbs[0, 0] = [0.0, 0.0]
    aabbs[0, 1] = [0.1, 0.1]

    cd.addOneObject(aabbs, 0, 0, 0)
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) == 0


def test_bvh_reset_clears_state_2d():
    from flatworld.bvh import CollisionDetector

    _init_taichi_cpu()
    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    aabbs[0, 0] = [0.0, 0.0]
    aabbs[0, 1] = [0.3, 0.3]
    aabbs[1, 0] = [0.2, 0.2]
    aabbs[1, 1] = [0.5, 0.5]

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([-1, -1], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) >= 1

    cd.reset()
    assert len(cd.get_collision_pairs()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])