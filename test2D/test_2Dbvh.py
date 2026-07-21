"""2D BVH CollisionDetector tests (Warp)."""

import os
import sys

import numpy as np
import pytest
import warp as wp
from test_utils import init_sim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

init_sim()

_AABBS_2D = None


def _ensure_aabbs_2d(capacity):
    """Return a (capacity, 2) wp.array of vec2 AABBs [lb, ub]."""
    global _AABBS_2D
    if _AABBS_2D is None or _AABBS_2D.shape[0] < capacity:
        _AABBS_2D = wp.zeros((capacity, 2), dtype=wp.vec2)
    return _AABBS_2D


def _set_aabb(aabbs, idx, lb, ub):
    np_a = aabbs.numpy()
    np_a[idx, 0] = lb
    np_a[idx, 1] = ub
    aabbs.assign(np_a)


def test_bvh_no_collision_2d():
    from flatworld.bvh import CollisionDetector

    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    _set_aabb(aabbs, 0, [0.0, 0.0], [0.1, 0.1])
    _set_aabb(aabbs, 1, [0.5, 0.5], [0.6, 0.6])

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([0, 1], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) == 0


def test_bvh_overlapping_2d():
    from flatworld.bvh import CollisionDetector

    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    _set_aabb(aabbs, 0, [0.0, 0.0], [0.3, 0.3])
    _set_aabb(aabbs, 1, [0.2, 0.2], [0.5, 0.5])

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([-1, -1], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) >= 1


def test_bvh_same_env_no_collision_2d():
    from flatworld.bvh import CollisionDetector

    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    _set_aabb(aabbs, 0, [0.0, 0.0], [0.5, 0.5])
    _set_aabb(aabbs, 1, [0.1, 0.1], [0.6, 0.6])

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([0, 0], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) == 0


def test_bvh_many_objects_2d():
    from flatworld.bvh import CollisionDetector

    n = 50
    cd = CollisionDetector(2, max_nodes=n * 4)
    cd.reset()

    aabbs = _ensure_aabbs_2d(256)
    np.random.seed(42)
    positions = np.random.rand(n, 2).astype(np.float32)
    np_a = aabbs.numpy()
    for i in range(n):
        np_a[i, 0] = positions[i]
        np_a[i, 1] = positions[i] + 0.05
    aabbs.assign(np_a)

    cd.addObjects_batch(aabbs, np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert isinstance(cd.get_collision_pairs(), np.ndarray)


def test_bvh_add_one_object_2d():
    from flatworld.bvh import CollisionDetector

    cd = CollisionDetector(2, max_nodes=16)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    _set_aabb(aabbs, 0, [0.0, 0.0], [0.1, 0.1])

    cd.addOneObject(aabbs, 0, 0, 0)
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) == 0


def test_bvh_reset_clears_state_2d():
    from flatworld.bvh import CollisionDetector

    cd = CollisionDetector(2, max_nodes=64)
    cd.reset()

    aabbs = _ensure_aabbs_2d(64)
    _set_aabb(aabbs, 0, [0.0, 0.0], [0.3, 0.3])
    _set_aabb(aabbs, 1, [0.2, 0.2], [0.5, 0.5])

    cd.addObjects_batch(aabbs, np.array([0, 1], dtype=np.int32), np.array([-1, -1], dtype=np.int32))
    cd.build()
    cd.detectInnerCollision()
    assert len(cd.get_collision_pairs()) >= 1

    cd.reset()
    assert len(cd.get_collision_pairs()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
