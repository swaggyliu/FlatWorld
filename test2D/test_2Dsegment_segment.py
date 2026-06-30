"""2D segment-segment closest-point tests."""

import math
import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld.utils import calMinDisSegment2Segment

if not ti.lang.impl.get_runtime().prog:
    ti.init(offline_cache=True, arch=ti.cpu)


def _distance(points):
    arr = np.array(points, dtype=np.float32)
    return math.hypot(arr[0] - arr[2], arr[1] - arr[3])


def test_segment_segment_parallel_2d():
    @ti.kernel
    def case_parallel() -> ti.types.vector(4, ti.f32):
        p, q, _, _ = calMinDisSegment2Segment(ti.Vector([0.0, 0.0]), ti.Vector([1.0, 0.0]), ti.Vector([0.0, 1.0]), ti.Vector([1.0, 1.0]))
        return ti.Vector([p[0], p[1], q[0], q[1]])

    assert abs(_distance(case_parallel()) - 1.0) < 1e-5


def test_segment_segment_intersecting_2d():
    @ti.kernel
    def case_intersecting() -> ti.types.vector(4, ti.f32):
        p, q, _, _ = calMinDisSegment2Segment(ti.Vector([0.0, 0.0]), ti.Vector([1.0, 1.0]), ti.Vector([0.0, 1.0]), ti.Vector([1.0, 0.0]))
        return ti.Vector([p[0], p[1], q[0], q[1]])

    assert _distance(case_intersecting()) < 1e-5