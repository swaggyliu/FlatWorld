"""2D segment-segment closest-point tests (Warp)."""

import math
import os
import sys

import numpy as np
import warp as wp
from test_utils import init_sim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld.utils import calMinDisSegment2Segment

init_sim()


@wp.kernel
def _k_parallel(out: wp.array(dtype=wp.vec4)):
    p, q, _, _ = calMinDisSegment2Segment(
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 0.0),
        wp.vec2(0.0, 1.0),
        wp.vec2(1.0, 1.0),
    )
    out[0] = wp.vec4(p[0], p[1], q[0], q[1])


@wp.kernel
def _k_intersecting(out: wp.array(dtype=wp.vec4)):
    p, q, _, _ = calMinDisSegment2Segment(
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 1.0),
        wp.vec2(0.0, 1.0),
        wp.vec2(1.0, 0.0),
    )
    out[0] = wp.vec4(p[0], p[1], q[0], q[1])


def _run_points(kernel):
    out = wp.zeros(1, dtype=wp.vec4)
    wp.launch(kernel, dim=1, inputs=[out])
    return out.numpy()[0]


def _distance(points):
    arr = np.array(points, dtype=np.float32)
    return math.hypot(arr[0] - arr[2], arr[1] - arr[3])


def test_segment_segment_parallel_2d():
    assert abs(_distance(_run_points(_k_parallel)) - 1.0) < 1e-5


def test_segment_segment_intersecting_2d():
    assert _distance(_run_points(_k_intersecting)) < 1e-5
