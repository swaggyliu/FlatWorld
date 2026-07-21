"""2D SAT collision tests (Warp)."""

import os
import sys

import warp as wp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from test_utils import init_sim
from flatworld.sat import (
    obb2d_signed_distance_quad_vs_circle,
    obb2d_signed_distance_quad_vs_quad,
    obb2d_signed_distance_quad_vs_segment,
)

init_sim()


@wp.kernel
def _k_quad_vs_quad_overlap(out: wp.array(dtype=float)):
    sd, _n = obb2d_signed_distance_quad_vs_quad(
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 0.0),
        wp.vec2(1.0, 1.0),
        wp.vec2(0.0, 1.0),
        wp.vec2(0.5, 0.0),
        wp.vec2(1.5, 0.0),
        wp.vec2(1.5, 1.0),
        wp.vec2(0.5, 1.0),
    )
    out[0] = sd


@wp.kernel
def _k_quad_vs_quad_separated(out: wp.array(dtype=float)):
    sd, _n = obb2d_signed_distance_quad_vs_quad(
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 0.0),
        wp.vec2(1.0, 1.0),
        wp.vec2(0.0, 1.0),
        wp.vec2(3.0, 0.0),
        wp.vec2(4.0, 0.0),
        wp.vec2(4.0, 1.0),
        wp.vec2(3.0, 1.0),
    )
    out[0] = sd


@wp.kernel
def _k_quad_vs_circle_inside(out: wp.array(dtype=float)):
    sd, _n = obb2d_signed_distance_quad_vs_circle(
        wp.vec2(0.0, 0.0),
        wp.vec2(2.0, 0.0),
        wp.vec2(2.0, 2.0),
        wp.vec2(0.0, 2.0),
        wp.vec2(1.0, 1.0),
        0.3,
    )
    out[0] = sd


@wp.kernel
def _k_quad_vs_circle_outside(out: wp.array(dtype=float)):
    sd, _n = obb2d_signed_distance_quad_vs_circle(
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 0.0),
        wp.vec2(1.0, 1.0),
        wp.vec2(0.0, 1.0),
        wp.vec2(5.0, 5.0),
        0.5,
    )
    out[0] = sd


@wp.kernel
def _k_quad_vs_segment_intersect(out: wp.array(dtype=float)):
    sd, _n = obb2d_signed_distance_quad_vs_segment(
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 0.0),
        wp.vec2(1.0, 1.0),
        wp.vec2(0.0, 1.0),
        wp.vec2(0.5, -0.5),
        wp.vec2(0.5, 0.5),
    )
    out[0] = sd


def _run(kernel) -> float:
    out = wp.zeros(1, dtype=float)
    wp.launch(kernel, dim=1, inputs=[out])
    return float(out.numpy()[0])


def test_sat2d_quad_overlap():
    assert _run(_k_quad_vs_quad_overlap) < 0


def test_sat2d_quad_separated():
    assert _run(_k_quad_vs_quad_separated) > 0


def test_sat2d_quad_circle_inside():
    assert _run(_k_quad_vs_circle_inside) < 0


def test_sat2d_quad_circle_outside():
    assert _run(_k_quad_vs_circle_outside) > 0


def test_sat2d_quad_segment_intersect():
    assert _run(_k_quad_vs_segment_intersect) < 0
