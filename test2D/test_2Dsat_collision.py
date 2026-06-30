"""2D SAT collision tests."""

import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld.sat import obb2d_signed_distance_quad_vs_circle, obb2d_signed_distance_quad_vs_quad, obb2d_signed_distance_quad_vs_segment


if not ti.lang.impl.get_runtime().prog:
    ti.init(offline_cache=True, arch=ti.cpu)


@ti.kernel
def _test_2d_quad_vs_quad_overlap() -> ti.f32:
    sd, _ = obb2d_signed_distance_quad_vs_quad(
        ti.Vector([0.0, 0.0]), ti.Vector([1.0, 0.0]), ti.Vector([1.0, 1.0]), ti.Vector([0.0, 1.0]),
        ti.Vector([0.5, 0.0]), ti.Vector([1.5, 0.0]), ti.Vector([1.5, 1.0]), ti.Vector([0.5, 1.0]),
    )
    return sd


@ti.kernel
def _test_2d_quad_vs_quad_separated() -> ti.f32:
    sd, _ = obb2d_signed_distance_quad_vs_quad(
        ti.Vector([0.0, 0.0]), ti.Vector([1.0, 0.0]), ti.Vector([1.0, 1.0]), ti.Vector([0.0, 1.0]),
        ti.Vector([3.0, 0.0]), ti.Vector([4.0, 0.0]), ti.Vector([4.0, 1.0]), ti.Vector([3.0, 1.0]),
    )
    return sd


@ti.kernel
def _test_2d_quad_vs_circle_inside() -> ti.f32:
    sd, _ = obb2d_signed_distance_quad_vs_circle(
        ti.Vector([0.0, 0.0]), ti.Vector([2.0, 0.0]), ti.Vector([2.0, 2.0]), ti.Vector([0.0, 2.0]),
        ti.Vector([1.0, 1.0]), 0.3,
    )
    return sd


@ti.kernel
def _test_2d_quad_vs_circle_outside() -> ti.f32:
    sd, _ = obb2d_signed_distance_quad_vs_circle(
        ti.Vector([0.0, 0.0]), ti.Vector([1.0, 0.0]), ti.Vector([1.0, 1.0]), ti.Vector([0.0, 1.0]),
        ti.Vector([5.0, 5.0]), 0.5,
    )
    return sd


@ti.kernel
def _test_2d_quad_vs_segment_intersect() -> ti.f32:
    sd, _ = obb2d_signed_distance_quad_vs_segment(
        ti.Vector([0.0, 0.0]), ti.Vector([1.0, 0.0]), ti.Vector([1.0, 1.0]), ti.Vector([0.0, 1.0]),
        ti.Vector([0.5, -0.5]), ti.Vector([0.5, 0.5]),
    )
    return sd


def test_sat2d_quad_overlap():
    assert _test_2d_quad_vs_quad_overlap() < 0


def test_sat2d_quad_separated():
    assert _test_2d_quad_vs_quad_separated() > 0


def test_sat2d_quad_circle_inside():
    assert _test_2d_quad_vs_circle_inside() < 0


def test_sat2d_quad_circle_outside():
    assert _test_2d_quad_vs_circle_outside() > 0


def test_sat2d_quad_segment_intersect():
    assert _test_2d_quad_vs_segment_intersect() < 0