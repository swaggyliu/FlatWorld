"""2D contact-detection and geometry-contact tests (Warp)."""

import os
import sys

import pytest
import warp as wp
from test_utils import init_sim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld.contact_detection import detectPointToAnalyticalPlane, pointToEdgeContact

init_sim()


@wp.kernel
def _k_plane_above(out: wp.array(dtype=float)):
    p, _n, _c = detectPointToAnalyticalPlane(
        wp.vec2(0.5, 1.0),
        wp.vec2(0.0, 0.0),
        wp.vec2(0.0, 1.0),
    )
    out[0] = p


@wp.kernel
def _k_plane_below(out: wp.array(dtype=float)):
    p, _n, _c = detectPointToAnalyticalPlane(
        wp.vec2(0.5, -0.5),
        wp.vec2(0.0, 0.0),
        wp.vec2(0.0, 1.0),
    )
    out[0] = p


@wp.kernel
def _k_plane_on(out: wp.array(dtype=float)):
    p, _n, _c = detectPointToAnalyticalPlane(
        wp.vec2(0.5, 0.0),
        wp.vec2(0.0, 0.0),
        wp.vec2(0.0, 1.0),
    )
    out[0] = p


@wp.kernel
def _k_edge_contact(out: wp.array(dtype=wp.vec2)):
    p, _n, _c, is_inside, _w = pointToEdgeContact(
        wp.vec2(0.5, 0.01),
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 0.0),
        2,
    )
    inside_f = 0.0
    if is_inside:
        inside_f = 1.0
    out[0] = wp.vec2(p, inside_f)


@wp.kernel
def _k_edge_outside(out: wp.array(dtype=int)):
    _p, _n, _c, is_inside, _w = pointToEdgeContact(
        wp.vec2(2.0, 0.0),
        wp.vec2(0.0, 0.0),
        wp.vec2(1.0, 0.0),
        2,
    )
    if is_inside:
        out[0] = 1
    else:
        out[0] = 0


def _run_float(kernel) -> float:
    out = wp.zeros(1, dtype=float)
    wp.launch(kernel, dim=1, inputs=[out])
    return float(out.numpy()[0])


def test_detect_point_to_plane_above_2d():
    assert _run_float(_k_plane_above) > 0


def test_detect_point_to_plane_below_2d():
    assert _run_float(_k_plane_below) < 0


def test_detect_point_on_plane_2d():
    assert abs(_run_float(_k_plane_on)) < 1e-5


def test_ball_on_analytical_plane_stops_2d():
    from flatworld import GroundDomain, BallRigid, ExplicitLoop, Gravity, RigidBodyDomain

    rigid = BallRigid(2, [0.5, 0.3], 0.05, 1.0)
    domain = RigidBodyDomain(rigid, [Gravity([0, -9.8])], considerContact=True)
    ground = GroundDomain(2, [0, 0.1], [0, 1])
    looper = ExplicitLoop(0.0, [domain, ground])

    for _ in range(120):
        looper.advanceWithTime(1.0 / 60.0)

    pos = domain.getCurrentRefPoint()
    assert pos[1] >= 0.1 - 0.01, f"Ball penetrated ground: y={pos[1]:.4f}"


def test_point_to_edge_contact_2d():
    out = wp.zeros(1, dtype=wp.vec2)
    wp.launch(_k_edge_contact, dim=1, inputs=[out])
    result = out.numpy()[0]
    assert result[0] < 1.0
    assert int(result[1]) == 1


def test_point_to_edge_outside_projection_2d():
    out = wp.zeros(1, dtype=int)
    wp.launch(_k_edge_outside, dim=1, inputs=[out])
    assert int(out.numpy()[0]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
