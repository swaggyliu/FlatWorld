"""2D contact-detection and geometry-contact tests."""

import os
import sys

import pytest
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


if not ti.lang.impl.get_runtime().prog:
    ti.init(offline_cache=True, arch=ti.cpu)


def test_detect_point_to_plane_above_2d():
    from flatworld.contact_detection import detectPointToAnalyticalPlane

    @ti.kernel
    def test_kernel() -> ti.f32:
        point = ti.Vector([0.5, 1.0])
        plane_point = ti.Vector([0.0, 0.0])
        plane_normal = ti.Vector([0.0, 1.0])
        p, _, _ = detectPointToAnalyticalPlane(point, plane_point, plane_normal)
        return p

    assert test_kernel() > 0


def test_detect_point_to_plane_below_2d():
    from flatworld.contact_detection import detectPointToAnalyticalPlane

    @ti.kernel
    def test_kernel() -> ti.f32:
        point = ti.Vector([0.5, -0.5])
        plane_point = ti.Vector([0.0, 0.0])
        plane_normal = ti.Vector([0.0, 1.0])
        p, _, _ = detectPointToAnalyticalPlane(point, plane_point, plane_normal)
        return p

    assert test_kernel() < 0


def test_detect_point_on_plane_2d():
    from flatworld.contact_detection import detectPointToAnalyticalPlane

    @ti.kernel
    def test_kernel() -> ti.f32:
        point = ti.Vector([0.5, 0.0])
        plane_point = ti.Vector([0.0, 0.0])
        plane_normal = ti.Vector([0.0, 1.0])
        p, _, _ = detectPointToAnalyticalPlane(point, plane_point, plane_normal)
        return p

    assert abs(test_kernel()) < 1e-5


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
    from flatworld.contact_detection import pointToEdgeContact

    @ti.kernel
    def test_kernel() -> ti.types.vector(2, ti.f32):
        point = ti.Vector([0.5, 0.01])
        edge_n0 = ti.Vector([0.0, 0.0])
        edge_n1 = ti.Vector([1.0, 0.0])
        p, _, _, is_inside, _ = pointToEdgeContact(point, edge_n0, edge_n1, 2)
        return ti.Vector([p, ti.cast(is_inside, ti.f32)])

    result = test_kernel()
    assert result[0] < 1.0
    assert int(result[1]) == 1


def test_point_to_edge_outside_projection_2d():
    from flatworld.contact_detection import pointToEdgeContact

    @ti.kernel
    def test_kernel() -> ti.i32:
        point = ti.Vector([2.0, 0.0])
        edge_n0 = ti.Vector([0.0, 0.0])
        edge_n1 = ti.Vector([1.0, 0.0])
        _, _, _, is_inside, _ = pointToEdgeContact(point, edge_n0, edge_n1, 2)
        return is_inside

    assert test_kernel() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])