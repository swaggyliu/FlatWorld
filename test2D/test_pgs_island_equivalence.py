"""Numerical equivalence tests for the optional contact-island PGS solver."""

import numpy as np
import warp as wp

import flatworld  # noqa: F401  (adds the package directory to sys.path)
from rigidmanager import (
    _build_pgs_islands_kernel,
    _solve_pgs_islands_kernel,
    _solve_pgs_kernel,
)
from wp_init import ensure_warp


def _make_problem():
    ensure_warp()
    device = "cpu"
    num_constraints = wp.array(np.array([4], dtype=np.int32), dtype=int, device=device)
    bodypair = wp.array(
        np.array([[0, 1], [0, 1], [2, 3], [2, 3]], dtype=np.int32),
        dtype=wp.vec2i,
        device=device,
    )
    jac_a = wp.array(
        np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    jac_b = wp.array(
        np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    rhs = wp.array(np.array([0.1, 0.0, 0.05, 0.0], dtype=np.float32), dtype=float, device=device)
    limits = wp.array(
        np.array([[0.0, 1e6], [-0.5, 0.5], [0.0, 1e6], [-0.5, 0.5]], dtype=np.float32),
        dtype=wp.vec2,
        device=device,
    )
    parent_row = wp.array(np.array([-1, 0, -1, 2], dtype=np.int32), dtype=int, device=device)
    initial_v = np.array([[0.0, -1.0], [0.0, 0.0], [0.0, -0.5], [0.0, 0.0]], dtype=np.float32)
    return device, num_constraints, bodypair, jac_a, jac_b, rhs, limits, parent_row, initial_v


def _state(initial_v, device):
    return (
        wp.array(initial_v.copy(), dtype=wp.vec2, device=device),
        wp.zeros(4, dtype=float, device=device),
        wp.array(np.ones(4, dtype=np.float32), dtype=float, device=device),
        wp.array(np.ones(4, dtype=np.float32), dtype=float, device=device),
        wp.zeros(4, dtype=float, device=device),
    )


def test_contact_island_pgs_matches_serial_symmetric_sweeps():
    (
        device, num_constraints, bodypair, jac_a, jac_b, rhs, limits,
        parent_row, initial_v,
    ) = _make_problem()
    serial_v, serial_rot_v, mass, inertia, serial_lambda = _state(initial_v, device)
    island_v, island_rot_v, _, _, island_lambda = _state(initial_v, device)

    wp.launch(
        _solve_pgs_kernel,
        dim=1,
        inputs=[25, 1e-8, num_constraints, 4, serial_v, serial_rot_v, mass, inertia,
                bodypair, jac_a, jac_b, rhs, limits, serial_lambda, parent_row],
        device=device,
    )

    max_bodies = 4
    island_parent = wp.zeros(max_bodies, dtype=int, device=device)
    island_rank = wp.zeros(max_bodies, dtype=int, device=device)
    island_offsets = wp.zeros(max_bodies, dtype=int, device=device)
    island_counts = wp.zeros(max_bodies, dtype=int, device=device)
    island_cursor = wp.zeros(max_bodies, dtype=int, device=device)
    island_rows = wp.zeros(4, dtype=int, device=device)
    wp.launch(
        _build_pgs_islands_kernel,
        dim=1,
        inputs=[num_constraints, 4, max_bodies, bodypair, island_parent, island_rank,
                island_offsets, island_counts, island_cursor, island_rows],
        device=device,
    )
    wp.launch(
        _solve_pgs_islands_kernel,
        dim=max_bodies,
        inputs=[25, 1e-8, max_bodies, island_parent, island_offsets, island_counts,
                island_rows, island_v, island_rot_v, mass, inertia, bodypair, jac_a,
                jac_b, rhs, limits, island_lambda, parent_row],
        device=device,
    )

    assert island_parent.numpy()[0] == island_parent.numpy()[1]
    assert island_parent.numpy()[2] == island_parent.numpy()[3]
    assert island_parent.numpy()[0] != island_parent.numpy()[2]
    np.testing.assert_allclose(island_v.numpy(), serial_v.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(island_rot_v.numpy(), serial_rot_v.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(island_lambda.numpy(), serial_lambda.numpy(), rtol=1e-6, atol=1e-6)
