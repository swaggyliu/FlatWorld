"""Regression test for host-side FEM count mirrors used by femStep."""

import numpy as np
import warp as wp

import flatworld  # noqa: F401  (adds the package directory to sys.path)
from femspringmanager import FemSpringManager, _integrate_nodes_kernel
from wp_init import ensure_warp


def test_fem_step_uses_host_counters_for_launch_dimensions(monkeypatch):
    manager = FemSpringManager.__new__(FemSpringManager)
    manager._n_fem_elems = 2
    manager._n_elems = 5
    manager._n_nodes = 4
    manager.aabb = None

    # femStep passes these buffers to Warp launches. Dummy objects are enough
    # because the test intercepts wp.launch and checks only launch dimensions.
    buffer_names = (
        "totalFEMElements", "totalElements", "totalNodes", "connectivity", "coords",
        "J_inv", "matType", "matE", "matNu", "matYieldStress", "matH",
        "plasticStrain", "eqPlasticStrain", "elemVolume", "elemStress", "elemStrain",
        "Fext", "V", "matDamping", "bcNodes", "bcValues", "mass", "A", "U",
    )
    for name in buffer_names:
        setattr(manager, name, object())

    launched_dims = []

    def record_launch(kernel, dim, inputs):
        launched_dims.append(dim)

    monkeypatch.setattr(wp, "launch", record_launch)
    manager.femStep(1.0 / 600.0, 0.0)

    assert launched_dims == [2, 5, 4]


def test_node_integration_clears_force_buffer_inline():
    ensure_warp()
    device = "cpu"
    total_nodes = wp.array([1], dtype=int, device=device)
    bc_nodes = wp.array([0], dtype=int, device=device)
    bc_values = wp.array([[0.0, 0.0]], dtype=wp.vec2, device=device)
    mass = wp.array([1.0], dtype=float, device=device)
    A = wp.array([[0.0, 0.0]], dtype=wp.vec2, device=device)
    V = wp.array([[0.0, 0.0]], dtype=wp.vec2, device=device)
    U = wp.array([[0.0, 0.0]], dtype=wp.vec2, device=device)
    coords = wp.array([[0.0, 0.0]], dtype=wp.vec2, device=device)
    Fext = wp.array([[2.0, 3.0]], dtype=wp.vec2, device=device)

    wp.launch(
        _integrate_nodes_kernel,
        dim=1,
        inputs=[0.1, 0.0, total_nodes, bc_nodes, bc_values, mass, A, V, U, coords, Fext],
        device=device,
    )

    np.testing.assert_allclose(Fext.numpy(), np.array([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(coords.numpy(), np.array([[0.02, 0.03]], dtype=np.float32), atol=1e-6)
