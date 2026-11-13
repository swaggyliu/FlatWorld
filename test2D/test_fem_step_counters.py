"""Regression test for host-side FEM count mirrors used by femStep."""

import warp as wp

from femspringmanager import FemSpringManager


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

    assert launched_dims == [2, 5, 4, 4]
