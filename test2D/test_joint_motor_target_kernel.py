"""Device-side joint motor target integration tests."""

import numpy as np
import warp as wp

import flatworld  # noqa: F401  (adds the package directory to sys.path)
from joint_kernels import vec6f
from rigidmanager import _update_joint_motor_velocity_targets_kernel
from wp_init import ensure_warp


def test_joint_motor_target_modes_match_python_reference():
    ensure_warp()
    device = "cpu"
    has_motor = wp.array(np.array([1, 1, 1, 0, 1], dtype=np.int32), dtype=int, device=device)
    modes = wp.array(np.array([0, 1, 2, 0, 1], dtype=np.int32), dtype=int, device=device)
    commands = wp.array(np.array([1.5, 10.0, 3.0, 4.0, -10.0], dtype=np.float32), dtype=float, device=device)
    motor_vel = wp.array(np.array([0.0, 0.5, 0.7, 0.9, 0.0], dtype=np.float32), dtype=float, device=device)
    params_np = np.zeros((5, 6), dtype=np.float32)
    params_np[:, 4] = [2.0, 1.0, 1.0, 1.0, 0.8]
    params = wp.array(params_np, dtype=vec6f, device=device)

    wp.launch(
        _update_joint_motor_velocity_targets_kernel,
        dim=5,
        inputs=[0.1, 5, has_motor, modes, commands, params, motor_vel],
        device=device,
    )

    expected = np.array([1.5, 1.0, 0.7, 0.9, -0.8], dtype=np.float32)
    np.testing.assert_allclose(motor_vel.numpy(), expected, rtol=0.0, atol=1e-6)
