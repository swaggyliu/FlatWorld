"""Warp joint PD velocity controller regression tests."""

import numpy as np
import warp as wp

import flatworld  # noqa: F401  (adds the package directory to sys.path)
from definitions import JointType
from joint_kernels import vec6f
from rigidmanager import _apply_joint_pd_velocity_kernel
from wp_init import ensure_warp


def test_joint_pd_velocity_targets_for_revolute_and_prismatic():
    ensure_warp()
    device = "cpu"
    joint_id_a = wp.array(np.array([0, 0], dtype=np.int32), dtype=int, device=device)
    joint_id_b = wp.array(np.array([1, 1], dtype=np.int32), dtype=int, device=device)
    joint_axis = wp.array(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=device)
    joint_type = wp.array(
        np.array([int(JointType.Revolute), int(JointType.Prismatic)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    kpd = wp.array(np.array([[2.0, 0.5], [2.0, 0.5]], dtype=np.float32), dtype=wp.vec2, device=device)
    params_np = np.zeros((2, 6), dtype=np.float32)
    params_np[0] = [0.0, 0.0, -1.0, 1.0, 2.0, 0.0]
    params_np[1] = [0.0, 0.0, 0.0, 0.0, 1.5, 0.0]
    joint_params = wp.array(params_np, dtype=vec6f, device=device)
    quat = wp.array(np.array([0.1, 0.4], dtype=np.float32), dtype=float, device=device)
    quat_initial = wp.array(np.array([0.0, 0.0], dtype=np.float32), dtype=float, device=device)
    rot_v = wp.array(np.array([0.1, 0.2], dtype=np.float32), dtype=float, device=device)
    control_target = wp.array(np.array([1.0, 0.8], dtype=np.float32), dtype=float, device=device)
    rigid_params_np = np.zeros((2, 2, 2), dtype=np.float32)
    rigid_params_np[1, 0] = [0.3, 0.0]
    rigid_params = wp.array(rigid_params_np, dtype=wp.vec2, device=device)
    lin_v = wp.array(np.array([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32), dtype=wp.vec2, device=device)
    motor_mode = wp.array(np.array([2, 1], dtype=np.int32), dtype=int, device=device)
    motor_vel = wp.array(np.array([0.0, 0.0], dtype=np.float32), dtype=float, device=device)

    wp.launch(
        _apply_joint_pd_velocity_kernel,
        dim=2,
        inputs=[
            2,
            0.1,
            int(JointType.Revolute),
            int(JointType.Prismatic),
            joint_id_a,
            joint_id_b,
            joint_axis,
            joint_type,
            kpd,
            joint_params,
            quat,
            quat_initial,
            rot_v,
            control_target,
            rigid_params,
            lin_v,
            motor_mode,
            motor_vel,
        ],
        device=device,
    )

    np.testing.assert_array_equal(motor_mode.numpy(), np.array([0, 1], dtype=np.int32))
    np.testing.assert_allclose(motor_vel.numpy(), np.array([2.0, 0.9], dtype=np.float32), atol=1e-6)
