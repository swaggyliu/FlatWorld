"""Warp joint PD torque controller regression tests."""

import numpy as np
import warp as wp

import flatworld  # noqa: F401
from definitions import JointType
from joint_kernels import vec6f
from rigidmanager import _apply_joint_pd_torque_kernel
from wp_init import ensure_warp


def test_joint_pd_torque_accumulates_shared_body_impulses():
    ensure_warp()
    device = "cpu"
    joint_id_a = wp.array(np.array([0, 0, 1], dtype=np.int32), dtype=int, device=device)
    joint_id_b = wp.array(np.array([1, 2, 2], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(
        np.array(
            [int(JointType.Revolute), int(JointType.Revolute), int(JointType.Prismatic)],
            dtype=np.int32,
        ),
        dtype=int,
        device=device,
    )
    kpd = wp.array(np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=device)
    params_np = np.zeros((3, 6), dtype=np.float32)
    joint_params = wp.array(params_np, dtype=vec6f, device=device)
    quat = wp.array(np.array([0.0, 0.1, 0.2], dtype=np.float32), dtype=float, device=device)
    quat_initial = wp.zeros(3, dtype=float, device=device)
    rot_v = wp.zeros(3, dtype=float, device=device)
    control_target = wp.zeros(3, dtype=float, device=device)
    inertia = wp.ones(3, dtype=float, device=device)
    motor_mode = wp.array(np.array([0, 0, 1], dtype=np.int32), dtype=int, device=device)
    accumulated = wp.array(np.array([0.5, 0.1, 0.2], dtype=np.float32), dtype=float, device=device)

    wp.launch(
        _apply_joint_pd_torque_kernel,
        dim=3,
        inputs=[
            3,
            0.01,
            int(JointType.Revolute),
            joint_id_a,
            joint_id_b,
            joint_type,
            kpd,
            joint_params,
            quat,
            quat_initial,
            rot_v,
            control_target,
            inertia,
            motor_mode,
            accumulated,
        ],
        device=device,
    )

    # Two independent joints share body 0. Their impulses must both be retained.
    np.testing.assert_array_equal(motor_mode.numpy(), np.array([2, 2, 1], dtype=np.int32))
    np.testing.assert_allclose(
        accumulated.numpy(),
        np.array([0.53, 0.09, 0.18], dtype=np.float32),
        atol=1e-6,
    )
