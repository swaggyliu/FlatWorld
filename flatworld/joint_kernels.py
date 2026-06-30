"""Unified Taichi kernels for joint constraint projection.

This module contains GPU-accelerated kernels that process all joints in a single pass,
eliminating per-joint kernel launches and improving performance significantly.

Joint types:
    0: RevoluteJoint - allows rotation around axis, constrains translation
    1: WeldJoint - fully constrains translation and rotation
    2: PrismaticJoint - allows sliding along axis, locks rotation
    3: SphericalJoint - constrains translation only, free rotation
"""

from definitions import *
import taichi as ti
from utils import cal2DRotationMat
ROW_INF = 1e10
EPS_NORM = 1e-9
EPS_QUAT = 1e-6


@ti.func
def _compute_rotation_matrix_2d(current_angle, initial_angle):
    """Compute 2D rotation matrix for relative rotation from initial orientation."""
    delta_angle = current_angle - initial_angle
    return cal2DRotationMat(delta_angle)



@ti.func
def _clamp_bias_velocity(bias_vec, max_bias_vel: ti.f32):
    eps = EPS_NORM
    bias_len = bias_vec.norm()
    if bias_len > max_bias_vel:
        bias_vec = bias_vec * (max_bias_vel / (bias_len + eps))
    return bias_vec


@ti.func
def _clamp_target_velocity(target: ti.f32, vel_limit: ti.f32):
    if vel_limit > 0.0:
        target = ti.min(ti.max(target, -vel_limit), vel_limit)
    return target


@ti.func
def _quat_to_rotvec(q_delta):
    w = ti.max(-1.0, ti.min(1.0, q_delta[0]))
    v = ti.Vector([q_delta[1], q_delta[2], q_delta[3]])
    v_norm = v.norm()
    rot = ti.Vector([0.0, 0.0, 0.0])
    if v_norm > EPS_NORM:
        angle = 2.0 * ti.atan2(v_norm, w)
        rot = angle * v / v_norm
    else:
        rot = 2.0 * v
    return rot


@ti.func
def _perp_component(vec, axis_u):
    return vec - axis_u * vec.dot(axis_u)


@ti.func
def _add_linear_constraint_row(
    manager: ti.template(), idx_a: ti.i32, idx_b: ti.i32, axis, r_a, r_b, rhs: ti.f32, lower: ti.f32, upper: ti.f32
):
    jac_a = ti.Vector.zero(ti.f32, 6)
    jac_b = ti.Vector.zero(ti.f32, 6)
    ang_a = r_a[0] * axis[1] - r_a[1] * axis[0]
    ang_b = r_b[0] * axis[1] - r_b[1] * axis[0]
    jac_a = ti.Vector([axis[0], axis[1], 0.0, 0.0, 0.0, ang_a])
    jac_b = ti.Vector([axis[0], axis[1], 0.0, 0.0, 0.0, ang_b])

    manager._add_pgs_row(idx_a, idx_b, jac_a, jac_b, rhs, lower, upper, -1)


@ti.func
def _add_angular_constraint_row(
    manager: ti.template(), idx_a: ti.i32, idx_b: ti.i32, axis, rhs: ti.f32, lower: ti.f32, upper: ti.f32
):
    jac_a = ti.Vector.zero(ti.f32, 6)
    jac_b = ti.Vector.zero(ti.f32, 6)
    jac_a = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, axis])
    jac_b = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, axis])
    manager._add_pgs_row(idx_a, idx_b, jac_a, jac_b, rhs, lower, upper, -1)


@ti.func
def _assemble_point_lock_rows(manager: ti.template(), idx_a: ti.i32, idx_b: ti.i32, r_a, r_b, bias_term):
    _add_linear_constraint_row(
        manager, idx_a, idx_b, ti.Vector([1.0, 0.0]), r_a, r_b, -bias_term[0], -ROW_INF, ROW_INF
    )
    _add_linear_constraint_row(
        manager, idx_a, idx_b, ti.Vector([0.0, 1.0]), r_a, r_b, -bias_term[1], -ROW_INF, ROW_INF
    )
    

@ti.func
def _assemble_prismatic_rows(manager: ti.template(), idx_a: ti.i32, idx_b: ti.i32, axis_world, r_a, r_b, bias_term):
    eps = EPS_NORM
    axis_len = axis_world.norm()
    if axis_len > eps:
        axis_u = axis_world / axis_len
        perp = ti.Vector([-axis_u[1], axis_u[0]])
        _add_linear_constraint_row(manager, idx_a, idx_b, perp, r_a, r_b, -bias_term.dot(perp), -ROW_INF, ROW_INF)


@ti.func
def _assemble_weld_angular_rows(
    manager: ti.template(), dt: ti.f32, idx_a: ti.i32, idx_b: ti.i32, q0_rel_inv, angular_bias: ti.f32
):
    eps = EPS_QUAT

    angle_error = manager.quat[idx_a][0] - manager.quat[idx_b][0] - q0_rel_inv[0]
    bias_term = angular_bias * angle_error / (dt + eps)
    _add_angular_constraint_row(manager, idx_a, idx_b, 1.0, -bias_term, -ROW_INF, ROW_INF)


@ti.func
def _assemble_angular_limit_row(
    manager: ti.template(),
    dt: ti.f32,
    idx_a: ti.i32,
    idx_b: ti.i32,
    axis_world,
    current_angle: ti.f32,
    lower_limit: ti.f32,
    upper_limit: ti.f32,
):
    eps = EPS_NORM
    restitution = 0.1
    rel_velocity = 0.0

    rel_velocity = manager.RotV[idx_b][0] - manager.RotV[idx_a][0]

    predicted_angle = current_angle + rel_velocity * dt
    violation = 0.0
    active_limit = 0
    if predicted_angle < lower_limit:
        # Use predicted angle error so correction reacts before deep penetration.
        violation = predicted_angle - lower_limit
        active_limit = 1
    elif predicted_angle > upper_limit:
        # Use predicted angle error so correction reacts before deep penetration.
        violation = predicted_angle - upper_limit
        active_limit = 2

    if active_limit > 0:
        bias_velocity = 0.0
        if ti.abs(violation) > eps:
            bias_velocity = 0.3 * violation / (dt + eps)

        target_velocity = 0.0
        if active_limit == 1:
            if rel_velocity < 0.0:
                target_velocity = -restitution * rel_velocity
        else:
            if rel_velocity > 0.0:
                target_velocity = -restitution * rel_velocity

        rhs = bias_velocity - target_velocity
        lower = 0.0
        upper = ROW_INF
        if active_limit == 1:
            lower = -ROW_INF
            upper = 0.0

        _add_angular_constraint_row(manager, idx_a, idx_b, 1.0, rhs, lower, upper)


@ti.func
def _assemble_linear_limit_row(
    manager: ti.template(),
    dt: ti.f32,
    idx_a: ti.i32,
    idx_b: ti.i32,
    axis_world,
    current_distance: ti.f32,
    lower_limit: ti.f32,
    upper_limit: ti.f32,
    r_a,
    r_b,
):
    eps = EPS_NORM
    violation = 0.0
    active_limit = 0

    if current_distance < lower_limit:
        violation = current_distance - lower_limit
        active_limit = 1
    elif current_distance > upper_limit:
        violation = current_distance - upper_limit
        active_limit = 2

    lower = -ROW_INF
    upper = ROW_INF
    if active_limit > 0:
        bias_velocity = 0.2 * violation / (dt + eps)
        target_velocity = 0.0
        rhs = bias_velocity - target_velocity
        lower = 0.0
        upper = ROW_INF
        if active_limit == 1:
            lower = -ROW_INF
            upper = 0.0
        _add_linear_constraint_row(manager, idx_a, idx_b, axis_world, r_a, r_b, rhs, lower, upper)


@ti.func
def _assemble_revolute_motor_row(
    manager: ti.template(), idx_a: ti.i32, idx_b: ti.i32, axis_world, vel_limit: ti.f32, target: ti.f32
):
    # Velocity motor: (omega_b - omega_a) = target
    target = _clamp_target_velocity(target, vel_limit)
    _add_angular_constraint_row(manager, idx_a, idx_b, 1.0, -target, -ROW_INF, ROW_INF)


@ti.func
def _assemble_prismatic_motor_row(
    manager: ti.template(), idx_a: ti.i32, idx_b: ti.i32, axis_world, r_a, r_b, vel_limit: ti.f32, target: ti.f32
):
    # Velocity motor: (v_b - v_a) · axis = target
    target = _clamp_target_velocity(target, vel_limit)
    axis_len = axis_world.norm()
    if axis_len > EPS_NORM:
        axis_u = axis_world / axis_len
        _add_linear_constraint_row(manager, idx_a, idx_b, axis_u, r_a, r_b, -target, -ROW_INF, ROW_INF)


@ti.func
def assemble_single_joint_rows(manager: ti.template(), dt: ti.f32, j_idx: ti.i32):
    eps = EPS_NORM
    max_bias_vel = 5.0

    joint_type = manager.joint_type[j_idx]
    idx_a = manager.joint_id_a[j_idx]
    idx_b = manager.joint_id_b[j_idx]
    position_bias = manager.joint_params[j_idx][0]
    angular_bias = manager.joint_params[j_idx][1]
    lower_limit = manager.joint_params[j_idx][2]
    upper_limit = manager.joint_params[j_idx][3]
    vel_limit = manager.joint_params[j_idx][4]
    has_motor = manager.joint_has_motor[j_idx]
    motor_mode = manager.joint_motor_target_mode[j_idx]
    target = manager.joint_motor_target_vel[j_idx]
    q0_rel_inv = manager.joint_q0_rel_inv[j_idx]
    ax = manager.joint_axis[j_idx]

    R1 = ti.Matrix.identity(ti.f32, manager.d)
    R2 = ti.Matrix.identity(ti.f32, manager.d)
    R1 = _compute_rotation_matrix_2d(manager.quat[idx_a][0], manager.quat_initial[idx_a][0])
    R2 = _compute_rotation_matrix_2d(manager.quat[idx_b][0], manager.quat_initial[idx_b][0])

    l1 = R1 @ manager.joint_l1[j_idx]
    l2 = R2 @ manager.joint_l2[j_idx]
    r_a = -l1
    r_b = -l2

    point_a = manager.rigidParams[idx_a, 0] + r_a
    point_b = manager.rigidParams[idx_b, 0] + r_b
    C = point_a - point_b  # stabilization term for position error
    bias_term = _clamp_bias_velocity(position_bias * C / (dt + eps), max_bias_vel)

    axis_world = ti.Vector.zero(ti.f32, manager.d)
    axis_world = ax

    if joint_type == JointType.Prismatic:
        C_parallel = axis_world * 0.0
        axis_len = axis_world.norm()
        if axis_len > eps:
            axis_u = axis_world / axis_len
            C_parallel = C.dot(axis_u) * axis_u
        bias_term = _clamp_bias_velocity(position_bias * (C - C_parallel) / (dt + eps), max_bias_vel)
        _assemble_prismatic_rows(manager, idx_a, idx_b, axis_world, r_a, r_b, bias_term)
        if has_motor == 1 and motor_mode != 2:  # meaning not torque controlled
            _assemble_prismatic_motor_row(manager, idx_a, idx_b, axis_world, r_a, r_b, vel_limit, target)
        _assemble_weld_angular_rows(manager, dt, idx_a, idx_b, q0_rel_inv, angular_bias)
        if lower_limit < upper_limit and axis_world.norm() > eps:
            axis_u = axis_world / (axis_world.norm() + eps)
            current_distance = (manager.rigidParams[idx_b, 0] - manager.rigidParams[idx_a, 0]).dot(axis_u)
            _assemble_linear_limit_row(
                manager, dt, idx_a, idx_b, axis_u, current_distance, lower_limit, upper_limit, r_a, r_b
            )
    else:
        _assemble_point_lock_rows(manager, idx_a, idx_b, r_a, r_b, bias_term)
        if joint_type == JointType.Weld:
            _assemble_weld_angular_rows(manager, dt, idx_a, idx_b, q0_rel_inv, angular_bias)
        elif joint_type == JointType.Revolute:
            if has_motor == 1 and motor_mode != 2:  # meaning not torque controlled
                _assemble_revolute_motor_row(manager, idx_a, idx_b, axis_world, vel_limit, target)
            if lower_limit < upper_limit:
                current_angle = 0.0
                angle_a = manager.quat[idx_a][0] - manager.quat_initial[idx_a][0]
                angle_b = manager.quat[idx_b][0] - manager.quat_initial[idx_b][0]
                current_angle = angle_b - angle_a
                _assemble_angular_limit_row(
                    manager, dt, idx_a, idx_b, axis_world, current_angle, lower_limit, upper_limit
                )
