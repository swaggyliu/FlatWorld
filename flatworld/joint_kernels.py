"""Unified Warp kernels for joint constraint projection.

This module contains GPU-accelerated funcs that process joints, writing PGS
constraint rows into explicit Warp arrays (no manager template).

Joint types (see definitions.JointType):
    Revolute (1)  - constrains translation; free rotation (optional limits/motor)
    Spherical (2) - constrains translation only
    Weld (3)      - fully constrains translation and rotation
    Prismatic (7) - allows sliding along axis; locks rotation

Public entry point
------------------
``assemble_single_joint_rows`` — call once per joint index from a ``@wp.kernel``
(e.g. from RigidManager) with explicit buffers:

.. code-block:: python

    @wp.func
    def assemble_single_joint_rows(
        dt: float,
        j_idx: int,
        max_constraints: int,
        # joint attributes
        joint_type: wp.array(dtype=int),
        joint_id_a: wp.array(dtype=int),
        joint_id_b: wp.array(dtype=int),
        joint_params: wp.array(dtype=vec6f),       # [pos_bias, ang_bias, lo, hi, vel_lim, effort]
        joint_has_motor: wp.array(dtype=int),
        joint_motor_target_mode: wp.array(dtype=int),
        joint_motor_target_vel: wp.array(dtype=float),
        joint_q0_rel_inv: wp.array(dtype=float),   # relative rest angle (2D)
        joint_axis: wp.array(dtype=wp.vec2),
        joint_l1: wp.array(dtype=wp.vec2),
        joint_l2: wp.array(dtype=wp.vec2),
        # rigid body state (2D)
        rigidParams: wp.array(dtype=wp.vec2, ndim=2),  # [:, 0] = center
        quat: wp.array(dtype=float),                   # angle
        quat_initial: wp.array(dtype=float),
        RotV: wp.array(dtype=float),                   # angular velocity
        # PGS row buffers
        numConstraints: wp.array(dtype=int),           # shape (1,), atomic counter
        pgs_bodypair: wp.array(dtype=wp.vec2i),
        pgs_Jac_a: wp.array(dtype=wp.vec3),            # [vx, vy, omega]
        pgs_Jac_b: wp.array(dtype=wp.vec3),
        pgs_rhs: wp.array(dtype=float),
        pgs_limits: wp.array(dtype=wp.vec2),           # [lower, upper]
        pgs_lambda: wp.array(dtype=float),
        pgs_parent_row: wp.array(dtype=int),
    )

Manager call site (after RigidManager Warp migration)::

    assemble_single_joint_rows(
        dt, j_idx, self.MAX_CONSTRAINTS,
        self.joint_type, self.joint_id_a, self.joint_id_b, self.joint_params,
        self.joint_has_motor, self.joint_motor_target_mode, self.joint_motor_target_vel,
        self.joint_q0_rel_inv, self.joint_axis, self.joint_l1, self.joint_l2,
        self.rigidParams, self.quat, self.quat_initial, self.RotV,
        self.numConstraints, self.pgs_bodypair, self.pgs_Jac_a, self.pgs_Jac_b,
        self.pgs_rhs, self.pgs_limits, self.pgs_lambda, self.pgs_parent_row,
    )
"""

from definitions import *
import warp as wp
from utils import cal2DRotationMat

ROW_INF = 1e10
EPS_NORM = 1e-9
EPS_QUAT = 1e-6

# joint_params row: [position_bias, angular_bias, lower, upper, vel_limit, effort]
vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def _compute_rotation_matrix_2d(current_angle: float, initial_angle: float):
    """Compute 2D rotation matrix for relative rotation from initial orientation."""
    delta_angle = current_angle - initial_angle
    return cal2DRotationMat(delta_angle)


@wp.func
def _clamp_bias_velocity(bias_vec: wp.vec2, max_bias_vel: float):
    eps = EPS_NORM
    bias_len = wp.length(bias_vec)
    if bias_len > max_bias_vel:
        bias_vec = bias_vec * (max_bias_vel / (bias_len + eps))
    return bias_vec


@wp.func
def _clamp_target_velocity(target: float, vel_limit: float):
    if vel_limit > 0.0:
        target = wp.min(wp.max(target, -vel_limit), vel_limit)
    return target


@wp.func
def _quat_to_rotvec(q_delta: wp.vec4):
    w = wp.max(-1.0, wp.min(1.0, q_delta[0]))
    v = wp.vec3(q_delta[1], q_delta[2], q_delta[3])
    v_norm = wp.length(v)
    rot = wp.vec3(0.0, 0.0, 0.0)
    if v_norm > EPS_NORM:
        angle = 2.0 * wp.atan2(v_norm, w)
        rot = angle * v / v_norm
    else:
        rot = 2.0 * v
    return rot


@wp.func
def _perp_component(vec: wp.vec2, axis_u: wp.vec2):
    return vec - axis_u * wp.dot(vec, axis_u)


@wp.func
def _add_pgs_row(
    aid: int,
    bid: int,
    jac_a: wp.vec3,
    jac_b: wp.vec3,
    rhs: float,
    lower: float,
    upper: float,
    parent_row: int,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    flag = int(-1)
    ci = int(wp.atomic_add(numConstraints, 0, 1))
    if ci < max_constraints:
        pgs_bodypair[ci] = wp.vec2i(aid, bid)
        pgs_Jac_a[ci] = jac_a
        pgs_Jac_b[ci] = jac_b
        pgs_rhs[ci] = rhs
        pgs_limits[ci] = wp.vec2(lower, upper)
        pgs_lambda[ci] = 0.0
        pgs_parent_row[ci] = parent_row
        flag = ci
    return flag


@wp.func
def _add_linear_constraint_row(
    idx_a: int,
    idx_b: int,
    axis: wp.vec2,
    r_a: wp.vec2,
    r_b: wp.vec2,
    rhs: float,
    lower: float,
    upper: float,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    ang_a = r_a[0] * axis[1] - r_a[1] * axis[0]
    ang_b = r_b[0] * axis[1] - r_b[1] * axis[0]
    jac_a = wp.vec3(axis[0], axis[1], ang_a)
    jac_b = wp.vec3(axis[0], axis[1], ang_b)
    _add_pgs_row(
        idx_a,
        idx_b,
        jac_a,
        jac_b,
        rhs,
        lower,
        upper,
        -1,
        numConstraints,
        max_constraints,
        pgs_bodypair,
        pgs_Jac_a,
        pgs_Jac_b,
        pgs_rhs,
        pgs_limits,
        pgs_lambda,
        pgs_parent_row,
    )


@wp.func
def _add_angular_constraint_row(
    idx_a: int,
    idx_b: int,
    axis: float,
    rhs: float,
    lower: float,
    upper: float,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    jac_a = wp.vec3(0.0, 0.0, axis)
    jac_b = wp.vec3(0.0, 0.0, axis)
    _add_pgs_row(
        idx_a,
        idx_b,
        jac_a,
        jac_b,
        rhs,
        lower,
        upper,
        -1,
        numConstraints,
        max_constraints,
        pgs_bodypair,
        pgs_Jac_a,
        pgs_Jac_b,
        pgs_rhs,
        pgs_limits,
        pgs_lambda,
        pgs_parent_row,
    )


@wp.func
def _assemble_point_lock_rows(
    idx_a: int,
    idx_b: int,
    r_a: wp.vec2,
    r_b: wp.vec2,
    bias_term: wp.vec2,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    _add_linear_constraint_row(
        idx_a,
        idx_b,
        wp.vec2(1.0, 0.0),
        r_a,
        r_b,
        -bias_term[0],
        -ROW_INF,
        ROW_INF,
        numConstraints,
        max_constraints,
        pgs_bodypair,
        pgs_Jac_a,
        pgs_Jac_b,
        pgs_rhs,
        pgs_limits,
        pgs_lambda,
        pgs_parent_row,
    )
    _add_linear_constraint_row(
        idx_a,
        idx_b,
        wp.vec2(0.0, 1.0),
        r_a,
        r_b,
        -bias_term[1],
        -ROW_INF,
        ROW_INF,
        numConstraints,
        max_constraints,
        pgs_bodypair,
        pgs_Jac_a,
        pgs_Jac_b,
        pgs_rhs,
        pgs_limits,
        pgs_lambda,
        pgs_parent_row,
    )


@wp.func
def _assemble_prismatic_rows(
    idx_a: int,
    idx_b: int,
    axis_world: wp.vec2,
    r_a: wp.vec2,
    r_b: wp.vec2,
    bias_term: wp.vec2,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    eps = EPS_NORM
    axis_len = wp.length(axis_world)
    if axis_len > eps:
        axis_u = axis_world / axis_len
        perp = wp.vec2(-axis_u[1], axis_u[0])
        _add_linear_constraint_row(
            idx_a,
            idx_b,
            perp,
            r_a,
            r_b,
            -wp.dot(bias_term, perp),
            -ROW_INF,
            ROW_INF,
            numConstraints,
            max_constraints,
            pgs_bodypair,
            pgs_Jac_a,
            pgs_Jac_b,
            pgs_rhs,
            pgs_limits,
            pgs_lambda,
            pgs_parent_row,
        )


@wp.func
def _assemble_weld_angular_rows(
    dt: float,
    idx_a: int,
    idx_b: int,
    q0_rel_inv: float,
    angular_bias: float,
    quat: wp.array(dtype=float),
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    eps = EPS_QUAT
    angle_error = quat[idx_a] - quat[idx_b] - q0_rel_inv
    bias_term = angular_bias * angle_error / (dt + eps)
    _add_angular_constraint_row(
        idx_a,
        idx_b,
        1.0,
        -bias_term,
        -ROW_INF,
        ROW_INF,
        numConstraints,
        max_constraints,
        pgs_bodypair,
        pgs_Jac_a,
        pgs_Jac_b,
        pgs_rhs,
        pgs_limits,
        pgs_lambda,
        pgs_parent_row,
    )


@wp.func
def _assemble_angular_limit_row(
    dt: float,
    idx_a: int,
    idx_b: int,
    current_angle: float,
    lower_limit: float,
    upper_limit: float,
    RotV: wp.array(dtype=float),
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    eps = EPS_NORM
    restitution = 0.1
    rel_velocity = RotV[idx_b] - RotV[idx_a]

    predicted_angle = current_angle + rel_velocity * dt
    violation = 0.0
    active_limit = int(0)
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
        if wp.abs(violation) > eps:
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

        _add_angular_constraint_row(
            idx_a,
            idx_b,
            1.0,
            rhs,
            lower,
            upper,
            numConstraints,
            max_constraints,
            pgs_bodypair,
            pgs_Jac_a,
            pgs_Jac_b,
            pgs_rhs,
            pgs_limits,
            pgs_lambda,
            pgs_parent_row,
        )


@wp.func
def _assemble_linear_limit_row(
    dt: float,
    idx_a: int,
    idx_b: int,
    axis_world: wp.vec2,
    current_distance: float,
    lower_limit: float,
    upper_limit: float,
    r_a: wp.vec2,
    r_b: wp.vec2,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    eps = EPS_NORM
    violation = 0.0
    active_limit = int(0)

    if current_distance < lower_limit:
        violation = current_distance - lower_limit
        active_limit = 1
    elif current_distance > upper_limit:
        violation = current_distance - upper_limit
        active_limit = 2

    if active_limit > 0:
        bias_velocity = 0.2 * violation / (dt + eps)
        target_velocity = 0.0
        rhs = bias_velocity - target_velocity
        lower = 0.0
        upper = ROW_INF
        if active_limit == 1:
            lower = -ROW_INF
            upper = 0.0
        _add_linear_constraint_row(
            idx_a,
            idx_b,
            axis_world,
            r_a,
            r_b,
            rhs,
            lower,
            upper,
            numConstraints,
            max_constraints,
            pgs_bodypair,
            pgs_Jac_a,
            pgs_Jac_b,
            pgs_rhs,
            pgs_limits,
            pgs_lambda,
            pgs_parent_row,
        )


@wp.func
def _assemble_revolute_motor_row(
    idx_a: int,
    idx_b: int,
    vel_limit: float,
    target: float,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    # Velocity motor: (omega_b - omega_a) = target
    target = _clamp_target_velocity(target, vel_limit)
    _add_angular_constraint_row(
        idx_a,
        idx_b,
        1.0,
        -target,
        -ROW_INF,
        ROW_INF,
        numConstraints,
        max_constraints,
        pgs_bodypair,
        pgs_Jac_a,
        pgs_Jac_b,
        pgs_rhs,
        pgs_limits,
        pgs_lambda,
        pgs_parent_row,
    )


@wp.func
def _assemble_prismatic_motor_row(
    idx_a: int,
    idx_b: int,
    axis_world: wp.vec2,
    r_a: wp.vec2,
    r_b: wp.vec2,
    vel_limit: float,
    target: float,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    # Velocity motor: (v_b - v_a) · axis = target
    target = _clamp_target_velocity(target, vel_limit)
    axis_len = wp.length(axis_world)
    if axis_len > EPS_NORM:
        axis_u = axis_world / axis_len
        _add_linear_constraint_row(
            idx_a,
            idx_b,
            axis_u,
            r_a,
            r_b,
            -target,
            -ROW_INF,
            ROW_INF,
            numConstraints,
            max_constraints,
            pgs_bodypair,
            pgs_Jac_a,
            pgs_Jac_b,
            pgs_rhs,
            pgs_limits,
            pgs_lambda,
            pgs_parent_row,
        )


@wp.func
def assemble_single_joint_rows(
    dt: float,
    j_idx: int,
    max_constraints: int,
    joint_type: wp.array(dtype=int),
    joint_id_a: wp.array(dtype=int),
    joint_id_b: wp.array(dtype=int),
    joint_params: wp.array(dtype=vec6f),
    joint_has_motor: wp.array(dtype=int),
    joint_motor_target_mode: wp.array(dtype=int),
    joint_motor_target_vel: wp.array(dtype=float),
    joint_q0_rel_inv: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec2),
    joint_l1: wp.array(dtype=wp.vec2),
    joint_l2: wp.array(dtype=wp.vec2),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    quat: wp.array(dtype=float),
    quat_initial: wp.array(dtype=float),
    RotV: wp.array(dtype=float),
    numConstraints: wp.array(dtype=int),
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    eps = EPS_NORM
    max_bias_vel = 5.0

    jtype = joint_type[j_idx]
    idx_a = joint_id_a[j_idx]
    idx_b = joint_id_b[j_idx]
    params = joint_params[j_idx]
    position_bias = params[0]
    angular_bias = params[1]
    lower_limit = params[2]
    upper_limit = params[3]
    vel_limit = params[4]
    has_motor = joint_has_motor[j_idx]
    motor_mode = joint_motor_target_mode[j_idx]
    target = joint_motor_target_vel[j_idx]
    q0_rel_inv = joint_q0_rel_inv[j_idx]
    ax = joint_axis[j_idx]

    R1 = _compute_rotation_matrix_2d(quat[idx_a], quat_initial[idx_a])
    R2 = _compute_rotation_matrix_2d(quat[idx_b], quat_initial[idx_b])

    l1 = R1 @ joint_l1[j_idx]
    l2 = R2 @ joint_l2[j_idx]
    r_a = -l1
    r_b = -l2

    point_a = rigidParams[idx_a, 0] + r_a
    point_b = rigidParams[idx_b, 0] + r_b
    C = point_a - point_b  # stabilization term for position error
    bias_term = _clamp_bias_velocity(position_bias * C / (dt + eps), max_bias_vel)

    axis_world = ax

    if jtype == JointType.Prismatic:
        C_parallel = wp.vec2(0.0, 0.0)
        axis_len = wp.length(axis_world)
        if axis_len > eps:
            axis_u = axis_world / axis_len
            C_parallel = wp.dot(C, axis_u) * axis_u
        bias_term = _clamp_bias_velocity(position_bias * (C - C_parallel) / (dt + eps), max_bias_vel)
        _assemble_prismatic_rows(
            idx_a,
            idx_b,
            axis_world,
            r_a,
            r_b,
            bias_term,
            numConstraints,
            max_constraints,
            pgs_bodypair,
            pgs_Jac_a,
            pgs_Jac_b,
            pgs_rhs,
            pgs_limits,
            pgs_lambda,
            pgs_parent_row,
        )
        if has_motor == 1 and motor_mode != 2:  # meaning not torque controlled
            _assemble_prismatic_motor_row(
                idx_a,
                idx_b,
                axis_world,
                r_a,
                r_b,
                vel_limit,
                target,
                numConstraints,
                max_constraints,
                pgs_bodypair,
                pgs_Jac_a,
                pgs_Jac_b,
                pgs_rhs,
                pgs_limits,
                pgs_lambda,
                pgs_parent_row,
            )
        _assemble_weld_angular_rows(
            dt,
            idx_a,
            idx_b,
            q0_rel_inv,
            angular_bias,
            quat,
            numConstraints,
            max_constraints,
            pgs_bodypair,
            pgs_Jac_a,
            pgs_Jac_b,
            pgs_rhs,
            pgs_limits,
            pgs_lambda,
            pgs_parent_row,
        )
        if lower_limit < upper_limit and wp.length(axis_world) > eps:
            axis_u = axis_world / (wp.length(axis_world) + eps)
            current_distance = wp.dot(rigidParams[idx_b, 0] - rigidParams[idx_a, 0], axis_u)
            _assemble_linear_limit_row(
                dt,
                idx_a,
                idx_b,
                axis_u,
                current_distance,
                lower_limit,
                upper_limit,
                r_a,
                r_b,
                numConstraints,
                max_constraints,
                pgs_bodypair,
                pgs_Jac_a,
                pgs_Jac_b,
                pgs_rhs,
                pgs_limits,
                pgs_lambda,
                pgs_parent_row,
            )
    else:
        _assemble_point_lock_rows(
            idx_a,
            idx_b,
            r_a,
            r_b,
            bias_term,
            numConstraints,
            max_constraints,
            pgs_bodypair,
            pgs_Jac_a,
            pgs_Jac_b,
            pgs_rhs,
            pgs_limits,
            pgs_lambda,
            pgs_parent_row,
        )
        if jtype == JointType.Weld:
            _assemble_weld_angular_rows(
                dt,
                idx_a,
                idx_b,
                q0_rel_inv,
                angular_bias,
                quat,
                numConstraints,
                max_constraints,
                pgs_bodypair,
                pgs_Jac_a,
                pgs_Jac_b,
                pgs_rhs,
                pgs_limits,
                pgs_lambda,
                pgs_parent_row,
            )
        elif jtype == JointType.Revolute:
            if has_motor == 1 and motor_mode != 2:  # meaning not torque controlled
                _assemble_revolute_motor_row(
                    idx_a,
                    idx_b,
                    vel_limit,
                    target,
                    numConstraints,
                    max_constraints,
                    pgs_bodypair,
                    pgs_Jac_a,
                    pgs_Jac_b,
                    pgs_rhs,
                    pgs_limits,
                    pgs_lambda,
                    pgs_parent_row,
                )
            if lower_limit < upper_limit:
                angle_a = quat[idx_a] - quat_initial[idx_a]
                angle_b = quat[idx_b] - quat_initial[idx_b]
                current_angle = angle_b - angle_a
                _assemble_angular_limit_row(
                    dt,
                    idx_a,
                    idx_b,
                    current_angle,
                    lower_limit,
                    upper_limit,
                    RotV,
                    numConstraints,
                    max_constraints,
                    pgs_bodypair,
                    pgs_Jac_a,
                    pgs_Jac_b,
                    pgs_rhs,
                    pgs_limits,
                    pgs_lambda,
                    pgs_parent_row,
                )
