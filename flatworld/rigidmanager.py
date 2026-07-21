"""RigidManager — batched 2D rigid-body simulation (NVIDIA Warp).

Public API: substep, detect_all_contacts, solve_pgs, precompute_rigid_transforms,
drawAll, processDomains_, processJoints, etc.

PGS is serial: solve_pgs launches a dim=1 kernel that loops constraints forward
and backward. Joint rows use joint_kernels.assemble_single_joint_rows with the
explicit-array signature.
"""
from __future__ import annotations

from bvh import CollisionDetector
from contact_detection import (
    detectPointToAnalyticalPlane,
    detectPointToMeshBoundary,
    detectPointToPrimitive,
    detect_point_to_mesh_boundary_np,
    detect_point_to_primitive_np,
)
from definitions import *
from mesh import *
from joint_kernels import assemble_single_joint_rows, vec6f
import numpy as np
from operator import pos
from rigid import *
from sat import *
from spatialmanager import (
    SpatialHashManager,
    add_element,
    set_bounds,
    query_point,
    query_point_with_buffer,
)
import warp as wp
from utils import *
import time
from wp_init import ensure_warp

ensure_warp()

mat28 = wp.types.matrix(shape=(2, 8), dtype=wp.float32)


def _assign_scalar(arr: wp.array, value):
    np_arr = arr.numpy()
    if arr.dtype == wp.vec2:
        a = np.asarray(value, dtype=np.float32).reshape(-1)
        np_arr[0] = (float(a[0]), float(a[1]) if a.size > 1 else 0.0)
    else:
        np_arr[0] = value
    arr.assign(np_arr)


def _patch_array(arr: wp.array, index, value):
    np_arr = arr.numpy()
    # Host Warp arrays: coerce Python/numpy scalars and vec2-like values.
    if arr.dtype is float or arr.dtype == wp.float32:
        value = _as_float(value)
    elif arr.dtype == wp.vec2:
        a = np.asarray(value, dtype=np.float32).reshape(-1)
        value = (float(a[0]), float(a[1]) if a.size > 1 else 0.0)
    np_arr[index] = value
    arr.assign(np_arr)


def _patch_slice(arr: wp.array, start: int, values: np.ndarray):
    np_arr = arr.numpy()
    n = len(values)
    np_arr[start : start + n] = values
    arr.assign(np_arr)


def _fill_array(arr: wp.array, value):
    arr.fill_(value)


def _as_float(v) -> float:
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        return float(a[0])
    return float(v)


def _as_vec2(v):
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    return wp.vec2(float(a[0]), float(a[1]) if a.size > 1 else 0.0)


# ---------------------------------------------------------------------------
# Serial PGS (dim=1) — critical path kept as real Warp kernels
# ---------------------------------------------------------------------------

@wp.func
def _solve_pgs_single(
    i: int,
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=float),
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    bodypair = pgs_bodypair[i]
    aid = bodypair[0]
    bid = bodypair[1]
    if aid >= 0:
        jac_a = pgs_Jac_a[i]
        jac_b = pgs_Jac_b[i]
        va3 = wp.vec3(V[aid][0], V[aid][1], RotV[aid])
        vb3 = wp.vec3(0.0, 0.0, 0.0)
        inv_mass_a = 1.0 / (mass[aid] + 1e-12)
        inv_Ia = 1.0 / (inertia[aid] + 1e-12)
        massInvJacA = wp.vec3(jac_a[0] * inv_mass_a, jac_a[1] * inv_mass_a, jac_a[2] * inv_Ia)
        vel = wp.dot(jac_a, va3)
        W = wp.dot(jac_a, massInvJacA)
        has_b = bid >= 0
        massInvJacB = wp.vec3(0.0, 0.0, 0.0)
        if has_b:
            inv_mass_b = 1.0 / (mass[bid] + 1e-12)
            inv_Ib = 1.0 / (inertia[bid] + 1e-12)
            vb3 = wp.vec3(V[bid][0], V[bid][1], RotV[bid])
            vel = vel - wp.dot(jac_b, vb3)
            massInvJacB = wp.vec3(jac_b[0] * inv_mass_b, jac_b[1] * inv_mass_b, jac_b[2] * inv_Ib)
            W = W + wp.dot(jac_b, massInvJacB)
        rhs = pgs_rhs[i] - vel
        delta_lamb = rhs / (W + 1e-12)
        old_lamb = pgs_lambda[i]
        new_lamb = old_lamb + delta_lamb
        parent = pgs_parent_row[i]
        if parent >= 0:
            fric_lim_low = pgs_limits[i][0] * pgs_lambda[parent]
            fric_lim_upper = pgs_limits[i][1] * pgs_lambda[parent]
            new_lamb = wp.max(fric_lim_low, wp.min(fric_lim_upper, new_lamb))
        else:
            lower = pgs_limits[i][0]
            upper = pgs_limits[i][1]
            new_lamb = wp.max(lower, wp.min(upper, new_lamb))
        apply_lamb = new_lamb - old_lamb
        pgs_lambda[i] = new_lamb
        if apply_lamb != 0.0:
            deltaA = massInvJacA * apply_lamb
            V[aid] = V[aid] + wp.vec2(deltaA[0], deltaA[1])
            RotV[aid] = RotV[aid] + deltaA[2]
            if has_b:
                deltaB = massInvJacB * apply_lamb
                V[bid] = V[bid] - wp.vec2(deltaB[0], deltaB[1])
                RotV[bid] = RotV[bid] - deltaB[2]


@wp.kernel
def _solve_pgs_kernel(
    pgs_iters: int,
    numConstraints: wp.array(dtype=int),
    max_constraints: int,
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=float),
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    """Serial PGS — single thread loops all constraints forward then backward."""
    tid = wp.tid()
    if tid != 0:
        return
    n_constraints = int(wp.min(numConstraints[0], max_constraints))
    for _iter in range(pgs_iters):
        for i in range(n_constraints):
            _solve_pgs_single(
                i, V, RotV, mass, inertia, pgs_bodypair, pgs_Jac_a, pgs_Jac_b,
                pgs_rhs, pgs_limits, pgs_lambda, pgs_parent_row,
            )
        for i in range(n_constraints):
            k = n_constraints - 1 - i
            _solve_pgs_single(
                k, V, RotV, mass, inertia, pgs_bodypair, pgs_Jac_a, pgs_Jac_b,
                pgs_rhs, pgs_limits, pgs_lambda, pgs_parent_row,
            )


@wp.kernel
def _assemble_joint_constraints_wp(
    dt: float,
    numAnchors: int,
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
    tid = wp.tid()
    if tid != 0:
        return
    for j_idx in range(numAnchors):
        assemble_single_joint_rows(
            dt, j_idx, max_constraints,
            joint_type, joint_id_a, joint_id_b, joint_params,
            joint_has_motor, joint_motor_target_mode, joint_motor_target_vel,
            joint_q0_rel_inv, joint_axis, joint_l1, joint_l2,
            rigidParams, quat, quat_initial, RotV,
            numConstraints, pgs_bodypair, pgs_Jac_a, pgs_Jac_b,
            pgs_rhs, pgs_limits, pgs_lambda, pgs_parent_row,
        )


@wp.kernel
def _precompute_rigid_transforms_wp(
    num_total: int,
    quat: wp.array(dtype=float),
    visual_angle: wp.array(dtype=float),
    inertia: wp.array(dtype=float),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    cached_inertia_inv_2d: wp.array(dtype=float),
):
    i = wp.tid()
    if i >= num_total:
        return
    cached_rotation_matrix[i] = cal2DRotationMat(quat[i] + visual_angle[i])
    I = inertia[i]
    if I > 0.0:
        cached_inertia_inv_2d[i] = 1.0 / I
    else:
        cached_inertia_inv_2d[i] = 1.0 / 1e-6


# ---------------------------------------------------------------------------
# Critical-path kernels for rigid substep / AABB (ball + ground and beyond)
# ---------------------------------------------------------------------------

RIGID_TYPE_BALL = 0b00001
RIGID_TYPE_BOX = 0b00010
RIGID_TYPE_CAPSULE = 0b01000
RIGID_TYPE_MESH = 0b10000

CONTACT_BALLBALL = 0b00001
CONTACT_BOXBALL = 0b00011
CONTACT_CAPSULEBALL = 0b01001
CONTACT_BOXBOX = 0b00010
CONTACT_CAPSULEBOX = 0b01010
CONTACT_CAPSULECAPSULE = 0b01000

BC_ATYPE = 0b000000100
BC_FORCETYPE = 0b000001000
BC_GRAVITY = 0b000010000
BC_ROTATYPE = 0b010000000
BC_TORQUETYPE = 0b100000000
BC_VTYPE = 0b000000010
BC_UTYPE = 0b000000001
BC_RTYPE = 0b000100000
BC_ROTVTYPE = 0b001000000


@wp.func
def _mask_allows_pair_func(
    idx_a: int,
    idx_b: int,
    category_bits: wp.array(dtype=wp.uint32),
    collide_bits: wp.array(dtype=wp.uint32),
):
    allow_ab = (collide_bits[idx_a] & category_bits[idx_b]) != wp.uint32(0)
    allow_ba = (collide_bits[idx_b] & category_bits[idx_a]) != wp.uint32(0)
    return 1 if (allow_ab and allow_ba) else 0


@wp.kernel
def _classify_collision_pairs_wp(
    pairs: wp.array(dtype=wp.vec2i),
    num_pairs: int,
    num_rigids: int,
    max_collision_pairs: int,
    max_ground_pairs: int,
    domainToRigid: wp.array(dtype=int),
    rigidDomainIds: wp.array(dtype=wp.vec3i),
    compound_count: wp.array(dtype=int),
    rigid_env_id: wp.array(dtype=int),
    category_bits: wp.array(dtype=wp.uint32),
    collide_bits: wp.array(dtype=wp.uint32),
    num_primitive_pairs: wp.array(dtype=int),
    num_ball_ball_pairs: wp.array(dtype=int),
    num_box_box_pairs: wp.array(dtype=int),
    num_box_ball_pairs: wp.array(dtype=int),
    num_seg_point_pairs: wp.array(dtype=int),
    num_seg_ball_pairs: wp.array(dtype=int),
    num_seg_seg_pairs: wp.array(dtype=int),
    num_mesh_pairs: wp.array(dtype=int),
    num_mixed_pairs: wp.array(dtype=int),
    num_groundprim_pairs: wp.array(dtype=int),
    num_groundmesh_pairs: wp.array(dtype=int),
    primitive_pairs_buffer: wp.array(dtype=wp.vec2i),
    ball_ball_pairs_buffer: wp.array(dtype=wp.vec2i),
    box_box_pairs_buffer: wp.array(dtype=wp.vec2i),
    box_ball_pairs_buffer: wp.array(dtype=wp.vec2i),
    seg_point_pairs_buffer: wp.array(dtype=wp.vec2i),
    seg_ball_pairs_buffer: wp.array(dtype=wp.vec2i),
    seg_seg_pairs_buffer: wp.array(dtype=wp.vec2i),
    mesh_pairs_buffer: wp.array(dtype=wp.vec2i),
    mixed_pairs_buffer: wp.array(dtype=wp.vec2i),
    groundprim_pairs_buffer: wp.array(dtype=wp.vec2i),
    groundmesh_pairs_buffer: wp.array(dtype=wp.vec2i),
):
    tid = wp.tid()
    if tid != 0:
        return

    num_primitive_pairs[0] = 0
    num_ball_ball_pairs[0] = 0
    num_box_box_pairs[0] = 0
    num_box_ball_pairs[0] = 0
    num_seg_point_pairs[0] = 0
    num_seg_ball_pairs[0] = 0
    num_seg_seg_pairs[0] = 0
    num_mesh_pairs[0] = 0
    num_mixed_pairs[0] = 0
    num_groundprim_pairs[0] = 0
    num_groundmesh_pairs[0] = 0

    for i in range(num_pairs):
        domain_a = pairs[i][0]
        domain_b = pairs[i][1]
        rigid_a = domainToRigid[domain_a]
        rigid_b = domainToRigid[domain_b]

        if rigid_a < 0 or rigid_b < 0:
            continue
        if _mask_allows_pair_func(rigid_a, rigid_b, category_bits, collide_bits) == 0:
            continue

        is_anal_a = rigid_a >= num_rigids
        is_anal_b = rigid_b >= num_rigids
        type_a = rigidDomainIds[rigid_a][1]
        type_b = rigidDomainIds[rigid_b][1]
        is_mesh_a = (type_a == RIGID_TYPE_MESH) and (compound_count[rigid_a] == 0)
        is_mesh_b = (type_b == RIGID_TYPE_MESH) and (compound_count[rigid_b] == 0)

        if is_anal_a or is_anal_b:
            if is_anal_a and not is_anal_b:
                if is_mesh_b:
                    idx = int(wp.atomic_add(num_groundmesh_pairs, 0, 1))
                    if idx < max_ground_pairs:
                        groundmesh_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)
                else:
                    idx = int(wp.atomic_add(num_groundprim_pairs, 0, 1))
                    if idx < max_ground_pairs:
                        groundprim_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)
            elif is_anal_b and not is_anal_a:
                if is_mesh_a:
                    idx = int(wp.atomic_add(num_groundmesh_pairs, 0, 1))
                    if idx < max_ground_pairs:
                        groundmesh_pairs_buffer[idx] = wp.vec2i(rigid_b, rigid_a)
                else:
                    idx = int(wp.atomic_add(num_groundprim_pairs, 0, 1))
                    if idx < max_ground_pairs:
                        groundprim_pairs_buffer[idx] = wp.vec2i(rigid_b, rigid_a)
        else:
            consider = 1
            env_a = rigid_env_id[rigid_a]
            env_b = rigid_env_id[rigid_b]
            if env_a >= 0 and env_b >= 0 and env_a != env_b:
                consider = 0

            if consider == 1:
                if is_mesh_a and is_mesh_b:
                    idx = int(wp.atomic_add(num_mesh_pairs, 0, 1))
                    if idx < max_collision_pairs:
                        mesh_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)
                elif is_mesh_a or is_mesh_b:
                    idx = int(wp.atomic_add(num_mixed_pairs, 0, 1))
                    if idx < max_collision_pairs:
                        mixed_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)
                else:
                    contact_type = type_a | type_b
                    if contact_type == CONTACT_BALLBALL:
                        idx = int(wp.atomic_add(num_ball_ball_pairs, 0, 1))
                        if idx < max_collision_pairs:
                            ball_ball_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)
                    elif contact_type == CONTACT_BOXBOX:
                        idx = int(wp.atomic_add(num_box_box_pairs, 0, 1))
                        if idx < max_collision_pairs:
                            box_box_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)
                    elif contact_type == CONTACT_BOXBALL:
                        idx = int(wp.atomic_add(num_box_ball_pairs, 0, 1))
                        if idx < max_collision_pairs:
                            r_a = rigid_a
                            r_b = rigid_b
                            if type_a == RIGID_TYPE_BALL:
                                r_a = rigid_b
                                r_b = rigid_a
                            box_ball_pairs_buffer[idx] = wp.vec2i(r_a, r_b)
                    elif contact_type == CONTACT_CAPSULEBOX:
                        idx = int(wp.atomic_add(num_seg_point_pairs, 0, 1))
                        if idx < max_collision_pairs:
                            r_a = rigid_a
                            r_b = rigid_b
                            if type_a == RIGID_TYPE_BOX:
                                r_a = rigid_b
                                r_b = rigid_a
                            seg_point_pairs_buffer[idx] = wp.vec2i(r_a, r_b)
                    elif contact_type == CONTACT_CAPSULEBALL:
                        idx = int(wp.atomic_add(num_seg_ball_pairs, 0, 1))
                        if idx < max_collision_pairs:
                            r_a = rigid_a
                            r_b = rigid_b
                            if type_a == RIGID_TYPE_BALL:
                                r_a = rigid_b
                                r_b = rigid_a
                            seg_ball_pairs_buffer[idx] = wp.vec2i(r_a, r_b)
                    elif contact_type == CONTACT_CAPSULECAPSULE:
                        idx = int(wp.atomic_add(num_seg_seg_pairs, 0, 1))
                        if idx < max_collision_pairs:
                            seg_seg_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)

                    idx = int(wp.atomic_add(num_primitive_pairs, 0, 1))
                    if idx < max_collision_pairs:
                        primitive_pairs_buffer[idx] = wp.vec2i(rigid_a, rigid_b)


@wp.func
def _cache_contact_func(
    aid: int,
    bid: int,
    cpoint: wp.vec2,
    normal: wp.vec2,
    depth: float,
    max_contacts: int,
    restitution_velocity_threshold: float,
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    contactParams: wp.array(dtype=wp.vec2),
    rigid_env_id: wp.array(dtype=int),
    num_contacts: wp.array(dtype=int),
    contact_rigid_a: wp.array(dtype=int),
    contact_rigid_b: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec2),
    contact_normal: wp.array(dtype=wp.vec2),
    contact_depth: wp.array(dtype=float),
    contact_bounce_vel: wp.array(dtype=float),
    contact_tangent1: wp.array(dtype=wp.vec2),
    contact_count_per_rigid: wp.array(dtype=int),
    contact_env_count: wp.array(dtype=int),
    contact_env_idx: wp.array(dtype=int),
    max_envs_alloc: int,
    max_cc_per_env: int,
):
    idx = int(wp.atomic_add(num_contacts, 0, 1))
    if idx < max_contacts:
        contact_rigid_a[idx] = aid
        contact_rigid_b[idx] = bid
        contact_point[idx] = cpoint
        contact_normal[idx] = normal
        contact_depth[idx] = depth
        wp.atomic_add(contact_count_per_rigid, aid, 1)
        wp.atomic_add(contact_count_per_rigid, bid, 1)
        ra = cpoint - rigidParams[aid, 0]
        rb = cpoint - rigidParams[bid, 0]
        e = 0.5 * (contactParams[aid][1] + contactParams[bid][1])
        va = V[aid] + wp.vec2(-ra[1], ra[0]) * RotV[aid]
        vb = V[bid] + wp.vec2(-rb[1], rb[0]) * RotV[bid]
        vn_pre = wp.dot(va - vb, normal)
        if vn_pre < -restitution_velocity_threshold:
            contact_bounce_vel[idx] = -e * vn_pre
        else:
            contact_bounce_vel[idx] = 0.0
        contact_tangent1[idx] = wp.vec2(-normal[1], normal[0])
        env_id = wp.max(rigid_env_id[aid], 0)
        if env_id < max_envs_alloc:
            local_i = int(wp.atomic_add(contact_env_count, env_id, 1))
            if local_i < max_cc_per_env:
                contact_env_idx[env_id * max_cc_per_env + local_i] = idx


@wp.kernel
def _detect_primitive_contacts_wp(
    num_pairs: int,
    max_contacts: int,
    restitution_velocity_threshold: float,
    max_envs_alloc: int,
    max_cc_per_env: int,
    primitive_pairs_buffer: wp.array(dtype=wp.vec2i),
    rigidDomainIds: wp.array(dtype=wp.vec3i),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    radius: wp.array(dtype=float),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    contactParams: wp.array(dtype=wp.vec2),
    rigid_env_id: wp.array(dtype=int),
    num_contacts: wp.array(dtype=int),
    contact_rigid_a: wp.array(dtype=int),
    contact_rigid_b: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec2),
    contact_normal: wp.array(dtype=wp.vec2),
    contact_depth: wp.array(dtype=float),
    contact_bounce_vel: wp.array(dtype=float),
    contact_tangent1: wp.array(dtype=wp.vec2),
    contact_count_per_rigid: wp.array(dtype=int),
    contact_env_count: wp.array(dtype=int),
    contact_env_idx: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid != 0:
        return

    for i in range(num_pairs):
        pair = primitive_pairs_buffer[i]
        rigid_a = pair[0]
        rigid_b = pair[1]
        type_a = rigidDomainIds[rigid_a][1]
        type_b = rigidDomainIds[rigid_b][1]
        contact_type = type_a | type_b

        if contact_type == CONTACT_BALLBALL:
            rad = radius[rigid_a] + radius[rigid_b]
            p = rigidParams[rigid_a, 0] - rigidParams[rigid_b, 0]
            l = wp.length(p)
            if l < rad and l > 1e-12:
                n = p / l
                cpoint_mid = (rigidParams[rigid_a, 0] + rigidParams[rigid_b, 0]) * 0.5
                _cache_contact_func(
                    rigid_a,
                    rigid_b,
                    cpoint_mid,
                    n,
                    l - rad,
                    max_contacts,
                    restitution_velocity_threshold,
                    rigidParams,
                    V,
                    RotV,
                    contactParams,
                    rigid_env_id,
                    num_contacts,
                    contact_rigid_a,
                    contact_rigid_b,
                    contact_point,
                    contact_normal,
                    contact_depth,
                    contact_bounce_vel,
                    contact_tangent1,
                    contact_count_per_rigid,
                    contact_env_count,
                    contact_env_idx,
                    max_envs_alloc,
                    max_cc_per_env,
                )

        elif contact_type == CONTACT_BOXBOX:
            hit, penetration, normal_ij, cpoint = obb2d_contact_quad_vs_quad(
                rigidParams[rigid_a, 0],
                rigidParams[rigid_a, 1] * 0.5,
                cached_rotation_matrix[rigid_a],
                rigidParams[rigid_b, 0],
                rigidParams[rigid_b, 1] * 0.5,
                cached_rotation_matrix[rigid_b],
            )
            if hit == 1:
                _cache_contact_func(
                    rigid_a,
                    rigid_b,
                    cpoint,
                    -normal_ij,
                    penetration,
                    max_contacts,
                    restitution_velocity_threshold,
                    rigidParams,
                    V,
                    RotV,
                    contactParams,
                    rigid_env_id,
                    num_contacts,
                    contact_rigid_a,
                    contact_rigid_b,
                    contact_point,
                    contact_normal,
                    contact_depth,
                    contact_bounce_vel,
                    contact_tangent1,
                    contact_count_per_rigid,
                    contact_env_count,
                    contact_env_idx,
                    max_envs_alloc,
                    max_cc_per_env,
                )

        elif contact_type == CONTACT_BOXBALL:
            box_idx = rigid_a
            ball_idx = rigid_b
            if type_a == RIGID_TYPE_BALL:
                box_idx = rigid_b
                ball_idx = rigid_a
            pos = rigidParams[ball_idx, 0]
            l, n, _c = detectPointToPrimitive(
                pos,
                rigidDomainIds[box_idx][1],
                rigidParams[box_idx, 0],
                rigidParams[box_idx, 1],
                cached_rotation_matrix[box_idx],
                radius[box_idx],
            )
            l = l - radius[ball_idx]
            if l < 0.0:
                nlen = wp.length(n)
                if nlen > 1e-9:
                    n = n / nlen
                else:
                    diff = pos - rigidParams[box_idx, 0]
                    dlen = wp.length(diff)
                    n = diff / (dlen + 1e-9)
                cpoint = pos - n * radius[ball_idx]
                _cache_contact_func(
                    ball_idx,
                    box_idx,
                    cpoint,
                    n,
                    l,
                    max_contacts,
                    restitution_velocity_threshold,
                    rigidParams,
                    V,
                    RotV,
                    contactParams,
                    rigid_env_id,
                    num_contacts,
                    contact_rigid_a,
                    contact_rigid_b,
                    contact_point,
                    contact_normal,
                    contact_depth,
                    contact_bounce_vel,
                    contact_tangent1,
                    contact_count_per_rigid,
                    contact_env_count,
                    contact_env_idx,
                    max_envs_alloc,
                    max_cc_per_env,
                )

        elif contact_type == CONTACT_CAPSULECAPSULE:
            center1 = rigidParams[rigid_a, 0]
            lc1 = cached_rotation_matrix[rigid_a] @ rigidParams[rigid_a, 1] + center1
            uc1 = center1 * 2.0 - lc1
            r1 = radius[rigid_a]
            center2 = rigidParams[rigid_b, 0]
            lc2 = cached_rotation_matrix[rigid_b] @ rigidParams[rigid_b, 1] + center2
            uc2 = center2 * 2.0 - lc2
            r2 = radius[rigid_b]
            p, q, _t1, _t2 = calMinDisSegment2Segment(lc1, uc1, lc2, uc2)
            pq = q - p
            dis = wp.length(pq)
            normal = wp.vec2(1.0, 0.0)
            if dis > 1e-9:
                normal = pq / dis
            penetration = dis - (r1 + r2)
            if penetration < 0.0:
                cpoint_mid = (p + q) * 0.5
                _cache_contact_func(
                    rigid_a,
                    rigid_b,
                    cpoint_mid,
                    -normal,
                    penetration,
                    max_contacts,
                    restitution_velocity_threshold,
                    rigidParams,
                    V,
                    RotV,
                    contactParams,
                    rigid_env_id,
                    num_contacts,
                    contact_rigid_a,
                    contact_rigid_b,
                    contact_point,
                    contact_normal,
                    contact_depth,
                    contact_bounce_vel,
                    contact_tangent1,
                    contact_count_per_rigid,
                    contact_env_count,
                    contact_env_idx,
                    max_envs_alloc,
                    max_cc_per_env,
                )


@wp.func
def _cache_ground_contact_func(
    rid: int,
    cpoint: wp.vec2,
    normal: wp.vec2,
    ground_vel: wp.vec2,
    depth: float,
    max_ground_contacts: int,
    restitution_velocity_threshold: float,
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    contactParams: wp.array(dtype=wp.vec2),
    rigid_env_id: wp.array(dtype=int),
    num_ground_contacts: wp.array(dtype=int),
    ground_contact_rigid: wp.array(dtype=int),
    ground_contact_point: wp.array(dtype=wp.vec2),
    ground_contact_normal: wp.array(dtype=wp.vec2),
    ground_contact_vel: wp.array(dtype=wp.vec2),
    ground_contact_depth: wp.array(dtype=float),
    ground_contact_bounce_vel: wp.array(dtype=float),
    ground_contact_tangent1: wp.array(dtype=wp.vec2),
    ground_contact_env_count: wp.array(dtype=int),
    ground_contact_env_idx: wp.array(dtype=int),
    max_envs_alloc: int,
    max_gc_per_env: int,
):
    idx = int(wp.atomic_add(num_ground_contacts, 0, 1))
    if idx < max_ground_contacts:
        ground_contact_rigid[idx] = rid
        ground_contact_point[idx] = cpoint
        ground_contact_normal[idx] = normal
        ground_contact_vel[idx] = ground_vel
        ground_contact_depth[idx] = depth
        lr = cpoint - rigidParams[rid, 0]
        e = contactParams[rid][1]
        tlr = wp.vec2(-lr[1], lr[0])
        v_point = V[rid] + tlr * RotV[rid]
        vn_pre = wp.dot(v_point - ground_vel, normal)
        if vn_pre < -restitution_velocity_threshold:
            ground_contact_bounce_vel[idx] = -e * vn_pre
        else:
            ground_contact_bounce_vel[idx] = 0.0
        ground_contact_tangent1[idx] = wp.vec2(-normal[1], normal[0])
        env_id = wp.max(rigid_env_id[rid], 0)
        if env_id < max_envs_alloc:
            local_i = int(wp.atomic_add(ground_contact_env_count, env_id, 1))
            if local_i < max_gc_per_env:
                ground_contact_env_idx[env_id * max_gc_per_env + local_i] = idx


@wp.func
def _write_primitive_aabb(
    rigid_id: int,
    rigidDomainIds: wp.array(dtype=wp.vec3i),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    radius: wp.array(dtype=float),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
):
    rigid_type = rigidDomainIds[rigid_id][1]
    if rigid_type == RIGID_TYPE_MESH:
        return
    center = rigidParams[rigid_id, 0]
    rotMat = cached_rotation_matrix[rigid_id]
    lb = wp.vec2(0.0, 0.0)
    ub = wp.vec2(0.0, 0.0)
    primary = rigidParams[rigid_id, 1]
    r = radius[rigid_id]
    if rigid_type == RIGID_TYPE_BALL:
        lb, ub = getBallBBox(center, r, rotMat)
    elif rigid_type == RIGID_TYPE_BOX:
        info = getBoxBBox(center, primary, rotMat)
        lb = info[0]
        ub = info[1]
    elif rigid_type == RIGID_TYPE_CAPSULE:
        lb, ub = getCapsuleBBox(center, primary, r, rotMat)
    domain_idx = rigidDomainIds[rigid_id][0]
    aabb[domain_idx, 0] = lb
    aabb[domain_idx, 1] = ub


@wp.kernel
def _update_bbox_wp(
    num_rigids: int,
    num_analytical: int,
    moving_analytical: int,
    rigidDomainIds: wp.array(dtype=wp.vec3i),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    radius: wp.array(dtype=float),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
):
    tid = wp.tid()
    if tid != 0:
        return
    for i in range(num_rigids):
        _write_primitive_aabb(i, rigidDomainIds, rigidParams, radius, cached_rotation_matrix, aabb)

    if moving_analytical == 1:
        buffer = 0.1
        large_span = 100.0
        for i in range(num_analytical):
            idx = i + num_rigids
            normal_local = rigidParams[idx, 1]
            normal_world = cached_rotation_matrix[idx] @ normal_local
            p = rigidParams[idx, 0]
            tangent = wp.vec2(-normal_world[1], normal_world[0])
            lo_raw = p - tangent * large_span - normal_world * buffer
            hi_raw = p + tangent * large_span + normal_world * buffer
            lb = wp.vec2(wp.min(lo_raw[0], hi_raw[0]), wp.min(lo_raw[1], hi_raw[1]))
            ub = wp.vec2(wp.max(lo_raw[0], hi_raw[0]), wp.max(lo_raw[1], hi_raw[1]))
            domain_idx = rigidDomainIds[idx][0]
            aabb[domain_idx, 0] = lb
            aabb[domain_idx, 1] = ub


@wp.kernel
def _rigid_step_wp(
    dt: float,
    damping: float,
    num_rigids: int,
    num_analytical: int,
    bcNodes: wp.array(dtype=int),
    bcGValues: wp.array(dtype=wp.vec2),
    bcTValues: wp.array(dtype=wp.vec2),
    bcRValues: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=float),
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    accumulated_impulse: wp.array(dtype=wp.vec2),
    accumulated_rotational_impulse: wp.array(dtype=float),
    quat: wp.array(dtype=float),
    visual_angle: wp.array(dtype=float),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    cached_inertia_inv_2d: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid != 0:
        return

    n_total = num_rigids + num_analytical
    for i in range(n_total):
        cached_rotation_matrix[i] = cal2DRotationMat(quat[i] + visual_angle[i])
        I = inertia[i]
        if I > 0.0:
            cached_inertia_inv_2d[i] = 1.0 / I
        else:
            cached_inertia_inv_2d[i] = 1.0 / 1e-6

    for i in range(num_rigids):
        bc_type = bcNodes[i]
        if (bc_type & BC_ATYPE) != 0:
            accumulated_impulse[i] = wp.vec2(0.0, 0.0)
        else:
            if (bc_type & BC_GRAVITY) != 0:
                accumulated_impulse[i] = accumulated_impulse[i] + mass[i] * bcGValues[i] * dt
            if (bc_type & BC_FORCETYPE) != 0:
                accumulated_impulse[i] = accumulated_impulse[i] + bcTValues[i] * dt
        if (bc_type & BC_ROTATYPE) != 0:
            accumulated_rotational_impulse[i] = 0.0
        elif (bc_type & BC_TORQUETYPE) != 0:
            accumulated_rotational_impulse[i] = accumulated_rotational_impulse[i] + bcRValues[i] * dt

        V[i] = V[i] + accumulated_impulse[i] / mass[i]
        RotV[i] = RotV[i] + accumulated_rotational_impulse[i] / (inertia[i] + 1e-6)
        damp_factor = wp.max(0.0, 1.0 - damping * dt)
        V[i] = V[i] * damp_factor
        RotV[i] = RotV[i] * damp_factor

    for i in range(n_total):
        bc_type = bcNodes[i]
        if (bc_type & BC_ATYPE) != 0:
            V[i] = V[i] + bcTValues[i] * dt
        if (bc_type & BC_ROTATYPE) != 0:
            RotV[i] = RotV[i] + bcRValues[i] * dt


@wp.kernel
def _update_u_and_bbox_wp(
    dt: float,
    update_bbox: int,
    num_rigids: int,
    num_analytical: int,
    moving_analytical: int,
    bcNodes: wp.array(dtype=int),
    bcTValues: wp.array(dtype=wp.vec2),
    bcRValues: wp.array(dtype=float),
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    U: wp.array(dtype=wp.vec2),
    quat: wp.array(dtype=float),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    accumulated_impulse: wp.array(dtype=wp.vec2),
    accumulated_rotational_impulse: wp.array(dtype=float),
    rigidDomainIds: wp.array(dtype=wp.vec3i),
    radius: wp.array(dtype=float),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
):
    tid = wp.tid()
    if tid != 0:
        return

    n_total = num_rigids + num_analytical
    for i in range(n_total):
        bc_type = bcNodes[i]
        if (bc_type & BC_VTYPE) != 0:
            V[i] = bcTValues[i]
        elif (bc_type & BC_UTYPE) != 0:
            V[i] = wp.vec2(0.0, 0.0)
        elif (bc_type & BC_RTYPE) != 0:
            V[i] = wp.vec2(0.0, 0.0)
            RotV[i] = 0.0
        if (bc_type & BC_ROTVTYPE) != 0:
            RotV[i] = bcRValues[i]

        du = V[i] * dt
        U[i] = U[i] + du
        rigidParams[i, 0] = rigidParams[i, 0] + du
        quat[i] = quat[i] + RotV[i] * dt
        if quat[i] > 3.141592653589793:
            quat[i] = quat[i] - 2.0 * 3.141592653589793
        elif quat[i] < -3.141592653589793:
            quat[i] = quat[i] + 2.0 * 3.141592653589793

    for i in range(n_total):
        accumulated_impulse[i] = wp.vec2(0.0, 0.0)
        accumulated_rotational_impulse[i] = 0.0

    if update_bbox == 1:
        for i in range(num_rigids):
            _write_primitive_aabb(i, rigidDomainIds, rigidParams, radius, cached_rotation_matrix, aabb)
        if moving_analytical == 1:
            buffer = 0.1
            large_span = 100.0
            for i in range(num_analytical):
                idx = i + num_rigids
                normal_local = rigidParams[idx, 1]
                normal_world = cached_rotation_matrix[idx] @ normal_local
                p = rigidParams[idx, 0]
                tangent = wp.vec2(-normal_world[1], normal_world[0])
                lo_raw = p - tangent * large_span - normal_world * buffer
                hi_raw = p + tangent * large_span + normal_world * buffer
                lb = wp.vec2(wp.min(lo_raw[0], hi_raw[0]), wp.min(lo_raw[1], hi_raw[1]))
                ub = wp.vec2(wp.max(lo_raw[0], hi_raw[0]), wp.max(lo_raw[1], hi_raw[1]))
                domain_idx = rigidDomainIds[idx][0]
                aabb[domain_idx, 0] = lb
                aabb[domain_idx, 1] = ub


@wp.kernel
def _generate_ground_pairs_wp(
    num_rigids: int,
    num_analytical: int,
    max_ground_pairs: int,
    rigidDomainIds: wp.array(dtype=wp.vec3i),
    compound_count: wp.array(dtype=int),
    category_bits: wp.array(dtype=wp.uint32),
    collide_bits: wp.array(dtype=wp.uint32),
    num_primitive_pairs: wp.array(dtype=int),
    num_ball_ball_pairs: wp.array(dtype=int),
    num_box_box_pairs: wp.array(dtype=int),
    num_box_ball_pairs: wp.array(dtype=int),
    num_seg_point_pairs: wp.array(dtype=int),
    num_seg_ball_pairs: wp.array(dtype=int),
    num_seg_seg_pairs: wp.array(dtype=int),
    num_mesh_pairs: wp.array(dtype=int),
    num_mixed_pairs: wp.array(dtype=int),
    num_groundprim_pairs: wp.array(dtype=int),
    num_groundmesh_pairs: wp.array(dtype=int),
    groundprim_pairs_buffer: wp.array(dtype=wp.vec2i),
    groundmesh_pairs_buffer: wp.array(dtype=wp.vec2i),
):
    tid = wp.tid()
    if tid != 0:
        return
    num_primitive_pairs[0] = 0
    num_ball_ball_pairs[0] = 0
    num_box_box_pairs[0] = 0
    num_box_ball_pairs[0] = 0
    num_seg_point_pairs[0] = 0
    num_seg_ball_pairs[0] = 0
    num_seg_seg_pairs[0] = 0
    num_mesh_pairs[0] = 0
    num_mixed_pairs[0] = 0
    num_groundprim_pairs[0] = 0
    num_groundmesh_pairs[0] = 0

    for i in range(num_rigids):
        type_i = rigidDomainIds[i][1]
        is_mesh_i = (type_i == RIGID_TYPE_MESH) and (compound_count[i] == 0)
        for j in range(num_analytical):
            anal_idx = num_rigids + j
            if _mask_allows_pair_func(anal_idx, i, category_bits, collide_bits) == 0:
                continue
            if is_mesh_i:
                idx = int(wp.atomic_add(num_groundmesh_pairs, 0, 1))
                if idx < max_ground_pairs:
                    groundmesh_pairs_buffer[idx] = wp.vec2i(anal_idx, i)
            else:
                idx = int(wp.atomic_add(num_groundprim_pairs, 0, 1))
                if idx < max_ground_pairs:
                    groundprim_pairs_buffer[idx] = wp.vec2i(anal_idx, i)


@wp.kernel
def _reset_contact_caches_wp(
    num_rigids: int,
    num_analytical: int,
    num_envs: int,
    num_contacts: wp.array(dtype=int),
    num_ground_contacts: wp.array(dtype=int),
    prev_num_contacts: wp.array(dtype=int),
    prev_num_ground_contacts: wp.array(dtype=int),
    numConstraints: wp.array(dtype=int),
    contact_count_per_rigid: wp.array(dtype=int),
    contact_env_count: wp.array(dtype=int),
    ground_contact_env_count: wp.array(dtype=int),
    contact_force: wp.array(dtype=wp.vec2),
    contact_pgs_indices: wp.array(dtype=wp.vec3i),
    contact_bounce_vel: wp.array(dtype=float),
    ground_contact_force: wp.array(dtype=wp.vec2),
    ground_contact_pgs_indices: wp.array(dtype=wp.vec3i),
    ground_contact_bounce_vel: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid != 0:
        return
    prev_nc = num_contacts[0]
    prev_ngc = num_ground_contacts[0]
    prev_num_contacts[0] = prev_nc
    prev_num_ground_contacts[0] = prev_ngc
    num_contacts[0] = 0
    num_ground_contacts[0] = 0
    numConstraints[0] = 0
    n_envs = wp.max(num_envs, 1)
    for i in range(n_envs):
        ground_contact_env_count[i] = 0
        contact_env_count[i] = 0
    total_nodes = num_rigids + num_analytical
    for rid in range(total_nodes):
        contact_count_per_rigid[rid] = 0
    for j in range(prev_nc):
        contact_force[j] = wp.vec2(0.0, 0.0)
        contact_pgs_indices[j] = wp.vec3i(-1, -1, -1)
        contact_bounce_vel[j] = 0.0
    for k in range(prev_ngc):
        ground_contact_force[k] = wp.vec2(0.0, 0.0)
        ground_contact_pgs_indices[k] = wp.vec3i(-1, -1, -1)
        ground_contact_bounce_vel[k] = 0.0


@wp.func
def _get_box_vertex_func(
    rigid_idx: int,
    v_idx: int,
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
):
    center = rigidParams[rigid_idx, 0]
    extent = rigidParams[rigid_idx, 1]
    sx = -1.0 if (v_idx == 0 or v_idx == 3) else 1.0
    sy = -1.0 if (v_idx == 0 or v_idx == 1) else 1.0
    local_pos = 0.5 * wp.vec2(sx * extent[0], sy * extent[1])
    return center + cached_rotation_matrix[rigid_idx] @ local_pos


@wp.kernel
def _detect_analytical_prim_contacts_wp(
    num_pairs: int,
    contact_margin: float,
    max_ground_contacts: int,
    restitution_velocity_threshold: float,
    use_aabb_early_out: wp.array(dtype=int),
    groundprim_pairs_buffer: wp.array(dtype=wp.vec2i),
    rigidDomainIds: wp.array(dtype=wp.vec3i),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    radius: wp.array(dtype=float),
    V: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    contactParams: wp.array(dtype=wp.vec2),
    rigid_env_id: wp.array(dtype=int),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
    compound_count: wp.array(dtype=int),
    compound_offset: wp.array(dtype=int),
    compound_local_pos: wp.array(dtype=wp.vec2),
    compound_radius: wp.array(dtype=float),
    num_ground_contacts: wp.array(dtype=int),
    ground_contact_rigid: wp.array(dtype=int),
    ground_contact_point: wp.array(dtype=wp.vec2),
    ground_contact_normal: wp.array(dtype=wp.vec2),
    ground_contact_vel: wp.array(dtype=wp.vec2),
    ground_contact_depth: wp.array(dtype=float),
    ground_contact_bounce_vel: wp.array(dtype=float),
    ground_contact_tangent1: wp.array(dtype=wp.vec2),
    ground_contact_env_count: wp.array(dtype=int),
    ground_contact_env_idx: wp.array(dtype=int),
    max_envs_alloc: int,
    max_gc_per_env: int,
):
    tid = wp.tid()
    if tid != 0:
        return
    for p in range(num_pairs):
        anal_idx = groundprim_pairs_buffer[p][0]
        rigid_idx = groundprim_pairs_buffer[p][1]
        planepoint = rigidParams[anal_idx, 0]
        normal = rigidParams[anal_idx, 1]
        anal_vel = V[anal_idx]
        run_narrow = True
        if use_aabb_early_out[0] == 1:
            domain_idx = rigidDomainIds[rigid_idx][0]
            bbox_min = aabb[domain_idx, 0]
            bbox_max = aabb[domain_idx, 1]
            support = wp.vec2(
                bbox_max[0] if normal[0] < 0.0 else bbox_min[0],
                bbox_max[1] if normal[1] < 0.0 else bbox_min[1],
            )
            min_dist = wp.dot(support - planepoint, normal)
            run_narrow = min_dist <= contact_margin
        if not run_narrow:
            continue

        n_sub = compound_count[rigid_idx]
        if n_sub > 0:
            base = compound_offset[rigid_idx]
            parent_center = rigidParams[rigid_idx, 0]
            R = cached_rotation_matrix[rigid_idx]
            for k in range(n_sub):
                local_p = compound_local_pos[base + k]
                r_sub = compound_radius[base + k]
                world_p = R @ local_p + parent_center
                d_sub, _, _ = detectPointToAnalyticalPlane(world_p, planepoint, normal)
                if d_sub < r_sub + contact_margin:
                    cpoint = world_p - normal * r_sub
                    depth = d_sub - r_sub
                    _cache_ground_contact_func(
                        rigid_idx, cpoint, normal, anal_vel, depth, max_ground_contacts,
                        restitution_velocity_threshold, rigidParams, V, RotV, contactParams,
                        rigid_env_id, num_ground_contacts, ground_contact_rigid,
                        ground_contact_point, ground_contact_normal, ground_contact_vel,
                        ground_contact_depth, ground_contact_bounce_vel, ground_contact_tangent1,
                        ground_contact_env_count, ground_contact_env_idx, max_envs_alloc, max_gc_per_env,
                    )
        else:
            rtype = rigidDomainIds[rigid_idx][1]
            if rtype == RIGID_TYPE_BALL:
                center = rigidParams[rigid_idx, 0]
                r = radius[rigid_idx]
                d, _, _ = detectPointToAnalyticalPlane(center, planepoint, normal)
                if d < r + contact_margin:
                    cpoint = center - normal * r
                    depth = d - r
                    _cache_ground_contact_func(
                        rigid_idx, cpoint, normal, anal_vel, depth, max_ground_contacts,
                        restitution_velocity_threshold, rigidParams, V, RotV, contactParams,
                        rigid_env_id, num_ground_contacts, ground_contact_rigid,
                        ground_contact_point, ground_contact_normal, ground_contact_vel,
                        ground_contact_depth, ground_contact_bounce_vel, ground_contact_tangent1,
                        ground_contact_env_count, ground_contact_env_idx, max_envs_alloc, max_gc_per_env,
                    )
            elif rtype == RIGID_TYPE_CAPSULE:
                center = rigidParams[rigid_idx, 0]
                lcdir = rigidParams[rigid_idx, 1]
                lc = cached_rotation_matrix[rigid_idx] @ lcdir + center
                uc = center * 2.0 - lc
                r = radius[rigid_idx]
                for ep in range(2):
                    test_p = lc if ep == 0 else uc
                    d_ep, _, _ = detectPointToAnalyticalPlane(test_p, planepoint, normal)
                    if d_ep < r + contact_margin:
                        cpoint = test_p - normal * r
                        depth = d_ep - r
                        _cache_ground_contact_func(
                            rigid_idx, cpoint, normal, anal_vel, depth, max_ground_contacts,
                            restitution_velocity_threshold, rigidParams, V, RotV, contactParams,
                            rigid_env_id, num_ground_contacts, ground_contact_rigid,
                            ground_contact_point, ground_contact_normal, ground_contact_vel,
                            ground_contact_depth, ground_contact_bounce_vel, ground_contact_tangent1,
                            ground_contact_env_count, ground_contact_env_idx, max_envs_alloc, max_gc_per_env,
                        )
            elif rtype == RIGID_TYPE_BOX:
                # Check all 4 box vertices against the plane (same as host detectAnalaytical2Rigid).
                for vi in range(4):
                    pos = _get_box_vertex_func(rigid_idx, vi, rigidParams, cached_rotation_matrix)
                    d_v, _, _ = detectPointToAnalyticalPlane(pos, planepoint, normal)
                    if d_v < contact_margin:
                        _cache_ground_contact_func(
                            rigid_idx, pos, normal, anal_vel, d_v, max_ground_contacts,
                            restitution_velocity_threshold, rigidParams, V, RotV, contactParams,
                            rigid_env_id, num_ground_contacts, ground_contact_rigid,
                            ground_contact_point, ground_contact_normal, ground_contact_vel,
                            ground_contact_depth, ground_contact_bounce_vel, ground_contact_tangent1,
                            ground_contact_env_count, ground_contact_env_idx, max_envs_alloc, max_gc_per_env,
                        )


@wp.func
def _add_pgs_row_func(
    aid: int,
    bid: int,
    jac_a: wp.vec3,
    jac_b: wp.vec3,
    rhs: float,
    lower: float,
    upper: float,
    parent_row: int,
    max_constraints: int,
    numConstraints: wp.array(dtype=int),
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    ci = int(wp.atomic_add(numConstraints, 0, 1))
    if ci < max_constraints:
        pgs_bodypair[ci] = wp.vec2i(aid, bid)
        pgs_Jac_a[ci] = jac_a
        pgs_Jac_b[ci] = jac_b
        pgs_rhs[ci] = rhs
        pgs_limits[ci] = wp.vec2(lower, upper)
        pgs_lambda[ci] = 0.0
        pgs_parent_row[ci] = parent_row
        return ci
    return -1


@wp.kernel
def _assemble_ground_contact_constraints_wp(
    dt: float,
    contact_erp: float,
    max_constraints: int,
    num_ground_contacts: wp.array(dtype=int),
    ground_contact_rigid: wp.array(dtype=int),
    ground_contact_point: wp.array(dtype=wp.vec2),
    ground_contact_normal: wp.array(dtype=wp.vec2),
    ground_contact_vel: wp.array(dtype=wp.vec2),
    ground_contact_depth: wp.array(dtype=float),
    ground_contact_bounce_vel: wp.array(dtype=float),
    ground_contact_tangent1: wp.array(dtype=wp.vec2),
    ground_contact_pgs_indices: wp.array(dtype=wp.vec3i),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    contactParams: wp.array(dtype=wp.vec2),
    numConstraints: wp.array(dtype=int),
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid != 0:
        return
    n = num_ground_contacts[0]
    for idx in range(n):
        rid = ground_contact_rigid[idx]
        cpoint = ground_contact_point[idx]
        normal = ground_contact_normal[idx]
        ground_vel = ground_contact_vel[idx]
        depth = ground_contact_depth[idx]
        lr = cpoint - rigidParams[rid, 0]
        mu = contactParams[rid][0]
        bounce_vel = ground_contact_bounce_vel[idx]
        bias_vel = 0.0
        if depth < 0.0 and contact_erp > 0.0:
            bias_vel = wp.max(contact_erp * depth / dt, -5.0)
        target_vel = bounce_vel
        if -bias_vel > bounce_vel:
            target_vel = -bias_vel
        jac_n = wp.vec3(normal[0], normal[1], vectorCrossProduct(lr, normal)[0])
        normal_row = _add_pgs_row_func(
            rid, -1, jac_n, wp.vec3(0.0, 0.0, 0.0), target_vel + wp.dot(normal, ground_vel),
            0.0, 1e10, -1, max_constraints, numConstraints, pgs_bodypair, pgs_Jac_a, pgs_Jac_b,
            pgs_rhs, pgs_limits, pgs_lambda, pgs_parent_row,
        )
        t1_row = -1
        if mu > 1e-12 and normal_row >= 0:
            t1 = ground_contact_tangent1[idx]
            jac_t1 = wp.vec3(t1[0], t1[1], vectorCrossProduct(lr, t1)[0])
            rhs_t1 = wp.dot(t1, ground_vel)
            t1_row = _add_pgs_row_func(
                rid, -1, jac_t1, wp.vec3(0.0, 0.0, 0.0), rhs_t1, -mu, mu, normal_row,
                max_constraints, numConstraints, pgs_bodypair, pgs_Jac_a, pgs_Jac_b,
                pgs_rhs, pgs_limits, pgs_lambda, pgs_parent_row,
            )
        ground_contact_pgs_indices[idx] = wp.vec3i(normal_row, t1_row, -1)


@wp.kernel
def _assemble_pair_contact_constraints_wp(
    dt: float,
    contact_erp: float,
    max_constraints: int,
    num_contacts: wp.array(dtype=int),
    contact_rigid_a: wp.array(dtype=int),
    contact_rigid_b: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec2),
    contact_normal: wp.array(dtype=wp.vec2),
    contact_depth: wp.array(dtype=float),
    contact_bounce_vel: wp.array(dtype=float),
    contact_tangent1: wp.array(dtype=wp.vec2),
    contact_pgs_indices: wp.array(dtype=wp.vec3i),
    rigidParams: wp.array(dtype=wp.vec2, ndim=2),
    contactParams: wp.array(dtype=wp.vec2),
    numConstraints: wp.array(dtype=int),
    pgs_bodypair: wp.array(dtype=wp.vec2i),
    pgs_Jac_a: wp.array(dtype=wp.vec3),
    pgs_Jac_b: wp.array(dtype=wp.vec3),
    pgs_rhs: wp.array(dtype=float),
    pgs_limits: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
    pgs_parent_row: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid != 0:
        return
    n = num_contacts[0]
    for idx in range(n):
        aid = contact_rigid_a[idx]
        bid = contact_rigid_b[idx]
        cpoint = contact_point[idx]
        normal = contact_normal[idx]
        depth = contact_depth[idx]
        ra = cpoint - rigidParams[aid, 0]
        rb = cpoint - rigidParams[bid, 0]
        mu = 0.5 * (contactParams[aid][0] + contactParams[bid][0])
        bounce_vel = contact_bounce_vel[idx]
        bias_vel = 0.0
        if depth < 0.0 and contact_erp > 0.0:
            bias_vel = wp.max(contact_erp * depth / dt, -5.0)
        target_vel = bounce_vel
        if -bias_vel > bounce_vel:
            target_vel = -bias_vel
        jac_na = wp.vec3(normal[0], normal[1], vectorCrossProduct(ra, normal)[0])
        jac_nb = wp.vec3(normal[0], normal[1], vectorCrossProduct(rb, normal)[0])
        normal_row = _add_pgs_row_func(
            aid, bid, jac_na, jac_nb, target_vel, 0.0, 1e10, -1,
            max_constraints, numConstraints, pgs_bodypair, pgs_Jac_a, pgs_Jac_b,
            pgs_rhs, pgs_limits, pgs_lambda, pgs_parent_row,
        )
        t1_row = -1
        if mu > 1e-12 and normal_row >= 0:
            t1 = contact_tangent1[idx]
            jac_t1a = wp.vec3(t1[0], t1[1], vectorCrossProduct(ra, t1)[0])
            jac_t1b = wp.vec3(t1[0], t1[1], vectorCrossProduct(rb, t1)[0])
            t1_row = _add_pgs_row_func(
                aid, bid, jac_t1a, jac_t1b, 0.0, -mu, mu, normal_row,
                max_constraints, numConstraints, pgs_bodypair, pgs_Jac_a, pgs_Jac_b,
                pgs_rhs, pgs_limits, pgs_lambda, pgs_parent_row,
            )
        contact_pgs_indices[idx] = wp.vec3i(normal_row, t1_row, -1)


@wp.kernel
def _compute_contact_forces_wp(
    dt: float,
    num_ground_contacts: wp.array(dtype=int),
    num_contacts: wp.array(dtype=int),
    ground_contact_pgs_indices: wp.array(dtype=wp.vec3i),
    ground_contact_normal: wp.array(dtype=wp.vec2),
    ground_contact_tangent1: wp.array(dtype=wp.vec2),
    ground_contact_force: wp.array(dtype=wp.vec2),
    contact_pgs_indices: wp.array(dtype=wp.vec3i),
    contact_normal: wp.array(dtype=wp.vec2),
    contact_tangent1: wp.array(dtype=wp.vec2),
    contact_force: wp.array(dtype=wp.vec2),
    pgs_lambda: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid != 0:
        return
    dt_inv = 1.0 / (dt + 1e-12)
    ng = num_ground_contacts[0]
    for idx in range(ng):
        pgs_indices = ground_contact_pgs_indices[idx]
        lambda_n = pgs_lambda[pgs_indices[0]] if pgs_indices[0] >= 0 else 0.0
        lambda_t1 = pgs_lambda[pgs_indices[1]] if pgs_indices[1] >= 0 else 0.0
        force = lambda_n * ground_contact_normal[idx] + lambda_t1 * ground_contact_tangent1[idx]
        ground_contact_force[idx] = force * dt_inv
    nc = num_contacts[0]
    for idx in range(nc):
        pgs_indices = contact_pgs_indices[idx]
        lambda_n = pgs_lambda[pgs_indices[0]] if pgs_indices[0] >= 0 else 0.0
        lambda_t1 = pgs_lambda[pgs_indices[1]] if pgs_indices[1] >= 0 else 0.0
        force = lambda_n * contact_normal[idx] + lambda_t1 * contact_tangent1[idx]
        contact_force[idx] = force * dt_inv


class RigidManager:
    def __init__(self, d, domains, joints, bvh=None, skip_spatial_hash=False, considerRigidRigidContact=True, use_pd=0):
        """Initialize RigidManager and allocate Warp arrays for rigids and state.

        Args:
            d: Spatial dimension (must be 2)
            domains: List of domain objects
            joints: List of joint objects
            bvh: Shared BVH instance from ExplicitLoop (if None, creates local one for standalone use)
            skip_spatial_hash: If True, skip SpatialHashManager creation (saves ~8GB VRAM)
            considerRigidRigidContact: If True, consider rigid-rigid contact
            use_pd: 0 = no PD, 1 = velocity PD, 2 = torque PD
        """

        assert d == 2, "RigidManager is 2D-only"

        self.skip_spatial_hash = skip_spatial_hash
        self.use_pd = use_pd
        self.considerRigidRigidContact = considerRigidRigidContact
        # --- OPTIMIZATION: Pre-scan domains to allocate only needed memory ---
        count_rigid = 0
        count_anal = 0
        count_mesh_rigids = 0
        count_mesh_nodes = 0
        count_mesh_elems = 0
        count_compound_shapes = 0

        for dom in domains:
            if dom.type == DomainType.RIGID:
                count_rigid += 1
                if dom.rigid.rtype == RigidType.MESH:
                    count_mesh_rigids += 1
                    count_mesh_nodes += dom.rigid.mesh.numBoundNodes
                    count_mesh_elems += dom.rigid.mesh.numBoundElements
                # Count compound collision sub-shapes
                if hasattr(dom, "collision_shapes") and dom.collision_shapes:
                    count_compound_shapes += len(dom.collision_shapes)
            elif dom.type in [DomainType.ANALYTICAL, DomainType.HEIGHTFIELD, DomainType.VOXELMAP]:
                count_anal += 1

        # Dynamic allocation (count + buffer) - drastically reduces memory when few rigids/meshes are used
        num_joints = 0 if joints is None else len(joints)
        self.MAX_NODES = max(count_rigid + count_anal + 64, 128)

        print("Number of maxNodes allocated:", self.MAX_NODES)
        self.MAX_ANAL = max(count_anal + 8, 16)
        self.MAX_MESH = max(count_mesh_rigids + 18, 16)
        self.MAX_JOINTS = max(num_joints + 8, 128)

        self.d = d
        self.numDomains = len(domains)
        # keep a reference to domain objects for host-side operations (mesh handling)
        self.domains = domains
        self.rigidDomainIds = wp.zeros(self.MAX_NODES, dtype=wp.vec3i)
        self.category_bits = wp.zeros(self.MAX_NODES, dtype=wp.uint32)
        self.collide_bits = wp.zeros(self.MAX_NODES, dtype=wp.uint32)
        _fill_array(self.category_bits, COLLISION_CATEGORY_ROBOT)
        _fill_array(self.collide_bits, COLLISION_MASK_ALL)
        maxDomains = max(len(domains), self.MAX_NODES)
        self.domainToRigid = wp.zeros(maxDomains, dtype=int)
        _fill_array(self.domainToRigid, -1)
        # Packed per-rigid parameters: rows hold different vector groups per-rigid.
        # row 0: reference point (current coords)
        # row 1: primary shape params (extents, endpoint1 (for capsule), normal for analytical domain etc.)
        self.rigidParams = wp.zeros((self.MAX_NODES, 2), dtype=wp.vec2)

        self.numRigids = 0
        self.numAnalytical = 0
        self.numMesh = 0
        self.numMeshRigidInContact = 0
        self.numRigidInContact = 0
        self.numRigidGroundContact = 0
        # When 0, ground narrow-phase skips AABB early-out so stale AABBs
        # cannot suppress first-contact detection in no-collision fast path.
        self._ground_use_aabb_early_out = wp.zeros(1, dtype=int)
        _assign_scalar(self._ground_use_aabb_early_out, 1)
        self.hasHeightFieldOrVoxel = False  # Set True if any HeightField/Voxel domains exist

        self.contact_erp = 0.2  # Baumgarte error reduction parameter for ground contacts
        self.restitution_velocity_threshold = 1.0  # Ignore restitution for low-speed contacts
        self.skip_bvh = False  # When True, skip BVH broadphase and use direct ground pair generation
        self.control_dt = 1.0 / 60.0  # default: 60 Hz control
        # These are nodal data
        self.bcNodes = wp.zeros(self.MAX_NODES, dtype=int)
        self.bcGValues = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.bcTValues = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.bcRValues = wp.zeros(self.MAX_NODES, dtype=float)

        # Mesh rigid storage - boundary elements only for contact detection
        # Dynamic resizing based on pre-scan results (count_mesh_nodes is calculated above)
        self.meshRigidContactMarginRatio = 0.02
        if count_mesh_nodes > 0:
            self.MAX_BOUNDARY_NODES = max(count_mesh_nodes + 4096, 4096)
            self.MAX_BOUNDARY_ELEMENTS = max(count_mesh_elems + 8192, 8192)

            # Boundary node coordinates (world space and local)
            self.meshBoundaryCoords = wp.zeros(self.MAX_BOUNDARY_NODES, dtype=wp.vec2)
            # Boundary element connectivity (edges for 2D, triangles for 3D)
            # For 2D: each element has 2 node indices
            # For 3D: each element has 3 node indices
            self.meshBoundaryElements = wp.zeros(self.MAX_BOUNDARY_ELEMENTS, dtype=wp.vec3i)
            # Cached per-element AABBs (updated once per substep)
            self.meshElemLB = wp.zeros(self.MAX_BOUNDARY_ELEMENTS, dtype=wp.vec2)
            self.meshElemUB = wp.zeros(self.MAX_BOUNDARY_ELEMENTS, dtype=wp.vec2)
            self.meshElemMarginBase = wp.zeros(self.MAX_BOUNDARY_ELEMENTS, dtype=float)

            # Per-mesh bookkeeping
            self.meshBoundaryNodeCount = wp.zeros(self.MAX_MESH, dtype=int)  # number of boundary nodes per mesh
            self.meshBoundaryNodeOffset = wp.zeros(self.MAX_MESH, dtype=int)  # starting index in meshBoundaryCoords
            self.meshBoundaryElementCount = wp.zeros(self.MAX_MESH, dtype=int)  # number of boundary elements per mesh
            self.meshBoundaryElementOffset = wp.zeros(self.MAX_MESH, dtype=int)  # starting index in meshBoundaryElements

            # Index mappings
            self.mesh2RigidIndices = wp.zeros(self.MAX_MESH, dtype=int)
            _fill_array(self.mesh2RigidIndices, -1)
            self.rigid2MeshIndices = wp.zeros(self.MAX_NODES, dtype=int)
            _fill_array(self.rigid2MeshIndices, -1)

            # ==== MESH INSTANCING: Geometry Pool (shared boundary data) ====
            # Pool storage for unique mesh geometries
            self.MAX_POOL_GEOMETRIES = 256  # Max unique mesh shapes
            self.num_pool_geometries = 0

            # Pool only stores UNIQUE geometries; size matches total to avoid over-capacity
            # (when many clones exist, pool will naturally stop growing after unique meshes are added)
            # Falls back to legacy path gracefully if pool capacity is exceeded
            self.MAX_POOL_NODES = count_mesh_nodes + 4096
            self.MAX_POOL_ELEMENTS = count_mesh_elems + 8192

            # Pool boundary data (same structure as existing, but deduplicated)
            self.pool_boundary_lrs = wp.zeros(self.MAX_POOL_NODES, dtype=wp.vec2)
            self.pool_boundary_elements = wp.zeros(self.MAX_POOL_ELEMENTS, dtype=wp.vec3i)

            # Per-geometry bookkeeping in pool
            self.pool_node_count = wp.zeros(self.MAX_POOL_GEOMETRIES, dtype=int)
            self.pool_node_offset = wp.zeros(self.MAX_POOL_GEOMETRIES, dtype=int)
            self.pool_elem_count = wp.zeros(self.MAX_POOL_GEOMETRIES, dtype=int)
            self.pool_elem_offset = wp.zeros(self.MAX_POOL_GEOMETRIES, dtype=int)

            # Hash-based deduplication lookup (Python-side dict for Phase 1)
            self.pool_hash_to_id = {}  # maps mesh_hash -> pool_geometry_id

            # ==== MESH INSTANCING: Instance Manager (per-rigid transforms) ====
            # Maps each mesh rigid to its pool geometry + transform
            self.instance_pool_id = wp.zeros(self.MAX_NODES, dtype=int)  # rigid_idx -> pool_geom_id
            _fill_array(self.instance_pool_id, -1)  # -1 = not using pool (legacy path)

            self.total_pool_nodes = 0
            self.total_pool_elements = 0

            # Transform storage for mesh rigids (scale component)
            self.meshRigidScale = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
            # Initialize scale to [1,1,1] for all rigids
            for i in range(self.MAX_NODES):
                _patch_array(self.meshRigidScale, i, wp.vec2(1.0, 1.0))

            # Transform storage for mesh rigids (offset component)
            self.meshRigidOffset = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
            # Initialize offset to [0,0,0] for all rigids
            for i in range(self.MAX_NODES):
                _patch_array(self.meshRigidOffset, i, wp.vec2(0.0, 0.0))

            # Active mesh mask for spatial hash population optimization
            self.mesh_active = wp.zeros(self.MAX_MESH, dtype=int)
            _fill_array(self.mesh_active, 0)

        else:
            self.MAX_BOUNDARY_NODES = 1
            self.MAX_BOUNDARY_ELEMENTS = 1

            # Use 1-element fields to prevent compilation errors when no mesh rigids exist
            self.meshBoundaryCoords = wp.zeros(1, dtype=wp.vec2)
            self.meshBoundaryElements = wp.zeros(1, dtype=wp.vec3i)
            self.meshElemLB = wp.zeros(1, dtype=wp.vec2)
            self.meshElemUB = wp.zeros(1, dtype=wp.vec2)
            self.meshElemMarginBase = wp.zeros(1, dtype=float)
            self.meshBoundaryNodeCount = wp.zeros(1, dtype=int)
            self.meshBoundaryNodeOffset = wp.zeros(1, dtype=int)
            self.meshBoundaryElementCount = wp.zeros(1, dtype=int)
            self.meshBoundaryElementOffset = wp.zeros(1, dtype=int)
            self.mesh2RigidIndices = wp.zeros(1, dtype=int)
            self.meshRigidScale = wp.zeros(1, dtype=wp.vec2)
            self.meshRigidOffset = wp.zeros(1, dtype=wp.vec2)
            self.mesh_active = wp.zeros(1, dtype=int)

            self.rigid2MeshIndices = wp.zeros(1, dtype=int)

            # Mesh instancing fields (even when no mesh rigids exist)
            self.MAX_POOL_GEOMETRIES = 1
            self.num_pool_geometries = 0

            self.pool_boundary_lrs = wp.zeros(1, dtype=wp.vec2)
            self.pool_boundary_elements = wp.zeros(1, dtype=wp.vec3i)

            self.pool_node_count = wp.zeros(1, dtype=int)
            self.pool_node_offset = wp.zeros(1, dtype=int)
            self.pool_elem_count = wp.zeros(1, dtype=int)
            self.pool_elem_offset = wp.zeros(1, dtype=int)

            self.pool_hash_to_id = {}

            self.instance_pool_id = wp.zeros(self.MAX_NODES, dtype=int)
            _fill_array(self.instance_pool_id, -1)

            self.total_pool_nodes = 0
            self.total_pool_elements = 0

        print("Allocating RigidManager with MAX_NODES =", self.MAX_NODES, "MAX_JOINTS =", self.MAX_JOINTS)

        # Visual mesh data for VTU export (indexed by rigid body index).
        # Populated by RigidBodyDomain.attach() when the domain carries
        # a visual_mesh dict (typically from URDF import with separate
        # collision / visual geometry).
        # Each entry: {'rest_vertices': np.ndarray (N,3), 'elements': np.ndarray (M,3)}
        self.visual_mesh_data = {}

        # Environment ID for collision filtering (batched training)
        # -1 means no env (e.g., ground, single robot), >= 0 means belongs to env_id
        self.rigid_env_id = wp.zeros(self.MAX_NODES, dtype=int)
        _fill_array(self.rigid_env_id, -1)  # Default: no env filtering

        self.num_envs = 0
        self.joints_per_env = 0

        self.totalBoundaryNodes = 0
        self.totalBoundaryElements = 0

        self.U = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.V = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.accumulated_impulse = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        # Rotation representation: 2D uses scalar angle, 3D uses quaternion
        self.quat = wp.zeros(self.MAX_NODES, dtype=float)  # angle for 2D
        self.quat_initial = wp.zeros(self.MAX_NODES, dtype=float)  # initial orientation snapshot
        self.RotV = wp.zeros(self.MAX_NODES, dtype=float)
        self.accumulated_rotational_impulse = wp.zeros(self.MAX_NODES, dtype=float)

        _fill_array(self.V, 0.0)
        _fill_array(self.RotV, 0.0)

        # AABB is now stored in ExplicitLoop using global domain indices
        # RigidManager will receive a reference to the global aabb field
        self.aabb = None  # Will be set by ExplicitLoop after initialization

        # Pre-allocated fields for collision pair processing (avoid dynamic field allocation)
        self.MAX_COLLISION_PAIRS = 10000  # Maximum collision pairs per frame

        # Buffers for different collision pair types
        self.primitive_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        self.ball_ball_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        self.box_box_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        self.box_ball_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        self.seg_point_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        self.seg_ball_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        self.seg_seg_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)

        self.mesh_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        self.mixed_pairs_buffer = wp.zeros(self.MAX_COLLISION_PAIRS, dtype=wp.vec2i)
        # Ground collision pairs need larger buffer for batched environments
        # Each rigid can collide with ground, so need at least MAX_NODES capacity
        self.MAX_GROUND_PAIRS = max(self.MAX_NODES * 2, 4096)
        self.groundprim_pairs_buffer = wp.zeros(self.MAX_GROUND_PAIRS, dtype=wp.vec2i)
        self.groundmesh_pairs_buffer = wp.zeros(self.MAX_GROUND_PAIRS, dtype=wp.vec2i)

        # Counters for each pair type
        self.num_primitive_pairs = wp.zeros(1, dtype=int)
        self.num_ball_ball_pairs = wp.zeros(1, dtype=int)
        self.num_box_box_pairs = wp.zeros(1, dtype=int)
        self.num_box_ball_pairs = wp.zeros(1, dtype=int)
        self.num_seg_point_pairs = wp.zeros(1, dtype=int)
        self.num_seg_ball_pairs = wp.zeros(1, dtype=int)
        self.num_seg_seg_pairs = wp.zeros(1, dtype=int)

        self.num_mesh_pairs = wp.zeros(1, dtype=int)
        self.num_mixed_pairs = wp.zeros(1, dtype=int)
        self.num_groundprim_pairs = wp.zeros(1, dtype=int)
        self.num_groundmesh_pairs = wp.zeros(1, dtype=int)

        # ==== Contact Cache: Store detected contacts to avoid redundant detection in PGS iterations ====
        self.MAX_CONTACTS = max(self.MAX_NODES * 16, 10000)
        # For mesh rigids, ground contacts scale with boundary nodes (each below-ground node = 1 contact)
        # Use boundary-aware scaling: typically only ~25% of mesh nodes contact ground simultaneously
        if count_mesh_nodes > 0:
            self.MAX_GROUND_CONTACTS = max(min(count_mesh_nodes // 4, self.MAX_NODES * 500), 50000)
        else:
            self.MAX_GROUND_CONTACTS = max(self.MAX_NODES * 200, 50000)

        # Rigid-Rigid contacts (use applyImpulsePair)
        self.num_contacts = wp.zeros(1, dtype=int)
        _assign_scalar(self.num_contacts, 0)
        self.contact_rigid_a = wp.zeros(self.MAX_CONTACTS, dtype=int)
        self.contact_rigid_b = wp.zeros(self.MAX_CONTACTS, dtype=int)
        self.contact_point = wp.zeros(self.MAX_CONTACTS, dtype=wp.vec2)
        self.contact_normal = wp.zeros(self.MAX_CONTACTS, dtype=wp.vec2)
        # PGS row indices for each contact (to map pgs_lambda back to contact forces)
        # Using single field to avoid LLVM memory layout issues on Windows
        self.contact_pgs_indices = wp.zeros(self.MAX_CONTACTS, dtype=wp.vec3i)  # [normal, tangent1, tangent2]
        self.contact_depth = wp.zeros(self.MAX_CONTACTS, dtype=float)
        self.contact_bounce_vel = wp.zeros(self.MAX_CONTACTS, dtype=float)  # Restitution bounce target velocity (computed once at cache time)
        self.contact_tangent1 = wp.zeros(self.MAX_CONTACTS, dtype=wp.vec2)
        self.contact_force = wp.zeros(self.MAX_CONTACTS, dtype=wp.vec2)
        self.contact_count_per_rigid = wp.zeros(self.MAX_NODES, dtype=int)

        # Ground-Rigid contacts (use applyImpulseAtPoint)
        self.num_ground_contacts = wp.zeros(1, dtype=int)
        _assign_scalar(self.num_ground_contacts, 0)
        # Track previous frame's contact counts to bound reset loops
        self.prev_num_contacts = wp.zeros(1, dtype=int)
        _assign_scalar(self.prev_num_contacts, 0)
        self.prev_num_ground_contacts = wp.zeros(1, dtype=int)
        _assign_scalar(self.prev_num_ground_contacts, 0)
        self.ground_contact_rigid = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=int)
        self.ground_contact_point = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=wp.vec2)
        self.ground_contact_normal = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=wp.vec2)
        self.ground_contact_vel = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=wp.vec2)
        # PGS row indices for each ground contact
        # Using single field to avoid LLVM memory layout issues on Windows
        self.ground_contact_pgs_indices = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=wp.vec3i)  # [normal, tangent1, tangent2]
        self.ground_contact_force = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=wp.vec2)
        self.ground_contact_depth = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=float)  # Signed penetration depth (negative = penetrating)
        self.ground_contact_bounce_vel = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=float)  # Restitution bounce target velocity (computed once at cache time)
        # Fixed tangent basis per contact (computed once at contact creation, stable across PGS iterations)
        self.ground_contact_tangent1 = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=wp.vec2)
        self.ground_contact_tangent2 = wp.zeros(self.MAX_GROUND_CONTACTS, dtype=wp.vec2)  # only used in 3D

        # Per-env ground contact indexing for efficient PGS parallel scanning.
        # Without this, each of N env-threads scans ALL contacts → O(N × total_contacts).
        # With per-env indices, each thread accesses only its own contacts → O(contacts_per_env).
        self.MAX_ENVS_ALLOC = (
            4096  # Max envs we allocate for (can be set based on expected batch size, but keep reasonable upper bound)
        )
        self.MAX_GC_PER_ENV = max(self.MAX_GROUND_CONTACTS // self.MAX_ENVS_ALLOC, 1024)
        self.ground_contact_env_count = wp.zeros(self.MAX_ENVS_ALLOC, dtype=int)
        # Per-env index lists: ground_contact_env_idx[env_id * MAX_GC_PER_ENV + local_i] = global contact idx
        self.ground_contact_env_idx = wp.zeros(self.MAX_ENVS_ALLOC * self.MAX_GC_PER_ENV, dtype=int)
        # Same for rigid-rigid contacts
        self.MAX_CC_PER_ENV = max(self.MAX_CONTACTS // self.MAX_ENVS_ALLOC, 256)
        self.contact_env_count = wp.zeros(self.MAX_ENVS_ALLOC, dtype=int)
        self.contact_env_idx = wp.zeros(self.MAX_ENVS_ALLOC * self.MAX_CC_PER_ENV, dtype=int)

        print("Finished allocating contact cached ======")

        # Per-rigid friction coefficient (Coulomb, 0.0 = frictionless), first friction, second restitution
        self.contactParams = wp.zeros(self.MAX_NODES, dtype=wp.vec2)

        # Inertia storage: 2D uses scalar, 3D uses 3x3 matrix
        self.inertia = wp.zeros(self.MAX_NODES, dtype=float)
        # OPTIMIZATION: Cached values for 2D joint solving
        self.cached_rotation_matrix = wp.zeros(self.MAX_NODES, dtype=wp.mat22)
        self.cached_inertia_inv_2d = wp.zeros(self.MAX_NODES, dtype=float)
        self.visual_angle = wp.zeros(self.MAX_NODES, dtype=float)

        self.mass = wp.zeros(self.MAX_NODES, dtype=float)

        # per-rigid radius for capsule/ball where applicable
        self.radius = wp.zeros(self.MAX_NODES, dtype=float)

        # ── Compound collision shapes ────────────────────────────────────
        # Multiple collision primitives (e.g. 4 spheres on a foot) attached
        # to a single parent rigid body. Sub-colliders are stored in a flat
        # pool; per-rigid (count, offset) index into it.
        self.MAX_COMPOUND_SHAPES = max(count_compound_shapes + 64, 128)
        self.compound_count = wp.zeros(self.MAX_NODES, dtype=int)  # num sub-colliders per rigid (0 = use main shape)
        self.compound_offset = wp.zeros(self.MAX_NODES, dtype=int)  # start index in pool
        self.compound_local_pos = wp.zeros(self.MAX_COMPOUND_SHAPES, dtype=wp.vec2)  # body-local offset
        self.compound_radius = wp.zeros(self.MAX_COMPOUND_SHAPES, dtype=float)  # sub-collider radius
        self.compound_type = wp.zeros(self.MAX_COMPOUND_SHAPES, dtype=int)  # RigidType (BALL for now)
        self.num_compound_shapes = 0  # total allocated sub-colliders

        self.needUpdate = False

        self.stableTime = 1.0 / 1000.0  # Default stable time step
        # Process rigid domains and allocate data
        print("Processing domains for RigidManager...")
        self.processDomains_(domains)
        print("Processed", self.numRigids, "rigids and", self.numAnalytical, "analytical planes.")

        self.spatialHash = None
        self._sh_contact_margin = wp.zeros(1, dtype=float)
        _assign_scalar(self._sh_contact_margin, 0.0)
        self._sh_mesh_elapsed = 0.0
        self._sh_mesh_rebuild_interval = 1.0 / 500.0  # rebuild at most every ~2ms
        self._sh_mesh_needs_rebuild = True  # first call always rebuilds
        self._sh_mesh_max_v = 0.0  # max rigid velocity at last rebuild
        self._sh_unbounded_lb = wp.zeros(1, dtype=wp.vec2)
        self._sh_unbounded_ub = wp.zeros(1, dtype=wp.vec2)
        _assign_scalar(self._sh_unbounded_lb, wp.vec2(-1e9, -1e9))
        _assign_scalar(self._sh_unbounded_ub, wp.vec2(1e9, 1e9))
        if (self.numMeshRigidInContact > 0) and not self.skip_spatial_hash:
            # Size spatial hash based on actual mesh boundary element count (not hardcoded)
            total_elems = self.totalBoundaryElements if self.totalBoundaryElements > 0 else count_mesh_elems
            sh_max_elements = max(total_elems + 4096, 10000)
            sh_max_cells = min(sh_max_elements, 100000)
            # Query capacity: avoid frequent candidate truncation in dense mesh
            # contacts (e.g. hand links). Keep bounded to control stack usage.
            sh_max_query = min(max(total_elems // 8, 2048), 4096)
            # Per-cell cap: assume ~100-500 actual cells; allow generous headroom
            # so no elements are silently dropped due to overflow.
            sh_max_per_cell = min(max(total_elems // 50 + 10, 200), 2000)
            self.spatialHash = SpatialHashManager(
                self.d,
                max_elements=sh_max_elements,
                max_cells=sh_max_cells,
                max_elements_per_cell=sh_max_per_cell,
                max_query_results=sh_max_query,
            )
            print(
                f"Initialized SpatialHashManager for mesh-rigid contacts "
                f"({sh_max_elements} elements, {sh_max_cells} cells)."
            )
        elif self.skip_spatial_hash and self.numMeshRigidInContact > 0:
            print(f"  [SpatialHash] SKIPPED (skip_spatial_hash=True, saving ~GB of VRAM)")

        # ==== Dynamic joint anchors (as special points using main rigid arrays) ====
        self.jointDict = dict()
        for joint in joints:
            self.jointDict[joint.name] = joint
        self.joints = joints

        self.numAnchors = 0

        # ==== Joint storage fields for unified kernel processing ====
        self.joint_type = wp.zeros(self.MAX_JOINTS, dtype=int)
        self.joint_anchor = wp.zeros(self.MAX_JOINTS, dtype=wp.vec2)  # world-space anchor point for joint constraint
        self.joint_id_a = wp.zeros(self.MAX_JOINTS, dtype=int)
        self.joint_id_b = wp.zeros(self.MAX_JOINTS, dtype=int)

        # Local offset vectors (body-local coordinates)
        self.joint_l1 = wp.zeros(self.MAX_JOINTS, dtype=wp.vec2)
        self.joint_l2 = wp.zeros(self.MAX_JOINTS, dtype=wp.vec2)

        # Initial orientations for computing relative rotation
        self.joint_axis = wp.zeros(self.MAX_JOINTS, dtype=wp.vec2)
        self.joint_q0_rel_inv = wp.zeros(self.MAX_JOINTS, dtype=float)

        # Joint parameters: [position_bias, angular_bias, lower_limit, upper_limit, velocity_limit, effort_limit]
        self.joint_params = wp.zeros(self.MAX_JOINTS, dtype=vec6f)

        # Motor flag
        self.joint_has_motor = wp.zeros(self.MAX_JOINTS, dtype=int)
        self.joint_control_target = wp.zeros(self.MAX_JOINTS, dtype=float)  # target position for motor (angle for revolute, length for prismatic)
        # Motor command semantics:
        # 0 = velocity command (rad/s or m/s), 1 = acceleration command (rad/s^2 or m/s^2), 2 = torque motor
        self.joint_motor_target_mode = wp.zeros(self.MAX_JOINTS, dtype=int)
        # Runtime velocity target consumed by unified PGS motor rows.
        # This is updated per-substep from joint_control_target according to mode.
        self.joint_motor_target_vel = wp.zeros(self.MAX_JOINTS, dtype=float)
        self.kpd_field = wp.zeros(self.MAX_JOINTS, dtype=wp.vec2)  # [kp_pos, kp_rot] for PD control

        print("Processing joints for RigidManager...")
        self.processJoints(joints)
        print("Processed", len(joints), "joints.")

        # World-frame PD flag: when set to 1, the PD controller measures the
        # child body's absolute orientation (world frame) projected onto the
        # joint axis, instead of the parent-child relative angle.  This lets
        # ankle/hip joints sense the full body tilt even when the whole chain
        # tips as a unit.
        self.joint_pd_world_frame = wp.zeros(self.MAX_JOINTS, dtype=int)

        self.pgs_iterations = 200  # Max PGS iterations (with early convergence exit)
        self.pgs_tol = 1e-5  # Convergence tolerance for PGS early exit

        self.MAX_CONSTRAINTS = 16 * self.MAX_CONTACTS * 3 + 7 * self.MAX_JOINTS  # 3 per contact and 7 per joints
        self.numConstraints = wp.zeros(1, dtype=int)
        _assign_scalar(self.numConstraints, 0)
        # Per-constraint Jacobians for body A and body B:
        #   - ground-vs-rigid uses only pgs_Jac_a (B is zero)
        #   - rigid-vs-rigid uses both pgs_Jac_a and pgs_Jac_b
        self.pgs_Jac_a = wp.zeros(self.MAX_CONSTRAINTS, dtype=wp.vec3)
        self.pgs_Jac_b = wp.zeros(self.MAX_CONSTRAINTS, dtype=wp.vec3)
        self.pgs_rhs = wp.zeros(self.MAX_CONSTRAINTS, dtype=float)
        self.pgs_limits = wp.zeros(self.MAX_CONSTRAINTS, dtype=wp.vec2)  # [lower, upper]
        self.pgs_bodypair = wp.zeros(self.MAX_CONSTRAINTS, dtype=wp.vec2i)
        # Per-row accumulated impulse for projected GS and friction clamping.
        self.pgs_lambda = wp.zeros(self.MAX_CONSTRAINTS, dtype=float)
        # For friction rows, parent normal row index; -1 otherwise.
        self.pgs_parent_row = wp.zeros(self.MAX_CONSTRAINTS, dtype=int)
        self._pgs_check_interval = 5  # Check convergence every N iterations (after warm-up of 10)
        # Allocate PGS RotV snapshot with matching dimension
        self.pgsErrorNone = wp.zeros(1, dtype=float)
        _assign_scalar(self.pgsErrorNone, 0.0)

        # Snapshot fields for PGS convergence check
        self.V_prev = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.RotV_prev = None  # allocated below after RotV
        self.RotV_prev = wp.zeros(self.MAX_NODES, dtype=float)

        # Optional Python-side PGS kernel timing breakdown.
        self.pgs_profile_enabled = False
        self.pgs_profile_sync_kernels = False
        self.pgs_profile_print_every = 100
        self._pgs_profile_calls = 0
        self._pgs_profile_iters = 0
        self._pgs_profile_breaks = 0
        self._pgs_profile_time_total = 0.0
        self._pgs_profile_time_snapshot = 0.0
        self._pgs_profile_time_joints_fwd = 0.0
        self._pgs_profile_time_contacts = 0.0
        self._pgs_profile_time_joints_bwd = 0.0
        self._pgs_profile_time_delta = 0.0

        # Adjust rigid poses to satisfy joint limits
        # self.adjust_rigid_poses_for_joint_limits()

        # Process boundary conditions from domains and joints
        self.movingAnalytical = False
        print("Processing boundary conditions for RigidManager...")
        self.processConditions()

        # Materialize world-space mesh boundary coords for contact / drawing.
        if self.numMesh > 0:
            self.precompute_rigid_transforms()
            self.updateMeshCoords()

        # Normal mode: use shared BVH from ExplicitLoop
        self.bvh = bvh

        print(
            "RigidManager initialized with",
            self.numRigids,
            "rigids,",
            self.numAnalytical,
            "analytical planes,",
            self.numAnchors,
            "anchors.",
        )

        print(f"In total {self.numRigidInContact} rigids are considered for rigid-rigid contact detection.")
        print(f"In total {self.numRigidGroundContact} rigids are considered for rigid-ground contact detection.")

    def _compute_mesh_hash(self, mesh) -> int:
        """Compute hash of mesh geometry for deduplication.

        Hash is based on boundary node count, element count, and a sample of coordinates.
        This is a fast approximate hash - collisions will be rare for typical robot geometries.

        Args:
            mesh: Mesh object with boundaryNodes, boundaryElements, coords

        Returns:
            Integer hash value
        """
        import hashlib

        # Use boundary topology as primary hash component
        num_nodes = mesh.numBoundNodes
        num_elems = mesh.numBoundElements

        # Sample first/last boundary node coordinates (rounded to avoid float precision issues)
        sample_coords = []
        for i in [0, min(5, num_nodes - 1), num_nodes - 1]:  # sample 3 nodes
            if i < num_nodes:
                node_id = int(mesh.boundaryNodes[i])
                coord = mesh.coords[node_id]
                # Round to 6 decimal places to avoid floating point precision differences
                sample_coords.extend([round(float(coord[j]), 6) for j in range(self.d)])

        # Sample first/last boundary element connectivity
        sample_elems = []
        for i in [0, num_elems - 1]:
            if i < num_elems:
                elem = mesh.boundaryElements[i]
                sample_elems.extend([int(elem[j]) for j in range(2)])

        # Combine into hash string
        hash_str = f"{num_nodes}_{num_elems}_{sample_coords}_{sample_elems}"
        return int(hashlib.md5(hash_str.encode()).hexdigest()[:16], 16)

    def _add_mesh_to_pool(self, mesh, rigid_idx: int) -> int:
        """Add a unique mesh geometry to the pool and return its pool ID.

        Args:
            mesh: Mesh object to add
            rigid_idx: Index of the rigid using this mesh

        Returns:
            pool_geometry_id: Index in the pool where this geometry is stored
        """
        pool_id = self.num_pool_geometries

        if pool_id >= self.MAX_POOL_GEOMETRIES:
            print(
                f"\033[91mWarning: Exceeded MAX_POOL_GEOMETRIES ({self.MAX_POOL_GEOMETRIES}), falling back to legacy storage\033[0m"
            )
            return -1

        num_nodes = mesh.numBoundNodes
        num_elems = mesh.numBoundElements

        node_offset = self.total_pool_nodes
        elem_offset = self.total_pool_elements

        # Safety checks
        if (node_offset + num_nodes) > self.MAX_POOL_NODES:
            print(
                f"\033[91mWarning: Pool boundary nodes exceeded capacity ({node_offset + num_nodes} > {self.MAX_POOL_NODES}), falling back to legacy\033[0m"
            )
            return -1

        if (elem_offset + num_elems) > self.MAX_POOL_ELEMENTS:
            print(f"\033[91mWarning: Pool boundary elements exceeded capacity, falling back to legacy\033[0m")
            return -1

        # Record pool geometry metadata
        _patch_array(self.pool_node_offset, pool_id, node_offset)
        _patch_array(self.pool_node_count, pool_id, num_nodes)
        _patch_array(self.pool_elem_offset, pool_id, elem_offset)
        _patch_array(self.pool_elem_count, pool_id, num_elems)

        # Get reference point from rigid
        ref = np.asarray(self.rigidParams.numpy()[rigid_idx, 0], dtype=np.float32)

        # Store boundary node coordinates in pool
        boundary_node_map = {}
        for local_bid in range(num_nodes):
            global_nid = int(mesh.boundaryNodes[local_bid])
            boundary_node_map[global_nid] = local_bid

            coord = mesh.coords[global_nid]
            lr = coord - ref

            _patch_array(self.pool_boundary_lrs, node_offset + local_bid, lr)

        # Store boundary element connectivity (remapped to local indices)
        for eid in range(num_elems):
            elem_conn = mesh.boundaryElements[eid]
            local_n0 = boundary_node_map[int(elem_conn[0])]
            local_n1 = boundary_node_map[int(elem_conn[1])]
            _patch_array(self.pool_boundary_elements, elem_offset + eid, (local_n0, local_n1, -1))
          

        # Update pool counters
        self.total_pool_nodes += num_nodes
        self.total_pool_elements += num_elems
        self.num_pool_geometries += 1

        # print(
        #     f"[Pool] Added geometry #{pool_id}: {num_nodes} nodes, {num_elems} elements (pool total: {self.total_pool_nodes} nodes)"
        # )

        return pool_id

    def setGlobalAABB(self, aabb):
        """Set reference to ExplicitLoop's global AABB field.

        Args:
            aabb: Global AABB field indexed by domain index
        """
        self.aabb = aabb
        self.precompute_rigid_transforms()
        self.updateBBox()

    def calInertiaInv(self):
        for i in range(self.numRigids):
            # Compute inverse using Taichi's built-in matrix inverse
            det = self.inertia[i].determinant()  # Ensure determinant is computed for validation
            if det > 1e-30:
                I_inv = self.inertia[i].inverse()
                self.inertiaInv[i] = I_inv
            else:
                self.inertiaInv[i] = wp.mat33(0.0)  # zero inverse = infinite inertia (fixed body)

    def _copy_elements_from_pool_kernel(
        self, dst_offset: int, src_offset: int, count: int, node_offset: int
    ):
        """Copy element connectivity from pool to legacy array (host numpy path)."""
        pool_elems = self.pool_boundary_elements.numpy()
        for i in range(count):
            conn = pool_elems[src_offset + i]
            c0, c1, c2 = int(conn[0]), int(conn[1]), int(conn[2])
            if c0 >= 0:
                c0 += node_offset
            if c1 >= 0:
                c1 += node_offset
            if c2 >= 0:
                c2 += node_offset
            _patch_array(self.meshBoundaryElements, dst_offset + i, wp.vec3(c0, c1, c2))

    def _batch_copy_elements_kernel(self, dst_offset: int, node_offset: int, elem_data):
        """Copy element connectivity from numpy into meshBoundaryElements."""
        for i in range(elem_data.shape[0]):
            c0 = int(elem_data[i, 0])
            c1 = int(elem_data[i, 1])
            c2 = int(elem_data[i, 2])
            if c0 >= 0:
                c0 += node_offset
            if c1 >= 0:
                c1 += node_offset
            if c2 >= 0:
                c2 += node_offset
            _patch_array(self.meshBoundaryElements, dst_offset + i, wp.vec3(c0, c1, c2))

    def substep(self, collision_pairs_field, num_pairs, dt, damping, fem_lb=None, fem_ub=None, fem_margin=0.0):
        """High-level rigid update: integrate velocities, solve constraints, update positions.

        PERFORMANCE-CRITICAL: Fused kernels minimize Python↔Taichi round trips.
        Each kernel launch costs ~0.08-0.15ms of pure overhead.

        Fast path (no HeightField/Voxel): multiple small kernel launches instead of fewer large ones.
            1. classify_collision_pairs_kernel  (conditional, if broadphase pairs exist)
            1b. apply_motor_torques_kernel      (adds torque-mode motor forces to accumulated_rotational_impulse)
            2. _rigidStep_and_precompute_kernel (fused velocity integration + transform cache)
            3. reset + per-type contact detection + PGS split kernels
            4. _updateU_and_BBox_kernel         (fused position integration + AABB update)

        Slow path (HeightField/Voxel): original separate kernels for compatibility.
        """
        if not self.needUpdate:
            return

        # Accumulate time for lazy spatial hash rebuild
        self._sh_mesh_elapsed += dt
        if self._sh_mesh_elapsed >= self._sh_mesh_rebuild_interval:
            self._sh_mesh_needs_rebuild = True
        # Displacement-based trigger: rebuild if max displacement > 0.5 * cell_size
        if not self._sh_mesh_needs_rebuild:
            max_disp = self._sh_mesh_max_v * self._sh_mesh_elapsed
            if max_disp > 0.5 * self._sh_mesh_cell_size:
                self._sh_mesh_needs_rebuild = True
        # Adaptive query buffer: cell_size + estimated displacement, capped at 3x cell
        # Only update GPU field when rigid-rigid SH is actually used (avoids GPU sync).
        if self.considerRigidRigidContact:
            margin = self._sh_mesh_max_v * self._sh_mesh_elapsed
            _assign_scalar(self._sh_contact_margin, margin)

        self.update_joint_motor_velocity_targets_kernel(dt)
        if self.use_pd == 1:
            self.apply_joint_pd_velocity_kernel(dt)
        elif self.use_pd == 2:
            self.apply_joint_pd_torque_kernel(dt)

        if self.considerRigidRigidContact and num_pairs > 0:
            self.classify_collision_pairs_kernel(collision_pairs_field, num_pairs)
        elif self.numRigidGroundContact > 0:
            # Generate them directly to avoid scanning all BVH pairs.
            self._generate_ground_pairs_direct_kernel()

        # if (int(self.num_box_box_pairs.numpy()[0]) > 0):
        #     print("Num pairs:", num_pairs, "Ball-Ball:", int(self.num_ball_ball_pairs.numpy()[0]),
        #         "Box-Box:", int(self.num_box_box_pairs.numpy()[0]), "Box-Ball:", int(self.num_box_ball_pairs.numpy()[0]),
        #         "Seg-Point:", int(self.num_seg_point_pairs.numpy()[0]), "Seg-Ball:", int(self.num_seg_ball_pairs.numpy()[0]), "Seg-Seg:", int(self.num_seg_seg_pairs.numpy()[0]),
        # "Mesh-Mesh:", int(self.num_mesh_pairs.numpy()[0]), "Mixed:", int(self.num_mixed_pairs.numpy()[0]), "Ground-Prim:", int(self.num_groundprim_pairs.numpy()[0]), "Ground-Mesh:", int(self.num_groundmesh_pairs.numpy()[0]))
        has_rigid_rigid_contact = (
            int(self.num_ball_ball_pairs.numpy()[0])
            + int(self.num_box_box_pairs.numpy()[0])
            + int(self.num_box_ball_pairs.numpy()[0])
            + int(self.num_seg_point_pairs.numpy()[0])
            + int(self.num_seg_ball_pairs.numpy()[0])
            + int(self.num_seg_seg_pairs.numpy()[0])
            + int(self.num_mesh_pairs.numpy()[0])
            + int(self.num_mixed_pairs.numpy()[0])
        ) > 0
        has_rigid_ground_contact = (int(self.num_groundprim_pairs.numpy()[0]) + int(self.num_groundmesh_pairs.numpy()[0])) > 0
        has_contact = has_rigid_rigid_contact or has_rigid_ground_contact
        has_solve = self.numAnchors > 0 or has_contact

        # K1: velocity integration + rotation matrix cache
        self._rigidStep_and_precompute_kernel(dt, damping)
        if has_solve:
            if has_contact:
                # K2: reset + detect contacts (split into small per-type kernels for fast JIT)
                self.reset_contact_caches_kernel()

                if self.considerRigidRigidContact and has_rigid_rigid_contact:
                    self.detect_all_contacts()

                if has_rigid_ground_contact:
                    self.detectRigidGroundContact()

            has_joints = self.numAnchors > 0
            # K3: unified PGS solve path.
            # Joint/contact rows are assembled first, then solved by unified PGS.
            if has_joints:
                self._assemble_joint_constraints_kernel(dt)
            if has_contact:
                self._assemble_contact_constraints_kernel(dt)
            self.solve_pgs(self.pgs_iterations)
            # Compute contact forces from impulses (force = impulse / dt)
            if has_contact:
                self._compute_contact_forces_kernel(dt)
            _assign_scalar(self.numConstraints, 0)
        # K4: position integration + AABB update (2→1 kernel)
        update_bbox_coords = 1
        self._updateU_and_BBox_kernel(dt, update_bbox_coords)

        if not has_contact and self.spatialHash is not None:
            # Meaning FEM-rigid contact need this SH rebuild
            self.maybe_rebuild_spatial_hash()

    def reset(self):
        """
        Reset all rigid bodies to their initial state.

        This method:
        1. Resets positions by re-packing rigid parameters from initial origins
        2. Resets velocities and angular velocities to zero
        3. Resets rotations to identity (zero angle)
        4. Resets accumulated displacements
        5. Resets joint anchors to initial positions
        6. Updates bounding boxes and shape coordinates

        Note: Does NOT reset boundary conditions - those should be managed by the caller.
        """
        # Reset all dynamic state variables to zero
        _fill_array(self.V, 0.0)
        _fill_array(self.RotV, 0.0)
        _fill_array(self.U, 0.0)  # CRITICAL: Reset accumulated displacement

        # Re-pack rigid parameters from initial origins stored in rigid objects
        # This restores initial positions (and will set initial rotations)
        domain_ids = self.rigidDomainIds.numpy()
        for i in range(self.numRigids):
            domain_idx = int(domain_ids[i][0])
            domain = self.domains[domain_idx]
            if domain.type == DomainType.RIGID:
                # Re-pack parameters from the rigid object (which stores initial origin)
                self.resetRigidParams(domain.rigid, i)

        # Update the cached rotation matrices and inertia tensors based on reset positions and zero rotations
        self.precompute_rigid_transforms()
        # Update bounding boxes and shape coordinates after reset
        # This will recompute shapeCoords for boxes based on new positions and zero rotation
        self.updateBBox()

    @staticmethod
    def _resolve_initial_quat(rigid):
        if rigid.initial_quat is not None:
            return np.asarray(rigid.initial_quat, dtype=np.float32)

        euler = rigid.angle
        mag = wp.length(euler)
        if mag < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _resolve_visual_quat(rigid):
        if hasattr(rigid, "visual_quat") and rigid.visual_quat is not None:
            return np.asarray(rigid.visual_quat, dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def packRigidParams(self, rigid, idx: int):
        """Pack a rigid object's key parameters into the compact `rigidParams` layout.

        Layout (per-rigid index `idx`):
        - rigidParams[idx, 0] : reference point (vector, dim d)
        - rigidParams[idx, 1] : primary params (vector, dim d) e.g. extents or radius in component 0
        """
        # Reference point
        ref = rigid.getRefPoint()
        _patch_array(self.rigidParams, (idx, 0), ref)
        _patch_array(self.mass, idx, rigid.mass)

        # default zero for primary / aux
        zero_vec = wp.vec2(0.0, 0.0)
        _patch_array(self.rigidParams, (idx, 1), zero_vec)

        # fill according to rigid type if available
        if rigid.rtype == RigidType.BALL:
            # also keep scalar radius field
            _patch_array(self.radius, idx, rigid.getRadius())

        elif rigid.rtype == RigidType.BOX:
            # extents -> primary, angle -> aux
            _patch_array(self.rigidParams, (idx, 1), rigid.getPrimary())

        elif rigid.rtype == RigidType.CAPSULE:
            # store segment endpoints relative to the reference point (local offsets)
            # so that they can be rotated/translated in-kernel correctly.
            _patch_array(self.rigidParams, (idx, 1), rigid.getPrimary() - ref)
            _patch_array(self.radius, idx, rigid.getRadius())

        elif rigid.rtype == RigidType.MESH:
            # ==== MESH INSTANCING: Use geometry pool with deduplication ====
            mesh = rigid.mesh
            mesh_hash = self._compute_mesh_hash(mesh)

            # Check if this geometry already exists in pool
            if mesh_hash in self.pool_hash_to_id:
                pool_id = self.pool_hash_to_id[mesh_hash]
                print(f"Packing mesh rigid idx {idx}: REUSING pool geometry #{pool_id} (hash collision = efficient!)")
            else:
                # Add new unique geometry to pool
                pool_id = self._add_mesh_to_pool(mesh, idx)
                if pool_id >= 0:
                    self.pool_hash_to_id[mesh_hash] = pool_id
                    print(f"Packing mesh rigid idx {idx}: NEW pool geometry #{pool_id}")

            # Link this rigid instance to its pool geometry
            if pool_id >= 0:
                _patch_array(self.instance_pool_id, idx, pool_id)

            # Store scale and offset (needed for transform operations)
            if rigid.transform is not None:
                scale_vec = rigid.transform.scale
                _patch_array(self.meshRigidScale, idx, wp.vec2(float(scale_vec[0]), float(scale_vec[1])))
                # Absorb offset into the body center so that the physics
                # pivot (rigidParams[idx,0]) coincides with the geometry
                # center.  This ensures correct lever arms in impulse
                # computations (applyImpulseAtPoint).  Pool boundary lrs
                # remain relative to the original mesh center, and the
                # updateBBox formula  center + offset + R @ (lr*scale)
                # still produces the correct world coords because offset
                # is now zero.
                offset_vec = rigid.transform.offset
                offset_ti = wp.vec2(float(offset_vec[0]), float(offset_vec[1]))
                params = self.rigidParams.numpy()
                params[idx, 0] = (
                    float(params[idx, 0][0]) + float(offset_vec[0]),
                    float(params[idx, 0][1]) + float(offset_vec[1]),
                )
                self.rigidParams.assign(params)
                _patch_array(self.meshRigidOffset, idx, wp.vec2(0.0, 0.0))
            else:
                # Default to uniform scale of 1.0
                _patch_array(self.meshRigidScale, idx, wp.vec2(1.0, 1.0))
                _patch_array(self.meshRigidOffset, idx, wp.vec2(0.0, 0.0))

            # Maintain index mappings and metadata for backward compatibility
            mesh_local_idx = self.numMesh
            _patch_array(self.rigid2MeshIndices, idx, mesh_local_idx)
            _patch_array(self.mesh2RigidIndices, mesh_local_idx, idx)

            # ==== OPTIMIZATION: Fast element copy using mesh instancing ====
            # If this mesh reuses pool geometry, directly copy elements from pool
            # Otherwise, build new element array
            num_boundary_nodes = rigid.mesh.numBoundNodes
            num_boundary_elements = rigid.mesh.numBoundElements

            # Allocate space in legacy arrays (still needed for contact detection)
            node_offset = self.totalBoundaryNodes
            elem_offset = self.totalBoundaryElements

            # Safety checks
            if (node_offset + num_boundary_nodes) > self.MAX_BOUNDARY_NODES:
                print(
                    f"\033[91mError: Exceeded MAX_BOUNDARY_NODES ({self.MAX_BOUNDARY_NODES})! Current total: {node_offset}, attempting to add: {num_boundary_nodes}\033[0m"
                )
                raise RuntimeError("Exceeded MAX_BOUNDARY_NODES")

            if (elem_offset + num_boundary_elements) > self.MAX_BOUNDARY_ELEMENTS:
                print(
                    f"\033[91mError: Exceeded MAX_BOUNDARY_ELEMENTS ({self.MAX_BOUNDARY_ELEMENTS})! Current total: {elem_offset}, attempting to add: {num_boundary_elements}\033[0m"
                )
                raise RuntimeError("Exceeded MAX_BOUNDARY_ELEMENTS")

            # Record metadata
            _patch_array(self.meshBoundaryNodeOffset, mesh_local_idx, node_offset)
            _patch_array(self.meshBoundaryNodeCount, mesh_local_idx, num_boundary_nodes)
            _patch_array(self.meshBoundaryElementOffset, mesh_local_idx, elem_offset)
            _patch_array(self.meshBoundaryElementCount, mesh_local_idx, num_boundary_elements)

            # OPTIMIZATION: Reuse element connectivity from pool if available
            if pool_id >= 0:
                # Fast path: copy from pool using Taichi kernel
                pool_elem_offset = int(self.pool_elem_offset.numpy()[pool_id])
                self._copy_elements_from_pool_kernel(elem_offset, pool_elem_offset, num_boundary_elements, node_offset)
            else:
                # Slow path: build element array (only for first instance of unique geometry)
                boundary_node_map = {}
                for local_bid in range(num_boundary_nodes):
                    global_nid = int(mesh.boundaryNodes[local_bid])
                    boundary_node_map[global_nid] = local_bid

                # Build remapped element array in numpy
                elem_array = np.zeros((num_boundary_elements, 3), dtype=np.int32)
                for eid in range(num_boundary_elements):
                    elem_conn = mesh.boundaryElements[eid]
                    elem_array[eid] = [
                        boundary_node_map[int(elem_conn[0])],
                        boundary_node_map[int(elem_conn[1])],
                        -1,
                    ]


                # Batch copy using Taichi kernel (much faster than Python loop)
                self._batch_copy_elements_kernel(elem_offset, node_offset, elem_array)

            self.totalBoundaryNodes += num_boundary_nodes
            self.totalBoundaryElements += num_boundary_elements
            self.numMesh += 1

        # consider angle for mesh rigid, box, capsule
        # 2D: just store the scalar angle
        _patch_array(self.quat, idx, rigid.angle)
        _patch_array(self.quat_initial, idx, rigid.angle)
        _patch_array(self.visual_angle, idx, 0.0)

        # Store inertia if available on python-side rigid object
        ib = rigid.inertia_body
        _patch_array(self.inertia, idx, float(ib))
      

    def resetRigidParams(self, rigid, idx: int):
        # Reference point (absorb transform offset so physics center = geometry center)
        ref = np.asarray(rigid.getRefPoint(), dtype=np.float32).reshape(-1)
        if hasattr(rigid, "transform") and rigid.transform is not None:
            offset_vec = rigid.transform.offset
            ref = np.array(
                [float(ref[0]) + float(offset_vec[0]), float(ref[1]) + float(offset_vec[1])],
                dtype=np.float32,
            )
        _patch_array(self.rigidParams, (idx, 0), ref)

        # consider angle for mesh rigid, box, capsule
        if hasattr(rigid, "angle"):
            # 2D: just store the scalar angle
            _patch_array(self.quat, idx, rigid.angle)
            _patch_array(self.quat_initial, idx, rigid.angle)
            _patch_array(self.visual_angle, idx, 0.0)

        else:
            _patch_array(self.quat, idx, 0.0)
            _patch_array(self.quat_initial, idx, 0.0)
            _patch_array(self.visual_angle, idx, 0.0)

    def _register_compound_shapes(self, rigid_idx, collision_shapes):
        """Register compound sub-colliders for a rigid body.

        Args:
            rigid_idx: index of the parent rigid body
            collision_shapes: list of dicts with keys 'type' (RigidType), 'local_pos' (list/tuple), 'radius' (float)
        """
        n = len(collision_shapes)
        offset = self.num_compound_shapes
        if offset + n > self.MAX_COMPOUND_SHAPES:
            print(f"\033[91mError: Compound shape pool exhausted ({offset + n} > {self.MAX_COMPOUND_SHAPES})\033[0m")
            raise RuntimeError("Exceeded MAX_COMPOUND_SHAPES")
        _patch_array(self.compound_count, rigid_idx, n)
        _patch_array(self.compound_offset, rigid_idx, offset)
        for k, shape in enumerate(collision_shapes):
            idx = offset + k
            pos = shape["local_pos"]

            _patch_array(self.compound_local_pos, idx, wp.vec2(float(pos[0]), float(pos[1])))
            _patch_array(self.compound_radius, idx, float(shape["radius"]))
            _patch_array(self.compound_type, idx, int(shape)["type"])
        self.num_compound_shapes = offset + n

    def processDomains_(self, domains):
        # Process rigid domains - attach logic moved to RigidBodyDomain.attach()
        for i, domain in enumerate(domains):
            if domain.type == DomainType.RIGID:
                if self.numRigids >= self.MAX_NODES:
                    print(
                        f"\033[91mError: Exceeded MAX_NODES ({self.MAX_NODES}) for rigid bodies! Cannot add more rigids.\033[0m"
                    )
                    raise RuntimeError(f"Exceeded maximum rigid bodies ({self.MAX_NODES})")

                rigid_idx = self.numRigids
                domain.attach(self, rigid_idx, i)
                self.packRigidParams(domain.rigid, rigid_idx)

                # Register compound collision sub-shapes if present
                collision_shapes = getattr(domain, "collision_shapes", None)
                if collision_shapes and len(collision_shapes) > 0:
                    self._register_compound_shapes(rigid_idx, collision_shapes)

                # Set environment ID for collision filtering (batched training)
                if hasattr(domain, "env_id"):
                    _patch_array(self.rigid_env_id, rigid_idx, domain.env_id)

                self.numRigids += 1
                self.needUpdate = True
                # Here I assume the maximum velocity is 100.0
                # so the stable time should be at least 0.1 * rigid.minSize / 100.0
                # self.stableTime = min(self.stableTime, domain.rigid.minSize / 100.0)  # too large time step will lead to inaccurate results although stable
                if domain.considerContact:
                    self.numRigidGroundContact += 1
                    if self.considerRigidRigidContact:
                        self.numRigidInContact += 1
                    if domain.rigid.rtype == RigidType.MESH:
                        self.numMeshRigidInContact += 1
                elif domain.considerGroundContact:
                    self.numRigidGroundContact += 1

        # Process analytical domains (planes, heightfields, voxel maps)
        for i, domain in enumerate(domains):
            if (
                domain.type == DomainType.ANALYTICAL
                or domain.type == DomainType.HEIGHTFIELD
                or domain.type == DomainType.VOXELMAP
            ):
                if self.numAnalytical >= self.MAX_ANAL:
                    print(
                        f"\033[91mError: Exceeded MAX_ANAL ({self.MAX_ANAL}) for analytical domains! Cannot add more analytical domains.\033[0m"
                    )
                    raise RuntimeError(f"Exceeded maximum analytical domains ({self.MAX_ANAL})")

                _patch_array(
                    self.rigidDomainIds,
                    self.numAnalytical + self.numRigids,
                    wp.vec3i(i, 0, 1),
                )  # type 0 for ground, type 1 for considerContact as default
                anal_idx = self.numAnalytical + self.numRigids
                category_bits = int(getattr(domain, "category_bits", COLLISION_CATEGORY_GROUND)) & 0b11111111
                collide_bits = int(getattr(domain, "collide_bits", COLLISION_MASK_ALL)) & 0b11111111
                _patch_array(self.category_bits, anal_idx, category_bits)
                _patch_array(self.collide_bits, anal_idx, collide_bits)
                _patch_array(self.mass, self.numAnalytical + self.numRigids, 1e10)# very large mass to simulate immovable object
                domain.attach(self, self.numAnalytical + self.numRigids)
                _patch_array(self.domainToRigid, i, self.numAnalytical + self.numRigids)
                if domain.type == DomainType.ANALYTICAL:
                    _patch_array(self.rigidParams, (self.numRigids + self.numAnalytical, 0), domain.point)
                    _patch_array(self.rigidParams, (self.numRigids + self.numAnalytical, 1), domain.normal)
                    if len(domain.bcs) > 0:
                        self.needUpdate = True
                elif domain.type in (DomainType.HEIGHTFIELD, DomainType.VOXELMAP):
                    self.hasHeightFieldOrVoxel = True
                self.numAnalytical += 1


    def processJoints(self, joints):
        for i in range(len(joints)):
            joints[i].attach(self)
            self.numAnchors += 1
            self.needUpdate = True

    def _addBcValue(self, idx, type, value):
        cur = int(self.bcNodes.numpy()[idx])
        _patch_array(self.bcNodes, idx, cur | int(type))

        if type == UTYPE:
            _patch_array(self.bcTValues, idx, wp.vec2(0.0, 0.0))
            _patch_array(self.mass, idx, 1e10)  # large mass to fix in space
        elif type == RTYPE:
            _patch_array(self.bcTValues, idx, wp.vec2(0.0, 0.0))
            _patch_array(self.mass, idx, 1e10)  # large mass to fix in space
            _patch_array(self.inertia, idx, 1e10)
            _patch_array(self.bcRValues, idx, 0.0)
        elif type == VTYPE:
            _patch_array(self.bcTValues, idx, value)
            _patch_array(self.mass, idx, 1e10)

        elif type == ROTVTYPE:
            _patch_array(self.bcRValues, idx, value)
            _patch_array(self.inertia, idx, 1e10)

        elif type == ROTATYPE or type == TORQUETYPE:
            _patch_array(self.bcRValues, idx, value)

        elif type == GRAVITY:
            _patch_array(self.bcGValues, idx, value)
        else:
            _patch_array(self.bcTValues, idx, value)

    def processConditions(self):
        domain_ids = self.rigidDomainIds.numpy()
        for i in range(self.numRigids):
            idx = int(domain_ids[i][0])
            domain = self.domains[idx]
            for bc in domain.bcs:
                type, nodes, value = bc.processData()
                self._addBcValue(i, type, value)

        for i in range(self.numAnalytical):
            idx = int(domain_ids[i + self.numRigids][0])
            domain = self.domains[idx]
            for bc in domain.bcs:
                type, nodes, value = bc.processData()
                self._addBcValue(i + self.numRigids, type, value)
                self.movingAnalytical = True

        for i in range(self.numAnchors):
            joint = self.joints[i]
            for bc in joint.bcs:
                type, nodes, value = bc.processData()
                target_value = 0.0
                if type == ROTVTYPE or type == VTYPE:
                    _patch_array(self.joint_motor_target_mode, i, 0)
                elif type == ROTATYPE or type == ATYPE:
                    _patch_array(self.joint_motor_target_mode, i, 1)

                if isinstance(value, (list, tuple, np.ndarray)):
                    axis_np = self.joint_axis.numpy()[i]
                    axis_norm = np.linalg.norm(axis_np)
                    if axis_norm > 1e-9:
                        axis_np = axis_np / axis_norm
                        target_value = float(np.dot(np.asarray(value[: self.d], dtype=np.float32), axis_np))
                    else:
                        target_value = float(value[0])  # fallback to first component if axis is degenerate
                else:
                    target_value = float(value)

                _patch_array(self.joint_control_target, i, target_value)

    # -------  Rigid-Rigid contact related functions ----------------------------
    # ---------------------------------------------------------------------------

    def detect_all_contacts(self):
        """Python-side dispatcher for contact detection.

        Primitive rigid-rigid contacts now share one kernel entry and two
        bottom-layer geometry families:
        - convex-convex via shared GJK/EPA helpers
        - point-vs-primitive via shared SDF helpers

        Mesh kernels stay separately gated because they depend on spatial-hash
        infrastructure and should not be compiled when a scene has no meshes.
        """
        # Primitive contacts: use a single dispatch kernel so first-frame JIT
        # only needs one kernel entry for primitive rigid-rigid contacts.
        if int(self.num_primitive_pairs.numpy()[0]) > 0:
            self._detect_primitive_contacts_kernel()
        # Mesh and mixed: only compile/launch when spatialHash exists.
        # When spatialHash is None (no mesh rigids), these kernels reference
        has_mesh_related_pairs = (int(self.num_mesh_pairs.numpy()[0]) + int(self.num_mixed_pairs.numpy()[0])) > 0
        if self.spatialHash is not None and has_mesh_related_pairs:
            self.maybe_rebuild_spatial_hash()
            if int(self.num_mesh_pairs.numpy()[0]) > 0:
                self.detect_mesh_mesh_contacts_kernel()
            if int(self.num_mixed_pairs.numpy()[0]) > 0:
                self.detect_mixed_contacts_kernel()

    # ── Primitive contact dispatch ──
    # A single kernel entry reduces first-frame JIT overhead for scenes that
    # only need a small subset of primitive contact types.

    def _dispatch_primitive_contact(self, rigid_a: int, rigid_b: int):
        domain_ids = self.rigidDomainIds.numpy()
        type_a = int(domain_ids[rigid_a][1])
        type_b = int(domain_ids[rigid_b][1])
        contact_type = type_a | type_b

        if contact_type == RigidContactType.BALLBALL:
            self.detectBallBallContact_(rigid_a, rigid_b)

        elif contact_type == RigidContactType.BOXBOX:
            self.detectBoxBoxContact_(rigid_a, rigid_b)

        elif contact_type == RigidContactType.BOXBALL:
            box_idx = rigid_a
            ball_idx = rigid_b
            if type_a == RigidType.BALL:
                box_idx = rigid_b
                ball_idx = rigid_a
            self.detectBoxBallContact_(box_idx, ball_idx)

        elif contact_type == RigidContactType.CAPSULEBOX:
            seg_idx = rigid_a
            box_idx = rigid_b
            if type_a == RigidType.BOX:
                seg_idx = rigid_b
                box_idx = rigid_a
            self.detectSegmentBoxContact_(seg_idx, box_idx)

        elif contact_type == RigidContactType.CAPSULEBALL:
            seg_idx = rigid_a
            ball_idx = rigid_b
            if type_a == RigidType.BALL:
                seg_idx = rigid_b
                ball_idx = rigid_a
            self.detectSegmentBallContact_(seg_idx, ball_idx)

        elif contact_type == RigidContactType.CAPSULECAPSULE:
            self.detectSegmentSegmentContact_(rigid_a, rigid_b)

    def _detect_primitive_contacts_kernel(self):
        n = int(self.num_primitive_pairs.numpy()[0])
        if n <= 0:
            return
        wp.launch(
            _detect_primitive_contacts_wp,
            dim=1,
            inputs=[
                n,
                int(self.MAX_CONTACTS),
                float(self.restitution_velocity_threshold),
                int(self.MAX_ENVS_ALLOC),
                int(self.MAX_CC_PER_ENV),
                self.primitive_pairs_buffer,
                self.rigidDomainIds,
                self.rigidParams,
                self.radius,
                self.cached_rotation_matrix,
                self.V,
                self.RotV,
                self.contactParams,
                self.rigid_env_id,
                self.num_contacts,
                self.contact_rigid_a,
                self.contact_rigid_b,
                self.contact_point,
                self.contact_normal,
                self.contact_depth,
                self.contact_bounce_vel,
                self.contact_tangent1,
                self.contact_count_per_rigid,
                self.contact_env_count,
                self.contact_env_idx,
            ],
        )

    def detect_analyticalprim_contacts_kernel(self):
        """Detect and resolve collisions between analytical planes and rigids."""
        n = int(self.num_groundprim_pairs.numpy()[0])
        if n <= 0:
            return
        wp.launch(
            _detect_analytical_prim_contacts_wp,
            dim=1,
            inputs=[
                n,
                0.0005,
                int(self.MAX_GROUND_CONTACTS),
                float(self.restitution_velocity_threshold),
                self._ground_use_aabb_early_out,
                self.groundprim_pairs_buffer,
                self.rigidDomainIds,
                self.rigidParams,
                self.radius,
                self.V,
                self.RotV,
                self.contactParams,
                self.rigid_env_id,
                self.cached_rotation_matrix,
                self.aabb,
                self.compound_count,
                self.compound_offset,
                self.compound_local_pos,
                self.compound_radius,
                self.num_ground_contacts,
                self.ground_contact_rigid,
                self.ground_contact_point,
                self.ground_contact_normal,
                self.ground_contact_vel,
                self.ground_contact_depth,
                self.ground_contact_bounce_vel,
                self.ground_contact_tangent1,
                self.ground_contact_env_count,
                self.ground_contact_env_idx,
                int(self.MAX_ENVS_ALLOC),
                int(self.MAX_GC_PER_ENV),
            ],
        )

    def detect_mesh_mesh_contacts_kernel(self):
        """Detect and resolve collisions between two mesh rigids."""
        n = int(self.num_mesh_pairs.numpy()[0])
        if n <= 0:
            return
        pairs = self.mesh_pairs_buffer.numpy()
        for i in range(n):
            ic = int(pairs[i][0])
            jc = int(pairs[i][1])
            self.detectMeshMeshContact_(ic, jc)

    def detect_mixed_contacts_kernel(self):
        """Detect and resolve collisions between a mesh rigid and a primitive rigid."""
        n = int(self.num_mixed_pairs.numpy()[0])
        if n <= 0:
            return
        pairs = self.mixed_pairs_buffer.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        for i in range(n):
            ic = int(pairs[i][0])
            jc = int(pairs[i][1])
            rigid1Type = int(domain_ids[ic][1])

            # Determine which index is the mesh and which is primitive
            mesh_idx = jc
            other_idx = ic
            if rigid1Type == RigidType.MESH:
                mesh_idx = ic
                other_idx = jc

            self.detectMeshPrimitiveContact_(mesh_idx, other_idx)

    # ===========================================================================
    # ===== All the followings are rigid-rigid contact detection functions ======
    # ===========================================================================

    def _capsule_segment_endpoints(self, seg_id: int):
        params = self.rigidParams.numpy()
        rot = self.cached_rotation_matrix.numpy()
        center = params[seg_id, 0]
        lcdir = params[seg_id, 1]
        lc = rot[seg_id] @ lcdir + center
        uc = center * 2.0 - lc
        return lc, uc

    def detectBallBallContact_(self, ic, jc):
        """Handle ball-ball instantaneous collision response by velocity impulse."""
        params = self.rigidParams.numpy()
        radius_np = self.radius.numpy()
        radius = radius_np[ic] + radius_np[jc]
        p = params[ic, 0] - params[jc, 0]
        l = wp.length(p)
        if l < radius:
            n = p / l
            cpoint_mid = (params[ic, 0] + params[jc, 0]) * 0.5
            self.cacheContact(ic, jc, cpoint_mid, n, l - radius)

    def detectSegmentSegmentContact_(self, id1, id2):
        """Handle collision between two segment-like rigids (capsule) using GJK+EPA.

        - Capsule-capsule : use closest segment method
        """
        self.detectCapsuleCapsuleContact_(id1, id2)

    def detectCapsuleCapsuleContact_(self, id1, id2):
        """Check capsule-capsule contact using closest points on segments."""
        params = self.rigidParams.numpy()
        rot = self.cached_rotation_matrix.numpy()
        radius_np = self.radius.numpy()

        center1 = params[id1, 0]
        lcdir1 = params[id1, 1]
        lc1 = rot[id1] @ lcdir1 + center1
        uc1 = center1 * 2 - lc1
        r1 = radius_np[id1]

        center2 = params[id2, 0]
        lcdir2 = params[id2, 1]
        lc2 = rot[id2] @ lcdir2 + center2
        uc2 = center2 * 2 - lc2
        r2 = radius_np[id2]

        p, q, t1, t2 = calMinDisSegment2Segment(lc1, uc1, lc2, uc2)
        pq = q - p
        dis = wp.length(pq)

        # Compute normal direction from p to q (fallback unit-x)
        normal = wp.vec2(1.0, 0.0)
        if dis > 1e-9:
            normal = pq / dis
        penetration = 1.0

        # For capsule-vs-capsule the classic formula applies (point-sphere ends)
        penetration = dis - (r1 + r2)

        if penetration < 0.0:
            # apply symmetric impulses at the closest points p (on id1) and q (on id2)
            cpoint1 = p
            cpoint2 = q

            # apply a symmetric impulse pair at the midpoint between the segments
            cpoint_mid = (cpoint1 + cpoint2) * 0.5
            self.cacheContact(id1, id2, cpoint_mid, -normal, penetration)

    def detectBoxBoxContact_(self, ic, jc):
        """Detect box-box contact via 2D OBB resolver in sat.py."""
        params = self.rigidParams.numpy()
        rot = self.cached_rotation_matrix.numpy()
        center_i = params[ic, 0]
        center_j = params[jc, 0]
        half_i = params[ic, 1] * 0.5
        half_j = params[jc, 1] * 0.5
        hit, penetration, normal_ij, cpoint = obb2d_contact_quad_vs_quad(
            center_i,
            half_i,
            rot[ic],
            center_j,
            half_j,
            rot[jc],
        )
        if hit == 1:
            self.cacheContact(ic, jc, cpoint, -normal_ij, penetration)

    def detectBoxBallContact_(self, ic, jc):
        """Test box representative vertices vs a sphere and apply forces/torques."""
        params = self.rigidParams.numpy()
        rot = self.cached_rotation_matrix.numpy()
        radius_np = self.radius.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        pos = params[jc, 0]
        l, n, _ = detectPointToPrimitive(
            pos,
            int(domain_ids[ic][1]),
            params[ic, 0],
            params[ic, 1],
            rot[ic],
            radius_np[ic],
        )
        l -= radius_np[jc]
        if l < 0:
            n = (n / (wp.length(n) + 1e-9)) if wp.length(n) > 1e-9 else (pos - params[ic, 0]) / (
                wp.length(pos - params[ic, 0]) + 1e-9
            )
            cpoint = pos - n * radius_np[jc]
            self.cacheContact(jc, ic, cpoint, n, l)

    def detectSegmentBoxContact_(self, seg_id, other_id):
        """Detect capsule vs box using 2D SAT (OBB quad vs segment), with capsule radius."""
        radius_np = self.radius.numpy()
        a0 = self.get_box_vertex(other_id, 0)
        a1 = self.get_box_vertex(other_id, 1)
        a2 = self.get_box_vertex(other_id, 2)
        a3 = self.get_box_vertex(other_id, 3)
        lc, uc = self._capsule_segment_endpoints(seg_id)
        signed, normal = obb2d_signed_distance_quad_vs_segment(a0, a1, a2, a3, lc, uc)
        penetration = signed - radius_np[seg_id]
        if penetration < 0.0:
            cpoint = (lc + uc) * 0.5
            self.cacheContact(other_id, seg_id, cpoint, -normal, penetration)

    def detectSegmentBallContact_(self, seg_id, other_id):
        """segment-point contact using SDF queries (kept for ball interactions).

        - Capsule vs Ball (SDF method)
        """
        params = self.rigidParams.numpy()
        rot = self.cached_rotation_matrix.numpy()
        radius_np = self.radius.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        pos = params[other_id, 0]
        dis, normal, _ = detectPointToPrimitive(
            pos,
            int(domain_ids[seg_id][1]),
            params[seg_id, 0],
            params[seg_id, 1],
            rot[seg_id],
            radius_np[seg_id],
        )

        penetration = dis
        penetration -= radius_np[other_id]

        if penetration < 0.0:
            nrm = wp.length(normal)
            n = wp.vec2(0.0, 0.0)
            if nrm > 1e-9:
                n = normal / nrm
            else:
                n = wp.vec2(1.0, 0.0)

            cpoint = pos - normal * dis
            self.cacheContact(other_id, seg_id, cpoint, n, penetration)

    def detectMeshPrimitiveContact_(self, mesh_idx: int, other_idx: int):
        """Handle mesh-primitive contacts (Optimized with Spatial Hash).

        Uses Spatial Hash for primitive-sample-vs-mesh checks.
        Uses direct node iteration for mesh-node-vs-primitive checks.
        """
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        params = self.rigidParams.numpy()
        rot = self.cached_rotation_matrix.numpy()
        radius_np = self.radius.numpy()
        rigid2mesh = self.rigid2MeshIndices.numpy()
        mesh_node_off = self.meshBoundaryNodeOffset.numpy()
        mesh_node_cnt = self.meshBoundaryNodeCount.numpy()
        mesh_coords = self.meshBoundaryCoords.numpy()

        mesh_local_idx = int(rigid2mesh[mesh_idx])
        node_offset = int(mesh_node_off[mesh_local_idx])
        num_nodes = int(mesh_node_cnt[mesh_local_idx])

        mesh_dom = int(domain_ids[mesh_idx][0])
        other_dom = int(domain_ids[other_idx][0])
        intersect_lb = np.maximum(aabb[mesh_dom, 0], aabb[other_dom, 0])
        intersect_ub = np.minimum(aabb[mesh_dom, 1], aabb[other_dom, 1])

        limit_penetration = float(np.min(aabb[mesh_dom, 1] - aabb[mesh_dom, 0]) * 0.1)
        intersect_lb = intersect_lb - limit_penetration
        intersect_ub = intersect_ub + limit_penetration

        other_type = int(domain_ids[other_idx][1])

        test_points = mat28(0.0)
        num_samples = 0
        p_radius = 0.0

        if other_type == RigidType.BALL:
            center = params[other_idx, 0]
            test_points[0, 0] = float(center[0])
            test_points[1, 0] = float(center[1])
            p_radius = float(radius_np[other_idx])
            num_samples = 1

        elif other_type == RigidType.BOX:
            center = params[other_idx, 0]
            extent = params[other_idx, 1]
            half_ext = extent * 0.5
            rot_m = rot[other_idx]

            num_cnt = 2**self.d
            num_samples = num_cnt
            for nid in range(num_cnt):
                local_pos = np.zeros(self.d, dtype=np.float32)
                for k in range(self.d):
                    local_pos[k] = half_ext[k] if (nid >> k) & 1 else -half_ext[k]
                world = center + rot_m @ local_pos
                test_points[0, nid] = float(world[0])
                test_points[1, nid] = float(world[1])
            p_radius = 0.0

        elif other_type == RigidType.CAPSULE:
            center = params[other_idx, 0]
            lcdir = params[other_idx, 1]
            lc = rot[other_idx] @ lcdir + center
            uc = center * 2 - lc
            p_radius = float(radius_np[other_idx])

            num_s = 2
            num_samples = num_s
            for k in range(num_s):
                t = k / (num_s - 1) if num_s > 1 else 0.5
                sample = lc * (1 - t) + uc * t
                test_points[0, k] = float(sample[0])
                test_points[1, k] = float(sample[1])

        for nid_local in range(num_nodes):
            coord = mesh_coords[node_offset + nid_local]
            if (coord - intersect_ub).max() <= 0.0 and (intersect_lb - coord).max() <= 0.0:
                self._mesh_node_vs_primitive(mesh_idx, other_idx, coord)

        for k in range(num_samples):
            p = wp.vec2(test_points[0, k], test_points[1, k])
            self._prim_point_vs_mesh_with_sh(p, p_radius, mesh_idx, other_idx, limit_penetration)

    def _mesh_node_vs_primitive(self, mesh_idx: int, other_idx: int, pos):
        """Thin dispatch over shared point-vs-primitive SDF family helpers."""
        params = self.rigidParams.numpy()
        rot = self.cached_rotation_matrix.numpy()
        radius_np = self.radius.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        pos = np.asarray(pos, dtype=np.float32).reshape(-1)
        center = np.asarray(params[other_idx, 0], dtype=np.float32).reshape(-1)
        prim = np.asarray(params[other_idx, 1], dtype=np.float32).reshape(-1)
        R = np.asarray(rot[other_idx], dtype=np.float32).reshape(2, 2)
        l, n, _ = detect_point_to_primitive_np(
            pos,
            int(domain_ids[other_idx][1]),
            center,
            prim,
            R,
            float(radius_np[other_idx]),
        )
        if l < 0.0:
            cpoint = pos - np.asarray([float(n[0]), float(n[1])], dtype=np.float32) * float(l)
            self.cacheContact(mesh_idx, other_idx, cpoint, n, l)

    def _prim_point_vs_mesh_with_sh(
        self, point, radius: float, mesh_idx: int, other_idx: int, limit_penetration: float
    ):
        """Test a point (with optional radius) against mesh using spatial hash."""
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        point_wp = wp.vec2(float(point[0]), float(point[1]))
        cell_size = float(np.min(self.spatialHash.gridSize.numpy()[0]))
        if radius < cell_size * 4.0:
            numPotentials = self.spatialHash.queryPointWithBuffer(point_wp, radius, mesh_idx)
            mesh_elements = self.meshBoundaryElements.numpy()
            query_elids = self.spatialHash.queryElids.numpy()
            mesh_coords_arr = self.meshBoundaryCoords

            for pot_idx in range(numPotentials):
                eidx = int(query_elids[pot_idx])
                if eidx >= 0:
                    conn = mesh_elements[eidx]
                    mesh_coords_np = mesh_coords_arr.numpy()
                    penetration, normal, cpoint, _ = detect_point_to_mesh_boundary_np(
                        point, mesh_coords_np, conn, limit_penetration=limit_penetration + radius
                    )

                    l = penetration - radius
                    if l < 0.0:
                        self.cacheContact(other_idx, mesh_idx, cpoint, normal, l)

    def detectMeshMeshContact_(self, ic: int, jc: int):
        """Test mesh ic boundary nodes against mesh jc boundary elements using spatial hash acceleration.

        OPTIMIZED: Split into two separate passes to reduce JIT compilation complexity.
        Each pass is now a separate function to reduce branching and improve compilation time.

        TODO: if several nodes are in contact, we need a manifold reduction strategy to avoid over-correction during PGS iteration. 
        We can select representative contacts based on penetration depth and spatial distribution across the contact patch.
        """
        # First pass: ic nodes vs jc triangles
        self._mesh_nodes_vs_mesh_elements(ic, jc)

        # Second pass: jc nodes vs ic triangles
        self._mesh_nodes_vs_mesh_elements(jc, ic)

    def _mesh_nodes_vs_mesh_elements(self, node_mesh_rigid_idx: int, elem_mesh_rigid_idx: int):
        """Test boundary nodes of node_mesh against boundary elements of elem_mesh."""
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        rigid2mesh = self.rigid2MeshIndices.numpy()
        mesh_node_off = self.meshBoundaryNodeOffset.numpy()
        mesh_node_cnt = self.meshBoundaryNodeCount.numpy()
        mesh_coords = self.meshBoundaryCoords.numpy()
        mesh_elements = self.meshBoundaryElements.numpy()
        mesh_coords_arr = self.meshBoundaryCoords

        node_mesh_idx = int(rigid2mesh[node_mesh_rigid_idx])
        elem_mesh_idx = int(rigid2mesh[elem_mesh_rigid_idx])

        node_offset = int(mesh_node_off[node_mesh_idx])
        num_nodes = int(mesh_node_cnt[node_mesh_idx])

        node_dom = int(domain_ids[node_mesh_rigid_idx][0])
        elem_dom = int(domain_ids[elem_mesh_rigid_idx][0])
        intersect_lb = np.maximum(aabb[node_dom, 0], aabb[elem_dom, 0])
        intersect_ub = np.minimum(aabb[node_dom, 1], aabb[elem_dom, 1])
        limit_penetration = float(
            0.1 * np.min(aabb[elem_dom, 1] - aabb[elem_dom, 0])
        )

        intersect_lb = intersect_lb - limit_penetration
        intersect_ub = intersect_ub + limit_penetration

        for nidx in range(num_nodes):
            coord = mesh_coords[node_offset + nidx]
            coord_wp = wp.vec2(float(coord[0]), float(coord[1]))

            if ((coord - intersect_lb).min() > 0.0) and ((coord - intersect_ub).max() < 0.0):
                query_buf = float(self._sh_contact_margin.numpy()[0])
                numPotentials = self.spatialHash.queryPointWithBuffer(
                    coord_wp, query_buf, elem_mesh_rigid_idx
                )
                query_elids = self.spatialHash.queryElids.numpy()

                for pot_idx in range(numPotentials):
                    elem_idx = int(query_elids[pot_idx])
                    if elem_idx >= 0:
                        conn = mesh_elements[elem_idx]
                        conn_wp = wp.vec3i(int(conn[0]), int(conn[1]), int(conn[2]))

                        lb = np.array([1e30, 1e30], dtype=np.float32)
                        ub = np.array([-1e30, -1e30], dtype=np.float32)
                        for j in range(self.d):
                            if int(conn[j]) >= 0:
                                tri_coord = mesh_coords[int(conn[j])]
                                lb = np.minimum(lb, tri_coord[:2])
                                ub = np.maximum(ub, tri_coord[:2])

                        lb = lb - limit_penetration
                        ub = ub + limit_penetration

                        if np.max(lb - coord[:2]) > 0.0 or np.max(coord[:2] - ub) > 0.0:
                            continue

                        penetration, normal, cpoint, _ = detect_point_to_mesh_boundary_np(
                            coord,
                            mesh_coords,
                            conn,
                            limit_penetration=limit_penetration,
                        )

                        if penetration < 0.0:
                            self.cacheContact(node_mesh_rigid_idx, elem_mesh_rigid_idx, cpoint, normal, penetration)

    # ================== The end of contact detection functions ==========================

    # -------  End of Rigid-Rigid contact related functions -----------------------------
    # -----------------------------------------------------------------------------------

    # -------  Rigid-Ground contact related functions -----------------------------
    # ----------------------------------------------------------------------------------

    def detectRigidGroundContact(self):
        """Run analytical-vs-rigid and analytical-vs-mesh contact detectors.

        OPTIMIZATION: Analytical plane contacts use kernelized dispatch.
        HeightField/Voxel contacts require Python loops (wp.template() args),
        but are skipped entirely when hasHeightFieldOrVoxel is False.

        When heightfield/voxel domains exist, we must NOT run the blanket
        analytical-plane kernel for ALL groundprim pairs, because it would
        treat heightfield/voxel domains as flat planes and generate incorrect
        duplicate contacts.  Instead, we dispatch each pair to the correct
        handler in the Python loop.
        """
        if not self.hasHeightFieldOrVoxel:
            # ── Fast path: all analytical domains are simple planes ──
            # Always launch both kernels — they read pair counts from GPU
            # memory internally and exit immediately when count == 0.
            # This eliminates two GPU→CPU syncs per rigid substep.
            self.detect_analyticalprim_contacts_kernel()
            self.detect_analyticalmesh_contacts_kernel()
            return

        # ── Slow path: mix of analytical planes, heightfields, and voxelmaps ──
        # Primitive rigids vs ground domains
        n_gp = int(self.num_groundprim_pairs.numpy()[0])
        if n_gp > 0:
            gp_pairs = self.groundprim_pairs_buffer.numpy()
            domain_ids = self.rigidDomainIds.numpy()
            for i in range(n_gp):
                rigid_i, rigid_j = int(gp_pairs[i][0]), int(gp_pairs[i][1])  # rigid_i : ground, rigid_j : rigid
                anlDomain = self.domains[int(domain_ids[rigid_i][0])]
                if anlDomain.type == DomainType.HEIGHTFIELD and anlDomain.considerContact:
                    self.detectHeightField2Rigids_(rigid_j, anlDomain)
                elif anlDomain.type == DomainType.VOXELMAP and anlDomain.considerContact:
                    self.detectVoxel2Rigids_(rigid_j, anlDomain)
                elif anlDomain.considerContact:
                    # True analytical plane — single-pair kernel dispatch
                    self._detect_single_analyticalprim_kernel(int(rigid_i), int(rigid_j))

        # Mesh rigids vs ground domains
        n_gm = int(self.num_groundmesh_pairs.numpy()[0])
        if n_gm > 0:
            gm_pairs = self.groundmesh_pairs_buffer.numpy()
            domain_ids = self.rigidDomainIds.numpy()
            for i in range(n_gm):
                rigid_i, rigid_j = int(gm_pairs[i][0]), int(gm_pairs[i][1])  # rigid_i : ground, rigid_j : rigid
                anlDomain = self.domains[int(domain_ids[rigid_i][0])]
                if anlDomain.type == DomainType.HEIGHTFIELD and anlDomain.considerContact:
                    self.detectHeightField2MeshContacts_(rigid_j, anlDomain)
                elif anlDomain.type == DomainType.VOXELMAP and anlDomain.considerContact:
                    self.detectVoxel2MeshContacts_(rigid_j, anlDomain)
                elif anlDomain.considerContact:
                    # True analytical plane — single-pair kernel dispatch
                    self._detect_single_analyticalmesh_kernel(int(rigid_i), int(rigid_j))

    def _detect_single_analyticalprim_kernel(self, analIdx: int, rigidIdx: int):
        """Dispatch a single analytical-plane vs primitive-rigid contact check."""
        self.detectAnalaytical2Rigid(analIdx, rigidIdx)

    def _detect_single_analyticalmesh_kernel(self, analIdx: int, rigidIdx: int):
        """Dispatch a single analytical-plane vs mesh-rigid contact check."""
        self.detectAnalytical2MeshPair(analIdx, rigidIdx)

    def detect_analyticalmesh_contacts_kernel(self):
        """Detect analytical-plane vs mesh-rigid contacts, parallelized over ground-mesh pairs.

        Each GPU thread handles one (analIdx, rigidIdx) pair and iterates over its
        boundary nodes internally. This pair-level dispatch avoids the global atomic
        contention that node-level expansion would cause (2.6M atomics for 1024 envs).
        """
        n = int(self.num_groundmesh_pairs.numpy()[0])
        if n <= 0:
            return
        pairs = self.groundmesh_pairs_buffer.numpy()
        for i in range(n):
            analIdx = int(pairs[i][0])
            rigidIdx = int(pairs[i][1])
            self.detectAnalytical2MeshPair(analIdx, rigidIdx)

    def detectAnalaytical2Rigid(self, analIdx, rigidIdx):
        """Detect and resolve collisions between analytical plane and rigids."""
        params = self.rigidParams.numpy()
        V = self.V.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        radius_np = self.radius.numpy()
        rot = self.cached_rotation_matrix.numpy()
        compound_count = self.compound_count.numpy()
        compound_offset = self.compound_offset.numpy()
        compound_local_pos = self.compound_local_pos.numpy()
        compound_radius = self.compound_radius.numpy()

        planepoint = params[analIdx, 0]
        normal = params[analIdx, 1]
        anal_vel = V[analIdx]

        contact_margin = 0.0005
        run_narrow_phase = True

        if int(self._ground_use_aabb_early_out.numpy()[0]) == 1:
            domain_idx = int(domain_ids[rigidIdx][0])
            bbox_min = aabb[domain_idx, 0]
            bbox_max = aabb[domain_idx, 1]

            support_point = np.zeros(self.d, dtype=np.float32)
            for dim in range(self.d):
                support_point[dim] = bbox_max[dim] if normal[dim] < 0 else bbox_min[dim]

            min_dist = float(np.dot(support_point - planepoint, normal))
            run_narrow_phase = min_dist <= contact_margin

        if run_narrow_phase:
            n_sub = int(compound_count[rigidIdx])
            if n_sub > 0:
                base = int(compound_offset[rigidIdx])
                parent_center = params[rigidIdx, 0]
                R = rot[rigidIdx]
                for k in range(n_sub):
                    idx = base + k
                    local_p = compound_local_pos[idx]
                    r_sub = float(compound_radius[idx])
                    world_p = R @ local_p + parent_center
                    d_sub, _, _ = detectPointToAnalyticalPlane(world_p, planepoint, normal)
                    if d_sub < r_sub + contact_margin:
                        cpoint = world_p - normal * r_sub
                        depth = d_sub - r_sub
                        self.cacheGroundContact(rigidIdx, cpoint, normal, anal_vel, depth)
            else:
                type = int(domain_ids[rigidIdx][1])

                if type == RigidType.BOX:
                    num_verts = 4
                    for i in range(num_verts):
                        pos = self.get_box_vertex(rigidIdx, i)
                        d, _, _ = detectPointToAnalyticalPlane(pos, planepoint, normal)
                        if d < contact_margin:
                            self.cacheGroundContact(rigidIdx, pos, normal, anal_vel, d)

                elif type == RigidType.BALL:
                    center = params[rigidIdx, 0]
                    radius = float(radius_np[rigidIdx])
                    d, _, _ = detectPointToAnalyticalPlane(center, planepoint, normal)
                    if d < radius + contact_margin:
                        cpoint = center - normal * radius
                        depth = d - radius
                        self.cacheGroundContact(rigidIdx, cpoint, normal, anal_vel, depth)

                elif type == RigidType.CAPSULE:
                    center = params[rigidIdx, 0]
                    lcdir = params[rigidIdx, 1]
                    lc = rot[rigidIdx] @ lcdir + center
                    uc = center * 2.0 - lc
                    radius = float(radius_np[rigidIdx])

                    for ep in range(2):
                        test_p = lc if ep == 0 else uc
                        d_ep, _, _ = detectPointToAnalyticalPlane(test_p, planepoint, normal)
                        if d_ep < radius + contact_margin:
                            cpoint = test_p - normal * radius
                            depth = d_ep - radius
                            self.cacheGroundContact(rigidIdx, cpoint, normal, anal_vel, depth)

    def detectAnalytical2MeshPair(self, analIdx, rigidIdx):
        """Detect representative boundary-node contacts between an analytical plane
        and a mesh rigid.

        Instead of caching every penetrating boundary node (which can be dozens for
        mesh bodies and causes angular over-correction during PGS iteration), this
        selects up to 5 representative contacts that span the contact patch:
          - 1 deepest penetrating node
          - 2 extreme nodes along first tangent direction (max / min t1)
          - 2 extreme nodes along second tangent direction (max / min t2, 3D only)
        This matches the ~4-8 contacts that primitive box bodies generate.
        """
        params = self.rigidParams.numpy()
        V = self.V.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        rigid2mesh = self.rigid2MeshIndices.numpy()
        mesh_node_off = self.meshBoundaryNodeOffset.numpy()
        mesh_node_cnt = self.meshBoundaryNodeCount.numpy()
        mesh_coords = self.meshBoundaryCoords.numpy()

        planepoint = params[analIdx, 0]
        normal = params[analIdx, 1]
        domain_idx = int(domain_ids[rigidIdx][0])
        bbox_min = aabb[domain_idx, 0]
        bbox_max = aabb[domain_idx, 1]

        support_point = np.zeros(self.d, dtype=np.float32)
        for dim in range(self.d):
            support_point[dim] = bbox_max[dim] if normal[dim] < 0 else bbox_min[dim]
        min_dist = float(np.dot(support_point - planepoint, normal))

        if min_dist < 0.0:
            mesh_local_idx = int(rigid2mesh[rigidIdx])
            if mesh_local_idx >= 0:
                node_offset = int(mesh_node_off[mesh_local_idx])
                num_nodes = int(mesh_node_cnt[mesh_local_idx])
                anal_vel = V[analIdx]

                t1 = np.array([-normal[1], normal[0]], dtype=np.float32)

                has_contact = False
                deep_d = 0.0
                deep_pos = np.zeros(2, dtype=np.float32)

                max_t1_proj = -1e30
                max_t1_d = 0.0
                max_t1_pos = np.zeros(2, dtype=np.float32)
                min_t1_proj = 1e30
                min_t1_d = 0.0
                min_t1_pos = np.zeros(2, dtype=np.float32)

                for nidx in range(num_nodes):
                    pos = mesh_coords[node_offset + nidx]
                    d = float(np.dot(pos - planepoint, normal))
                    if d < 0.0:
                        has_contact = True

                        if d < deep_d:
                            deep_d = d
                            deep_pos = pos

                        p1 = float(np.dot(pos, t1))
                        if p1 > max_t1_proj:
                            max_t1_proj = p1
                            max_t1_d = d
                            max_t1_pos = pos
                        if p1 < min_t1_proj:
                            min_t1_proj = p1
                            min_t1_d = d
                            min_t1_pos = pos

                if has_contact:
                    self.cacheGroundContact(rigidIdx, deep_pos, normal, anal_vel, deep_d)

                    sep = 0.005
                    if float(np.linalg.norm(max_t1_pos - deep_pos)) > sep:
                        self.cacheGroundContact(rigidIdx, max_t1_pos, normal, anal_vel, max_t1_d)
                    if float(np.linalg.norm(min_t1_pos - deep_pos)) > sep and float(
                        np.linalg.norm(min_t1_pos - max_t1_pos)
                    ) > sep:
                        self.cacheGroundContact(rigidIdx, min_t1_pos, normal, anal_vel, min_t1_d)


    def detectHeightField2Rigids_(self, rigidIdx: int, hf):
        """Detect and resolve collisions between a heightfield ground and rigids."""
        if rigidIdx == -1:
            return
        j = int(rigidIdx)
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        params = self.rigidParams.numpy()
        radius_arr = self.radius.numpy()
        rot = self.cached_rotation_matrix.numpy()
        compound_count = self.compound_count.numpy()
        compound_offset = self.compound_offset.numpy()
        compound_local_pos = self.compound_local_pos.numpy()
        compound_radius = self.compound_radius.numpy()

        domain_idx = int(domain_ids[j][0])
        rigid_min_x = float(aabb[domain_idx, 0][0])
        rigid_max_x = float(aabb[domain_idx, 1][0])
        rigid_min_z = float(aabb[domain_idx, 0][self.d - 1])
        rigid_max_z = float(aabb[domain_idx, 1][self.d - 1])

        max_height_in_range, min_height_in_range = hf.get_maxmin_height_in_range_2d(rigid_min_x, rigid_max_x)
        if max_height_in_range < rigid_min_z or (hf.reverse and (min_height_in_range > rigid_max_z)):
            return

        n_sub = int(compound_count[j])
        if n_sub > 0:
            base = int(compound_offset[j])
            parent_center = params[j, 0]
            R = rot[j]
            for k in range(n_sub):
                sidx = base + k
                local_p = compound_local_pos[sidx]
                r_sub = float(compound_radius[sidx])
                world_p = R @ local_p + parent_center
                foot, n, signed = hf.nearest_on_curve_2d(float(world_p[0]), float(world_p[1]))
                penetration = r_sub - signed
                if penetration > 0.0:
                    cpoint = world_p - np.asarray(n, dtype=np.float32) * r_sub
                    self.cacheGroundContact(j, cpoint, n, (0.0, 0.0), signed - r_sub)
            return

        rtype = int(domain_ids[j][1])
        if rtype == RigidType.BOX:
            for vi in range(4):
                pos = self.get_box_vertex(j, vi)
                foot, n, signed = hf.nearest_on_curve_2d(float(pos[0]), float(pos[1]))
                if signed < 0.0:
                    self.cacheGroundContact(j, pos, n, (0.0, 0.0), signed)
        elif rtype == RigidType.BALL:
            center = params[j, 0]
            radius = float(radius_arr[j])
            foot, n, signed = hf.nearest_on_curve_2d(float(center[0]), float(center[1]))
            penetration = radius - signed
            if penetration > 0.0:
                n = np.asarray(n, dtype=np.float32)
                cpoint = center - n * radius
                self.cacheGroundContact(j, cpoint, n, (0.0, 0.0), signed - radius)
        elif rtype == RigidType.CAPSULE:
            center = params[j, 0]
            lcdir = params[j, 1]
            lc = rot[j] @ lcdir + center
            uc = center * 2.0 - lc
            radius = float(radius_arr[j])
            for test_p in (lc, uc):
                foot, n, signed = hf.nearest_on_curve_2d(float(test_p[0]), float(test_p[1]))
                penetration = radius - signed
                if penetration > 0.0:
                    n = np.asarray(n, dtype=np.float32)
                    cpoint = test_p - n * radius
                    self.cacheGroundContact(j, cpoint, n, (0.0, 0.0), signed - radius)

    def detectHeightField2MeshContacts_(self, rigidIdx: int, hf):
        """Heightfield vs mesh rigid boundary nodes."""
        if rigidIdx == -1:
            return
        j = int(rigidIdx)
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        mesh_ids = self.rigid2MeshIndices.numpy()
        node_off = self.meshBoundaryNodeOffset.numpy()
        node_cnt = self.meshBoundaryNodeCount.numpy()
        coords = self.meshBoundaryCoords.numpy()

        domain_idx = int(domain_ids[j][0])
        rigid_min_x = float(aabb[domain_idx, 0][0])
        rigid_max_x = float(aabb[domain_idx, 1][0])
        rigid_min_z = float(aabb[domain_idx, 0][self.d - 1])
        rigid_max_z = float(aabb[domain_idx, 1][self.d - 1])
        max_height_in_range, min_height_in_range = hf.get_maxmin_height_in_range_2d(rigid_min_x, rigid_max_x)
        if max_height_in_range < rigid_min_z or (hf.reverse and (min_height_in_range > rigid_max_z)):
            return

        mesh_local = int(mesh_ids[j])
        if mesh_local < 0:
            return
        off = int(node_off[mesh_local])
        nnodes = int(node_cnt[mesh_local])
        for nidx in range(nnodes):
            pos = coords[off + nidx]
            foot, n, signed = hf.nearest_on_curve_2d(float(pos[0]), float(pos[1]))
            if signed < 0.0:
                self.cacheGroundContact(j, pos, n, (0.0, 0.0), signed)

    def detectVoxel2Rigids_(self, rigidIdx: int, vox):
        """Voxel map vs primitive rigid."""
        if rigidIdx == -1:
            return
        j = int(rigidIdx)
        domain_ids = self.rigidDomainIds.numpy()
        params = self.rigidParams.numpy()
        radius_arr = self.radius.numpy()
        rot = self.cached_rotation_matrix.numpy()
        rtype = int(domain_ids[j][1])
        if rtype == RigidType.BOX:
            for vi in range(4):
                pos = self.get_box_vertex(j, vi)
                d, n, c = vox.signed_distance_to_edges_2d(pos, 0.0)
                if d < 0.0:
                    self.cacheGroundContact(j, c, n, (0.0, 0.0), d)
        elif rtype == RigidType.BALL:
            center = params[j, 0]
            radius = float(radius_arr[j])
            d, n, c = vox.signed_distance_to_edges_2d(center, radius)
            if d < 0.0:
                self.cacheGroundContact(j, c, n, (0.0, 0.0), d)
        elif rtype == RigidType.CAPSULE:
            center = params[j, 0]
            lcdir = params[j, 1]
            lc = rot[j] @ lcdir + center
            uc = center * 2.0 - lc
            radius = float(radius_arr[j])
            for test_p in (lc, uc):
                d, n, c = vox.signed_distance_to_edges_2d(test_p, radius)
                if d < 0.0:
                    self.cacheGroundContact(j, c, n, (0.0, 0.0), d)

    def detectVoxel2MeshContacts_(self, rigidIdx: int, vox):
        """Voxel map vs mesh rigid boundary nodes."""
        if rigidIdx == -1:
            return
        j = int(rigidIdx)
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        mesh_ids = self.rigid2MeshIndices.numpy()
        node_off = self.meshBoundaryNodeOffset.numpy()
        node_cnt = self.meshBoundaryNodeCount.numpy()
        coords = self.meshBoundaryCoords.numpy()

        domain_idx = int(domain_ids[j][0])
        mesh_local = int(mesh_ids[j])
        if mesh_local < 0:
            return
        off = int(node_off[mesh_local])
        nnodes = int(node_cnt[mesh_local])
        minExtent = aabb[domain_idx, 1] - aabb[domain_idx, 0]
        extent_scale = 0.2 * float(np.linalg.norm(minExtent))
        for nidx in range(nnodes):
            pos = coords[off + nidx]
            d, n, c = vox.signed_distance_to_edges_2d(pos, extent_scale)
            if d < 0.0:
                self.cacheGroundContact(j, c, n, (0.0, 0.0), d)

    # -------  End of Rigid-Ground contact related functions -----------------------------
    # ----------------------------------------------------------------------------------

    def cacheContact(self, aid, bid, cpoint, normal, depth: float):
        """Cache a rigid-rigid contact (host path; Warp kernels use _cache_*_func)."""
        nc = self.num_contacts.numpy()
        idx = int(nc[0])
        if idx >= self.MAX_CONTACTS:
            return
        nc[0] = idx + 1
        self.num_contacts.assign(nc)

        cpoint = np.asarray(cpoint, dtype=np.float32).reshape(-1)
        normal = np.asarray(normal, dtype=np.float32).reshape(-1)
        aid = int(aid)
        bid = int(bid)
        _patch_array(self.contact_rigid_a, idx, aid)
        _patch_array(self.contact_rigid_b, idx, bid)
        _patch_array(self.contact_point, idx, (float(cpoint[0]), float(cpoint[1])))
        _patch_array(self.contact_normal, idx, (float(normal[0]), float(normal[1])))
        _patch_array(self.contact_depth, idx, float(depth))

        cpr = self.contact_count_per_rigid.numpy()
        cpr[aid] = int(cpr[aid]) + 1
        cpr[bid] = int(cpr[bid]) + 1
        self.contact_count_per_rigid.assign(cpr)

        params = self.rigidParams.numpy()
        V = self.V.numpy()
        RotV = self.RotV.numpy()
        contactParams = self.contactParams.numpy()
        ra = cpoint[:2] - params[aid, 0]
        rb = cpoint[:2] - params[bid, 0]
        e = 0.5 * (float(contactParams[aid][1]) + float(contactParams[bid][1]))
        v_threshold = float(self.restitution_velocity_threshold)
        va = V[aid] + np.array([-ra[1], ra[0]], dtype=np.float32) * float(RotV[aid])
        vb = V[bid] + np.array([-rb[1], rb[0]], dtype=np.float32) * float(RotV[bid])
        vn_pre = float(np.dot(va - vb, normal[:2]))
        bounce = -e * vn_pre if vn_pre < -v_threshold else 0.0
        _patch_array(self.contact_bounce_vel, idx, bounce)
        _patch_array(self.contact_tangent1, idx, (float(-normal[1]), float(normal[0])))

        env_ids = self.rigid_env_id.numpy()
        env_id = max(int(env_ids[aid]), 0)
        if env_id < self.MAX_ENVS_ALLOC:
            env_count = self.contact_env_count.numpy()
            local_i = int(env_count[env_id])
            env_count[env_id] = local_i + 1
            self.contact_env_count.assign(env_count)
            if local_i < self.MAX_CC_PER_ENV:
                _patch_array(self.contact_env_idx, env_id * self.MAX_CC_PER_ENV + local_i, idx)

    def cacheGroundContact(self, rid, cpoint, normal, ground_vel, depth: float):
        """Cache a ground-rigid contact (host path; Warp kernels use _cache_ground_contact_func)."""
        ng = self.num_ground_contacts.numpy()
        idx = int(ng[0])
        if idx >= self.MAX_GROUND_CONTACTS:
            return
        ng[0] = idx + 1
        self.num_ground_contacts.assign(ng)

        cpoint = np.asarray(cpoint, dtype=np.float32).reshape(-1)
        normal = np.asarray(normal, dtype=np.float32).reshape(-1)
        ground_vel = np.asarray(ground_vel, dtype=np.float32).reshape(-1)
        rid = int(rid)
        _patch_array(self.ground_contact_rigid, idx, rid)
        _patch_array(self.ground_contact_point, idx, (float(cpoint[0]), float(cpoint[1])))
        _patch_array(self.ground_contact_normal, idx, (float(normal[0]), float(normal[1])))
        _patch_array(
            self.ground_contact_vel,
            idx,
            (float(ground_vel[0]), float(ground_vel[1]) if ground_vel.size > 1 else 0.0),
        )
        _patch_array(self.ground_contact_depth, idx, float(depth))

        params = self.rigidParams.numpy()
        V = self.V.numpy()
        RotV = self.RotV.numpy()
        contactParams = self.contactParams.numpy()
        lr = cpoint[:2] - params[rid, 0]
        e = float(contactParams[rid][1])
        v_threshold = float(self.restitution_velocity_threshold)
        tlr = np.array([-lr[1], lr[0]], dtype=np.float32)
        gv = (
            ground_vel[:2]
            if ground_vel.size >= 2
            else np.array([float(ground_vel[0]), 0.0], dtype=np.float32)
        )
        v_point = V[rid] + tlr * float(RotV[rid])
        vn_pre = float(np.dot(v_point - gv, normal[:2]))
        bounce = -e * vn_pre if vn_pre < -v_threshold else 0.0
        _patch_array(self.ground_contact_bounce_vel, idx, bounce)
        _patch_array(self.ground_contact_tangent1, idx, (float(-normal[1]), float(normal[0])))

        env_ids = self.rigid_env_id.numpy()
        env_id = max(int(env_ids[rid]), 0)
        if env_id < self.MAX_ENVS_ALLOC:
            env_count = self.ground_contact_env_count.numpy()
            local_i = int(env_count[env_id])
            env_count[env_id] = local_i + 1
            self.ground_contact_env_count.assign(env_count)
            if local_i < self.MAX_GC_PER_ENV:
                _patch_array(self.ground_contact_env_idx, env_id * self.MAX_GC_PER_ENV + local_i, idx)

    def _add_pgs_row(
        self,
        aid,
        bid,
        jac_a,
        jac_b,
        rhs,
        lower,
        upper,
        parent_row,
    ):
        flag = -1
        ci = wp.atomic_add(self.numConstraints, 0, 1)
        if ci < self.MAX_CONSTRAINTS:
            self.pgs_bodypair[ci] = wp.vec2(aid, bid)
            self.pgs_Jac_a[ci] = jac_a
            self.pgs_Jac_b[ci] = jac_b
            self.pgs_rhs[ci] = rhs
            self.pgs_limits[ci] = wp.vec2(lower, upper)
            self.pgs_lambda[ci] = 0.0
            self.pgs_parent_row[ci] = parent_row
            flag = ci
        return flag

    def _assemble_ground_contact_rows(self, idx: int, dt: float):
        rid = self.ground_contact_rigid[idx]
        cpoint = self.ground_contact_point[idx]
        normal = self.ground_contact_normal[idx]
        ground_vel = self.ground_contact_vel[idx]
        depth = self.ground_contact_depth[idx]
        lr = cpoint - self.rigidParams[rid, 0]
        mu = self.contactParams[rid][0]

        bounce_vel = self.ground_contact_bounce_vel[idx]

        bias_vel = 0.0
        if depth < 0.0 and self.contact_erp > 0.0:
            bias_vel = wp.max(self.contact_erp * depth / dt, -5.0)

        # Properly decouple restitution and baumgarte separation velocities
        target_vel = bounce_vel
        if -bias_vel > bounce_vel:
            target_vel = -bias_vel

        jac_n = wp.vec3(normal[0], normal[1], vectorCrossProduct(lr, normal)[0])

        normal_row = self._add_pgs_row(
            rid, -1, jac_n, wp.vec3(0.0, 0.0, 0.0), target_vel + normal.dot(ground_vel), 0.0, 1e10, -1
        )
        self.ground_contact_pgs_indices[idx] = wp.vec3(normal_row, -1, -1)
        if mu > 1e-12 and normal_row >= 0:
            t1 = self.ground_contact_tangent1[idx]
            jac_t1 = wp.vec3(t1[0], t1[1], vectorCrossProduct(lr, t1)[0])
            rhs_t1 = t1.dot(ground_vel)
            tangent1_row = self._add_pgs_row(rid, -1, jac_t1, wp.vec3(0.0, 0.0, 0.0), rhs_t1, -mu, mu, normal_row)
            self.ground_contact_pgs_indices[idx][1] = tangent1_row


    def _assemble_pair_contact_rows(self, idx: int, dt: float):
        aid = self.contact_rigid_a[idx]
        bid = self.contact_rigid_b[idx]
        cpoint = self.contact_point[idx]
        normal = self.contact_normal[idx]
        depth = self.contact_depth[idx]
        ra = cpoint - self.rigidParams[aid, 0]
        rb = cpoint - self.rigidParams[bid, 0]
        mu = 0.5 * (self.contactParams[aid][0] + self.contactParams[bid][0])

        bounce_vel = self.contact_bounce_vel[idx]

        bias_vel = 0.0
        if depth < 0.0 and self.contact_erp > 0.0:
            bias_vel = wp.max(self.contact_erp * depth / dt, -5.0)

        # Properly decouple restitution and baumgarte separation velocities
        # Both are positive requested separation velocities
        target_vel = bounce_vel
        if -bias_vel > bounce_vel:
            target_vel = -bias_vel

        jac_na = wp.vec3(normal[0], normal[1], vectorCrossProduct(ra, normal)[0])
        jac_nb = wp.vec3(normal[0], normal[1], vectorCrossProduct(rb, normal)[0])

        normal_row = self._add_pgs_row(aid, bid, jac_na, jac_nb, target_vel, 0.0, 1e10, -1)
        self.contact_pgs_indices[idx] = wp.vec3(normal_row, -1, -1)

        if mu > 1e-12 and normal_row >= 0:
            t1 = self.contact_tangent1[idx]
            jac_t1a = wp.vec3(t1[0], t1[1], vectorCrossProduct(ra, t1)[0])
            jac_t1b = wp.vec3(t1[0], t1[1], vectorCrossProduct(rb, t1)[0])
            tangent1_row = self._add_pgs_row(aid, bid, jac_t1a, jac_t1b, 0.0, -mu, mu, normal_row)
            self.contact_pgs_indices[idx][1] = tangent1_row


    def _assemble_contact_constraints_kernel(self, dt: float):
        wp.launch(
            _assemble_ground_contact_constraints_wp,
            dim=1,
            inputs=[
                float(dt),
                float(self.contact_erp),
                int(self.MAX_CONSTRAINTS),
                self.num_ground_contacts,
                self.ground_contact_rigid,
                self.ground_contact_point,
                self.ground_contact_normal,
                self.ground_contact_vel,
                self.ground_contact_depth,
                self.ground_contact_bounce_vel,
                self.ground_contact_tangent1,
                self.ground_contact_pgs_indices,
                self.rigidParams,
                self.contactParams,
                self.numConstraints,
                self.pgs_bodypair,
                self.pgs_Jac_a,
                self.pgs_Jac_b,
                self.pgs_rhs,
                self.pgs_limits,
                self.pgs_lambda,
                self.pgs_parent_row,
            ],
        )
        if int(self.num_contacts.numpy()[0]) > 0:
            wp.launch(
                _assemble_pair_contact_constraints_wp,
                dim=1,
                inputs=[
                    float(dt),
                    float(self.contact_erp),
                    int(self.MAX_CONSTRAINTS),
                    self.num_contacts,
                    self.contact_rigid_a,
                    self.contact_rigid_b,
                    self.contact_point,
                    self.contact_normal,
                    self.contact_depth,
                    self.contact_bounce_vel,
                    self.contact_tangent1,
                    self.contact_pgs_indices,
                    self.rigidParams,
                    self.contactParams,
                    self.numConstraints,
                    self.pgs_bodypair,
                    self.pgs_Jac_a,
                    self.pgs_Jac_b,
                    self.pgs_rhs,
                    self.pgs_limits,
                    self.pgs_lambda,
                    self.pgs_parent_row,
                ],
            )

    def _compute_contact_forces_kernel(self, dt: float):
        """Compute contact forces from accumulated impulses after PGS solve."""
        wp.launch(
            _compute_contact_forces_wp,
            dim=1,
            inputs=[
                float(dt),
                self.num_ground_contacts,
                self.num_contacts,
                self.ground_contact_pgs_indices,
                self.ground_contact_normal,
                self.ground_contact_tangent1,
                self.ground_contact_force,
                self.contact_pgs_indices,
                self.contact_normal,
                self.contact_tangent1,
                self.contact_force,
                self.pgs_lambda,
            ],
        )

    def _assemble_joint_constraints_kernel(self, dt: float):
        """Assemble joint PGS rows via joint_kernels.assemble_single_joint_rows."""
        wp.launch(
            _assemble_joint_constraints_wp,
            dim=1,
            inputs=[
                float(dt),
                int(self.numAnchors),
                int(self.MAX_CONSTRAINTS),
                self.joint_type,
                self.joint_id_a,
                self.joint_id_b,
                self.joint_params,
                self.joint_has_motor,
                self.joint_motor_target_mode,
                self.joint_motor_target_vel,
                self.joint_q0_rel_inv,
                self.joint_axis,
                self.joint_l1,
                self.joint_l2,
                self.rigidParams,
                self.quat,
                self.quat_initial,
                self.RotV,
                self.numConstraints,
                self.pgs_bodypair,
                self.pgs_Jac_a,
                self.pgs_Jac_b,
                self.pgs_rhs,
                self.pgs_limits,
                self.pgs_lambda,
                self.pgs_parent_row,
            ],
        )

    def reset_contact_caches_kernel(self):
        """Reset contact counters and force arrays in a single kernel launch."""
        wp.launch(
            _reset_contact_caches_wp,
            dim=1,
            inputs=[
                int(self.numRigids),
                int(self.numAnalytical),
                int(max(self.num_envs, 1)),
                self.num_contacts,
                self.num_ground_contacts,
                self.prev_num_contacts,
                self.prev_num_ground_contacts,
                self.numConstraints,
                self.contact_count_per_rigid,
                self.contact_env_count,
                self.ground_contact_env_count,
                self.contact_force,
                self.contact_pgs_indices,
                self.contact_bounce_vel,
                self.ground_contact_force,
                self.ground_contact_pgs_indices,
                self.ground_contact_bounce_vel,
            ],
        )

    def solve_pgs(self, pgs_iters: int):
        """Serial PGS via dim=1 Warp kernel (forward + backward sweeps)."""
        wp.launch(
            _solve_pgs_kernel,
            dim=1,
            inputs=[
                int(pgs_iters),
                self.numConstraints,
                int(self.MAX_CONSTRAINTS),
                self.V,
                self.RotV,
                self.mass,
                self.inertia,
                self.pgs_bodypair,
                self.pgs_Jac_a,
                self.pgs_Jac_b,
                self.pgs_rhs,
                self.pgs_limits,
                self.pgs_lambda,
                self.pgs_parent_row,
            ],
        )

    def solve_pgs_single_func(self, i: int):
        bodypair = self.pgs_bodypair[i]
        aid = bodypair[0]
        bid = bodypair[1]
        if aid >= 0:
            jac_a = self.pgs_Jac_a[i]
            jac_b = self.pgs_Jac_b[i]

            va3 = wp.vec3(self.V[aid][0], self.V[aid][1], self.RotV[aid])
            vb3 = wp.vec3(0.0, 0.0, 0.0)

            massInvA = wp.mat33(0.0)
            massInvB = wp.mat33(0.0)

            inv_mass_a = 1.0 / (self.mass[aid] + 1e-12)
            massInvA[0, 0] = inv_mass_a
            massInvA[1, 1] = inv_mass_a
            inv_Ia = 1.0 / (self.inertia[aid] + 1e-12)
            massInvA[2, 2] = inv_Ia

            vel = jac_a.dot(va3)
            massInvJacA = massInvA @ jac_a
            W = jac_a.dot(massInvJacA)

            has_b = bid >= 0
            massInvJacB = wp.vec3(0.0, 0.0, 0.0)
            if has_b:
                inv_mass_b = 1.0 / (self.mass[bid] + 1e-12)
                massInvB[0, 0] = inv_mass_b
                massInvB[1, 1] = inv_mass_b
                inv_Ib = 1.0 / (self.inertia[bid] + 1e-12)
                massInvB[2, 2] = inv_Ib

                vb3 = wp.vec3(self.V[bid][0], self.V[bid][1], self.RotV[bid])
                vel -= jac_b.dot(vb3)
                massInvJacB = massInvB @ jac_b
                W += jac_b.dot(massInvJacB)

            rhs = self.pgs_rhs[i] - vel
            delta_lamb = rhs / (W + 1e-12)

            old_lamb = self.pgs_lambda[i]
            new_lamb = old_lamb + delta_lamb

            parent = self.pgs_parent_row[i]
            if parent >= 0:
                fric_lim_low = self.pgs_limits[i][0] * self.pgs_lambda[parent]
                fric_lim_upper = self.pgs_limits[i][1] * self.pgs_lambda[parent]
                new_lamb = wp.max(fric_lim_low, wp.min(fric_lim_upper, new_lamb))
            else:
                lower = self.pgs_limits[i][0]
                upper = self.pgs_limits[i][1]
                new_lamb = wp.max(lower, wp.min(upper, new_lamb))

            apply_lamb = new_lamb - old_lamb
            self.pgs_lambda[i] = new_lamb

            if apply_lamb != 0.0:
                deltaA = massInvJacA * apply_lamb

                self.V[aid] += wp.vec2(deltaA[0], deltaA[1])
                self.RotV[aid] += deltaA[2]

                if has_b:
                    deltaB = massInvJacB * apply_lamb

                    self.V[bid] -= wp.vec2(deltaB[0], deltaB[1])
                    self.RotV[bid] -= deltaB[2]

    def precompute_rigid_transforms(self):
        n = int(self.numRigids + self.numAnalytical)
        if n <= 0:
            return
        wp.launch(
            _precompute_rigid_transforms_wp,
            dim=n,
            inputs=[
                n,
                self.quat,
                self.visual_angle,
                self.inertia,
                self.cached_rotation_matrix,
                self.cached_inertia_inv_2d,
            ],
        )

    def _mask_allows_pair(self, idx_a: int, idx_b: int) -> int:
        allow_ab = (self.collide_bits[idx_a] & self.category_bits[idx_b]) != wp.uint32(0)
        allow_ba = (self.collide_bits[idx_b] & self.category_bits[idx_a]) != wp.uint32(0)
        return int(allow_ab and allow_ba, int)

    def classify_collision_pairs_kernel(self, pairs, num_pairs: int):
        """Classify collision pairs into typed buffers (Warp kernel)."""
        wp.launch(
            _classify_collision_pairs_wp,
            dim=1,
            inputs=[
                pairs,
                int(num_pairs),
                int(self.numRigids),
                int(self.MAX_COLLISION_PAIRS),
                int(self.MAX_GROUND_PAIRS),
                self.domainToRigid,
                self.rigidDomainIds,
                self.compound_count,
                self.rigid_env_id,
                self.category_bits,
                self.collide_bits,
                self.num_primitive_pairs,
                self.num_ball_ball_pairs,
                self.num_box_box_pairs,
                self.num_box_ball_pairs,
                self.num_seg_point_pairs,
                self.num_seg_ball_pairs,
                self.num_seg_seg_pairs,
                self.num_mesh_pairs,
                self.num_mixed_pairs,
                self.num_groundprim_pairs,
                self.num_groundmesh_pairs,
                self.primitive_pairs_buffer,
                self.ball_ball_pairs_buffer,
                self.box_box_pairs_buffer,
                self.box_ball_pairs_buffer,
                self.seg_point_pairs_buffer,
                self.seg_ball_pairs_buffer,
                self.seg_seg_pairs_buffer,
                self.mesh_pairs_buffer,
                self.mixed_pairs_buffer,
                self.groundprim_pairs_buffer,
                self.groundmesh_pairs_buffer,
            ],
        )

    def maybe_rebuild_spatial_hash(self, fem_lb=None, fem_ub=None, fem_margin=0.0):
        """Rebuild the mesh spatial hash if enough time has elapsed or
        max displacement since last rebuild exceeds 0.5 * cell_size.
        Called before detect_mesh_mesh_contacts_kernel each substep."""

        if self.spatialHash is None or not self._sh_mesh_needs_rebuild:
            return

        if self.considerRigidRigidContact:
            self.populate_spatial_hash_filtered()
        else:
            self.populate_spatial_hash_filtered(fem_lb=fem_lb, fem_ub=fem_ub)

        cell_size = float(float(np.max(self.spatialHash.gridSize.numpy()[0])))
        self._sh_mesh_max_v = self.get_max_linear_velocity()
        _assign_scalar(self._sh_contact_margin, cell_size)
        self._sh_mesh_cell_size = cell_size

        # print(
        #     f"\033[93m[Contact] Rebuilding mesh spatial hash... "
        #     f"(max displacement: {self._sh_mesh_max_v * self._sh_mesh_elapsed:.4f} m, "
        #     f"elapsed time: {self._sh_mesh_elapsed:.4f} s, "
        #     f"cell size: {self._sh_mesh_cell_size:.4f} m)\033[0m"
        # )

        self._sh_mesh_elapsed = 0.0
        self._sh_mesh_needs_rebuild = False

    def populate_spatial_hash_filtered(self, fem_lb=None, fem_ub=None):
        """Populate spatial hash with mesh elements filtered by FEM AABB intersection.

        For each mesh rigid, computes the intersection of the rigid AABB with
        the FEM contact node AABB.  Only elements overlapping this intersection
        (expanded by margin) are inserted, dramatically reducing the number of
        hashed elements when FEM nodes only cover part of the rigid surface.
        If fem_lb or fem_ub is None, inserts all mesh elements.
        """
        if self.spatialHash is None or self.numMesh == 0:
            return

        if fem_lb is None or fem_ub is None:
            fem_lb_field = self._sh_unbounded_lb
            fem_ub_field = self._sh_unbounded_ub
        else:
            fem_lb_field = fem_lb
            fem_ub_field = fem_ub

        self.spatialHash.reset()
        contact_margin = float(self._sh_contact_margin.numpy()[0])
        velocity_buffer = 2.0 * self._sh_mesh_max_v * self._sh_mesh_rebuild_interval
        self._populate_spatial_hash_filtered_kernel(
            float(velocity_buffer),
            float(contact_margin),
            fem_lb_field,
            fem_ub_field,
        )
        self.spatialHash.build()
        _assign_scalar(self._sh_contact_margin, float(contact_margin) + float(velocity_buffer))

    def _populate_spatial_hash_filtered_kernel(
        self, velocity_buffer: float, contact_margin: float, fem_lb_field, fem_ub_field
    ):
        """Insert only mesh elements that overlap the FEM-rigid AABB intersection."""
        fem_lb_np = np.asarray(fem_lb_field.numpy()[0], dtype=np.float32).reshape(-1)
        fem_ub_np = np.asarray(fem_ub_field.numpy()[0], dtype=np.float32).reshape(-1)

        mesh2rigid = self.mesh2RigidIndices.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        aabb = self.aabb.numpy()
        elem_offsets = self.meshBoundaryElementOffset.numpy()
        elem_counts = self.meshBoundaryElementCount.numpy()
        elements = self.meshBoundaryElements.numpy()
        coords = self.meshBoundaryCoords.numpy()

        global_lb = np.array([1e9, 1e9], dtype=np.float32)
        global_ub = np.array([-1e9, -1e9], dtype=np.float32)

        for k in range(self.numMesh):
            rigid_id = int(mesh2rigid[k])
            domain_id = int(domain_ids[rigid_id][0])
            rigid_lb_k = aabb[domain_id, 0]
            rigid_ub_k = aabb[domain_id, 1]
            isect_lb_k = np.maximum(rigid_lb_k, fem_lb_np) - contact_margin
            isect_ub_k = np.minimum(rigid_ub_k, fem_ub_np) + contact_margin
            if np.all(isect_lb_k <= isect_ub_k):
                global_lb = np.minimum(global_lb, isect_lb_k)
                global_ub = np.maximum(global_ub, isect_ub_k)

        if global_lb[0] <= global_ub[0]:
            expand = float(np.max(global_ub - global_lb)) * 0.05 + contact_margin
            global_lb = global_lb - expand
            global_ub = global_ub + expand

            for k in range(self.numMesh):
                rigid_id = int(mesh2rigid[k])
                domain_id = int(domain_ids[rigid_id][0])
                rigid_lb = aabb[domain_id, 0]
                rigid_ub = aabb[domain_id, 1]
                isect_lb = np.maximum(rigid_lb, fem_lb_np) - contact_margin
                isect_ub = np.minimum(rigid_ub, fem_ub_np) + contact_margin
                has_isect = bool(np.all(isect_lb <= isect_ub))

                elem_offset = int(elem_offsets[k])
                num_elems = int(elem_counts[k])
                for eidx in range(num_elems):
                    global_eidx = elem_offset + eidx
                    conn = elements[global_eidx]
                    lb = np.array([1e30, 1e30], dtype=np.float32)
                    ub = np.array([-1e30, -1e30], dtype=np.float32)
                    for j in range(3):
                        if int(conn[j]) >= 0:
                            coord = coords[int(conn[j])]
                            lb = np.minimum(lb, coord[:2])
                            ub = np.maximum(ub, coord[:2])

                    _patch_array(self.meshElemLB, global_eidx, (float(lb[0]), float(lb[1])))
                    _patch_array(self.meshElemUB, global_eidx, (float(ub[0]), float(ub[1])))
                    margin_base = self.meshRigidContactMarginRatio * float(np.linalg.norm(ub - lb)) + velocity_buffer
                    _patch_array(self.meshElemMarginBase, global_eidx, margin_base)

                    if has_isect:
                        overlaps = bool(np.all(ub >= isect_lb) and np.all(lb <= isect_ub))
                        if overlaps:
                            self.spatialHash.addElement(lb, ub, rigid_id, global_eidx, velocity_buffer)
        else:
            global_lb = np.array([0.0, 0.0], dtype=np.float32)
            global_ub = np.array([1.0, 1.0], dtype=np.float32)

        self.spatialHash.setBounds(global_lb, global_ub)

    def update_mesh_element_aabbs(self):
        """Refresh cached mesh element AABBs from current transformed coords."""
        elem_offsets = self.meshBoundaryElementOffset.numpy()
        elem_counts = self.meshBoundaryElementCount.numpy()
        elements = self.meshBoundaryElements.numpy()
        coords = self.meshBoundaryCoords.numpy()
        ratio = self.meshRigidContactMarginRatio

        for k in range(self.numMesh):
            elem_offset = int(elem_offsets[k])
            num_elems = int(elem_counts[k])
            for eidx in range(num_elems):
                global_eidx = elem_offset + eidx
                conn = elements[global_eidx]
                lb = wp.vec2(1e30, 1e30)
                ub = wp.vec2(-1e30, -1e30)
                for j in range(3):
                    if conn[j] >= 0:
                        coord = coords[conn[j]]
                        c = wp.vec2(float(coord[0]), float(coord[1]))
                        lb = wp.min(lb, c)
                        ub = wp.max(ub, c)
                _patch_array(self.meshElemLB, global_eidx, lb)
                _patch_array(self.meshElemUB, global_eidx, ub)
                _patch_array(
                    self.meshElemMarginBase,
                    global_eidx,
                    ratio * (ub - wp.length(lb)),
                )

    # ════════════════════════════════════════════════════════════════════
    # FUSED KERNELS — minimize kernel launch overhead (~0.12ms per launch)
    # ════════════════════════════════════════════════════════════════════

    def _rigidStep_and_precompute_kernel(self, dt: float, damping: float):
        """Fused: precompute_rigid_transforms + rigidStep in 1 kernel launch."""
        wp.launch(
            _rigid_step_wp,
            dim=1,
            inputs=[
                float(dt),
                float(damping),
                int(self.numRigids),
                int(self.numAnalytical),
                self.bcNodes,
                self.bcGValues,
                self.bcTValues,
                self.bcRValues,
                self.mass,
                self.inertia,
                self.V,
                self.RotV,
                self.accumulated_impulse,
                self.accumulated_rotational_impulse,
                self.quat,
                self.visual_angle,
                self.cached_rotation_matrix,
                self.cached_inertia_inv_2d,
            ],
        )

    def _updateU_and_BBox_kernel(self, dt: float, update_bbox: int):
        """Fused: position integration (updateU) + AABB update (updateBBox)."""
        wp.launch(
            _update_u_and_bbox_wp,
            dim=1,
            inputs=[
                float(dt),
                int(update_bbox),
                int(self.numRigids),
                int(self.numAnalytical),
                1 if self.movingAnalytical else 0,
                self.bcNodes,
                self.bcTValues,
                self.bcRValues,
                self.V,
                self.RotV,
                self.U,
                self.quat,
                self.rigidParams,
                self.accumulated_impulse,
                self.accumulated_rotational_impulse,
                self.rigidDomainIds,
                self.radius,
                self.cached_rotation_matrix,
                self.aabb,
            ],
        )
        # Mesh world coords / AABBs are not covered by the primitive AABB kernel.
        if self.numMesh > 0:
            self.precompute_rigid_transforms()
            self.updateMeshCoords()
            self._sh_mesh_needs_rebuild = True

    def _generate_ground_pairs_direct_kernel(self):
        """Generate ground-rigid contact pairs directly without BVH broadphase."""
        wp.launch(
            _generate_ground_pairs_wp,
            dim=1,
            inputs=[
                int(self.numRigids),
                int(self.numAnalytical),
                int(self.MAX_GROUND_PAIRS),
                self.rigidDomainIds,
                self.compound_count,
                self.category_bits,
                self.collide_bits,
                self.num_primitive_pairs,
                self.num_ball_ball_pairs,
                self.num_box_box_pairs,
                self.num_box_ball_pairs,
                self.num_seg_point_pairs,
                self.num_seg_ball_pairs,
                self.num_seg_seg_pairs,
                self.num_mesh_pairs,
                self.num_mixed_pairs,
                self.num_groundprim_pairs,
                self.num_groundmesh_pairs,
                self.groundprim_pairs_buffer,
                self.groundmesh_pairs_buffer,
            ],
        )

    def update_joint_motor_velocity_targets_kernel(self, dt: float):
        """Convert command targets to per-substep velocity targets for motor rows."""
        if self.numAnchors <= 0:
            return
        for j in range(self.numAnchors):
            if int(self.joint_has_motor.numpy()[j]) == 0:
                continue
            mode = int(self.joint_motor_target_mode.numpy()[j])
            cmd = float(self.joint_control_target.numpy()[j])
            vel_limit = float(self.joint_params.numpy()[j][4])
            if mode == 2:
                continue
            if mode == 1:
                vel_target = float(self.joint_motor_target_vel.numpy()[j]) + cmd * float(dt)
                if vel_limit > 0.0:
                    vel_target = max(min(vel_target, vel_limit), -vel_limit)
                _patch_array(self.joint_motor_target_vel, j, vel_target)
            else:
                _patch_array(self.joint_motor_target_vel, j, cmd)

    def apply_joint_pd_velocity_kernel(self, dt: float):
        """PD control with per-joint stiffness/damping — VELOCITY-MOTOR output.

        Computes a velocity target from the PD position error and writes it
        to the anchor's RotV so the constraint-solver velocity motor enforces
        it implicitly.  This avoids the instability caused by coupling
        explicit accumulated_rotational_impulse torques with the iterative PGS position constraint.

        Velocity target:  ω_target = kp · (θ_target − θ)
        The kd gain is used as optional pre-damping: the effective velocity
        written is  ω_target − (kd/kp) · ω_rel  when kp > 0.
        """
        if self.numAnchors <= 0:
            return
        joint_id_a = self.joint_id_a.numpy()
        joint_id_b = self.joint_id_b.numpy()
        joint_axis = self.joint_axis.numpy()
        joint_type = self.joint_type.numpy()
        kpd = self.kpd_field.numpy()
        joint_params = self.joint_params.numpy()
        quat = self.quat.numpy()
        quat_initial = self.quat_initial.numpy()
        rot_v = self.RotV.numpy()
        control_target = self.joint_control_target.numpy()
        rigid_params = self.rigidParams.numpy()
        lin_v = self.V.numpy()
        motor_modes = self.joint_motor_target_mode.numpy().copy()
        motor_vels = self.joint_motor_target_vel.numpy().copy()

        for j in range(self.numAnchors):
            rigid_a = int(joint_id_a[j])
            rigid_b = int(joint_id_b[j])
            axis_local = joint_axis[j]
            jointType = int(joint_type[j])

            kp = float(kpd[j][0])
            kd = float(kpd[j][1])
            lower_limit = float(joint_params[j][2])
            upper_limit = float(joint_params[j][3])
            vel_limit = float(joint_params[j][4])

            if jointType == JointType.Revolute:
                motor_modes[j] = 0
                qa = float(quat[rigid_a])
                q0a = float(quat_initial[rigid_a])
                qb = float(quat[rigid_b])
                q0b = float(quat_initial[rigid_b])
                wa = float(rot_v[rigid_a])
                wb = float(rot_v[rigid_b])

                angle = (qb - q0b) - (qa - q0a)
                target_pos = float(control_target[j])

                if lower_limit < upper_limit:
                    target_pos = min(max(target_pos, lower_limit), upper_limit)

                pos_err = float(np.arctan2(np.sin(target_pos - angle), np.cos(target_pos - angle)))
                w_rel = wb - wa
                ctrl_dt = float(self.control_dt)
                vel_target = kp / ctrl_dt * pos_err - kd * w_rel
                max_step_gain = 0.5
                if ctrl_dt > 0.0:
                    max_vel_from_error = max_step_gain * abs(pos_err) / ctrl_dt
                    vel_target = min(max(vel_target, -max_vel_from_error), max_vel_from_error)
                if vel_limit > 0.0:
                    vel_target = min(max(vel_target, -vel_limit), vel_limit)
                motor_vels[j] = vel_target
            elif jointType == JointType.Prismatic:
                target_pos = float(control_target[j])
                posA = rigid_params[rigid_a, 0]
                posB = rigid_params[rigid_b, 0]
                axis_world = np.asarray(axis_local, dtype=np.float32).reshape(-1)
                rel_pos = float(np.dot(posB - posA, axis_world[:2]))
                pos_err = target_pos - rel_pos

                vel_err = float(np.dot(lin_v[rigid_b] - lin_v[rigid_a], axis_world[:2]))
                vel_target = kp * pos_err - kd * vel_err

                max_step_gain = 0.5
                ctrl_dt = float(self.control_dt)
                if ctrl_dt > 0.0:
                    max_vel_from_error = max_step_gain * abs(pos_err) / ctrl_dt
                    vel_target = min(max(vel_target, -max_vel_from_error), max_vel_from_error)

                if vel_limit > 0.0:
                    vel_target = min(max(vel_target, -vel_limit), vel_limit)

                motor_vels[j] = vel_target

        self.joint_motor_target_mode.assign(motor_modes)
        self.joint_motor_target_vel.assign(motor_vels)

    def apply_joint_pd_torque_kernel(self, dt: float):
        """PD control with per-joint stiffness/damping — TORQUE output.

        Computes τ = kp * (target − angle) − kd * ω_rel

        The physics engine's ``apply_motor_torques_kernel`` will convert
        to world frame and apply ±τ on the connected bodies as accumulated_rotational_impulse
        (external torque), identical to how MuJoCo / Isaac Lab operate.
        """
        if self.numAnchors <= 0:
            return
        joint_id_a = self.joint_id_a.numpy()
        joint_id_b = self.joint_id_b.numpy()
        joint_type = self.joint_type.numpy()
        kpd = self.kpd_field.numpy()
        joint_params = self.joint_params.numpy()
        quat = self.quat.numpy()
        quat_initial = self.quat_initial.numpy()
        rot_v = self.RotV.numpy()
        control_target = self.joint_control_target.numpy()
        inertia = self.inertia.numpy()
        motor_modes = self.joint_motor_target_mode.numpy().copy()
        accum_rot = self.accumulated_rotational_impulse.numpy().copy()

        for j in range(self.numAnchors):
            rigid_a = int(joint_id_a[j])
            rigid_b = int(joint_id_b[j])
            jointType = int(joint_type[j])

            kp = float(kpd[j][0])
            kd = float(kpd[j][1])
            vel_limit = float(joint_params[j][4])
            effort_lim = float(joint_params[j][5])
            target_pos = float(control_target[j])

            if jointType == 1:  # JointType.Revolute
                motor_modes[j] = 2
                qa = float(quat[rigid_a])
                q0a = float(quat_initial[rigid_a])
                qb = float(quat[rigid_b])
                q0b = float(quat_initial[rigid_b])
                wa = float(rot_v[rigid_a])
                wb = float(rot_v[rigid_b])

                angle = (qb - q0b) - (qa - q0a)
                w_rel = wb - wa
                pos_err = float(np.arctan2(np.sin(target_pos - angle), np.cos(target_pos - angle)))
                torque_mag = kp * pos_err - kd * w_rel

                if abs(pos_err) < 1e-3 and abs(w_rel) < 1e-3:
                    torque_mag = 0.0

                if effort_lim > 0.0:
                    torque_mag = effort_lim * float(np.tanh(torque_mag / (effort_lim + 1e-6)))
                    torque_mag = min(max(torque_mag, -effort_lim), effort_lim)

                inertia_a = float(inertia[rigid_a])
                inertia_b = float(inertia[rigid_b])
                alpha_per_tau = 1.0 / (inertia_a + 1e-6) + 1.0 / (inertia_b + 1e-6)
                if vel_limit > 0.0 and dt > 0.0 and alpha_per_tau > 0.0:
                    tau_min = (-vel_limit - w_rel) / (dt * alpha_per_tau)
                    tau_max = (vel_limit - w_rel) / (dt * alpha_per_tau)
                    torque_mag = min(max(torque_mag, tau_min), tau_max)

                if effort_lim > 0.0 and dt > 0.0 and alpha_per_tau > 0.0:
                    max_acc = effort_lim * alpha_per_tau
                    stopping_err = 0.5 * w_rel * w_rel / (max_acc + 1e-6)
                    if abs(pos_err) < stopping_err and pos_err * w_rel > 0.0:
                        brake_tau = -w_rel / (dt * (alpha_per_tau + 1e-9))
                        torque_mag = min(max(brake_tau, -effort_lim), effort_lim)

                accum_rot[rigid_a] -= torque_mag * dt
                accum_rot[rigid_b] += torque_mag * dt

        self.joint_motor_target_mode.assign(motor_modes)
        self.accumulated_rotational_impulse.assign(accum_rot)

    def _calculate_bc_for_index(self, idx, dt):
        """Accumulate boundary-condition forces/torques into accumulated_impulse
        and accumulated_rotational_impulse.

        NOTE: Also handles ROTVTYPE here so the motor target is available
        during PGS solve (before _updateU_and_BBox_kernel).
        """
        bc_nodes = self.bcNodes.numpy()
        bc_type = int(bc_nodes[idx])
        mass = self.mass.numpy()
        bc_g = self.bcGValues.numpy()
        bc_t = self.bcTValues.numpy()
        bc_r = self.bcRValues.numpy()
        accum = self.accumulated_impulse.numpy().copy()
        accum_rot = self.accumulated_rotational_impulse.numpy().copy()

        # Linear: accumulate into accumulated_impulse
        if (bc_type & ATYPE) != 0:
            accum[idx] = (0.0, 0.0)
        else:
            if (bc_type & GRAVITY) != 0:
                accum[idx] = (
                    float(accum[idx][0]) + float(mass[idx]) * float(bc_g[idx][0]) * float(dt),
                    float(accum[idx][1]) + float(mass[idx]) * float(bc_g[idx][1]) * float(dt),
                )
            if (bc_type & FORCETYPE) != 0:
                accum[idx] = (
                    float(accum[idx][0]) + float(bc_t[idx][0]) * float(dt),
                    float(accum[idx][1]) + float(bc_t[idx][1]) * float(dt),
                )

        # Angular: accumulate into accumulated_rotational_impulse
        if (bc_type & ROTATYPE) != 0:
            accum_rot[idx] = 0.0
        elif (bc_type & TORQUETYPE) != 0:
            accum_rot[idx] = float(accum_rot[idx]) + float(bc_r[idx]) * float(dt)

        self.accumulated_impulse.assign(accum)
        self.accumulated_rotational_impulse.assign(accum_rot)

    def _update_bc_for_index(self, i):
        """Apply boundary conditions for rigid index i."""
        bc_nodes = self.bcNodes.numpy()
        bc_type = int(bc_nodes[i])
        bc_t = self.bcTValues.numpy()
        bc_r = self.bcRValues.numpy()
        if (bc_type & VTYPE) != 0:
            _patch_array(self.V, i, bc_t[i])
        elif (bc_type & UTYPE) != 0:
            _patch_array(self.V, i, (0.0, 0.0))
        elif (bc_type & RTYPE) != 0:
            _patch_array(self.V, i, (0.0, 0.0))
            _patch_array(self.RotV, i, 0.0)
        if (bc_type & ROTVTYPE) != 0:
            _patch_array(self.RotV, i, float(bc_r[i]))

    def get_box_vertex(self, rigidIdx: int, v_idx: int):
        params = self.rigidParams.numpy()
        center = params[rigidIdx, 0]
        extent = params[rigidIdx, 1]

        # 0: - -, 1: + -, 2: + +, 3: - +
        sx = -1.0 if (v_idx == 0 or v_idx == 3) else 1.0
        sy = -1.0 if (v_idx == 0 or v_idx == 1) else 1.0

        local_pos = 0.5 * np.array([sx * extent[0], sy * extent[1]], dtype=np.float32)
        rotMat = self.cached_rotation_matrix.numpy()[rigidIdx]
        return center + rotMat @ local_pos

    def getPrimitiveRigidBBox(self, rigidId: int):
        """Compute and store AABB for a single primitive rigid based on packed params."""
        # Host path: launch full bbox update (cheap for small scenes).
        self.updateBBox()

    def updateMeshCoords(self):
        """Transform mesh boundary vertices to world space and refresh mesh AABBs."""
        if self.numMesh <= 0:
            return

        mesh2rigid = self.mesh2RigidIndices.numpy()
        node_off = self.meshBoundaryNodeOffset.numpy()
        node_cnt = self.meshBoundaryNodeCount.numpy()
        pool_ids = self.instance_pool_id.numpy()
        pool_node_off = self.pool_node_offset.numpy()
        pool_node_cnt = self.pool_node_count.numpy()
        pool_lrs = self.pool_boundary_lrs.numpy()
        params = self.rigidParams.numpy()
        quat = self.quat.numpy()
        scales = self.meshRigidScale.numpy()
        offsets = self.meshRigidOffset.numpy()
        domain_ids = self.rigidDomainIds.numpy()
        coords = self.meshBoundaryCoords.numpy().copy()
        aabb = self.aabb.numpy().copy() if self.aabb is not None else None

        for mid in range(self.numMesh):
            rid = int(mesh2rigid[mid])
            if rid < 0:
                continue
            pool_id = int(pool_ids[rid])
            if pool_id < 0:
                continue
            n_off = int(node_off[mid])
            n_cnt = int(node_cnt[mid])
            p_off = int(pool_node_off[pool_id])
            p_cnt = int(pool_node_cnt[pool_id])
            n_use = min(n_cnt, p_cnt)
            if n_use <= 0:
                continue

            center = params[rid, 0]
            angle = float(quat[rid])
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            scale = scales[rid]
            offset = offsets[rid]

            lb = np.array([1e30, 1e30], dtype=np.float32)
            ub = np.array([-1e30, -1e30], dtype=np.float32)
            for i in range(n_use):
                lr = pool_lrs[p_off + i]
                local = np.array([float(lr[0]) * float(scale[0]), float(lr[1]) * float(scale[1])], dtype=np.float32)
                world = center + offset + R @ local
                coords[n_off + i] = world
                lb = np.minimum(lb, world)
                ub = np.maximum(ub, world)

            if aabb is not None:
                domain_idx = int(domain_ids[rid][0])
                aabb[domain_idx, 0] = lb
                aabb[domain_idx, 1] = ub

        self.meshBoundaryCoords.assign(coords)
        if aabb is not None:
            self.aabb.assign(aabb)

    def updateBBox(self):
        """Recompute all primitive and mesh rigid bounding boxes (kernel)."""
        if self.aabb is None:
            return
        wp.launch(
            _update_bbox_wp,
            dim=1,
            inputs=[
                int(self.numRigids),
                int(self.numAnalytical),
                1 if self.movingAnalytical else 0,
                self.rigidDomainIds,
                self.rigidParams,
                self.radius,
                self.cached_rotation_matrix,
                self.aabb,
            ],
        )

    # ===============================================================================
    # Some util functions
    # ===============================================================================

    def get_max_linear_velocity(self):
        """Return max boundary-point linear speed across all rigids.

        Uses an upper bound per rigid:
            |v_boundary|max <= |v_center| + |omega| * r_max
        where r_max is the farthest boundary distance from rigid center.
        """
        n = self.numRigids
        if n == 0:
            return 0.0

        V_np = self.V.numpy()[:n]
        W_np = self.RotV.numpy()[:n]

        rigid_ids = self.rigidDomainIds.numpy()[:n]
        rigid_type = rigid_ids[:, 1]
        rigid_params_np = self.rigidParams.numpy()[:n]
        centers = rigid_params_np[:, 0, :]
        primary = rigid_params_np[:, 1, :]
        radius_np = self.radius.numpy()[:n]

        boundary_radius = np.zeros(n, dtype=np.float32)
        center_speed = np.linalg.norm(V_np, axis=1)
        # 2D RotV is a scalar per rigid (1-D); 3D would be a vector.
        W_np = np.asarray(W_np)
        if W_np.ndim == 1:
            omega_speed = np.abs(W_np.astype(np.float32))
        else:
            omega_speed = np.linalg.norm(W_np, axis=1)

        # Primitive shapes
        ball_mask = rigid_type == int(RigidType.BALL)
        box_mask = rigid_type == int(RigidType.BOX)
        cyl_cap_mask = rigid_type == int(RigidType.CAPSULE)
        boundary_radius[ball_mask] = radius_np[ball_mask]
        boundary_radius[box_mask] = 0.5 * np.linalg.norm(primary[box_mask], axis=1)
        boundary_radius[cyl_cap_mask] = np.linalg.norm(primary[cyl_cap_mask], axis=1) + radius_np[cyl_cap_mask]

        # Compound sub-colliders: override with sub-collider envelope.
        compound_count = self.compound_count.numpy()[:n]
        compound_offset = self.compound_offset.numpy()[:n]
        if np.any(compound_count > 0):
            sub_pos = self.compound_local_pos.numpy()
            sub_radius = self.compound_radius.numpy()
            for rid in np.where(compound_count > 0)[0]:
                off = int(compound_offset[rid])
                cnt = int(compound_count[rid])
                if cnt <= 0:
                    continue
                p = sub_pos[off : off + cnt]
                r = sub_radius[off : off + cnt]
                boundary_radius[rid] = float(np.max(np.linalg.norm(p, axis=1) + r))

        # Mesh rigids: exact farthest transformed boundary node from center.
        mesh_mask = rigid_type == int(RigidType.MESH)
        if np.any(mesh_mask):
            rigid2mesh = self.rigid2MeshIndices.numpy()[:n]
            mesh_node_off = self.meshBoundaryNodeOffset.numpy()
            mesh_node_cnt = self.meshBoundaryNodeCount.numpy()
            mesh_coords = self.meshBoundaryCoords.numpy()
            for rid in np.where(mesh_mask)[0]:
                mid = int(rigid2mesh[rid])
                if mid < 0:
                    continue
                off = int(mesh_node_off[mid])
                cnt = int(mesh_node_cnt[mid])
                if cnt <= 0:
                    continue
                pts = mesh_coords[off : off + cnt]
                c = centers[rid]
                boundary_radius[rid] = float(np.max(np.linalg.norm(pts - c, axis=1)))

        boundary_speed_upper = center_speed + omega_speed * boundary_radius
        return float(boundary_speed_upper.max())

    def get_sh_update_interval(self):
        return self._sh_mesh_rebuild_interval

    def drawAll(self, gui, domains, colors=None, resolution=10):
        """Batch draw all rigids efficiently by caching to_numpy() calls.

        Args:
            gui: Viewer object (circle / line / lines)
            domains: List of domain objects
            colors: List of colors (one per domain), or None for default colors
            resolution: Legacy pixel scale for GUIs without world-space helpers
        """
        # Prefer true world-space drawing when Viewer provides it (avoids
        # radius*resolution vs window-width mismatch that oversized spheres).
        circle = getattr(gui, "circle_world", None) or (
            lambda pos, radius, color=0xFFFFFF: gui.circle(pos, color=color, radius=radius * resolution)
        )
        line = getattr(gui, "line_world", None) or (
            lambda a, b, radius=0.002, color=0xFFFFFF: gui.line(a, b, radius=radius * resolution, color=color)
        )
        lines = gui.lines
        # Cache all numpy arrays once (Warp 1.14+ forbids host item indexing).
        if self.numMesh > 0:
            all_boundary_coords = self.meshBoundaryCoords.numpy()
            all_boundary_elements = self.meshBoundaryElements.numpy()
            mesh_node_offsets = self.meshBoundaryNodeOffset.numpy()
            mesh_node_counts = self.meshBoundaryNodeCount.numpy()
            mesh_elem_offsets = self.meshBoundaryElementOffset.numpy()
            mesh_elem_counts = self.meshBoundaryElementCount.numpy()
            rigid2mesh = self.rigid2MeshIndices.numpy()

        all_rigid_params = self.rigidParams.numpy()
        all_domain_ids = self.rigidDomainIds.numpy()
        all_rot = self.cached_rotation_matrix.numpy()
        all_radius = self.radius.numpy()
        # all_shape_coords removed

        # Draw each domain using cached data
        for i, domain in enumerate(domains):
            if domain.type != DomainType.RIGID:
                continue

            color = colors[i] if colors is not None and i < len(colors) else 0xFFFFFF
            ndOffset = domain.ndOffset
            rtype = int(all_domain_ids[ndOffset][1])
            rotMat = all_rot[ndOffset]

            if rtype == RigidType.MESH:
                # Draw mesh rigid
                mesh_local_id = int(rigid2mesh[ndOffset])
                node_offset = int(mesh_node_offsets[mesh_local_id])
                num_nodes = int(mesh_node_counts[mesh_local_id])
                elem_offset = int(mesh_elem_offsets[mesh_local_id])
                num_elems = int(mesh_elem_counts[mesh_local_id])

                # Slice from cached arrays
                pos = all_boundary_coords[node_offset : node_offset + num_nodes, :2]
                elements = all_boundary_elements[elem_offset : elem_offset + num_elems]

                # 2D: draw edges
                a, b = elements[:, 0], elements[:, 1]
                lines(pos[a], pos[b], radius=2, color=color)
                
            elif rtype == RigidType.BALL:
                # Draw ball
                center = all_rigid_params[ndOffset, 0, :2]
                radius = float(all_radius[ndOffset])
                circle(center, radius=radius, color=color)

            elif rtype == RigidType.BOX:
                # Draw box
                center = all_rigid_params[ndOffset, 0]
                extent = all_rigid_params[ndOffset, 1]

                # 2D Box
                half_ext = 0.5 * extent
                corners_local = np.array(
                    [
                        [-half_ext[0], -half_ext[1]],
                        [half_ext[0], -half_ext[1]],
                        [half_ext[0], half_ext[1]],
                        [-half_ext[0], half_ext[1]],
                    ]
                )

                vertices = (rotMat @ corners_local.T).T + center

                for j in range(4):
                    line(vertices[j], vertices[(j + 1) % 4], radius=0.002, color=color)
           

            elif rtype == RigidType.CAPSULE:
                # Draw capsule
                center = all_rigid_params[ndOffset, 0, :2]
                lcdir = all_rigid_params[ndOffset, 1, :]
                lc = (rotMat @ lcdir)[:2] + center
                uc = center * 2 - lc
                radius = float(all_radius[ndOffset])

                # Draw center line
                line(lc, uc, radius=radius, color=color)

                if rtype == RigidType.CAPSULE:
                    # Draw end caps
                    circle(lc, radius=radius, color=color)
                    circle(uc, radius=radius, color=color)
