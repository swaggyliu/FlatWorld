"""Batched mixed-domain contact kernels for ExplicitLoop (NVIDIA Warp).

Handles penalty contact between FEM/Spring domains and analytical planes,
other FEM domains, rigid bodies, height fields, and voxel maps.  Work lists
are built once at init; each substep launches O(1) kernels per contact type.

Expected duck-typed manager / host attributes
--------------------------------------------
**FemSpringManager**
  ``coords``, ``V``, ``Fext`` : ``wp.array(dtype=wp.vec2)``
  ``boundaryElements`` : ``wp.array(dtype=wp.vec3i)``
  ``boundaryNodes``, ``domainNodeOffset``, ``domainBoundaryNodeOffset``,
  ``domainBoundaryNodeCount``, ``domainBoundaryElemOffset``, ``femDomainIds``
  ``boundaryNodeNormals`` : ``wp.array(dtype=wp.vec2)``
  ``spatialHash`` : ``SpatialHashManager`` (or None) with ``gridSize``,
  ``globalbbox``, ``cellNumbers``, ``total_cells``, ``cellStart``, ``cellEnd``,
  ``_sortedElemIdx``, ``domainIds``, ``queryElids``, ``MAX_QUERY``
  ``maybe_rebuild_fem_spatial_hash(dt)``, ``MAX_NODES``

**RigidManager**
  ``rigidParams`` : ``wp.array(dtype=wp.vec2, ndim=2)``  # [:,0]=center, [:,1]=prim
  ``rigidDomainIds`` : ``wp.array(dtype=wp.vec2i)``       # [0]=domain, [1]=type
  ``cached_rotation_matrix`` : ``wp.array(dtype=wp.mat22)``
  ``radius``, ``RotV`` : ``wp.array(dtype=float)``
  ``V``, ``accumulated_impulse`` : ``wp.array(dtype=wp.vec2)``
  ``accumulated_rotational_impulse`` : ``wp.array(dtype=float)``  # 2D omega
  ``meshBoundaryElements`` : ``wp.array(dtype=wp.vec3i)``
  ``meshBoundaryCoords``, ``meshElemLB``, ``meshElemUB`` : ``wp.vec2`` arrays
  ``meshElemMarginBase`` : ``wp.array(dtype=float)``
  ``spatialHash`` : same layout as FEM SH (or None)
  ``update_mesh_element_aabbs()``

**ExplicitLoop (host fields used by kernels)**
  ``aabb`` : ``wp.array(dtype=wp.vec2, ndim=2)`` shape ``(max_domains, 2)``
  ``_max_domains``, ``skip_bvh``, ``dt``, ``d`` (must be 2)
  ``_use_bvh_domain_mask``, ``_bvh_domain_stamp``, ``_bvh_active_stamp``
    length-1 / per-domain ``wp.array``; device uses ``[0]``, host ``.numpy()[0]``
"""

from __future__ import annotations

import numpy as np
import warp as wp

from contact_detection import detectPointToMeshBoundary, detectPointToPrimitive, pointToEdgeContact
from definitions import *
from spatialmanager import query_point_with_buffer
from wp_init import ensure_warp


# ---------------------------------------------------------------------------
# Host helpers (Warp 1.14: no host __setitem__ on arrays)
# ---------------------------------------------------------------------------


def _assign_scalar(arr: wp.array, value):
    np_arr = arr.numpy()
    np_arr[0] = value
    arr.assign(np_arr)


def _host_np(arr):
    """Read manager array as numpy (Warp / legacy Taichi / plain)."""
    if hasattr(arr, "numpy"):
        return arr.numpy()
    if hasattr(arr, "to_numpy"):
        return arr.numpy()
    return np.asarray(arr)


def _wp_int(n: int) -> wp.array:
    return wp.zeros(max(n, 1), dtype=int)


def _wp_float(n: int) -> wp.array:
    return wp.zeros(max(n, 1), dtype=float)


def _wp_vec2(n: int) -> wp.array:
    return wp.zeros(max(n, 1), dtype=wp.vec2)


def _wp_vec2i(n: int) -> wp.array:
    return wp.zeros(max(n, 1), dtype=wp.vec2i)


def _wp_vec3i(n: int) -> wp.array:
    return wp.zeros(max(n, 1), dtype=wp.vec3i)


def _from_numpy_i32(data) -> wp.array:
    return wp.array(np.asarray(data, dtype=np.int32), dtype=int)


def _from_numpy_f32(data) -> wp.array:
    return wp.array(np.asarray(data, dtype=np.float32), dtype=float)


def _from_numpy_vec2(data) -> wp.array:
    return wp.array(np.asarray(data, dtype=np.float32).reshape(-1, 2), dtype=wp.vec2)


def _from_numpy_vec2i(data) -> wp.array:
    return wp.array(np.asarray(data, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@wp.func
def _vec2_min_comp(v: wp.vec2):
    return wp.min(v[0], v[1])


@wp.func
def _point_outside_aabb(p: wp.vec2, lb: wp.vec2, ub: wp.vec2):
    return (_vec2_min_comp(p - lb) < 0.0) or (_vec2_min_comp(ub - p) < 0.0)


@wp.func
def _point_aabb_distance_sq(p: wp.vec2, lb: wp.vec2, ub: wp.vec2):
    dist2 = float(0.0)
    for d in range(2):
        v = float(0.0)
        if p[d] < lb[d]:
            v = lb[d] - p[d]
        elif p[d] > ub[d]:
            v = p[d] - ub[d]
        dist2 = dist2 + v * v
    return dist2


@wp.func
def _eval_fixed_tie_point(
    elem_idx: int,
    weights: wp.vec2,
    boundary_elements: wp.array(dtype=wp.vec3i),
    boundary_coords: wp.array(dtype=wp.vec2),
):
    conn = boundary_elements[elem_idx]
    cp = wp.vec2(0.0, 0.0)
    cp = cp + boundary_coords[conn[0]] * weights[0]
    cp = cp + boundary_coords[conn[1]] * weights[1]
    return cp


@wp.func
def _fem_fem_tie_elem_normal(
    elem_idx: int,
    boundary_elements: wp.array(dtype=wp.vec3i),
    coords: wp.array(dtype=wp.vec2),
):
    conn = boundary_elements[elem_idx]
    p0 = coords[conn[0]]
    p1 = coords[conn[1]]
    edge = p1 - p0
    elen = wp.length(edge)
    t = wp.vec2(1.0, 0.0)
    if elen > 1e-10:
        t = edge / elen
    return wp.vec2(-t[1], t[0])


@wp.func
def _nearest_on_curve_2d(
    x: float,
    z: float,
    height: wp.array(dtype=float),
    nx: int,
    lb_x: float,
    ub_x: float,
    reverse: int,
):
    """Device port of HeightFieldDomain.nearest_on_curve_2d."""
    span = ub_x - lb_x
    u = float(0.0)
    if span > 1e-12:
        u = (x - lb_x) / span
    u = wp.clamp(u, 0.0, 1.0)
    s = u * float(nx - 1)
    i0 = int(wp.floor(s))
    i1 = i0 + 1
    if i1 >= nx:
        i1 = nx - 1
    if i0 < 0:
        i0 = 0
    t = s - float(i0)
    h = height[i0] + (height[i1] - height[i0]) * t

    # central difference normal
    i_round = int(wp.round(s))
    if i_round < 0:
        i_round = 0
    if i_round >= nx:
        i_round = nx - 1
    j0 = i_round - 1
    j1 = i_round + 1
    if j0 < 0:
        j0 = 0
    if j1 >= nx:
        j1 = nx - 1
    dx_world = span / float(nx - 1)
    denom = dx_world * float(j1 - j0)
    dhdx = float(0.0)
    if denom > 1e-6:
        dhdx = (height[j1] - height[j0]) / denom
    n = wp.vec2(-dhdx, 1.0)
    if reverse == 1:
        n = -n
    nlen = wp.length(n)
    if nlen > 1e-12:
        n = n / nlen
    foot = wp.vec2(x, h)
    signed = wp.dot(wp.vec2(x, z) - foot, n)
    return foot, n, signed


@wp.func
def _voxel_signed_distance_2d(
    p: wp.vec2,
    occ: wp.array(dtype=int, ndim=2),
    nx: int,
    ny: int,
    lb: wp.vec2,
    dx: float,
    dz: float,
):
    """Device port of VoxelGridDomain.signed_distance_to_edges_2d (interior)."""
    best_d = float(1e9)
    best_n = wp.vec2(0.0, 1.0)
    best_c = p

    i = int(wp.floor((p[0] - lb[0]) / dx))
    j = int(wp.floor((p[1] - lb[1]) / dz))

    if 0 <= i < nx and 0 <= j < ny:
        if occ[i, j] == 1:
            dist_neg_x = float(1e9)
            for step in range(nx):
                idx = i - 1 - step
                if idx < 0 or occ[idx, j] == 0:
                    boundary_x = lb[0] + float(idx + 1) * dx
                    dist_neg_x = p[0] - boundary_x
                    break

            dist_pos_x = float(1e9)
            for step in range(nx):
                idx = i + 1 + step
                if idx >= nx or occ[idx, j] == 0:
                    boundary_x = lb[0] + float(idx) * dx
                    dist_pos_x = boundary_x - p[0]
                    break

            dist_neg_y = float(1e9)
            for step in range(ny):
                idx = j - 1 - step
                if idx < 0 or occ[i, idx] == 0:
                    boundary_y = lb[1] + float(idx + 1) * dz
                    dist_neg_y = p[1] - boundary_y
                    break

            dist_pos_y = float(1e9)
            for step in range(ny):
                idx = j + 1 + step
                if idx >= ny or occ[i, idx] == 0:
                    boundary_y = lb[1] + float(idx) * dz
                    dist_pos_y = boundary_y - p[1]
                    break

            # pick min face distance
            min_v = dist_neg_x
            min_face = 0
            if dist_pos_x < min_v:
                min_v = dist_pos_x
                min_face = 1
            if dist_neg_y < min_v:
                min_v = dist_neg_y
                min_face = 2
            if dist_pos_y < min_v:
                min_v = dist_pos_y
                min_face = 3

            if min_face == 0:
                best_d = -dist_neg_x
                best_n = wp.vec2(-1.0, 0.0)
                best_c = wp.vec2(p[0] - dist_neg_x, p[1])
            elif min_face == 1:
                best_d = -dist_pos_x
                best_n = wp.vec2(1.0, 0.0)
                best_c = wp.vec2(p[0] + dist_pos_x, p[1])
            elif min_face == 2:
                best_d = -dist_neg_y
                best_n = wp.vec2(0.0, -1.0)
                best_c = wp.vec2(p[0], p[1] - dist_neg_y)
            else:
                best_d = -dist_pos_y
                best_n = wp.vec2(0.0, 1.0)
                best_c = wp.vec2(p[0], p[1] + dist_pos_y)

    return best_d, best_n, best_c


@wp.func
def _cell_id_to_bounds(
    cid: int,
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
):
    grid_lb = global_bbox[0]
    grid_sz = grid_size[0]
    cn = cell_numbers[0]
    nx = cn[0]
    ix = cid % nx
    iy = cid // nx
    cell_lb = grid_lb + wp.vec2(float(ix) * grid_sz[0], float(iy) * grid_sz[1])
    cell_ub = cell_lb + grid_sz
    return cell_lb, cell_ub


@wp.func
def _point_to_cell_coord(
    pos: wp.vec2,
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
):
    gs = grid_size[0]
    cn = cell_numbers[0]
    delta = pos - global_bbox[0]
    rel = wp.vec2i(int(wp.floor(delta[0] / gs[0])), int(wp.floor(delta[1] / gs[1])))
    rx = rel[0]
    ry = rel[1]
    if rx < 0:
        rx = 0
    if ry < 0:
        ry = 0
    if rx > cn[0] - 1:
        rx = cn[0] - 1
    if ry > cn[1] - 1:
        ry = cn[1] - 1
    return wp.vec2i(rx, ry)


@wp.func
def _clear_rigid_mesh_friction_state(
    i: int,
    fric_prev_elem: wp.array(dtype=int),
    fric_prev_valid: wp.array(dtype=int),
    fric_prev_weights: wp.array(dtype=wp.vec2),
    fric_prev_force: wp.array(dtype=wp.vec2),
    fric_prev_penetration: wp.array(dtype=float),
):
    fric_prev_elem[i] = -1
    fric_prev_valid[i] = 0
    fric_prev_weights[i] = wp.vec2(0.0, 0.0)
    fric_prev_force[i] = wp.vec2(0.0, 0.0)
    fric_prev_penetration[i] = 0.0


@wp.func
def _apply_one_fixed_rigid_mesh_tie(
    nid: int,
    rigid_idx: int,
    cpoint: wp.vec2,
    penalty: float,
    dt: float,
    coords: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
    rigid_params: wp.array(dtype=wp.vec2, ndim=2),
    accum_impulse: wp.array(dtype=wp.vec2),
    accum_rot_impulse: wp.array(dtype=float),
):
    node_coord = coords[nid]
    move = cpoint - node_coord
    wp.atomic_add(Fext, nid, penalty * move)
    wp.atomic_add(accum_impulse, rigid_idx, -(penalty * move) * dt)
    lever = cpoint - rigid_params[rigid_idx, 0]
    force = -(penalty * move)
    torque = (lever[0] * force[1] - lever[1] * force[0]) * dt
    wp.atomic_add(accum_rot_impulse, rigid_idx, torque)


@wp.func
def _process_rigid_sh_cell(
    cid: int,
    node_coord: wp.vec2,
    target_rigid_did: int,
    penetration: float,
    normal: wp.vec2,
    cpoint: wp.vec2,
    weights: wp.vec2,
    elem_idx: int,
    rigid_did: int,
    found: int,
    i_a: int,
    has_cand: int,
    total_cells: wp.array(dtype=int),
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
    cell_start: wp.array(dtype=int),
    cell_end: wp.array(dtype=int),
    sorted_elem_idx: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    mesh_boundary_elements: wp.array(dtype=wp.vec3i),
    mesh_boundary_coords: wp.array(dtype=wp.vec2),
    mesh_elem_margin_base: wp.array(dtype=float),
    cache_elem: wp.array(dtype=wp.vec2i),
):
    """Test all elements in one SH cell against a query node."""
    tc = total_cells[0]
    if 0 <= cid < tc:
        skip_cell = int(0)
        best_abs = wp.abs(penetration)
        if found == 1 and best_abs < 1e8:
            c_lb, c_ub = _cell_id_to_bounds(cid, global_bbox, grid_size, cell_numbers)
            if _point_aabb_distance_sq(node_coord, c_lb, c_ub) > best_abs * best_abs:
                skip_cell = 1

        if skip_cell == 0:
            start = cell_start[cid]
            end = cell_end[cid]
            for p in range(start, end):
                ei = sorted_elem_idx[p]
                if ei >= 0:
                    did = domain_ids[ei][0]
                    global_eidx = domain_ids[ei][1]
                    if did != target_rigid_did:
                        continue
                    has_cand = 1
                    conn = mesh_boundary_elements[global_eidx]
                    el_lb = wp.vec2(1e30, 1e30)
                    el_ub = wp.vec2(-1e30, -1e30)
                    for vj in range(2):
                        c = mesh_boundary_coords[conn[vj]]
                        el_lb = wp.min(el_lb, c)
                        el_ub = wp.max(el_ub, c)
                    margin = wp.max(1e-4, mesh_elem_margin_base[global_eidx])
                    el_lb = el_lb - wp.vec2(margin, margin)
                    el_ub = el_ub + wp.vec2(margin, margin)

                    best_abs2 = wp.abs(penetration)
                    if found == 1 and best_abs2 < 1e8:
                        if _point_aabb_distance_sq(node_coord, el_lb, el_ub) > best_abs2 * best_abs2:
                            continue

                    if not _point_outside_aabb(node_coord, el_lb, el_ub):
                        conn2 = mesh_boundary_elements[global_eidx]
                        pen, norm, cp, curr_weights = detectPointToMeshBoundary(
                            node_coord, mesh_boundary_coords, conn2, margin
                        )
                        if wp.abs(pen) < wp.abs(penetration):
                            penetration = pen
                            normal = norm
                            cpoint = cp
                            weights = curr_weights
                            elem_idx = global_eidx
                            rigid_did = did
                            found = 1
                            cache_elem[i_a] = wp.vec2i(global_eidx, did)

    return penetration, normal, cpoint, weights, elem_idx, rigid_did, found, has_cand


# ---------------------------------------------------------------------------
# Kernels — pair activation / prefilters
# ---------------------------------------------------------------------------


@wp.kernel
def _activate_pairs_by_aabb_kernel(
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    active: wp.array(dtype=int),
    num_contact_pairs: int,
    aabb: wp.array(dtype=wp.vec2, ndim=2),
    max_domains: int,
):
    i = wp.tid()
    if i >= num_contact_pairs:
        return
    a = pair_a[i]
    b = pair_b[i]
    is_active = 1
    if 0 <= a < max_domains and 0 <= b < max_domains:
        for k in range(2):
            if aabb[a, 1][k] < aabb[b, 0][k] or aabb[b, 1][k] < aabb[a, 0][k]:
                is_active = 0
    active[i] = is_active


@wp.kernel
def _activate_pairs_kernel(
    pairs: wp.array(dtype=wp.vec2i),
    num_pairs: int,
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    active: wp.array(dtype=int),
    num_contact_pairs: int,
):
    i = wp.tid()
    if i < num_contact_pairs:
        active[i] = 0
    # second pass: mark overlaps (each thread scans all BVH pairs — O(n*m) like Taichi)
    if i < num_contact_pairs:
        for p in range(num_pairs):
            a = pairs[p][0]
            b = pairs[p][1]
            pa = wp.max(a, b)
            pb = wp.min(a, b)
            if pair_a[i] == pa and pair_b[i] == pb:
                active[i] = 1


@wp.kernel
def _build_active_rigid_mesh_workset_kernel(
    total_mesh_items: int,
    bc_rigid_mesh_idx: wp.array(dtype=int),
    bc_rigid_node: wp.array(dtype=int),
    bc_rigid_tied: wp.array(dtype=int),
    bc_rigid_pair: wp.array(dtype=int),
    bc_rigid_active: wp.array(dtype=int),
    bc_rigid_did: wp.array(dtype=int),
    bc_rigid_skip_epoch: wp.array(dtype=int),
    bc_rigid_node_near_any: wp.array(dtype=int),
    bc_rigid_node_mask: wp.array(dtype=int),
    bc_rigid_use_node_mask: wp.array(dtype=int),
    bc_rigid_mesh_active_count: wp.array(dtype=int),
    bc_rigid_mesh_active_idx: wp.array(dtype=int),
    fric_prev_elem: wp.array(dtype=int),
    fric_prev_valid: wp.array(dtype=int),
    fric_prev_weights: wp.array(dtype=wp.vec2),
    fric_prev_force: wp.array(dtype=wp.vec2),
    fric_prev_penetration: wp.array(dtype=float),
    rigid_domain_ids: wp.array(dtype=wp.vec3i),
    coords: wp.array(dtype=wp.vec2),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
):
    i = wp.tid()
    if i >= total_mesh_items:
        return

    wm = bc_rigid_mesh_idx[i]
    nid = bc_rigid_node[wm]
    tied = bc_rigid_tied[wm]

    if tied == 1:
        return

    if bc_rigid_node_near_any[nid] == 0:
        _clear_rigid_mesh_friction_state(
            i, fric_prev_elem, fric_prev_valid, fric_prev_weights, fric_prev_force, fric_prev_penetration
        )
        return

    pair_idx = bc_rigid_pair[wm]
    keep = bc_rigid_active[pair_idx] != 0

    if keep:
        if bc_rigid_skip_epoch[i] == 1:
            keep = False
        else:
            rid = bc_rigid_did[wm]
            if bc_rigid_use_node_mask[0] == 1 and rid < 31:
                if (bc_rigid_node_mask[nid] & (1 << rid)) == 0:
                    keep = False
            else:
                rigid_domain_idx = rigid_domain_ids[rid][0]
                node_coord = coords[nid]
                margin = float(1e-4)
                m = wp.vec2(margin, margin)
                rigid_lb = aabb[rigid_domain_idx, 0] - m
                rigid_ub = aabb[rigid_domain_idx, 1] + m
                if _point_outside_aabb(node_coord, rigid_lb, rigid_ub):
                    keep = False

    if keep:
        dst = int(wp.atomic_add(bc_rigid_mesh_active_count, 0, 1))
        bc_rigid_mesh_active_idx[dst] = i
    else:
        _clear_rigid_mesh_friction_state(
            i, fric_prev_elem, fric_prev_valid, fric_prev_weights, fric_prev_force, fric_prev_penetration
        )


@wp.kernel
def _reset_rigid_mesh_active_count_kernel(bc_rigid_mesh_active_count: wp.array(dtype=int)):
    bc_rigid_mesh_active_count[0] = 0


@wp.kernel
def _compute_fem_contact_aabb_kernel(
    num_nodes: int,
    bc_rigid_mesh_nodes: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec2),
    fem_contact_aabb_lb: wp.array(dtype=wp.vec2),
    fem_contact_aabb_ub: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    if i >= num_nodes:
        return
    nid = bc_rigid_mesh_nodes[i]
    coord = coords[nid]
    # Per-component atomic min/max on vec2
    wp.atomic_min(fem_contact_aabb_lb, 0, coord)
    wp.atomic_max(fem_contact_aabb_ub, 0, coord)


@wp.kernel
def _init_fem_contact_aabb_kernel(
    fem_contact_aabb_lb: wp.array(dtype=wp.vec2),
    fem_contact_aabb_ub: wp.array(dtype=wp.vec2),
):
    fem_contact_aabb_lb[0] = wp.vec2(1e9, 1e9)
    fem_contact_aabb_ub[0] = wp.vec2(-1e9, -1e9)


@wp.kernel
def _prefilter_mesh_node_activity_kernel(
    bc_rigid_mesh_node_count: wp.array(dtype=int),
    bc_rigid_mesh_nodes: wp.array(dtype=int),
    bc_rigid_mesh_rigid_count: wp.array(dtype=int),
    bc_rigid_mesh_rigid_ids: wp.array(dtype=int),
    bc_rigid_mesh_rigid_margin: wp.array(dtype=float),
    bc_rigid_node_near_any: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec2),
    rigid_domain_ids: wp.array(dtype=wp.vec3i),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
):
    i = wp.tid()
    n_nodes = bc_rigid_mesh_node_count[0]
    if i >= n_nodes:
        return
    nid = bc_rigid_mesh_nodes[i]
    coord = coords[nid]
    near_any = int(0)
    n_rigid = bc_rigid_mesh_rigid_count[0]
    for j in range(n_rigid):
        if near_any == 0:
            rid = bc_rigid_mesh_rigid_ids[j]
            domain_idx = rigid_domain_ids[rid][0]
            m = bc_rigid_mesh_rigid_margin[j]
            mv = wp.vec2(m, m)
            r_lb = aabb[domain_idx, 0] - mv
            r_ub = aabb[domain_idx, 1] + mv
            if not _point_outside_aabb(coord, r_lb, r_ub):
                near_any = 1
    if near_any == 1:
        bc_rigid_node_near_any[nid] = 1


# ---------------------------------------------------------------------------
# Kernels — analytical / FEM-FEM
# ---------------------------------------------------------------------------


@wp.kernel
def _batched_analytical_contact_kernel(
    total: int,
    bc_anal_pair: wp.array(dtype=int),
    bc_anal_active: wp.array(dtype=int),
    bc_anal_node: wp.array(dtype=int),
    bc_anal_pp: wp.array(dtype=wp.vec2),
    bc_anal_pn: wp.array(dtype=wp.vec2),
    bc_anal_penalty: wp.array(dtype=float),
    coords: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
):
    w = wp.tid()
    if w >= total:
        return
    pair_idx = bc_anal_pair[w]
    if bc_anal_active[pair_idx] == 0:
        return
    nid = bc_anal_node[w]
    node_coord = coords[nid]
    pp = bc_anal_pp[w]
    pn = bc_anal_pn[w]
    pen = wp.dot(node_coord - pp, pn)
    if pen < 0.0:
        wp.atomic_add(Fext, nid, -(pn * bc_anal_penalty[w] * pen))


@wp.kernel
def _initialize_fem_fem_ties_once_kernel(
    total: int,
    bc_flex_tied: wp.array(dtype=int),
    bc_flex_tie_resolved: wp.array(dtype=int),
    bc_flex_node: wp.array(dtype=int),
    bc_flex_be_off: wp.array(dtype=int),
    bc_flex_be_cnt: wp.array(dtype=int),
    bc_flex_target_did: wp.array(dtype=int),
    bc_flex_tie_elem: wp.array(dtype=int),
    bc_flex_tie_weights: wp.array(dtype=wp.vec2),
    bc_flex_tie_gap: wp.array(dtype=float),
    fixed_flexflex_tie_found_count: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec2),
    boundary_elements: wp.array(dtype=wp.vec3i),
    # spatial hash
    sh_max_query: int,
    sh_total_cells: wp.array(dtype=int),
    sh_global_bbox: wp.array(dtype=wp.vec2),
    sh_grid_size: wp.array(dtype=wp.vec2),
    sh_cell_numbers: wp.array(dtype=wp.vec2i),
    sh_cell_start: wp.array(dtype=int),
    sh_cell_end: wp.array(dtype=int),
    sh_sorted_elem_idx: wp.array(dtype=int),
    sh_domain_ids: wp.array(dtype=wp.vec2i),
    sh_query_elids: wp.array(dtype=int),
):
    w = wp.tid()
    if w >= total:
        return
    if bc_flex_tied[w] == 0:
        return
    if bc_flex_tie_resolved[w] == 1:
        return

    nid = bc_flex_node[w]
    node_coord = coords[nid]
    target_did = bc_flex_target_did[w]

    best_pen = float(1e9)
    best_elem = int(-1)
    best_weights = wp.vec2(0.0, 0.0)
    found = int(0)

    gs = sh_grid_size[0]
    min_buf = wp.max(gs[0], gs[1])
    query_buf = wp.max(wp.max(1e-4, min_buf), 0.001)

    # Note: shared query_elids races under parallel launches (same as Taichi).
    num_potentials = query_point_with_buffer(
        node_coord,
        query_buf,
        target_did,
        sh_max_query,
        sh_total_cells,
        sh_global_bbox,
        sh_grid_size,
        sh_cell_numbers,
        sh_cell_start,
        sh_cell_end,
        sh_sorted_elem_idx,
        sh_domain_ids,
        sh_query_elids,
    )
    for j in range(num_potentials):
        elem_idx = sh_query_elids[j]
        elem_conn = boundary_elements[elem_idx]
        pen, normal, cp, curr_weights = detectPointToMeshBoundary(
            node_coord, coords, elem_conn, query_buf
        )
        if wp.abs(pen) < wp.abs(best_pen) and wp.abs(pen) < query_buf:
            best_weights = curr_weights
            found = 1
            best_elem = elem_idx
            best_pen = pen

    if found == 1:
        cpoint = _eval_fixed_tie_point(best_elem, best_weights, boundary_elements, coords)
        gap = node_coord - cpoint
        normal = _fem_fem_tie_elem_normal(best_elem, boundary_elements, coords)
        bc_flex_tie_resolved[w] = 1
        bc_flex_tie_elem[w] = best_elem
        bc_flex_tie_weights[w] = best_weights
        bc_flex_tie_gap[w] = wp.dot(gap, normal)
        wp.atomic_add(fixed_flexflex_tie_found_count, 0, 1)


@wp.kernel
def _apply_fem_fem_ties_fixed_kernel(
    total: int,
    bc_flex_tied: wp.array(dtype=int),
    bc_flex_tie_resolved: wp.array(dtype=int),
    bc_flex_node: wp.array(dtype=int),
    bc_flex_tie_elem: wp.array(dtype=int),
    bc_flex_tie_weights: wp.array(dtype=wp.vec2),
    bc_flex_tie_gap: wp.array(dtype=float),
    bc_flex_penalty: wp.array(dtype=float),
    coords: wp.array(dtype=wp.vec2),
    boundary_elements: wp.array(dtype=wp.vec3i),
    Fext: wp.array(dtype=wp.vec2),
):
    w = wp.tid()
    if w >= total:
        return
    if bc_flex_tied[w] == 0:
        return
    if bc_flex_tie_resolved[w] == 0:
        return

    nid = bc_flex_node[w]
    elem_idx = bc_flex_tie_elem[w]
    weights = bc_flex_tie_weights[w]
    gap = bc_flex_tie_gap[w]
    penalty = bc_flex_penalty[w]

    cpoint = _eval_fixed_tie_point(elem_idx, weights, boundary_elements, coords)
    normal = _fem_fem_tie_elem_normal(elem_idx, boundary_elements, coords)
    target = cpoint + gap * normal
    move = target - coords[nid]
    total_force = penalty * move
    wp.atomic_add(Fext, nid, total_force)
    conn = boundary_elements[elem_idx]
    wp.atomic_add(Fext, conn[0], -(total_force * weights[0]))
    wp.atomic_add(Fext, conn[1], -(total_force * weights[1]))


@wp.kernel
def _batched_flexflex_contact_kernel(
    total: int,
    bc_flex_pair: wp.array(dtype=int),
    bc_flex_active: wp.array(dtype=int),
    bc_flex_target_did: wp.array(dtype=int),
    bc_flex_node: wp.array(dtype=int),
    bc_flex_penalty: wp.array(dtype=float),
    bc_flex_be_off: wp.array(dtype=int),
    bc_flex_be_cnt: wp.array(dtype=int),
    bc_flex_friction: wp.array(dtype=float),
    use_bvh_domain_mask: wp.array(dtype=int),
    bvh_domain_stamp: wp.array(dtype=int),
    bvh_active_stamp: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec2),
    V: wp.array(dtype=wp.vec2),
    boundary_elements: wp.array(dtype=wp.vec3i),
    Fext: wp.array(dtype=wp.vec2),
):
    w = wp.tid()
    if w >= total:
        return
    pair_idx = bc_flex_pair[w]
    if bc_flex_active[pair_idx] == 0:
        return
    target_did = bc_flex_target_did[w]
    if use_bvh_domain_mask[0] == 1 and bvh_domain_stamp[target_did] != bvh_active_stamp[0]:
        return

    nid = bc_flex_node[w]
    penalty = bc_flex_penalty[w]
    be_off = bc_flex_be_off[w]
    be_cnt = bc_flex_be_cnt[w]
    friction_coeff = bc_flex_friction[w]

    node_coord = coords[nid]
    node_vel = V[nid]

    best_pen = float(1e9)
    best_normal = wp.vec2(0.0, 0.0)
    best_weights = wp.vec2(0.0, 0.0)
    best_n0 = int(0)
    best_n1 = int(0)
    found = int(0)

    for j in range(be_cnt):
        elem_conn = boundary_elements[be_off + j]
        n0 = coords[elem_conn[0]]
        n1 = coords[elem_conn[1]]
        pen, normal, cp, is_inside, weights = pointToEdgeContact(node_coord, n0, n1, 2)
        if pen < best_pen and wp.abs(pen) < 1.0 and is_inside:
            best_pen = pen
            best_normal = normal
            best_weights = weights
            best_n0 = elem_conn[0]
            best_n1 = elem_conn[1]
            found = 1

    if found == 1 and best_pen < 0.0:
        normal_force = -best_normal * penalty * best_pen
        total_force = normal_force
        if friction_coeff > 1e-9:
            surf_vel = V[best_n0] * best_weights[0] + V[best_n1] * best_weights[1]
            relative_vel = node_vel - surf_vel
            tangential_vel = relative_vel - wp.dot(relative_vel, best_normal) * best_normal
            tlen = wp.length(tangential_vel)
            if tlen > 1e-9:
                friction_dir = -tangential_vel / tlen
                total_force = total_force + friction_dir * friction_coeff * wp.length(normal_force)
        wp.atomic_add(Fext, nid, total_force)


@wp.kernel
def _batched_flexflex_contact_kernel_sh(
    total: int,
    bc_flex_tied: wp.array(dtype=int),
    bc_flex_pair: wp.array(dtype=int),
    bc_flex_active: wp.array(dtype=int),
    bc_flex_target_did: wp.array(dtype=int),
    bc_flex_node: wp.array(dtype=int),
    bc_flex_penalty: wp.array(dtype=float),
    bc_flex_be_off: wp.array(dtype=int),
    bc_flex_be_cnt: wp.array(dtype=int),
    bc_flex_friction: wp.array(dtype=float),
    bc_flex_cache_elem: wp.array(dtype=int),
    use_bvh_domain_mask: wp.array(dtype=int),
    bvh_domain_stamp: wp.array(dtype=int),
    bvh_active_stamp: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec2),
    V: wp.array(dtype=wp.vec2),
    boundary_elements: wp.array(dtype=wp.vec3i),
    Fext: wp.array(dtype=wp.vec2),
    sh_max_query: int,
    sh_total_cells: wp.array(dtype=int),
    sh_global_bbox: wp.array(dtype=wp.vec2),
    sh_grid_size: wp.array(dtype=wp.vec2),
    sh_cell_numbers: wp.array(dtype=wp.vec2i),
    sh_cell_start: wp.array(dtype=int),
    sh_cell_end: wp.array(dtype=int),
    sh_sorted_elem_idx: wp.array(dtype=int),
    sh_domain_ids: wp.array(dtype=wp.vec2i),
    sh_query_elids: wp.array(dtype=int),
):
    w = wp.tid()
    if w >= total:
        return
    if bc_flex_tied[w] == 1:
        return
    pair_idx = bc_flex_pair[w]
    if bc_flex_active[pair_idx] == 0:
        return
    target_did = bc_flex_target_did[w]
    if use_bvh_domain_mask[0] == 1 and bvh_domain_stamp[target_did] != bvh_active_stamp[0]:
        return

    nid = bc_flex_node[w]
    friction_coeff = bc_flex_friction[w]
    be_off = bc_flex_be_off[w]
    be_cnt = bc_flex_be_cnt[w]
    node_coord = coords[nid]
    node_vel = V[nid]
    penalty = bc_flex_penalty[w]

    gs = sh_grid_size[0]
    min_buf = wp.max(gs[0], gs[1])
    query_buf = wp.max(wp.max(1e-4, min_buf), 0.016)

    best_pen = float(1e9)
    best_normal = wp.vec2(0.0, 0.0)
    weights = wp.vec2(0.0, 0.0)
    best_n0 = int(0)
    best_n1 = int(0)
    found = int(0)

    cached_elem = bc_flex_cache_elem[w]
    if cached_elem >= be_off and cached_elem < be_off + be_cnt:
        curr_conn = boundary_elements[cached_elem]
        lb = wp.vec2(1e30, 1e30)
        ub = wp.vec2(-1e30, -1e30)
        for k in range(2):
            coord = coords[curr_conn[k]]
            lb = wp.min(lb, coord)
            ub = wp.max(ub, coord)
        margin = wp.max(1e-4, 1e-2 * wp.length(ub - lb))
        lb = lb - wp.vec2(margin, margin)
        ub = ub + wp.vec2(margin, margin)
        if not _point_outside_aabb(node_coord, lb, ub):
            pen, normal, cp, curr_weights = detectPointToMeshBoundary(
                node_coord, coords, curr_conn, margin
            )
            if pen < 1e-4 and wp.abs(pen) < margin:
                best_pen = pen
                best_normal = normal
                weights = curr_weights
                best_n0 = curr_conn[0]
                best_n1 = curr_conn[1]
                found = 1

    if found == 0:
        num_potentials = query_point_with_buffer(
            node_coord,
            query_buf,
            target_did,
            sh_max_query,
            sh_total_cells,
            sh_global_bbox,
            sh_grid_size,
            sh_cell_numbers,
            sh_cell_start,
            sh_cell_end,
            sh_sorted_elem_idx,
            sh_domain_ids,
            sh_query_elids,
        )
        for j in range(num_potentials):
            global_elem_idx = sh_query_elids[j]
            curr_conn = boundary_elements[global_elem_idx]
            pen, normal, cp, curr_weights = detectPointToMeshBoundary(
                node_coord, coords, curr_conn, query_buf
            )
            if pen < best_pen and pen < 1e-4 and wp.abs(pen) < query_buf:
                best_pen = pen
                best_normal = normal
                weights = curr_weights
                best_n0 = curr_conn[0]
                best_n1 = curr_conn[1]
                found = 1
                bc_flex_cache_elem[w] = global_elem_idx

    if found == 0:
        bc_flex_cache_elem[w] = -1

    if found == 1 and best_pen < 0.0:
        normal_force = -best_normal * penalty * best_pen
        total_force = normal_force
        if friction_coeff > 1e-9:
            surf_vel = V[best_n0] * weights[0] + V[best_n1] * weights[1]
            relative_vel = node_vel - surf_vel
            tangential_vel = relative_vel - wp.dot(relative_vel, best_normal) * best_normal
            tlen = wp.length(tangential_vel)
            if tlen > 1e-9:
                friction_dir = -tangential_vel / tlen
                total_force = total_force + friction_dir * friction_coeff * wp.length(normal_force)
        wp.atomic_add(Fext, nid, total_force)
        wp.atomic_add(Fext, best_n0, -(total_force * weights[0]))
        wp.atomic_add(Fext, best_n1, -(total_force * weights[1]))


# ---------------------------------------------------------------------------
# Kernels — rigid primitive / mesh
# ---------------------------------------------------------------------------


@wp.kernel
def _batched_rigid_prim_kernel(
    num_prim: int,
    bc_rigid_prim_idx: wp.array(dtype=int),
    bc_rigid_pair: wp.array(dtype=int),
    bc_rigid_active: wp.array(dtype=int),
    bc_rigid_node: wp.array(dtype=int),
    bc_rigid_idx: wp.array(dtype=int),
    bc_rigid_tied: wp.array(dtype=int),
    bc_rigid_penalty: wp.array(dtype=float),
    bc_rigid_friction: wp.array(dtype=float),
    coords: wp.array(dtype=wp.vec2),
    V_fem: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
    rigid_domain_ids: wp.array(dtype=wp.vec3i),
    rigid_params: wp.array(dtype=wp.vec2, ndim=2),
    cached_rotation_matrix: wp.array(dtype=wp.mat22),
    radius: wp.array(dtype=float),
    V_rigid: wp.array(dtype=wp.vec2),
    RotV: wp.array(dtype=float),
    accum_impulse: wp.array(dtype=wp.vec2),
    accum_rot_impulse: wp.array(dtype=float),
    dt: float,
):
    wp_i = wp.tid()
    if wp_i >= num_prim:
        return
    w = bc_rigid_prim_idx[wp_i]
    pair_idx = bc_rigid_pair[w]
    if bc_rigid_active[pair_idx] == 0:
        return
    nid = bc_rigid_node[w]
    rigid_idx = bc_rigid_idx[w]
    tied = bc_rigid_tied[w]

    domain_i = rigid_domain_ids[rigid_idx][0]
    lb = aabb[domain_i, 0]
    ub = aabb[domain_i, 1]
    node_coord = coords[nid]
    node_vel = V_fem[nid]
    center = rigid_params[rigid_idx, 0]
    if _point_outside_aabb(node_coord, lb, ub):
        return

    rigid_type = rigid_domain_ids[rigid_idx][1]
    prim = rigid_params[rigid_idx, 1]
    rot = cached_rotation_matrix[rigid_idx]
    rad = radius[rigid_idx]

    pen, norm, cp = detectPointToPrimitive(node_coord, rigid_type, center, prim, rot, rad)
    margin = wp.length(ub - lb) * 0.5

    if tied == 1 and pen < margin:
        penalty = bc_rigid_penalty[w]
        move = cp - node_coord
        wp.atomic_add(Fext, nid, penalty * move)
        wp.atomic_add(accum_impulse, rigid_idx, -(penalty * move) * dt)
        lever = cp - center
        force = -(penalty * move)
        torque = (lever[0] * force[1] - lever[1] * force[0]) * dt
        wp.atomic_add(accum_rot_impulse, rigid_idx, torque)

    elif pen < margin:
        penetration = pen - 1e-4
        if penetration < 0.0:
            penalty = bc_rigid_penalty[w]
            friction_coeff = bc_rigid_friction[w]
            normal_force = -norm * penalty * penetration
            total_force = normal_force

            if friction_coeff > 1e-9:
                rigid_vel = V_rigid[rigid_idx]
                rigid_omega = RotV[rigid_idx]
                r_contact = cp - center
                surface_vel = rigid_vel + wp.vec2(
                    -rigid_omega * r_contact[1], rigid_omega * r_contact[0]
                )
                relative_vel = node_vel - surface_vel
                tangential_vel = relative_vel - wp.dot(relative_vel, norm) * norm
                tlen = wp.length(tangential_vel)
                if tlen > 1e-9:
                    friction_dir = -tangential_vel / tlen
                    total_force = total_force + friction_dir * friction_coeff * wp.length(normal_force)

            wp.atomic_add(Fext, nid, total_force)
            wp.atomic_add(accum_impulse, rigid_idx, -total_force * dt)
            lever = cp - center
            force = -total_force
            torque = (lever[0] * force[1] - lever[1] * force[0]) * dt
            wp.atomic_add(accum_rot_impulse, rigid_idx, torque)


@wp.kernel
def _batched_rigid_mesh_kernel_sh(
    batched_rigid_mesh_count: int,
    run_full: int,
    bc_rigid_mesh_active_count: wp.array(dtype=int),
    bc_rigid_mesh_active_idx: wp.array(dtype=int),
    bc_rigid_mesh_idx: wp.array(dtype=int),
    bc_rigid_node: wp.array(dtype=int),
    bc_rigid_penalty: wp.array(dtype=float),
    bc_rigid_friction: wp.array(dtype=float),
    bc_rigid_did: wp.array(dtype=int),
    bc_rigid_tied: wp.array(dtype=int),
    bc_rigid_cache_elem: wp.array(dtype=wp.vec2i),
    bc_rigid_skip_epoch: wp.array(dtype=int),
    fric_prev_elem: wp.array(dtype=int),
    fric_prev_valid: wp.array(dtype=int),
    fric_prev_weights: wp.array(dtype=wp.vec2),
    fric_prev_force: wp.array(dtype=wp.vec2),
    fric_prev_penetration: wp.array(dtype=float),
    coords: wp.array(dtype=wp.vec2),
    V_fem: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
    rigid_params: wp.array(dtype=wp.vec2, ndim=2),
    accum_impulse: wp.array(dtype=wp.vec2),
    accum_rot_impulse: wp.array(dtype=float),
    mesh_boundary_elements: wp.array(dtype=wp.vec3i),
    mesh_boundary_coords: wp.array(dtype=wp.vec2),
    mesh_elem_lb: wp.array(dtype=wp.vec2),
    mesh_elem_ub: wp.array(dtype=wp.vec2),
    mesh_elem_margin_base: wp.array(dtype=float),
    sh_total_cells: wp.array(dtype=int),
    sh_global_bbox: wp.array(dtype=wp.vec2),
    sh_grid_size: wp.array(dtype=wp.vec2),
    sh_cell_numbers: wp.array(dtype=wp.vec2i),
    sh_cell_start: wp.array(dtype=int),
    sh_cell_end: wp.array(dtype=int),
    sh_sorted_elem_idx: wp.array(dtype=int),
    sh_domain_ids: wp.array(dtype=wp.vec2i),
    dt: float,
):
    k = wp.tid()
    if k >= batched_rigid_mesh_count:
        return
    if k >= bc_rigid_mesh_active_count[0]:
        return

    i_a = bc_rigid_mesh_active_idx[k]
    wm = bc_rigid_mesh_idx[i_a]
    nid = bc_rigid_node[wm]
    penalty = bc_rigid_penalty[wm]
    friction_coeff = bc_rigid_friction[wm]
    target_rigid_did = bc_rigid_did[wm]
    tied = bc_rigid_tied[wm]
    if tied == 1:
        return

    node_coord = coords[nid]
    gs = sh_grid_size[0]
    min_buf = wp.max(gs[0], gs[1])
    query_buf_small = wp.max(1e-4, min_buf)

    penetration = float(1e9)
    normal = wp.vec2(0.0, 0.0)
    cpoint = wp.vec2(0.0, 0.0)
    curr_weights = wp.vec2(0.0, 0.0)
    curr_elem = int(-1)
    rigid_did = int(0)
    found = int(0)

    cached_el = bc_rigid_cache_elem[i_a][0]
    cached_did = bc_rigid_cache_elem[i_a][1]
    if cached_el >= 0 and cached_did == target_rigid_did:
        lb = mesh_elem_lb[cached_el]
        ub = mesh_elem_ub[cached_el]
        margin = wp.max(1e-4, mesh_elem_margin_base[cached_el])
        lb = lb - wp.vec2(margin, margin)
        ub = ub + wp.vec2(margin, margin)
        if not _point_outside_aabb(node_coord, lb, ub):
            conn = mesh_boundary_elements[cached_el]
            pen, norm, cp, weights = detectPointToMeshBoundary(
                node_coord, mesh_boundary_coords, conn, margin
            )
            if wp.abs(pen) < wp.abs(penetration):
                penetration = pen
                normal = norm
                cpoint = cp
                curr_weights = weights
                curr_elem = cached_el
                rigid_did = cached_did
                found = 1

    if found == 0 and bc_rigid_skip_epoch[i_a] == 1:
        _clear_rigid_mesh_friction_state(
            i_a, fric_prev_elem, fric_prev_valid, fric_prev_weights, fric_prev_force, fric_prev_penetration
        )
        return
    if found == 0 and run_full == 0:
        _clear_rigid_mesh_friction_state(
            i_a, fric_prev_elem, fric_prev_valid, fric_prev_weights, fric_prev_force, fric_prev_penetration
        )
        return

    if found == 0:
        sh_has_cand = int(0)
        if sh_total_cells[0] > 0:
            qlb = node_coord - wp.vec2(query_buf_small, query_buf_small)
            qub = node_coord + wp.vec2(query_buf_small, query_buf_small)
            origin = sh_global_bbox[0]
            cn = sh_cell_numbers[0]
            dlb = qlb - origin
            dub = qub - origin
            lx = int(wp.floor(dlb[0] / gs[0]))
            ly = int(wp.floor(dlb[1] / gs[1]))
            ux = int(wp.floor(dub[0] / gs[0]))
            uy = int(wp.floor(dub[1] / gs[1]))
            lx = int(wp.max(lx, 0))
            ly = int(wp.max(ly, 0))
            lx = int(wp.min(lx, cn[0] - 1))
            ly = int(wp.min(ly, cn[1] - 1))
            ux = int(wp.max(ux, 0))
            uy = int(wp.max(uy, 0))
            ux = int(wp.min(ux, cn[0] - 1))
            uy = int(wp.min(uy, cn[1] - 1))

            use_local_first = int(0)
            hint_lx = 0
            hint_ly = 0
            hint_ux = 0
            hint_uy = 0
            if cached_el >= 0 and cached_did == target_rigid_did:
                use_local_first = 1
                hint_center = 0.5 * (mesh_elem_lb[cached_el] + mesh_elem_ub[cached_el])
                hint_cell = _point_to_cell_coord(hint_center, sh_global_bbox, sh_grid_size, sh_cell_numbers)
                hint_lx = int(wp.max(hint_cell[0], lx))
                hint_ly = int(wp.max(hint_cell[1], ly))
                hint_ux = int(wp.min(hint_cell[0], ux))
                hint_uy = int(wp.min(hint_cell[1], uy))

            if use_local_first == 1:
                for ix in range(hint_lx, hint_ux + 1):
                    for iy in range(hint_ly, hint_uy + 1):
                        cid = ix + iy * cn[0]
                        (
                            penetration,
                            normal,
                            cpoint,
                            curr_weights,
                            curr_elem,
                            rigid_did,
                            found,
                            sh_has_cand,
                        ) = _process_rigid_sh_cell(
                            cid,
                            node_coord,
                            target_rigid_did,
                            penetration,
                            normal,
                            cpoint,
                            curr_weights,
                            curr_elem,
                            rigid_did,
                            found,
                            i_a,
                            sh_has_cand,
                            sh_total_cells,
                            sh_global_bbox,
                            sh_grid_size,
                            sh_cell_numbers,
                            sh_cell_start,
                            sh_cell_end,
                            sh_sorted_elem_idx,
                            sh_domain_ids,
                            mesh_boundary_elements,
                            mesh_boundary_coords,
                            mesh_elem_margin_base,
                            bc_rigid_cache_elem,
                        )

            for ix in range(lx, ux + 1):
                for iy in range(ly, uy + 1):
                    if use_local_first == 1:
                        if ix >= hint_lx and ix <= hint_ux and iy >= hint_ly and iy <= hint_uy:
                            continue
                    cid = ix + iy * cn[0]
                    (
                        penetration,
                        normal,
                        cpoint,
                        curr_weights,
                        curr_elem,
                        rigid_did,
                        found,
                        sh_has_cand,
                    ) = _process_rigid_sh_cell(
                        cid,
                        node_coord,
                        target_rigid_did,
                        penetration,
                        normal,
                        cpoint,
                        curr_weights,
                        curr_elem,
                        rigid_did,
                        found,
                        i_a,
                        sh_has_cand,
                        sh_total_cells,
                        sh_global_bbox,
                        sh_grid_size,
                        sh_cell_numbers,
                        sh_cell_start,
                        sh_cell_end,
                        sh_sorted_elem_idx,
                        sh_domain_ids,
                        mesh_boundary_elements,
                        mesh_boundary_coords,
                        mesh_elem_margin_base,
                        bc_rigid_cache_elem,
                    )

            if sh_has_cand == 0:
                bc_rigid_skip_epoch[i_a] = 1

    if found == 1 and penetration < 0.0:
        rigid_idx = rigid_did
        center = rigid_params[rigid_idx, 0]
        normal_force = -normal * penalty * penetration
        total_force = normal_force
        friction_force = wp.vec2(0.0, 0.0)

        if friction_coeff > 1e-9:
            Fy = friction_coeff * wp.length(normal_force)
            k_t = penalty * 1.0
            prev_force = wp.vec2(0.0, 0.0)
            slip = wp.vec2(0.0, 0.0)
            if fric_prev_valid[i_a] == 1:
                prev_elem = fric_prev_elem[i_a]
                prev_conn = mesh_boundary_elements[prev_elem]
                prev_weights = fric_prev_weights[i_a]
                curr_conn = mesh_boundary_elements[curr_elem]
                curr_surface_point = (
                    mesh_boundary_coords[curr_conn[0]] * curr_weights[0]
                    + mesh_boundary_coords[curr_conn[1]] * curr_weights[1]
                )
                prev_surface_point = (
                    mesh_boundary_coords[prev_conn[0]] * prev_weights[0]
                    + mesh_boundary_coords[prev_conn[1]] * prev_weights[1]
                )
                slip = curr_surface_point - prev_surface_point
                slip = slip - wp.dot(slip, normal) * normal
                prev_force = fric_prev_force[i_a]
                prev_force = prev_force - wp.dot(prev_force, normal) * normal

            f_trial = prev_force + k_t * slip
            f_trial_norm = wp.length(f_trial)
            if f_trial_norm <= Fy:
                friction_force = f_trial
            else:
                friction_force = f_trial / f_trial_norm * Fy

            fric_prev_elem[i_a] = curr_elem
            fric_prev_valid[i_a] = 1
            fric_prev_weights[i_a] = curr_weights
            fric_prev_force[i_a] = friction_force
            fric_prev_penetration[i_a] = penetration
            total_force = total_force + friction_force
        else:
            _clear_rigid_mesh_friction_state(
                i_a, fric_prev_elem, fric_prev_valid, fric_prev_weights, fric_prev_force, fric_prev_penetration
            )

        wp.atomic_add(Fext, nid, total_force)
        wp.atomic_add(accum_impulse, rigid_idx, -total_force * dt)
        lever = cpoint - center
        force = -total_force
        torque = (lever[0] * force[1] - lever[1] * force[0]) * dt
        wp.atomic_add(accum_rot_impulse, rigid_idx, torque)
    else:
        _clear_rigid_mesh_friction_state(
            i_a, fric_prev_elem, fric_prev_valid, fric_prev_weights, fric_prev_force, fric_prev_penetration
        )


@wp.kernel
def _initialize_fem_rigid_mesh_ties_once_kernel(
    total_mesh_items: int,
    bc_rigid_mesh_idx: wp.array(dtype=int),
    bc_rigid_tied: wp.array(dtype=int),
    bc_rigid_tie_resolved: wp.array(dtype=int),
    bc_rigid_node: wp.array(dtype=int),
    bc_rigid_did: wp.array(dtype=int),
    bc_rigid_cache_elem: wp.array(dtype=wp.vec2i),
    bc_rigid_tie_weights: wp.array(dtype=wp.vec2),
    fixed_rigid_mesh_tie_found_count: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec2),
    mesh_boundary_elements: wp.array(dtype=wp.vec3i),
    mesh_boundary_coords: wp.array(dtype=wp.vec2),
    mesh_elem_margin_base: wp.array(dtype=float),
    sh_total_cells: wp.array(dtype=int),
    sh_global_bbox: wp.array(dtype=wp.vec2),
    sh_grid_size: wp.array(dtype=wp.vec2),
    sh_cell_numbers: wp.array(dtype=wp.vec2i),
    sh_cell_start: wp.array(dtype=int),
    sh_cell_end: wp.array(dtype=int),
    sh_sorted_elem_idx: wp.array(dtype=int),
    sh_domain_ids: wp.array(dtype=wp.vec2i),
):
    i_a = wp.tid()
    if i_a >= total_mesh_items:
        return
    wm = bc_rigid_mesh_idx[i_a]
    if bc_rigid_tied[wm] == 0:
        return
    if bc_rigid_tie_resolved[i_a] == 1:
        return

    nid = bc_rigid_node[wm]
    target_rigid_did = bc_rigid_did[wm]
    node_coord = coords[nid]

    gs = sh_grid_size[0]
    min_buf = wp.max(gs[0], gs[1])
    query_buf = wp.max(1e-4, min_buf)

    penetration = float(1e9)
    normal = wp.vec2(0.0, 0.0)
    cpoint = wp.vec2(0.0, 0.0)
    weights = wp.vec2(0.0, 0.0)
    elem_idx = int(-1)
    rigid_did = int(-1)
    found = int(0)
    has_cand = int(0)

    if sh_total_cells[0] > 0:
        qlb = node_coord - wp.vec2(query_buf, query_buf)
        qub = node_coord + wp.vec2(query_buf, query_buf)
        origin = sh_global_bbox[0]
        cn = sh_cell_numbers[0]
        dlb = qlb - origin
        dub = qub - origin
        lx = int(wp.floor(dlb[0] / gs[0]))
        ly = int(wp.floor(dlb[1] / gs[1]))
        ux = int(wp.floor(dub[0] / gs[0]))
        uy = int(wp.floor(dub[1] / gs[1]))
        lx = int(wp.max(lx, 0))
        ly = int(wp.max(ly, 0))
        lx = int(wp.min(lx, cn[0] - 1))
        ly = int(wp.min(ly, cn[1] - 1))
        ux = int(wp.max(ux, 0))
        uy = int(wp.max(uy, 0))
        ux = int(wp.min(ux, cn[0] - 1))
        uy = int(wp.min(uy, cn[1] - 1))

        for ix in range(lx, ux + 1):
            for iy in range(ly, uy + 1):
                cid = ix + iy * cn[0]
                (
                    penetration,
                    normal,
                    cpoint,
                    weights,
                    elem_idx,
                    rigid_did,
                    found,
                    has_cand,
                ) = _process_rigid_sh_cell(
                    cid,
                    node_coord,
                    target_rigid_did,
                    penetration,
                    normal,
                    cpoint,
                    weights,
                    elem_idx,
                    rigid_did,
                    found,
                    i_a,
                    has_cand,
                    sh_total_cells,
                    sh_global_bbox,
                    sh_grid_size,
                    sh_cell_numbers,
                    sh_cell_start,
                    sh_cell_end,
                    sh_sorted_elem_idx,
                    sh_domain_ids,
                    mesh_boundary_elements,
                    mesh_boundary_coords,
                    mesh_elem_margin_base,
                    bc_rigid_cache_elem,
                )

    if found == 1:
        bc_rigid_tie_resolved[i_a] = 1
        bc_rigid_cache_elem[i_a] = wp.vec2i(elem_idx, rigid_did)
        bc_rigid_tie_weights[i_a] = weights
        wp.atomic_add(fixed_rigid_mesh_tie_found_count, 0, 1)


@wp.kernel
def _apply_fem_rigid_mesh_ties_fixed_kernel(
    total_mesh_items: int,
    bc_rigid_mesh_idx: wp.array(dtype=int),
    bc_rigid_tied: wp.array(dtype=int),
    bc_rigid_tie_resolved: wp.array(dtype=int),
    bc_rigid_node: wp.array(dtype=int),
    bc_rigid_penalty: wp.array(dtype=float),
    bc_rigid_cache_elem: wp.array(dtype=wp.vec2i),
    bc_rigid_tie_weights: wp.array(dtype=wp.vec2),
    coords: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
    rigid_params: wp.array(dtype=wp.vec2, ndim=2),
    accum_impulse: wp.array(dtype=wp.vec2),
    accum_rot_impulse: wp.array(dtype=float),
    mesh_boundary_elements: wp.array(dtype=wp.vec3i),
    mesh_boundary_coords: wp.array(dtype=wp.vec2),
    dt: float,
):
    i_a = wp.tid()
    if i_a >= total_mesh_items:
        return
    wm = bc_rigid_mesh_idx[i_a]
    if bc_rigid_tied[wm] == 0:
        return
    if bc_rigid_tie_resolved[i_a] == 0:
        return

    nid = bc_rigid_node[wm]
    penalty = bc_rigid_penalty[wm]
    elem_idx = bc_rigid_cache_elem[i_a][0]
    rigid_idx = bc_rigid_cache_elem[i_a][1]
    weights = bc_rigid_tie_weights[i_a]
    cpoint = _eval_fixed_tie_point(elem_idx, weights, mesh_boundary_elements, mesh_boundary_coords)
    _apply_one_fixed_rigid_mesh_tie(
        nid,
        rigid_idx,
        cpoint,
        penalty,
        dt,
        coords,
        Fext,
        rigid_params,
        accum_impulse,
        accum_rot_impulse,
    )


@wp.kernel
def _batched_heightfield_contact_kernel(
    total: int,
    dt: float,
    bc_hf_pair: wp.array(dtype=int),
    bc_hf_active: wp.array(dtype=int),
    bc_hf_node: wp.array(dtype=int),
    bc_hf_penalty: wp.array(dtype=float),
    coords: wp.array(dtype=wp.vec2),
    V: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
    hf_height: wp.array(dtype=float),
    hf_nx: int,
    hf_lb_x: float,
    hf_ub_x: float,
    hf_reverse: int,
):
    w = wp.tid()
    if w >= total:
        return
    pair_idx = bc_hf_pair[w]
    if bc_hf_active[pair_idx] == 0:
        return
    nid = bc_hf_node[w]
    penalty = bc_hf_penalty[w]
    node_pos = coords[nid]
    node_vel = V[nid]
    damping_coeff = float(0.5)
    foot, n, signed = _nearest_on_curve_2d(
        node_pos[0], node_pos[1], hf_height, hf_nx, hf_lb_x, hf_ub_x, hf_reverse
    )
    if signed < 0.0:
        contact_force = -n * penalty * signed
        vn = wp.dot(node_vel, n)
        if vn < 0.0:
            contact_force = contact_force + (-n * damping_coeff * penalty * vn * dt)
        wp.atomic_add(Fext, nid, contact_force)


@wp.kernel
def _batched_voxelmap_contact_kernel(
    total: int,
    dt: float,
    bc_voxel_pair: wp.array(dtype=int),
    bc_voxel_active: wp.array(dtype=int),
    bc_voxel_node: wp.array(dtype=int),
    bc_voxel_penalty: wp.array(dtype=float),
    coords: wp.array(dtype=wp.vec2),
    V: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
    voxel_occ: wp.array(dtype=int, ndim=2),
    voxel_nx: int,
    voxel_ny: int,
    voxel_lb: wp.vec2,
    voxel_dx: float,
    voxel_dz: float,
):
    w = wp.tid()
    if w >= total:
        return
    pair_idx = bc_voxel_pair[w]
    if bc_voxel_active[pair_idx] == 0:
        return
    nid = bc_voxel_node[w]
    penalty = bc_voxel_penalty[w]
    node_pos = coords[nid]
    node_vel = V[nid]
    damping_coeff = float(0.5)
    signed_dist, n, cpoint = _voxel_signed_distance_2d(
        node_pos, voxel_occ, voxel_nx, voxel_ny, voxel_lb, voxel_dx, voxel_dz
    )
    if signed_dist < 0.0:
        penetration = -signed_dist
        contact_force = n * penalty * penetration
        vn = wp.dot(node_vel, n)
        if vn < 0.0:
            contact_force = contact_force + (-n * damping_coeff * penalty * vn * dt)
        wp.atomic_add(Fext, nid, contact_force)


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


class MixedContact:
    """Mixin providing batched contact build/run/kernels for ExplicitLoop."""

    def run_batched_contacts(self, pairs_field, num_pairs_work):
        """Execute all batched contact kernels for one substep."""
        ensure_warp()
        if self._batched_anal_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_anal_pair_a, self._bc_anal_pair_b, self._bc_anal_active
            )
            self._launch_batched_analytical_contact(self._batched_anal_count)

        if self._batched_flex_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_flex_pair_a, self._bc_flex_pair_b, self._bc_flex_active
            )

            flex_tie_initialized = (
                hasattr(self, "_fixed_flexflex_tie_initialized")
                and int(_host_np(self._fixed_flexflex_tie_initialized)[0]) == 1
            )

            need_flex_hash_refresh = self._nontied_flex_item_count > 0 or (
                self._tied_flex_item_count > 0 and not flex_tie_initialized
            )
            if self._use_fem_spatial_hash and need_flex_hash_refresh:
                if self.femSpringManager is not None:
                    self.femSpringManager.maybe_rebuild_fem_spatial_hash(self.dt)

            if self._tied_flex_item_count > 0:
                if int(_host_np(self._fixed_flexflex_tie_initialized)[0]) == 0:
                    self._launch_initialize_fem_fem_ties_once(self._batched_flex_count)
                    _assign_scalar(self._fixed_flexflex_tie_initialized, 1)
                    print(
                        f"Initialized FEM-FEM ties: resolved "
                        f"{int(_host_np(self._fixed_flexflex_tie_found_count)[0])} / "
                        f"{self._tied_flex_item_count} work items"
                    )
                self._launch_apply_fem_fem_ties_fixed(self._batched_flex_count)

            if self._nontied_flex_item_count > 0:
                if self._use_fem_spatial_hash:
                    self._launch_batched_flexflex_contact_sh(self._batched_flex_count)
                else:
                    self._launch_batched_flexflex_contact(self._batched_flex_count)

        if self._batched_rigid_count > 0:
            if self._rigid_contact_prefilter_dirty:
                if self.skip_bvh:
                    self._launch_activate_pairs_by_aabb(
                        self._bc_rigid_pair_a,
                        self._bc_rigid_pair_b,
                        self._bc_rigid_active,
                        self._bc_rigid_active.shape[0],
                    )
                else:
                    self._activate_or_fill_pairs(
                        pairs_field,
                        num_pairs_work,
                        self._bc_rigid_pair_a,
                        self._bc_rigid_pair_b,
                        self._bc_rigid_active,
                    )
            if self._batched_rigid_prim_count > 0:
                self._launch_batched_rigid_prim(self._batched_rigid_prim_count)
            if self._batched_rigid_mesh_count > 0:
                tie_initialized = (
                    hasattr(self, "_fixed_rigid_mesh_tie_initialized")
                    and int(_host_np(self._fixed_rigid_mesh_tie_initialized)[0]) == 1
                )
                needs_mesh_element_aabb = self._nontied_rigid_mesh_item_count > 0 or (
                    self._tied_rigid_mesh_item_count > 0 and not tie_initialized
                )
                if self._rigid_mesh_aabb_dirty and needs_mesh_element_aabb:
                    self.rigidManager.update_mesh_element_aabbs()
                    self._rigid_mesh_aabb_dirty = False

                if self._tied_rigid_mesh_item_count > 0:
                    if int(_host_np(self._fixed_rigid_mesh_tie_initialized)[0]) == 0:
                        self._launch_initialize_fem_rigid_mesh_ties(self._batched_rigid_mesh_count)
                        _assign_scalar(self._fixed_rigid_mesh_tie_initialized, 1)
                    self._launch_apply_fem_rigid_mesh_ties(self._batched_rigid_mesh_count)

                if self._nontied_rigid_mesh_item_count > 0:
                    run_full_mesh_contact = 1
                    if self._rigid_contact_subcycle > 1 and (self._counter_py % self._rigid_contact_subcycle) != 0:
                        run_full_mesh_contact = 0

                    self._bc_rigid_node_near_any.fill_(0)
                    self._launch_prefilter_mesh_node_activity()
                    self._launch_build_active_rigid_mesh_workset(self._batched_rigid_mesh_count)
                    self._launch_batched_rigid_mesh_sh(run_full_mesh_contact)

        if self._batched_hf_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_hf_pair_a, self._bc_hf_pair_b, self._bc_hf_active
            )
            self._launch_batched_heightfield(self._batched_hf_count, float(self.dt))

        if self._batched_voxel_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_voxel_pair_a, self._bc_voxel_pair_b, self._bc_voxel_active
            )
            self._launch_batched_voxelmap(self._batched_voxel_count, float(self.dt))

    def _build_batched_contacts(self):
        """Build flat Warp work-lists for all contact pairs."""
        ensure_warp()
        from femcontact import (
            ContactFlexAnalytical,
            ContactFlexFlex,
            ContactFlexHeightField,
            ContactFlexRigid,
            ContactFlexVoxelMap,
            ContactSpringAnalytical,
            ContactSpringHeightField,
            ContactSpringRigid,
        )

        anal_contacts = []
        flex_contacts = []
        rigid_contacts = []
        hf_contacts = []
        voxel_contacts = []
        unbatched = []
        unbatched_pair_ids = []
        pair_ids_np = self.contactPairIds.numpy() if self.contactPairIds is not None else None

        for idx, c in enumerate(self.contacts):
            if isinstance(c, (ContactFlexAnalytical, ContactSpringAnalytical)):
                anal_contacts.append((c, idx))
            elif isinstance(c, ContactFlexFlex):
                flex_contacts.append((c, idx))
            elif isinstance(c, (ContactFlexRigid, ContactSpringRigid)):
                rigid_contacts.append((c, idx))
            elif isinstance(c, (ContactFlexHeightField, ContactSpringHeightField)):
                hf_contacts.append((c, idx))
            elif isinstance(c, ContactFlexVoxelMap):
                voxel_contacts.append((c, idx))
            else:
                unbatched.append(c)
                unbatched_pair_ids.append((int(pair_ids_np[idx][0]), int(pair_ids_np[idx][1])))

        # ── Analytical ──
        self._batched_anal_count = 0
        if anal_contacts and self.femSpringManager is not None:
            work_bn = []
            work_penalty = []
            work_plane_pt = []
            work_plane_nm = []
            work_dom_did = []
            work_pair = []

            self._anal_pair_domain_ids = [
                (int(pair_ids_np[orig_idx][0]), int(pair_ids_np[orig_idx][1]))
                for _, (_, orig_idx) in enumerate(anal_contacts)
            ]

            fsm = self.femSpringManager
            node_off = _host_np(fsm.domainNodeOffset)
            bn_off = _host_np(fsm.domainBoundaryNodeOffset)
            bn_cnt = _host_np(fsm.domainBoundaryNodeCount)
            bnodes = _host_np(fsm.boundaryNodes)

            for pair_idx, (c, _) in enumerate(anal_contacts):
                is_spring = isinstance(c, ContactSpringAnalytical)
                d_idx = c.domain1.domainIdx
                anal_idx = c.domain2.ndOffset
                rp = _host_np(c.domain2.rigidManager.rigidParams)
                pp = [float(rp[anal_idx, 0][k]) for k in range(2)]
                pn = [float(rp[anal_idx, 1][k]) for k in range(2)]

                if is_spring:
                    off = int(node_off[d_idx])
                    cnt = c.domain1.nnodes
                    for i in range(cnt):
                        work_bn.append(off + i)
                        work_penalty.append(float(c.penalty))
                        work_plane_pt.append(pp)
                        work_plane_nm.append(pn)
                        work_dom_did.append(d_idx)
                        work_pair.append(pair_idx)
                else:
                    off = int(bn_off[d_idx])
                    cnt = int(bn_cnt[d_idx])
                    for i in range(cnt):
                        work_bn.append(int(bnodes[off + i]))
                        work_penalty.append(float(c.penalty))
                        work_plane_pt.append(pp)
                        work_plane_nm.append(pn)
                        work_dom_did.append(d_idx)
                        work_pair.append(pair_idx)

            total = len(work_bn)
            if total > 0:
                self._bc_anal_node = _from_numpy_i32(work_bn)
                self._bc_anal_penalty = _from_numpy_f32(work_penalty)
                self._bc_anal_pp = _from_numpy_vec2(work_plane_pt)
                self._bc_anal_pn = _from_numpy_vec2(work_plane_nm)
                self._bc_anal_dom_did = _from_numpy_i32(work_dom_did)
                self._bc_anal_pair = _from_numpy_i32(work_pair)
                self._bc_anal_pair_a, self._bc_anal_pair_b, self._bc_anal_active = self._create_pair_activation_fields(
                    self._anal_pair_domain_ids
                )
                self._batched_anal_count = total
                print(f"  Batched Analytical: {len(anal_contacts)} pairs → 1 kernel ({total} work items)")

        # ── FlexFlex ──
        self._batched_flex_count = 0
        self._use_fem_spatial_hash = False
        self._tied_flex_item_count = 0
        self._nontied_flex_item_count = 0
        if flex_contacts and self.femSpringManager is not None:
            work_node = []
            work_penalty = []
            work_be_off = []
            work_be_cnt = []
            work_pair = []
            work_friction = []
            work_target_did = []
            work_tied = []

            fsm = self.femSpringManager
            bn_off = _host_np(fsm.domainBoundaryNodeOffset)
            bn_cnt = _host_np(fsm.domainBoundaryNodeCount)
            be_off_arr = _host_np(fsm.domainBoundaryElemOffset)
            bnodes = _host_np(fsm.boundaryNodes)

            for pair_idx, (c, _) in enumerate(flex_contacts):
                fric = max(c.domain1.friction, c.domain2.friction)
                pen = float(c.penalty)
                tied = 1 if (hasattr(c, "tied") and c.tied) else 0

                d2_idx = c.domain2.domainIdx
                off2 = int(bn_off[d2_idx])
                cnt2 = int(bn_cnt[d2_idx])
                d1_be_off = int(be_off_arr[c.domain1.domainIdx])
                d1_be_cnt = int(c.domain1.mesh.numBoundElements)
                d1_did = int(c.domain1.domainIdx)
                for i in range(cnt2):
                    work_node.append(int(bnodes[off2 + i]))
                    work_penalty.append(pen)
                    work_be_off.append(d1_be_off)
                    work_be_cnt.append(d1_be_cnt)
                    work_pair.append(pair_idx)
                    work_friction.append(fric)
                    work_target_did.append(d1_did)
                    work_tied.append(tied)

                d1_idx = c.domain1.domainIdx
                off1 = int(bn_off[d1_idx])
                cnt1 = int(bn_cnt[d1_idx])
                d2_be_off = int(be_off_arr[c.domain2.domainIdx])
                d2_be_cnt = int(c.domain2.mesh.numBoundElements)
                d2_did = int(c.domain2.domainIdx)
                for i in range(cnt1):
                    work_node.append(int(bnodes[off1 + i]))
                    work_penalty.append(pen)
                    work_be_off.append(d2_be_off)
                    work_be_cnt.append(d2_be_cnt)
                    work_pair.append(pair_idx)
                    work_friction.append(fric)
                    work_target_did.append(d2_did)
                    work_tied.append(tied)

            total = len(work_node)
            if total > 0:
                self._bc_flex_node = _from_numpy_i32(work_node)
                self._bc_flex_penalty = _from_numpy_f32(work_penalty)
                self._bc_flex_be_off = _from_numpy_i32(work_be_off)
                self._bc_flex_be_cnt = _from_numpy_i32(work_be_cnt)
                self._bc_flex_pair = _from_numpy_i32(work_pair)
                self._bc_flex_friction = _from_numpy_f32(work_friction)
                self._bc_flex_target_did = _from_numpy_i32(work_target_did)
                self._bc_flex_cache_elem = _wp_int(total)
                self._bc_flex_cache_elem.fill_(-1)
                self._bc_flex_kind = _wp_int(total)
                self._bc_flex_tied = _from_numpy_i32(work_tied)
                work_tied_np = np.array(work_tied, dtype=np.int32)

                self._tied_flex_item_count = int(work_tied_np.sum())
                self._nontied_flex_item_count = total - self._tied_flex_item_count
                if self._tied_flex_item_count > 0:
                    self._bc_flex_tie_resolved = _wp_int(total)
                    self._bc_flex_tie_elem = _wp_int(total)
                    self._bc_flex_tie_weights = _wp_vec2(total)
                    self._bc_flex_tie_gap = _wp_float(total)
                    self._bc_flex_tie_resolved.fill_(0)
                    self._bc_flex_tie_elem.fill_(-1)
                    self._bc_flex_tie_weights.fill_(0.0)
                    self._bc_flex_tie_gap.fill_(0.0)
                    self._fixed_flexflex_tie_initialized = wp.zeros(1, dtype=int)
                    self._fixed_flexflex_tie_found_count = wp.zeros(1, dtype=int)

                self._flex_pair_domain_ids = [
                    (int(pair_ids_np[orig_idx][0]), int(pair_ids_np[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(flex_contacts)
                ]
                self._bc_flex_pair_a, self._bc_flex_pair_b, self._bc_flex_active = self._create_pair_activation_fields(
                    self._flex_pair_domain_ids
                )
                self._batched_flex_count = total
                self._use_fem_spatial_hash = self.femSpringManager.spatialHash is not None
                print(
                    f"  Batched FlexFlex: {len(flex_contacts)} pairs → 1 kernel "
                    f"({total} work items, tied={self._tied_flex_item_count}, "
                    f"spatial_hash={'ON' if self._use_fem_spatial_hash else 'OFF'})"
                )

        # ── Rigid ──
        self._batched_rigid_count = 0
        self._batched_rigid_mesh_count = 0
        self._batched_rigid_prim_count = 0
        self._use_rigid_spatial_hash = False
        self._tied_rigid_mesh_item_count = 0
        self._nontied_rigid_mesh_item_count = 0
        if rigid_contacts and self.femSpringManager is not None and self.rigidManager is not None:
            work_node = []
            work_normals = []
            work_rigid_idx = []
            work_penalty = []
            work_friction = []
            work_is_mesh = []
            work_fem_did = []
            work_rigid_did = []
            work_pair = []
            work_tied = []

            fsm = self.femSpringManager
            node_off = _host_np(fsm.domainNodeOffset)
            bn_off = _host_np(fsm.domainBoundaryNodeOffset)
            bn_cnt = _host_np(fsm.domainBoundaryNodeCount)
            bnodes = _host_np(fsm.boundaryNodes)
            bnormals = _host_np(fsm.boundaryNodeNormals)
            fem_ids = _host_np(fsm.femDomainIds)

            for pair_idx, (c, orig_idx) in enumerate(rigid_contacts):
                is_spring = isinstance(c, ContactSpringRigid)
                d_idx = c.domain1.domainIdx
                rigid_idx = c.domain2.ndOffset
                pen = float(c.penalty)
                fric = float(c.domain2.friction)
                tied = 1 if (hasattr(c, "tied") and c.tied) else 0

                rids = _host_np(c.domain2.rigidManager.rigidDomainIds)
                rigid_type_val = int(rids[rigid_idx][1])
                is_mesh = 1 if rigid_type_val == int(RigidType.MESH) else 0
                fem_gid = int(fem_ids[c.domain1.domainIdx]) if is_mesh else 0
                rigid_gid = int(rigid_idx) if is_mesh else 0

                if is_spring:
                    off = int(node_off[d_idx])
                    cnt = c.domain1.nnodes
                    for i in range(cnt):
                        work_node.append(off + i)
                        work_rigid_idx.append(rigid_idx)
                        work_penalty.append(pen)
                        work_friction.append(fric)
                        work_is_mesh.append(is_mesh)
                        work_fem_did.append(fem_gid)
                        work_rigid_did.append(rigid_gid)
                        work_pair.append(pair_idx)
                        work_tied.append(tied)
                else:
                    off = int(bn_off[d_idx])
                    cnt = int(bn_cnt[d_idx])
                    for i in range(cnt):
                        nd = int(bnodes[off + i])
                        work_node.append(nd)
                        work_normals.append([float(bnormals[nd][0]), float(bnormals[nd][1])])
                        work_rigid_idx.append(rigid_idx)
                        work_penalty.append(pen)
                        work_friction.append(fric)
                        work_is_mesh.append(is_mesh)
                        work_fem_did.append(fem_gid)
                        work_rigid_did.append(rigid_gid)
                        work_pair.append(pair_idx)
                        work_tied.append(tied)

            total = len(work_node)
            if total > 0:
                self._bc_rigid_node = _from_numpy_i32(work_node)
                self._bc_rigid_idx = _from_numpy_i32(work_rigid_idx)
                self._bc_rigid_penalty = _from_numpy_f32(work_penalty)
                self._bc_rigid_friction = _from_numpy_f32(work_friction)
                self._bc_rigid_is_mesh = _from_numpy_i32(work_is_mesh)
                self._bc_rigid_fem_did = _from_numpy_i32(work_fem_did)
                self._bc_rigid_did = _from_numpy_i32(work_rigid_did)
                self._bc_rigid_pair = _from_numpy_i32(work_pair)
                self._bc_rigid_tied = _from_numpy_i32(work_tied)

                self._rigid_pair_domain_ids = [
                    (int(pair_ids_np[orig_idx][0]), int(pair_ids_np[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(rigid_contacts)
                ]
                self._bc_rigid_pair_a, self._bc_rigid_pair_b, self._bc_rigid_active = (
                    self._create_pair_activation_fields(self._rigid_pair_domain_ids)
                )
                self._batched_rigid_count = total

                mesh_indices = [i for i, m in enumerate(work_is_mesh) if m == 1]
                prim_indices = [i for i, m in enumerate(work_is_mesh) if m == 0]
                self._batched_rigid_mesh_count = len(mesh_indices)
                self._batched_rigid_prim_count = len(prim_indices)
                self._tied_rigid_mesh_item_count = sum(1 for i in mesh_indices if work_tied[i] == 1)
                self._nontied_rigid_mesh_item_count = len(mesh_indices) - self._tied_rigid_mesh_item_count

                if mesh_indices:
                    n_mesh = len(mesh_indices)
                    self._bc_rigid_mesh_idx = _from_numpy_i32(mesh_indices)
                    # normals: keep Taichi-era shape (may truncate / pad)
                    normals_np = np.zeros((n_mesh, 2), dtype=np.float32)
                    if work_normals:
                        src = np.asarray(work_normals, dtype=np.float32)
                        n_copy = min(len(src), n_mesh)
                        normals_np[:n_copy] = src[:n_copy]
                    self._bc_rigid_normals = _from_numpy_vec2(normals_np)
                    self._bc_rigid_mesh_active_idx = _wp_int(n_mesh)
                    self._bc_rigid_mesh_active_count = wp.zeros(1, dtype=int)
                    _assign_scalar(self._bc_rigid_mesh_active_count, n_mesh)
                    self._bc_rigid_node_resolved = _wp_int(self.femSpringManager.MAX_NODES)
                    self._bc_rigid_cache_elem = _wp_vec2i(n_mesh)
                    self._bc_rigid_cache_elem.fill_(-1)
                    self._bc_rigid_fric_prev_elem = _wp_int(n_mesh)
                    self._bc_rigid_fric_prev_valid = _wp_int(n_mesh)
                    self._bc_rigid_fric_prev_force = _wp_vec2(n_mesh)
                    self._bc_rigid_fric_prev_weights = _wp_vec2(n_mesh)
                    self._bc_rigid_fric_prev_penetration = _wp_float(n_mesh)
                    self._bc_rigid_fric_prev_elem.fill_(-1)
                    self._bc_rigid_fric_prev_valid.fill_(0)
                    self._bc_rigid_fric_prev_force.fill_(0.0)
                    self._bc_rigid_fric_prev_weights.fill_(0.0)
                    self._bc_rigid_fric_prev_penetration.fill_(0.0)
                    self._bc_rigid_tie_resolved = _wp_int(n_mesh)
                    self._bc_rigid_tie_resolved.fill_(0)
                    self._bc_rigid_tie_weights = _wp_vec2(n_mesh)
                    self._bc_rigid_tie_weights.fill_(0.0)
                    self._fixed_rigid_mesh_tie_enabled = wp.zeros(1, dtype=int)
                    _assign_scalar(self._fixed_rigid_mesh_tie_enabled, 1)
                    self._fixed_rigid_mesh_tie_initialized = wp.zeros(1, dtype=int)
                    self._fixed_rigid_mesh_tie_found_count = wp.zeros(1, dtype=int)
                    self._bc_rigid_skip_epoch = _wp_int(n_mesh)
                    self._bc_rigid_skip_epoch.fill_(-1)

                    mesh_nodes_unique = sorted(set(work_node[i] for i in mesh_indices))
                    mesh_rigid_ids = sorted(set(work_rigid_did[i] for i in mesh_indices))
                    rigid_margin = [1e-4] * len(mesh_rigid_ids)

                    self._bc_rigid_mesh_nodes = _from_numpy_i32(mesh_nodes_unique)
                    self._bc_rigid_mesh_node_count = wp.zeros(1, dtype=int)
                    _assign_scalar(self._bc_rigid_mesh_node_count, len(mesh_nodes_unique))
                    self._bc_rigid_mesh_rigid_ids = _from_numpy_i32(mesh_rigid_ids)
                    self._bc_rigid_mesh_rigid_margin = _from_numpy_f32(rigid_margin)
                    self._bc_rigid_mesh_rigid_count = wp.zeros(1, dtype=int)
                    _assign_scalar(self._bc_rigid_mesh_rigid_count, len(mesh_rigid_ids))

                    use_node_mask = 0
                    self._bc_rigid_node_mask = _wp_int(self.femSpringManager.MAX_NODES)
                    self._bc_rigid_node_near_any = _wp_int(self.femSpringManager.MAX_NODES)
                    self._bc_rigid_use_node_mask = wp.zeros(1, dtype=int)
                    _assign_scalar(self._bc_rigid_use_node_mask, use_node_mask)

                if prim_indices:
                    self._bc_rigid_prim_idx = _from_numpy_i32(prim_indices)
                    if not hasattr(self, "_bc_rigid_node_resolved"):
                        self._bc_rigid_node_resolved = _wp_int(self.femSpringManager.MAX_NODES)

                has_mesh = self._batched_rigid_mesh_count > 0
                self._use_rigid_spatial_hash = (
                    has_mesh and self.rigidManager is not None and self.rigidManager.spatialHash is not None
                )
                self._rigid_sh_contact_margin = 0.0
                self._fem_contact_aabb_lb = wp.zeros(1, dtype=wp.vec2)
                self._fem_contact_aabb_ub = wp.zeros(1, dtype=wp.vec2)
                print(
                    f"  Batched Rigid: {len(rigid_contacts)} pairs → split kernels "
                    f"({self._batched_rigid_mesh_count} mesh + {self._batched_rigid_prim_count} prim items, "
                    f"spatial_hash={'ON' if self._use_rigid_spatial_hash else 'OFF'})"
                )

        # ── HeightField ──
        self._batched_hf_count = 0
        if hf_contacts and self.femSpringManager is not None:
            hf_domains = {}
            for c, idx in hf_contacts:
                key = id(c.domain2)
                if key not in hf_domains:
                    hf_domains[key] = c.domain2

            if len(hf_domains) <= 1:
                self._hf_domain = list(hf_domains.values())[0]
                work_node = []
                work_penalty = []
                work_dom_did = []
                work_pair = []

                self._hf_pair_domain_ids = [
                    (int(pair_ids_np[orig_idx][0]), int(pair_ids_np[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(hf_contacts)
                ]

                fsm = self.femSpringManager
                node_off = _host_np(fsm.domainNodeOffset)
                bn_off = _host_np(fsm.domainBoundaryNodeOffset)
                bn_cnt = _host_np(fsm.domainBoundaryNodeCount)
                bnodes = _host_np(fsm.boundaryNodes)

                for pair_idx, (c, _) in enumerate(hf_contacts):
                    is_spring = isinstance(c, ContactSpringHeightField)
                    d_idx = c.domain1.domainIdx
                    pen = float(c.penalty)
                    if is_spring:
                        off = int(node_off[d_idx])
                        cnt = c.domain1.nnodes
                        for i in range(cnt):
                            work_node.append(off + i)
                            work_penalty.append(pen)
                            work_dom_did.append(d_idx)
                            work_pair.append(pair_idx)
                    else:
                        off = int(bn_off[d_idx])
                        cnt = int(bn_cnt[d_idx])
                        for i in range(cnt):
                            work_node.append(int(bnodes[off + i]))
                            work_penalty.append(pen)
                            work_dom_did.append(d_idx)
                            work_pair.append(pair_idx)

                total = len(work_node)
                if total > 0:
                    self._bc_hf_node = _from_numpy_i32(work_node)
                    self._bc_hf_penalty = _from_numpy_f32(work_penalty)
                    self._bc_hf_dom_did = _from_numpy_i32(work_dom_did)
                    self._bc_hf_pair = _from_numpy_i32(work_pair)
                    self._bc_hf_pair_a, self._bc_hf_pair_b, self._bc_hf_active = self._create_pair_activation_fields(
                        self._hf_pair_domain_ids
                    )
                    # Upload heightfield sample data for device kernels
                    self._hf_height = _from_numpy_f32(self._hf_domain.height)
                    self._hf_nx = int(self._hf_domain.nx)
                    self._hf_lb_x = float(self._hf_domain.lb[0])
                    self._hf_ub_x = float(self._hf_domain.ub[0])
                    self._hf_reverse = 1 if self._hf_domain.reverse else 0
                    self._batched_hf_count = total
                    print(f"  Batched HeightField: {len(hf_contacts)} pairs → 1 kernel ({total} work items)")
            else:
                for c, idx in hf_contacts:
                    unbatched.append(c)
                    unbatched_pair_ids.append((int(pair_ids_np[idx][0]), int(pair_ids_np[idx][1])))
                print(f"  HeightField: {len(hf_domains)} unique domains, keeping {len(hf_contacts)} unbatched")

        # ── VoxelMap ──
        self._batched_voxel_count = 0
        if voxel_contacts and self.femSpringManager is not None:
            voxel_domains = {}
            for c, idx in voxel_contacts:
                key = id(c.domain2)
                if key not in voxel_domains:
                    voxel_domains[key] = c.domain2

            if len(voxel_domains) <= 1:
                self._voxel_domain = list(voxel_domains.values())[0]
                work_node = []
                work_penalty = []
                work_dom_did = []
                work_pair = []

                self._voxel_pair_domain_ids = [
                    (int(pair_ids_np[orig_idx][0]), int(pair_ids_np[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(voxel_contacts)
                ]

                fsm = self.femSpringManager
                bn_off = _host_np(fsm.domainBoundaryNodeOffset)
                bn_cnt = _host_np(fsm.domainBoundaryNodeCount)
                bnodes = _host_np(fsm.boundaryNodes)

                for pair_idx, (c, _) in enumerate(voxel_contacts):
                    d_idx = c.domain1.domainIdx
                    pen = float(c.penalty)
                    off = int(bn_off[d_idx])
                    cnt = int(bn_cnt[d_idx])
                    for i in range(cnt):
                        work_node.append(int(bnodes[off + i]))
                        work_penalty.append(pen)
                        work_dom_did.append(d_idx)
                        work_pair.append(pair_idx)

                total = len(work_node)
                if total > 0:
                    self._bc_voxel_node = _from_numpy_i32(work_node)
                    self._bc_voxel_penalty = _from_numpy_f32(work_penalty)
                    self._bc_voxel_dom_did = _from_numpy_i32(work_dom_did)
                    self._bc_voxel_pair = _from_numpy_i32(work_pair)
                    self._bc_voxel_pair_a, self._bc_voxel_pair_b, self._bc_voxel_active = (
                        self._create_pair_activation_fields(self._voxel_pair_domain_ids)
                    )
                    vd = self._voxel_domain
                    self._voxel_occ = wp.array(np.asarray(vd.occ, dtype=np.int32), dtype=int)
                    self._voxel_nx = int(vd.nx)
                    self._voxel_ny = int(vd.ny)
                    self._voxel_lb = wp.vec2(float(vd.lb[0]), float(vd.lb[1]))
                    self._voxel_dx = float(vd.dx)
                    self._voxel_dz = float(vd.dz)
                    self._batched_voxel_count = total
                    print(f"  Batched VoxelMap: {len(voxel_contacts)} pairs → 1 kernel ({total} work items)")
            else:
                for c, idx in voxel_contacts:
                    unbatched.append(c)
                    unbatched_pair_ids.append((int(pair_ids_np[idx][0]), int(pair_ids_np[idx][1])))
                print(f"  VoxelMap: {len(voxel_domains)} unique domains, keeping {len(voxel_contacts)} unbatched")

        self._unbatched_contacts = unbatched
        self._unbatched_contacts_count = len(unbatched)
        self._unbatched_pair_ids = unbatched_pair_ids
        if self._unbatched_contacts_count > 0:
            self._bc_unbatched_pair_a, self._bc_unbatched_pair_b, self._bc_unbatched_active = (
                self._create_pair_activation_fields(self._unbatched_pair_ids)
            )

    # ── Pair activation helpers ────────────────────────────────────

    def _create_pair_activation_fields(self, pair_domain_ids):
        """Create Warp arrays (pair_a, pair_b, active) for BVH pair activation."""
        n = len(pair_domain_ids)
        pair_a = _from_numpy_i32([p[0] for p in pair_domain_ids])
        pair_b = _from_numpy_i32([p[1] for p in pair_domain_ids])
        active = _wp_int(n)
        active.fill_(0)
        return pair_a, pair_b, active

    def _activate_or_fill_pairs(self, pairs_field, num_pairs_work, pair_a, pair_b, active):
        """Activate/deactivate contact pairs based on BVH collision results."""
        n = active.shape[0]
        if not self.skip_bvh:
            if num_pairs_work > 0:
                self._launch_activate_pairs(pairs_field, num_pairs_work, pair_a, pair_b, active, n)
            else:
                active.fill_(0)
        else:
            active.fill_(1)

    # ── Launch wrappers (public names used by ExplicitLoop) ────────

    def _activate_pairs_by_aabb_kernel(self, pair_a, pair_b, active, num_contact_pairs):
        self._launch_activate_pairs_by_aabb(pair_a, pair_b, active, num_contact_pairs)

    def _launch_activate_pairs_by_aabb(self, pair_a, pair_b, active, num_contact_pairs):
        wp.launch(
            _activate_pairs_by_aabb_kernel,
            dim=num_contact_pairs,
            inputs=[pair_a, pair_b, active, int(num_contact_pairs), self.aabb, int(self._max_domains)],
        )

    def _activate_pairs_kernel(self, pairs, num_pairs, pair_a, pair_b, active, num_contact_pairs):
        self._launch_activate_pairs(pairs, num_pairs, pair_a, pair_b, active, num_contact_pairs)

    def _launch_activate_pairs(self, pairs, num_pairs, pair_a, pair_b, active, num_contact_pairs):
        wp.launch(
            _activate_pairs_kernel,
            dim=num_contact_pairs,
            inputs=[pairs, int(num_pairs), pair_a, pair_b, active, int(num_contact_pairs)],
        )

    def _build_active_rigid_mesh_workset(self, total_mesh_items):
        self._launch_build_active_rigid_mesh_workset(total_mesh_items)

    def _launch_build_active_rigid_mesh_workset(self, total_mesh_items):
        rm = self.rigidManager
        fsm = self.femSpringManager
        wp.launch(_reset_rigid_mesh_active_count_kernel, dim=1, inputs=[self._bc_rigid_mesh_active_count])
        wp.launch(
            _build_active_rigid_mesh_workset_kernel,
            dim=total_mesh_items,
            inputs=[
                int(total_mesh_items),
                self._bc_rigid_mesh_idx,
                self._bc_rigid_node,
                self._bc_rigid_tied,
                self._bc_rigid_pair,
                self._bc_rigid_active,
                self._bc_rigid_did,
                self._bc_rigid_skip_epoch,
                self._bc_rigid_node_near_any,
                self._bc_rigid_node_mask,
                self._bc_rigid_use_node_mask,
                self._bc_rigid_mesh_active_count,
                self._bc_rigid_mesh_active_idx,
                self._bc_rigid_fric_prev_elem,
                self._bc_rigid_fric_prev_valid,
                self._bc_rigid_fric_prev_weights,
                self._bc_rigid_fric_prev_force,
                self._bc_rigid_fric_prev_penetration,
                rm.rigidDomainIds,
                fsm.coords,
                self.aabb,
            ],
        )

    def _compute_fem_contact_aabb(self, num_nodes):
        """Compute AABB of FEM nodes in rigid mesh contact.

        ``num_nodes`` may be a Python int or length-1 array scalar (host).
        """
        n = int(num_nodes) if not hasattr(num_nodes, "numpy") else int(_host_np(num_nodes)[0])
        fsm = self.femSpringManager
        wp.launch(
            _init_fem_contact_aabb_kernel,
            dim=1,
            inputs=[self._fem_contact_aabb_lb, self._fem_contact_aabb_ub],
        )
        if n > 0:
            wp.launch(
                _compute_fem_contact_aabb_kernel,
                dim=n,
                inputs=[
                    n,
                    self._bc_rigid_mesh_nodes,
                    fsm.coords,
                    self._fem_contact_aabb_lb,
                    self._fem_contact_aabb_ub,
                ],
            )

    def _prefilter_mesh_node_activity(self):
        self._launch_prefilter_mesh_node_activity()

    def _launch_prefilter_mesh_node_activity(self):
        rm = self.rigidManager
        fsm = self.femSpringManager
        n = int(_host_np(self._bc_rigid_mesh_node_count)[0])
        if n <= 0:
            return
        wp.launch(
            _prefilter_mesh_node_activity_kernel,
            dim=n,
            inputs=[
                self._bc_rigid_mesh_node_count,
                self._bc_rigid_mesh_nodes,
                self._bc_rigid_mesh_rigid_count,
                self._bc_rigid_mesh_rigid_ids,
                self._bc_rigid_mesh_rigid_margin,
                self._bc_rigid_node_near_any,
                fsm.coords,
                rm.rigidDomainIds,
                self.aabb,
            ],
        )

    def _batched_analytical_contact_kernel(self, total):
        self._launch_batched_analytical_contact(total)

    def _launch_batched_analytical_contact(self, total):
        fsm = self.femSpringManager
        wp.launch(
            _batched_analytical_contact_kernel,
            dim=total,
            inputs=[
                int(total),
                self._bc_anal_pair,
                self._bc_anal_active,
                self._bc_anal_node,
                self._bc_anal_pp,
                self._bc_anal_pn,
                self._bc_anal_penalty,
                fsm.coords,
                fsm.Fext,
            ],
        )

    def _initialize_fem_fem_ties_once_kernel(self, total):
        self._launch_initialize_fem_fem_ties_once(total)

    def _launch_initialize_fem_fem_ties_once(self, total):
        fsm = self.femSpringManager
        sh = fsm.spatialHash
        wp.launch(
            _initialize_fem_fem_ties_once_kernel,
            dim=total,
            inputs=[
                int(total),
                self._bc_flex_tied,
                self._bc_flex_tie_resolved,
                self._bc_flex_node,
                self._bc_flex_be_off,
                self._bc_flex_be_cnt,
                self._bc_flex_target_did,
                self._bc_flex_tie_elem,
                self._bc_flex_tie_weights,
                self._bc_flex_tie_gap,
                self._fixed_flexflex_tie_found_count,
                fsm.coords,
                fsm.boundaryElements,
                int(sh.MAX_QUERY),
                sh.total_cells,
                sh.globalbbox,
                sh.gridSize,
                sh.cellNumbers,
                sh.cellStart,
                sh.cellEnd,
                sh._sortedElemIdx,
                sh.domainIds,
                sh.queryElids,
            ],
        )

    def _apply_fem_fem_ties_fixed_kernel(self, total):
        self._launch_apply_fem_fem_ties_fixed(total)

    def _launch_apply_fem_fem_ties_fixed(self, total):
        fsm = self.femSpringManager
        wp.launch(
            _apply_fem_fem_ties_fixed_kernel,
            dim=total,
            inputs=[
                int(total),
                self._bc_flex_tied,
                self._bc_flex_tie_resolved,
                self._bc_flex_node,
                self._bc_flex_tie_elem,
                self._bc_flex_tie_weights,
                self._bc_flex_tie_gap,
                self._bc_flex_penalty,
                fsm.coords,
                fsm.boundaryElements,
                fsm.Fext,
            ],
        )

    def _batched_flexflex_contact_kernel(self, total):
        self._launch_batched_flexflex_contact(total)

    def _launch_batched_flexflex_contact(self, total):
        fsm = self.femSpringManager
        wp.launch(
            _batched_flexflex_contact_kernel,
            dim=total,
            inputs=[
                int(total),
                self._bc_flex_pair,
                self._bc_flex_active,
                self._bc_flex_target_did,
                self._bc_flex_node,
                self._bc_flex_penalty,
                self._bc_flex_be_off,
                self._bc_flex_be_cnt,
                self._bc_flex_friction,
                self._use_bvh_domain_mask,
                self._bvh_domain_stamp,
                self._bvh_active_stamp,
                fsm.coords,
                fsm.V,
                fsm.boundaryElements,
                fsm.Fext,
            ],
        )

    def _batched_flexflex_contact_kernel_sh(self, total):
        self._launch_batched_flexflex_contact_sh(total)

    def _launch_batched_flexflex_contact_sh(self, total):
        fsm = self.femSpringManager
        sh = fsm.spatialHash
        wp.launch(
            _batched_flexflex_contact_kernel_sh,
            dim=total,
            inputs=[
                int(total),
                self._bc_flex_tied,
                self._bc_flex_pair,
                self._bc_flex_active,
                self._bc_flex_target_did,
                self._bc_flex_node,
                self._bc_flex_penalty,
                self._bc_flex_be_off,
                self._bc_flex_be_cnt,
                self._bc_flex_friction,
                self._bc_flex_cache_elem,
                self._use_bvh_domain_mask,
                self._bvh_domain_stamp,
                self._bvh_active_stamp,
                fsm.coords,
                fsm.V,
                fsm.boundaryElements,
                fsm.Fext,
                int(sh.MAX_QUERY),
                sh.total_cells,
                sh.globalbbox,
                sh.gridSize,
                sh.cellNumbers,
                sh.cellStart,
                sh.cellEnd,
                sh._sortedElemIdx,
                sh.domainIds,
                sh.queryElids,
            ],
        )

    def _batched_rigid_prim_kernel(self, num_prim):
        self._launch_batched_rigid_prim(num_prim)

    def _launch_batched_rigid_prim(self, num_prim):
        fsm = self.femSpringManager
        rm = self.rigidManager
        wp.launch(
            _batched_rigid_prim_kernel,
            dim=num_prim,
            inputs=[
                int(num_prim),
                self._bc_rigid_prim_idx,
                self._bc_rigid_pair,
                self._bc_rigid_active,
                self._bc_rigid_node,
                self._bc_rigid_idx,
                self._bc_rigid_tied,
                self._bc_rigid_penalty,
                self._bc_rigid_friction,
                fsm.coords,
                fsm.V,
                fsm.Fext,
                self.aabb,
                rm.rigidDomainIds,
                rm.rigidParams,
                rm.cached_rotation_matrix,
                rm.radius,
                rm.V,
                rm.RotV,
                rm.accumulated_impulse,
                rm.accumulated_rotational_impulse,
                float(self.dt),
            ],
        )

    def _batched_rigid_mesh_kernel_sh(self, run_full):
        self._launch_batched_rigid_mesh_sh(run_full)

    def _launch_batched_rigid_mesh_sh(self, run_full):
        fsm = self.femSpringManager
        rm = self.rigidManager
        sh = rm.spatialHash
        wp.launch(
            _batched_rigid_mesh_kernel_sh,
            dim=self._batched_rigid_mesh_count,
            inputs=[
                int(self._batched_rigid_mesh_count),
                int(run_full),
                self._bc_rigid_mesh_active_count,
                self._bc_rigid_mesh_active_idx,
                self._bc_rigid_mesh_idx,
                self._bc_rigid_node,
                self._bc_rigid_penalty,
                self._bc_rigid_friction,
                self._bc_rigid_did,
                self._bc_rigid_tied,
                self._bc_rigid_cache_elem,
                self._bc_rigid_skip_epoch,
                self._bc_rigid_fric_prev_elem,
                self._bc_rigid_fric_prev_valid,
                self._bc_rigid_fric_prev_weights,
                self._bc_rigid_fric_prev_force,
                self._bc_rigid_fric_prev_penetration,
                fsm.coords,
                fsm.V,
                fsm.Fext,
                rm.rigidParams,
                rm.accumulated_impulse,
                rm.accumulated_rotational_impulse,
                rm.meshBoundaryElements,
                rm.meshBoundaryCoords,
                rm.meshElemLB,
                rm.meshElemUB,
                rm.meshElemMarginBase,
                sh.total_cells,
                sh.globalbbox,
                sh.gridSize,
                sh.cellNumbers,
                sh.cellStart,
                sh.cellEnd,
                sh._sortedElemIdx,
                sh.domainIds,
                float(self.dt),
            ],
        )

    def _initialize_fem_rigid_mesh_ties_once_kernel(self, total_mesh_items):
        self._launch_initialize_fem_rigid_mesh_ties(total_mesh_items)

    def _launch_initialize_fem_rigid_mesh_ties(self, total_mesh_items):
        fsm = self.femSpringManager
        rm = self.rigidManager
        sh = rm.spatialHash
        wp.launch(
            _initialize_fem_rigid_mesh_ties_once_kernel,
            dim=total_mesh_items,
            inputs=[
                int(total_mesh_items),
                self._bc_rigid_mesh_idx,
                self._bc_rigid_tied,
                self._bc_rigid_tie_resolved,
                self._bc_rigid_node,
                self._bc_rigid_did,
                self._bc_rigid_cache_elem,
                self._bc_rigid_tie_weights,
                self._fixed_rigid_mesh_tie_found_count,
                fsm.coords,
                rm.meshBoundaryElements,
                rm.meshBoundaryCoords,
                rm.meshElemMarginBase,
                sh.total_cells,
                sh.globalbbox,
                sh.gridSize,
                sh.cellNumbers,
                sh.cellStart,
                sh.cellEnd,
                sh._sortedElemIdx,
                sh.domainIds,
            ],
        )

    def _apply_fem_rigid_mesh_ties_fixed_kernel(self, total_mesh_items):
        self._launch_apply_fem_rigid_mesh_ties(total_mesh_items)

    def _launch_apply_fem_rigid_mesh_ties(self, total_mesh_items):
        fsm = self.femSpringManager
        rm = self.rigidManager
        wp.launch(
            _apply_fem_rigid_mesh_ties_fixed_kernel,
            dim=total_mesh_items,
            inputs=[
                int(total_mesh_items),
                self._bc_rigid_mesh_idx,
                self._bc_rigid_tied,
                self._bc_rigid_tie_resolved,
                self._bc_rigid_node,
                self._bc_rigid_penalty,
                self._bc_rigid_cache_elem,
                self._bc_rigid_tie_weights,
                fsm.coords,
                fsm.Fext,
                rm.rigidParams,
                rm.accumulated_impulse,
                rm.accumulated_rotational_impulse,
                rm.meshBoundaryElements,
                rm.meshBoundaryCoords,
                float(self.dt),
            ],
        )

    def _batched_heightfield_contact_kernel(self, total, dt):
        self._launch_batched_heightfield(total, dt)

    def _launch_batched_heightfield(self, total, dt):
        fsm = self.femSpringManager
        wp.launch(
            _batched_heightfield_contact_kernel,
            dim=total,
            inputs=[
                int(total),
                float(dt),
                self._bc_hf_pair,
                self._bc_hf_active,
                self._bc_hf_node,
                self._bc_hf_penalty,
                fsm.coords,
                fsm.V,
                fsm.Fext,
                self._hf_height,
                int(self._hf_nx),
                float(self._hf_lb_x),
                float(self._hf_ub_x),
                int(self._hf_reverse),
            ],
        )

    def _batched_voxelmap_contact_kernel(self, total, dt):
        self._launch_batched_voxelmap(total, dt)

    def _launch_batched_voxelmap(self, total, dt):
        fsm = self.femSpringManager
        wp.launch(
            _batched_voxelmap_contact_kernel,
            dim=total,
            inputs=[
                int(total),
                float(dt),
                self._bc_voxel_pair,
                self._bc_voxel_active,
                self._bc_voxel_node,
                self._bc_voxel_penalty,
                fsm.coords,
                fsm.V,
                fsm.Fext,
                self._voxel_occ,
                int(self._voxel_nx),
                int(self._voxel_ny),
                self._voxel_lb,
                float(self._voxel_dx),
                float(self._voxel_dz),
            ],
        )
