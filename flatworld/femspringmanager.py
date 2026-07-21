"""
FemSpringManager - Unified manager for FEM and SpringMass domains (NVIDIA Warp).

Batches all FEM/Spring computations into GPU kernels for massive speedup
(expected 5-10x improvement over per-domain loops).

Key optimizations:
1. Unified data layout: all nodes/elements from all domains in single Warp arrays
2. Batched force computation: single kernel processes all domains
3. Batched velocity/position update: single kernel for all nodes
4. Reduced CPU-GPU synchronization overhead

Author: Dongyu Liu
Date: 2025-12-19
"""

from __future__ import annotations

import numpy as np
import warp as wp

from definitions import *
from materials import getStress2D, misesReturnMap2D
from spatialmanager import SpatialHashManager, add_element, set_bounds
from wp_init import ensure_warp

# Local matrix/vector types (wp.mat36 may be unavailable)
_mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)
_vec6 = wp.types.vector(length=6, dtype=wp.float32)


def _assign_scalar(arr: wp.array, value):
    """Host write to a length-1 Warp array (no ``arr[0] =`` on Warp 1.14+)."""
    np_arr = arr.numpy()
    np_arr[0] = value
    arr.assign(np_arr)


def _patch_array(arr: wp.array, index: int, value):
    """Host write a single element via numpy round-trip."""
    np_arr = arr.numpy()
    np_arr[index] = value
    arr.assign(np_arr)


def _patch_slice(arr: wp.array, start: int, values: np.ndarray):
    """Host write a contiguous slice via numpy round-trip."""
    np_arr = arr.numpy()
    n = len(values)
    np_arr[start : start + n] = values
    arr.assign(np_arr)


# ---------------------------------------------------------------------------
# Device helpers (2D triangle FEM)
# ---------------------------------------------------------------------------


@wp.func
def get_jacobian_2d(c1: wp.vec2, c2: wp.vec2, c3: wp.vec2):
    """Compute the 2x2 Jacobian for a linear triangle (nodes c1,c2,c3)."""
    return wp.mat22(c1[0] - c3[0], c2[0] - c3[0], c1[1] - c3[1], c2[1] - c3[1])


@wp.func
def get_weights_2d(jac: wp.mat22):
    """Return triangle area weight = 0.5 * |det(J)|."""
    return wp.abs(jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]) * 0.5


@wp.func
def get_b_matrix_2d(F: wp.mat22, J_inv: wp.mat22):
    """Assemble the 3x6 B-matrix for linear triangular elasticity."""
    # shapeDparam^T @ J_inv  →  3x2 (dN/dx)
    # shapeDparam = [[1,0,-1],[0,1,-1]]
    dndx00 = J_inv[0, 0]
    dndx01 = J_inv[0, 1]
    dndx10 = J_inv[1, 0]
    dndx11 = J_inv[1, 1]
    dndx20 = -J_inv[0, 0] - J_inv[1, 0]
    dndx21 = -J_inv[0, 1] - J_inv[1, 1]

    BMat = _mat36(0.0)
    # node 0
    BMat[0, 0] = dndx00 * F[0, 0]
    BMat[0, 1] = dndx00 * F[1, 0]
    BMat[1, 0] = dndx01 * F[0, 1]
    BMat[1, 1] = dndx01 * F[1, 1]
    BMat[2, 0] = dndx01 * F[0, 0] + dndx00 * F[0, 1]
    BMat[2, 1] = dndx00 * F[1, 1] + dndx01 * F[1, 0]
    # node 1
    BMat[0, 2] = dndx10 * F[0, 0]
    BMat[0, 3] = dndx10 * F[1, 0]
    BMat[1, 2] = dndx11 * F[0, 1]
    BMat[1, 3] = dndx11 * F[1, 1]
    BMat[2, 2] = dndx11 * F[0, 0] + dndx10 * F[0, 1]
    BMat[2, 3] = dndx10 * F[1, 1] + dndx11 * F[1, 0]
    # node 2
    BMat[0, 4] = dndx20 * F[0, 0]
    BMat[0, 5] = dndx20 * F[1, 0]
    BMat[1, 4] = dndx21 * F[0, 1]
    BMat[1, 5] = dndx21 * F[1, 1]
    BMat[2, 4] = dndx21 * F[0, 0] + dndx20 * F[0, 1]
    BMat[2, 5] = dndx20 * F[1, 1] + dndx21 * F[1, 0]
    return BMat


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _prepare_fem_data_kernel(
    total_fem_elements: wp.array(dtype=int),
    connectivity: wp.array(dtype=wp.vec4i),
    coords: wp.array(dtype=wp.vec2),
    mat_rho: wp.array(dtype=float),
    elem_volume: wp.array(dtype=float),
    J_inv: wp.array(dtype=wp.mat22),
    mass: wp.array(dtype=float),
):
    i = wp.tid()
    n_fem = total_fem_elements[0]
    if i >= n_fem:
        return
    conn = connectivity[i]
    n1 = conn[0]
    n2 = conn[1]
    n3 = conn[2]
    Jac = get_jacobian_2d(coords[n1], coords[n2], coords[n3])
    vol = get_weights_2d(Jac)
    elem_volume[i] = vol
    J_inv[i] = wp.inverse(Jac)
    nd_mass = vol * mat_rho[i] / 3.0
    wp.atomic_add(mass, n1, nd_mass)
    wp.atomic_add(mass, n2, nd_mass)
    wp.atomic_add(mass, n3, nd_mass)


@wp.kernel
def _mass_scaling_kernel(
    dt_target: float,
    total_fem_elements: wp.array(dtype=int),
    total_elements: wp.array(dtype=int),
    connectivity: wp.array(dtype=wp.vec4i),
    coords: wp.array(dtype=wp.vec2),
    mat_e: wp.array(dtype=float),
    mat_rho: wp.array(dtype=float),
    elem_volume: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    added_mass_out: wp.array(dtype=float),
):
    """Add mass to under-resolved FEM elements and springs (one thread)."""
    tid = wp.tid()
    if tid > 0:
        return

    added_mass = float(0.0)
    safety = float(0.4)
    n_fem = total_fem_elements[0]
    n_tot = total_elements[0]

    for i in range(n_fem):
        conn = connectivity[i]
        n1 = conn[0]
        n2 = conn[1]
        n3 = conn[2]
        E_val = mat_e[i]
        rho_val = mat_rho[i]
        V_e = elem_volume[i]

        e01 = wp.length(coords[n2] - coords[n1])
        e12 = wp.length(coords[n3] - coords[n2])
        e20 = wp.length(coords[n1] - coords[n3])
        L_e = wp.min(e01, wp.min(e12, e20))
        n_nodes = float(3.0)

        c_e = wp.sqrt(E_val / rho_val)
        dt_elem = safety * L_e / c_e

        if dt_elem < dt_target and L_e > 1e-15:
            rho_new = E_val * (dt_target / (safety * L_e)) * (dt_target / (safety * L_e))
            dm = (rho_new - rho_val) * V_e / n_nodes
            mass[n1] = mass[n1] + dm
            mass[n2] = mass[n2] + dm
            mass[n3] = mass[n3] + dm
            added_mass = added_mass + (rho_new - rho_val) * V_e

    for i in range(n_fem, n_tot):
        conn = connectivity[i]
        k_spring = mat_e[i]
        m_min = float(4.0) * k_spring * dt_target * dt_target
        for nd in range(2):
            nid = conn[nd]
            if mass[nid] < m_min:
                dm = m_min - mass[nid]
                mass[nid] = m_min
                added_mass = added_mass + dm

    added_mass_out[0] = added_mass


@wp.kernel
def _total_mass_kernel(
    total_nodes: wp.array(dtype=int),
    mass: wp.array(dtype=float),
    total_out: wp.array(dtype=float),
):
    i = wp.tid()
    n = total_nodes[0]
    if i >= n:
        return
    wp.atomic_add(total_out, 0, mass[i])


@wp.kernel
def _fem_internal_force_kernel(
    total_fem_elements: wp.array(dtype=int),
    connectivity: wp.array(dtype=wp.vec4i),
    coords: wp.array(dtype=wp.vec2),
    J_inv_arr: wp.array(dtype=wp.mat22),
    mat_type: wp.array(dtype=int),
    mat_e: wp.array(dtype=float),
    mat_nu: wp.array(dtype=float),
    mat_yield: wp.array(dtype=float),
    mat_h: wp.array(dtype=float),
    plastic_strain: wp.array(dtype=wp.vec3),
    eq_plastic_strain: wp.array(dtype=float),
    elem_volume: wp.array(dtype=float),
    elem_stress: wp.array(dtype=wp.vec3),
    elem_strain: wp.array(dtype=wp.vec3),
    Fext: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    n_fem = total_fem_elements[0]
    if i >= n_fem:
        return

    conn = connectivity[i]
    n1 = conn[0]
    n2 = conn[1]
    n3 = conn[2]

    Jac = get_jacobian_2d(coords[n1], coords[n2], coords[n3])
    J_inv = J_inv_arr[i]
    F = Jac @ J_inv
    BMat = get_b_matrix_2d(F, J_inv)

    sigma = wp.vec3(0.0, 0.0, 0.0)
    strain_voigt = wp.vec3(0.0, 0.0, 0.0)

    if mat_type[i] == MaterialType.MISES:
        sigma, eps_p_new, eqps_new, strain_voigt = misesReturnMap2D(
            mat_e[i],
            mat_nu[i],
            mat_yield[i],
            mat_h[i],
            F,
            plastic_strain[i],
            eq_plastic_strain[i],
        )
        plastic_strain[i] = eps_p_new
        eq_plastic_strain[i] = eqps_new
    else:
        sigma, strain_voigt = getStress2D(mat_type[i], mat_e[i], mat_nu[i], wp.vec2(0.0, 0.0), F)

    # PK2 → Cauchy
    S = wp.mat22(sigma[0], sigma[2], sigma[2], sigma[1])
    Jdet = wp.determinant(F)
    sigma_cauchy = (F @ S @ wp.transpose(F)) / wp.max(Jdet, float(1e-12))
    elem_stress[i] = wp.vec3(sigma_cauchy[0, 0], sigma_cauchy[1, 1], sigma_cauchy[0, 1])
    elem_strain[i] = strain_voigt

    el_fint = wp.transpose(BMat) @ sigma
    el_fint = el_fint * elem_volume[i]

    wp.atomic_add(Fext, n1, -wp.vec2(el_fint[0], el_fint[1]))
    wp.atomic_add(Fext, n2, -wp.vec2(el_fint[2], el_fint[3]))
    wp.atomic_add(Fext, n3, -wp.vec2(el_fint[4], el_fint[5]))


@wp.kernel
def _spring_force_kernel(
    total_fem_elements: wp.array(dtype=int),
    total_elements: wp.array(dtype=int),
    connectivity: wp.array(dtype=wp.vec4i),
    coords: wp.array(dtype=wp.vec2),
    V: wp.array(dtype=wp.vec2),
    mat_e: wp.array(dtype=float),
    mat_damping: wp.array(dtype=float),
    elem_volume: wp.array(dtype=float),
    Fext: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    n_fem = total_fem_elements[0]
    n_tot = total_elements[0]
    if i < n_fem or i >= n_tot:
        return

    conn = connectivity[i]
    ia = conn[0]
    ib = conn[1]
    xab = coords[ia] - coords[ib]
    lnew = wp.length(xab)
    dir_v = xab / lnew

    nd_force = mat_e[i] * (lnew / elem_volume[i] - 1.0) * dir_v
    vab = V[ia] - V[ib]
    nd_force = nd_force + mat_damping[i] * wp.dot(vab, dir_v) * dir_v
    wp.atomic_add(Fext, ia, -nd_force)
    wp.atomic_add(Fext, ib, nd_force)


@wp.kernel
def _integrate_nodes_kernel(
    dt: float,
    damping: float,
    total_nodes: wp.array(dtype=int),
    bc_nodes: wp.array(dtype=int),
    bc_values: wp.array(dtype=wp.vec2),
    mass: wp.array(dtype=float),
    A: wp.array(dtype=wp.vec2),
    V: wp.array(dtype=wp.vec2),
    U: wp.array(dtype=wp.vec2),
    coords: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    n = total_nodes[0]
    if i >= n:
        return

    bc = bc_nodes[i]
    f = Fext[i]

    if (bc & UTYPE) != 0:
        f = wp.vec2(0.0, 0.0)
    elif (bc & ATYPE) != 0:
        f = mass[i] * bc_values[i]
    elif (bc & GRAVITY) != 0:
        f = f + mass[i] * bc_values[i]
    elif (bc & FORCETYPE) != 0:
        f = f + bc_values[i]

    Fext[i] = f
    a = A[i] + (f - damping * V[i]) * (1.0 / mass[i])
    v = V[i] + a * dt

    if (bc & VTYPE) != 0:
        v = bc_values[i]
    elif (bc & UTYPE) != 0:
        v = wp.vec2(0.0, 0.0)

    V[i] = v
    du = v * dt
    U[i] = U[i] + du
    coords[i] = coords[i] + du


@wp.kernel
def _update_bbox_kernel(
    num_fem_domains: wp.array(dtype=int),
    num_spring_domains: wp.array(dtype=int),
    fem_domain_ids: wp.array(dtype=int),
    domain_boundary_node_offset: wp.array(dtype=int),
    domain_boundary_node_count: wp.array(dtype=int),
    boundary_nodes: wp.array(dtype=int),
    domain_node_offset: wp.array(dtype=int),
    domain_node_count: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec2),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
):
    """One thread: update FEM/spring domain AABBs into shared aabb array."""
    tid = wp.tid()
    if tid > 0:
        return

    n_fem = num_fem_domains[0]
    n_spring = num_spring_domains[0]

    for femidx in range(n_fem):
        lb = wp.vec2(1e30, 1e30)
        ub = wp.vec2(-1e30, -1e30)
        node_offset = domain_boundary_node_offset[femidx]
        node_count = domain_boundary_node_count[femidx]
        for I in range(node_offset, node_offset + node_count):
            nid = boundary_nodes[I]
            coord = coords[nid]
            lb = wp.min(lb, coord)
            ub = wp.max(ub, coord)
        domain_idx = fem_domain_ids[femidx]
        aabb[domain_idx, 0] = lb
        aabb[domain_idx, 1] = ub

    offset = n_fem
    for femidx in range(offset, offset + n_spring):
        lb = wp.vec2(1e30, 1e30)
        ub = wp.vec2(-1e30, -1e30)
        node_offset = domain_node_offset[femidx]
        node_count = domain_node_count[femidx]
        for I in range(node_offset, node_offset + node_count):
            coord = coords[I]
            lb = wp.min(lb, coord)
            ub = wp.max(ub, coord)
        domain_idx = fem_domain_ids[femidx]
        aabb[domain_idx, 0] = lb
        aabb[domain_idx, 1] = ub


@wp.kernel
def _clear_fext_kernel(
    total_nodes: wp.array(dtype=int),
    Fext: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    if i < total_nodes[0]:
        Fext[i] = wp.vec2(0.0, 0.0)


@wp.kernel
def _populate_fem_spatial_hash_kernel(
    velocity_buffer: float,
    num_fem_domains: wp.array(dtype=int),
    fem_domain_ids: wp.array(dtype=int),
    aabb: wp.array(dtype=wp.vec2, ndim=2),
    domain_boundary_elem_offset: wp.array(dtype=int),
    domain_boundary_element_count: wp.array(dtype=int),
    boundary_elements: wp.array(dtype=wp.vec3i),
    coords: wp.array(dtype=wp.vec2),
    max_elements: int,
    num_elements: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    element_bbox: wp.array(dtype=wp.vec2, ndim=2),
    estimate_size: wp.array(dtype=float),
    sum_size: wp.array(dtype=float),
    global_bbox: wp.array(dtype=wp.vec2),
):
    """Insert all FEM boundary elements into spatial hash (serial, one thread)."""
    tid = wp.tid()
    if tid > 0:
        return

    n_fem = num_fem_domains[0]
    global_lb = wp.vec2(1e9, 1e9)
    global_ub = wp.vec2(-1e9, -1e9)

    for fem_local_idx in range(n_fem):
        global_did = fem_domain_ids[fem_local_idx]
        global_lb = wp.min(global_lb, aabb[global_did, 0])
        global_ub = wp.max(global_ub, aabb[global_did, 1])

    extent = wp.max(global_ub[0] - global_lb[0], global_ub[1] - global_lb[1])
    expand = extent * 0.01 + velocity_buffer
    buf = wp.vec2(expand, expand)
    global_lb = global_lb - buf
    global_ub = global_ub + buf

    for fem_local_idx in range(n_fem):
        elem_offset = domain_boundary_elem_offset[fem_local_idx]
        elem_count = domain_boundary_element_count[fem_local_idx]
        for local_elem_idx in range(elem_count):
            global_elem_idx = elem_offset + local_elem_idx
            elem_conn = boundary_elements[global_elem_idx]
            lb = wp.vec2(1e30, 1e30)
            ub = wp.vec2(-1e30, -1e30)
            for k in range(3):
                node_id = elem_conn[k]
                if node_id >= 0:
                    coord = coords[node_id]
                    lb = wp.min(lb, coord)
                    ub = wp.max(ub, coord)
            add_element(
                lb,
                ub,
                fem_local_idx,
                global_elem_idx,
                velocity_buffer,
                max_elements,
                num_elements,
                domain_ids,
                element_bbox,
                estimate_size,
                sum_size,
            )

    set_bounds(global_lb, global_ub, global_bbox)


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------


class FemSpringManager:
    """
    Manages all FEM and SpringMass domains in a unified data structure.

    Architecture:
    - Global node arrays: position, velocity, acceleration, force, mass
    - Global element arrays: connectivity, properties, volumes
    - Domain metadata: offsets and counts for indexing
    - Batched kernels: process all domains in parallel

    Scalar counters are length-1 Warp arrays. In device kernels use ``arr[0]``;
    from host Python use ``arr.numpy()[0]`` (Warp 1.14+ has no host ``arr[i]``).
    """

    def __init__(self, d, domains, spatial_hash=None):
        """
        Initialize FemSpringManager with pre-allocated buffers.

        Args:
            d: Spatial dimension (2 only in Warp migration)
            domains: List of FEM / SpringMass domain objects
            spatial_hash: Unused (API compat); manager owns SpatialHashManager
        """
        ensure_warp()
        if d != 2:
            raise ValueError(f"FemSpringManager Warp migration supports d=2 only (got {d})")

        self.d = d
        self._sh_fem_elapsed = 0.0
        self._sh_rebuild_interval = 1.0 / 500.0
        self.aabb = None  # set via setGlobalAABB (wp.array shape (N,2) dtype=vec2)

        count_nodes = 0
        count_elements = 0
        count_fem_domains = 0
        count_spring_domains = 0
        count_boundary_nodes = 0
        count_boundary_elements = 0

        for dom in domains:
            if dom.type == DomainType.FEM:
                count_nodes += dom.nnodes
                count_elements += dom.nelements
                count_fem_domains += 1
                count_boundary_nodes += dom.mesh.numBoundNodes
                count_boundary_elements += dom.mesh.numBoundElements
            elif dom.type == DomainType.SPRINGMASS:
                count_nodes += dom.nnodes
                count_elements += dom.nelements
                count_spring_domains += 1

        self.MAX_NODES = max(count_nodes + 1024, 2048)
        self.MAX_ELEMENTS = max(count_elements + 2048, 4096)
        self.MAX_BOUNDARY_NODES = max(count_boundary_nodes + 1024, self.MAX_NODES)
        self.MAX_BOUNDARY_ELEMENTS = max(count_boundary_elements + 1024, self.MAX_NODES)

        print(f"[FemSpringManager] Memory allocation: {count_nodes} nodes + buffer → {self.MAX_NODES} MAX_NODES")
        print(
            f"[FemSpringManager] Memory allocation: {count_elements} elements + buffer → {self.MAX_ELEMENTS} MAX_ELEMENTS"
        )

        self.spatialHash = SpatialHashManager(d, max_elements=self.MAX_BOUNDARY_ELEMENTS)

        # Domain counters (scalar → length-1 arrays, access [0])
        self.numFemDomains = wp.zeros(1, dtype=int)
        self.numSpringDomains = wp.zeros(1, dtype=int)
        self.totalNodes = wp.zeros(1, dtype=int)
        self.totalElements = wp.zeros(1, dtype=int)
        self.totalFEMElements = wp.zeros(1, dtype=int)
        self.totalFEMNodes = wp.zeros(1, dtype=int)

        # Node data
        self.coords = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.U = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.V = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.A = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.Fext = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.mass = wp.zeros(self.MAX_NODES, dtype=float)

        self.bcNodes = wp.zeros(self.MAX_NODES, dtype=int)
        self.bcValues = wp.zeros(self.MAX_NODES, dtype=wp.vec2)

        # Element data
        self.connectivity = wp.zeros(self.MAX_ELEMENTS, dtype=wp.vec4i)
        self.elemType = wp.zeros(self.MAX_ELEMENTS, dtype=int)
        self.elemNodesCount = wp.zeros(self.MAX_ELEMENTS, dtype=int)
        self.elemVolume = wp.zeros(self.MAX_ELEMENTS, dtype=float)
        self.J_inv = wp.zeros(self.MAX_ELEMENTS, dtype=wp.mat22)

        self.matE = wp.zeros(self.MAX_ELEMENTS, dtype=float)
        self.matNu = wp.zeros(self.MAX_ELEMENTS, dtype=float)
        self.matRho = wp.zeros(self.MAX_ELEMENTS, dtype=float)
        self.matDamping = wp.zeros(self.MAX_ELEMENTS, dtype=float)
        self.matType = wp.zeros(self.MAX_ELEMENTS, dtype=int)
        self.matYieldStress = wp.zeros(self.MAX_ELEMENTS, dtype=float)
        self.matH = wp.zeros(self.MAX_ELEMENTS, dtype=float)

        self.plasticStrain = wp.zeros(self.MAX_ELEMENTS, dtype=wp.vec3)
        self.eqPlasticStrain = wp.zeros(self.MAX_ELEMENTS, dtype=float)
        self.elemStress = wp.zeros(self.MAX_ELEMENTS, dtype=wp.vec3)
        self.elemStrain = wp.zeros(self.MAX_ELEMENTS, dtype=wp.vec3)

        self.MAX_DOMAINS = max(len(domains), 128)
        self.domainNodeOffset = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.domainNodeCount = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.domainElemOffset = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.domainElemCount = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.domainType = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.boundaryNodes = wp.zeros(self.MAX_BOUNDARY_NODES, dtype=int)
        self.boundaryElements = wp.zeros(self.MAX_BOUNDARY_ELEMENTS, dtype=wp.vec3i)
        self.domainBoundaryNodeOffset = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.domainBoundaryNodeCount = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.domainBoundaryElemOffset = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.domainBoundaryElementCount = wp.zeros(self.MAX_DOMAINS, dtype=int)
        self.boundaryNodeNormals = wp.zeros(self.MAX_NODES, dtype=wp.vec2)
        self.boundaryNodeCounts = wp.zeros(1, dtype=int)
        _assign_scalar(self.boundaryNodeCounts, count_boundary_nodes)
        self.boundaryElementCounts = wp.zeros(1, dtype=int)
        _assign_scalar(self.boundaryElementCounts, count_boundary_elements)
        self.femDomainIds = wp.zeros(self.MAX_DOMAINS, dtype=int)

        self.domains = []
        self.domainMeshes = []

        # Host-side counter mirrors (synced into wp scalars after domain load)
        self._n_fem = 0
        self._n_spring = 0
        self._n_nodes = 0
        self._n_elems = 0
        self._n_fem_elems = 0
        self._n_fem_nodes = 0

        self._added_mass_buf = wp.zeros(1, dtype=float)
        self._total_mass_buf = wp.zeros(1, dtype=float)

        self.stableTime = 1.0

        self.processDomains(domains)
        self._sync_counters()
        self.processConditions()

        if self._n_fem_elems > 0:
            self.prepareFEMData_()

    def _sync_counters(self):
        _assign_scalar(self.numFemDomains, self._n_fem)
        _assign_scalar(self.numSpringDomains, self._n_spring)
        _assign_scalar(self.totalNodes, self._n_nodes)
        _assign_scalar(self.totalElements, self._n_elems)
        _assign_scalar(self.totalFEMElements, self._n_fem_elems)
        _assign_scalar(self.totalFEMNodes, self._n_fem_nodes)

    def processDomains(self, domains):
        """Add FEM and SpringMass domains to this manager."""
        for domain_idx, dom in enumerate(domains):
            if dom.type == DomainType.FEM:
                fem_idx = self.addFemDomain(dom)
            elif dom.type == DomainType.SPRINGMASS:
                fem_idx = self.addSpringDomain(dom)
            else:
                continue

            _patch_array(self.femDomainIds, fem_idx, domain_idx)
            self.stableTime = min(dom.stableTime, self.stableTime)

    def processConditions(self):
        bc_nodes = self.bcNodes.numpy()
        bc_values = self.bcValues.numpy()
        V = self.V.numpy()
        offsets = self.domainNodeOffset.numpy()

        for domain_idx, domain in enumerate(self.domains):
            offset = int(offsets[domain_idx])
            for bc in domain.bcs:
                type_, nodes, values = bc.processData()
                if values is None:
                    values = [0.0] * domain.d
                if nodes is None:
                    nodes = range(domain.nnodes)
                for idx in nodes:
                    nd = offset + int(idx)
                    bc_nodes[nd] |= int(type_)
                    bc_values[nd, 0] = float(values[0])
                    bc_values[nd, 1] = float(values[1])

            for ic in domain.initials:
                self._apply_initial_condition_np(ic, offset, domain.nnodes, V)

        self.bcNodes.assign(bc_nodes)
        self.bcValues.assign(bc_values)
        self.V.assign(V)

    def _apply_initial_condition_np(self, ic, offset: int, num_nodes: int, V: np.ndarray):
        """Host-side IC apply into a numpy velocity buffer."""
        vel = ic.vel
        vx, vy = float(vel[0]), float(vel[1])
        if getattr(ic, "is_all_nodes", False):
            V[offset : offset + num_nodes, 0] = vx
            V[offset : offset + num_nodes, 1] = vy
        else:
            nds = ic.nds
            if hasattr(nds, "numpy"):
                nds_np = nds.numpy()
            elif hasattr(nds, "to_numpy"):
                nds_np = nds.numpy()
            else:
                nds_np = np.asarray(nds)
            for nid in nds_np:
                V[offset + int(nid), 0] = vx
                V[offset + int(nid), 1] = vy

    def addFemDomain(self, domain):
        domain_idx = len(self.domains)
        if domain_idx >= self.MAX_DOMAINS:
            print(f"\033[91mError: Exceeded MAX_DOMAINS ({self.MAX_DOMAINS})! Cannot add more domains.\033[0m")
            raise RuntimeError(f"Exceeded maximum domains ({self.MAX_DOMAINS})")

        node_offset = self._n_nodes
        elem_offset = self._n_elems

        mesh = domain.mesh
        nnodes = mesh.numNodes
        nelems = mesh.numElements

        if node_offset + nnodes > self.MAX_NODES:
            print(
                f"\033[91mError: Exceeded MAX_NODES ({self.MAX_NODES})! "
                f"Trying to add {nnodes} nodes, current total: {node_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum nodes ({self.MAX_NODES})")
        if elem_offset + nelems > self.MAX_ELEMENTS:
            print(
                f"\033[91mError: Exceeded MAX_ELEMENTS ({self.MAX_ELEMENTS})! "
                f"Trying to add {nelems} elements, current total: {elem_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum elements ({self.MAX_ELEMENTS})")

        coords_np = np.asarray(mesh.coords, dtype=np.float32)
        conns_np = np.asarray(mesh.connectivity, dtype=np.int32)

        _patch_array(self.domainNodeOffset, domain_idx, node_offset)
        _patch_array(self.domainNodeCount, domain_idx, nnodes)
        _patch_array(self.domainElemOffset, domain_idx, elem_offset)
        _patch_array(self.domainElemCount, domain_idx, nelems)
        _patch_array(self.domainType, domain_idx, 1)  # FEM

        _patch_slice(self.coords, node_offset, coords_np[:, :2])

        mat = domain.prop.getMaterial()
        conn_buf = np.full((nelems, 4), -1, dtype=np.int32)
        for i in range(nelems):
            conn = conns_np[i]
            for j in range(len(conn)):
                conn_buf[i, j] = node_offset + int(conn[j])
        _patch_slice(self.connectivity, elem_offset, conn_buf)

        elem_type = np.full(nelems, 1, dtype=np.int32)
        elem_nc = np.full(nelems, 3, dtype=np.int32)
        mat_e = np.full(nelems, float(mat.E), dtype=np.float32)
        mat_nu = np.full(nelems, float(mat.nu), dtype=np.float32)
        mat_rho = np.full(nelems, float(mat.rho), dtype=np.float32)
        mat_type = np.full(nelems, int(mat.type), dtype=np.int32)
        mat_y = np.full(nelems, float(getattr(mat, "sigma_y", 0.0)), dtype=np.float32)
        mat_h = np.full(nelems, float(getattr(mat, "H", 0.0)), dtype=np.float32)
        _patch_slice(self.elemType, elem_offset, elem_type)
        _patch_slice(self.elemNodesCount, elem_offset, elem_nc)
        _patch_slice(self.matE, elem_offset, mat_e)
        _patch_slice(self.matNu, elem_offset, mat_nu)
        _patch_slice(self.matRho, elem_offset, mat_rho)
        _patch_slice(self.matType, elem_offset, mat_type)
        _patch_slice(self.matYieldStress, elem_offset, mat_y)
        _patch_slice(self.matH, elem_offset, mat_h)

        bn_off_arr = self.domainBoundaryNodeOffset.numpy()
        bn_cnt_arr = self.domainBoundaryNodeCount.numpy()
        if domain_idx > 0:
            bn_off_arr[domain_idx] = int(bn_off_arr[domain_idx - 1]) + int(bn_cnt_arr[domain_idx - 1])
        bn_off = int(bn_off_arr[domain_idx])
        bn_cnt_arr[domain_idx] = domain.mesh.numBoundNodes
        self.domainBoundaryNodeOffset.assign(bn_off_arr)
        self.domainBoundaryNodeCount.assign(bn_cnt_arr)

        bn_ids = np.array(
            [node_offset + int(domain.mesh.boundaryNodes[i]) for i in range(domain.mesh.numBoundNodes)],
            dtype=np.int32,
        )
        if len(bn_ids):
            _patch_slice(self.boundaryNodes, bn_off, bn_ids)

        be_off_arr = self.domainBoundaryElemOffset.numpy()
        be_cnt_arr = self.domainBoundaryElementCount.numpy()
        if domain_idx > 0:
            be_off_arr[domain_idx] = int(be_off_arr[domain_idx - 1]) + int(be_cnt_arr[domain_idx - 1])
        be_off = int(be_off_arr[domain_idx])
        be_cnt_arr[domain_idx] = domain.mesh.numBoundElements
        self.domainBoundaryElemOffset.assign(be_off_arr)
        self.domainBoundaryElementCount.assign(be_cnt_arr)

        if domain.mesh.numBoundElements > 0:
            be_buf = np.full((domain.mesh.numBoundElements, 3), -1, dtype=np.int32)
            for i in range(domain.mesh.numBoundElements):
                conn = domain.mesh.boundaryElements[i]
                for j in range(len(conn)):
                    be_buf[i, j] = node_offset + int(conn[j])
            _patch_slice(self.boundaryElements, be_off, be_buf)

        self._n_nodes = node_offset + nnodes
        self._n_elems = elem_offset + nelems
        self._n_fem_elems += nelems
        self._n_fem_nodes += nnodes
        self._n_fem += 1

        boundary_normals = np.asarray(mesh.computeBoundaryNodeNormals(), dtype=np.float32)
        _patch_slice(self.boundaryNodeNormals, node_offset, boundary_normals[:nnodes, :2])

        self.domains.append(domain)
        self.domainMeshes.append(mesh)
        domain.attach(self, domain_idx)
        return domain_idx

    def addSpringDomain(self, domain):
        """Add a SpringMass domain to the manager."""
        domain_idx = len(self.domains)
        if domain_idx >= self.MAX_DOMAINS:
            print(f"\033[91mError: Exceeded MAX_DOMAINS ({self.MAX_DOMAINS})! Cannot add more domains.\033[0m")
            raise RuntimeError(f"Exceeded maximum domains ({self.MAX_DOMAINS})")

        node_offset = self._n_nodes
        elem_offset = self._n_elems

        nnodes = domain.nnodes
        nelems = domain.nelements

        if node_offset + nnodes > self.MAX_NODES:
            print(
                f"\033[91mError: Exceeded MAX_NODES ({self.MAX_NODES})! "
                f"Trying to add {nnodes} nodes, current total: {node_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum nodes ({self.MAX_NODES})")
        if elem_offset + nelems > self.MAX_ELEMENTS:
            print(
                f"\033[91mError: Exceeded MAX_ELEMENTS ({self.MAX_ELEMENTS})! "
                f"Trying to add {nelems} elements, current total: {elem_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum elements ({self.MAX_ELEMENTS})")

        coords_np = np.asarray(domain.coords, dtype=np.float32)
        conns_np = np.asarray(domain.connectivity, dtype=np.int32)
        rest_len_np = np.asarray(domain.restLength, dtype=np.float32)

        _patch_array(self.domainNodeOffset, domain_idx, node_offset)
        _patch_array(self.domainNodeCount, domain_idx, nnodes)
        _patch_array(self.domainElemOffset, domain_idx, elem_offset)
        _patch_array(self.domainElemCount, domain_idx, nelems)
        _patch_array(self.domainType, domain_idx, 0)  # Spring

        _patch_slice(self.coords, node_offset, coords_np[:, :2])
        mass_slice = np.full(nnodes, float(domain.mass), dtype=np.float32)
        _patch_slice(self.mass, node_offset, mass_slice)

        conn_buf = np.full((nelems, 4), -1, dtype=np.int32)
        conn_buf[:, 0] = node_offset + conns_np[:, 0]
        conn_buf[:, 1] = node_offset + conns_np[:, 1]
        _patch_slice(self.connectivity, elem_offset, conn_buf)
        _patch_slice(self.elemType, elem_offset, np.zeros(nelems, dtype=np.int32))
        _patch_slice(self.elemNodesCount, elem_offset, np.full(nelems, 2, dtype=np.int32))
        _patch_slice(self.elemVolume, elem_offset, rest_len_np)
        _patch_slice(self.matE, elem_offset, np.full(nelems, float(domain.spring), dtype=np.float32))
        _patch_slice(self.matDamping, elem_offset, np.full(nelems, float(domain.damping), dtype=np.float32))

        domain.attach(self, domain_idx)
        self._n_nodes = node_offset + nnodes
        self._n_elems = elem_offset + nelems
        self._n_spring += 1
        self.domains.append(domain)
        return domain_idx

    def setGlobalAABB(self, aabb):
        """Attach shared domain AABB buffer.

        ``aabb`` must be a Warp array of shape ``(max_domains, 2)`` with
        ``dtype=wp.vec2`` (index ``[domain_id, 0/1]`` for lb/ub).
        """
        self.aabb = aabb
        self.substep(0.0, 0.0)

    def prepareFEMData_(self):
        n = int(self.totalFEMElements.numpy()[0])
        if n <= 0:
            return
        wp.launch(
            _prepare_fem_data_kernel,
            dim=n,
            inputs=[
                self.totalFEMElements,
                self.connectivity,
                self.coords,
                self.matRho,
                self.elemVolume,
                self.J_inv,
                self.mass,
            ],
        )

    def applyMassScaling(self, dt_target: float):
        """Scale nodal masses so every element's critical time step >= dt_target."""
        self._added_mass_buf.zero_()
        wp.launch(
            _mass_scaling_kernel,
            dim=1,
            inputs=[
                float(dt_target),
                self.totalFEMElements,
                self.totalElements,
                self.connectivity,
                self.coords,
                self.matE,
                self.matRho,
                self.elemVolume,
                self.mass,
                self._added_mass_buf,
            ],
        )
        added = float(self._added_mass_buf.numpy()[0])
        total_mass = self._total_mass()
        pct = added / total_mass * 100.0 if total_mass > 0 else 0.0
        print(
            f"[MassScaling] dt_target={dt_target:.3e}  added_mass={added:.4e}  "
            f"total_mass={total_mass:.4e}  ({pct:.2f}%)"
        )

    def _total_mass(self) -> float:
        self._total_mass_buf.zero_()
        n = int(self.totalNodes.numpy()[0])
        if n <= 0:
            return 0.0
        wp.launch(
            _total_mass_kernel,
            dim=n,
            inputs=[self.totalNodes, self.mass, self._total_mass_buf],
        )
        return float(self._total_mass_buf.numpy()[0])

    def substep(self, dt: float, damping: float):
        self.femStep(float(dt), float(damping))

    def femStep(self, dt: float, damping: float):
        n_fem = int(self.totalFEMElements.numpy()[0])
        n_tot = int(self.totalElements.numpy()[0])
        n_nodes = int(self.totalNodes.numpy()[0])

        if n_fem > 0:
            wp.launch(
                _fem_internal_force_kernel,
                dim=n_fem,
                inputs=[
                    self.totalFEMElements,
                    self.connectivity,
                    self.coords,
                    self.J_inv,
                    self.matType,
                    self.matE,
                    self.matNu,
                    self.matYieldStress,
                    self.matH,
                    self.plasticStrain,
                    self.eqPlasticStrain,
                    self.elemVolume,
                    self.elemStress,
                    self.elemStrain,
                    self.Fext,
                ],
            )

        if n_tot > n_fem:
            wp.launch(
                _spring_force_kernel,
                dim=n_tot,
                inputs=[
                    self.totalFEMElements,
                    self.totalElements,
                    self.connectivity,
                    self.coords,
                    self.V,
                    self.matE,
                    self.matDamping,
                    self.elemVolume,
                    self.Fext,
                ],
            )

        if n_nodes > 0:
            wp.launch(
                _integrate_nodes_kernel,
                dim=n_nodes,
                inputs=[
                    float(dt),
                    float(damping),
                    self.totalNodes,
                    self.bcNodes,
                    self.bcValues,
                    self.mass,
                    self.A,
                    self.V,
                    self.U,
                    self.coords,
                    self.Fext,
                ],
            )

        self.updateBBox()

        if n_nodes > 0:
            wp.launch(
                _clear_fext_kernel,
                dim=n_nodes,
                inputs=[self.totalNodes, self.Fext],
            )

    def updateBBox(self):
        """Compute FEM/spring domain bboxes into ``self.aabb``."""
        if self.aabb is None:
            return
        wp.launch(
            _update_bbox_kernel,
            dim=1,
            inputs=[
                self.numFemDomains,
                self.numSpringDomains,
                self.femDomainIds,
                self.domainBoundaryNodeOffset,
                self.domainBoundaryNodeCount,
                self.boundaryNodes,
                self.domainNodeOffset,
                self.domainNodeCount,
                self.coords,
                self.aabb,
            ],
        )

    def maybe_rebuild_fem_spatial_hash(self, dt: float):
        self._sh_fem_elapsed += dt
        if self._sh_fem_elapsed >= self._sh_rebuild_interval:
            self.populate_fem_spatial_hash(velocity_buffer=0.0)
            self._sh_fem_elapsed = 0.0

    def populate_fem_spatial_hash(self, velocity_buffer: float = 0.0):
        """Populate spatial hash with FEM boundary elements for FEM-FEM contact."""
        if self.spatialHash is None or self.aabb is None:
            return

        num_fem = int(self.numFemDomains.numpy()[0])
        if num_fem == 0:
            return

        sh = self.spatialHash
        sh.reset()
        wp.launch(
            _populate_fem_spatial_hash_kernel,
            dim=1,
            inputs=[
                float(velocity_buffer),
                self.numFemDomains,
                self.femDomainIds,
                self.aabb,
                self.domainBoundaryElemOffset,
                self.domainBoundaryElementCount,
                self.boundaryElements,
                self.coords,
                sh.MAX_ELEMENTS,
                sh.numElements,
                sh.domainIds,
                sh.elementbbox,
                sh.estimateSize,
                sh._sumSize,
                sh.globalbbox,
            ],
        )
        sh.build()

    def drawMesh(self, gui, color, resoultion=800):
        """Draw nodes as circles and element edges as lines (Viewer API)."""
        totalEls = int(self.totalElements.numpy()[0])
        totalNds = int(self.totalNodes.numpy()[0])
        if totalNds <= 0:
            return
        pos = self.coords.numpy()[:totalNds]
        gui.circles(pos, radius=2, color=0xFFAA33)
        if totalEls <= 0:
            return
        e2n = self.connectivity.numpy()[:totalEls]
        # Triangle edges (skip padded -1 for springs)
        i0 = e2n[:, 0]
        i1 = e2n[:, 1]
        i2 = e2n[:, 2]
        valid_tri = i2 >= 0
        if np.any(valid_tri):
            a = pos[i0[valid_tri]]
            b = pos[i1[valid_tri]]
            c = pos[i2[valid_tri]]
            gui.lines(a, b, radius=1, color=color)
            gui.lines(b, c, radius=1, color=color)
            gui.lines(c, a, radius=1, color=color)
        valid_spring = (i2 < 0) & (i0 >= 0) & (i1 >= 0)
        if np.any(valid_spring):
            gui.lines(pos[i0[valid_spring]], pos[i1[valid_spring]], radius=1, color=color)
