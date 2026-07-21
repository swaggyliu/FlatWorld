"""
Public contact detection functions for different object types.
These functions work directly with RigidManager data for maximum efficiency.
"""

from typing import Any
import numpy as np

from definitions import *
import warp as wp
from utils import capsuleSDFQuery, obbSDFQuery, sphereSDFQuery


@wp.func
def pointToEdgeContact(point: wp.vec2, edge_n0: wp.vec2, edge_n1: wp.vec2, d: int):
    """2D: Find closest point on edge and check penetration.

    Args:
        point: Query point
        edge_n0: Edge start point
        edge_n1: Edge end point
        d: Dimensionality (2 for 2D); kept for call-site compatibility

    Returns:
        penetration: Signed distance (negative = outside, positive = inside)
        normal: Contact normal (points from edge to point)
        cpoint: Closest point on edge
        is_inside: True if point projects inside edge
        weights: (w0, w1) nodal weights where cpoint = w0*edge_n0 + w1*edge_n1
    """
    edge = edge_n1 - edge_n0
    edge_len = wp.length(edge)

    penetration = 1e9
    normal = wp.vec2(0.0, 0.0)
    cpoint = point
    t = 0.0
    is_inside = False
    weights = wp.vec2(0.0, 0.0)

    if edge_len > 1e-9:
        edge_dir = edge / edge_len
        normal = wp.vec2(edge_dir[1], -edge_dir[0])
        # Project point onto edge
        to_point = point - edge_n0
        t = wp.dot(to_point, edge_dir)
        # Normalize t to [0,1]
        t = t / edge_len

        if 0.0 <= t <= 1.0:
            is_inside = True
            penetration = wp.dot(to_point, normal)

            if penetration < 0.0:
                cpoint = edge_n0 + edge_dir * t * edge_len

            # Compute nodal weights: w0 = 1-t, w1 = t
            weights = wp.vec2(1.0 - t, t)

    return penetration, normal, cpoint, is_inside, weights


@wp.func
def detectPointToMeshBoundary(
    point: wp.vec2, mesh_boundary_coords: Any, elem_conn: Any, limit_penetration: float
):
    pen = 1e9
    best_penetration = 1e9
    best_normal = wp.vec2(0.0, 0.0)
    best_cpoint = point
    normal = wp.vec2(0.0, 0.0)
    weights = wp.vec2(0.0, 0.0)
    cp = point
    is_inside = False

    # 2D: edge contact
    n0_idx = elem_conn[0]
    n1_idx = elem_conn[1]

    n0 = mesh_boundary_coords[n0_idx]
    n1 = mesh_boundary_coords[n1_idx]

    pen, normal, cp, is_inside, weights = pointToEdgeContact(point, n0, n1, 2)

    if wp.abs(pen) < limit_penetration and is_inside:
        best_penetration = pen
        best_normal = normal
        best_cpoint = cp

    return best_penetration, best_normal, best_cpoint, weights


@wp.func
def detectPointToMeshBoundaries(
    point: wp.vec2,
    mesh_boundary_coords: Any,
    mesh_boundary_elements: Any,
    elem_offset: int,
    num_elems: int,
    limit_penetration: float,
):
    """Detect contact between a point and mesh boundary elements.

    Args:
        point: Query point
        mesh_boundary_coords: Global boundary node coordinates field
        mesh_boundary_elements: Global boundary element connectivity field
        elem_offset: Starting index for this mesh's elements
        num_elems: Number of boundary elements for this mesh
        limit_penetration: Maximum penetration depth to consider contact

    Returns:
        penetration: Signed distance (negative = penetration)
        normal: Contact normal
        cpoint: Contact point on mesh surface
    """
    best_penetration = 1e9
    best_normal = wp.vec2(0.0, 0.0)
    best_cpoint = point

    # Test all boundary elements
    for eid in range(num_elems):
        elem_conn = mesh_boundary_elements[elem_offset + eid]

        pen = 1e9
        normal = wp.vec2(0.0, 0.0)
        cp = point
        is_inside = False

        # 2D: edge contact
        n0_idx = elem_conn[0]
        n1_idx = elem_conn[1]

        n0 = mesh_boundary_coords[n0_idx]
        n1 = mesh_boundary_coords[n1_idx]

        pen, normal, cp, is_inside, weights = pointToEdgeContact(point, n0, n1, 2)

        if wp.abs(pen) < best_penetration and wp.abs(pen) < limit_penetration and is_inside:
            best_penetration = pen
            best_normal = normal
            best_cpoint = cp

    return best_penetration, best_normal, best_cpoint


@wp.func
def detectPointToPrimitive(
    point: wp.vec2, rigid_type: int, center: wp.vec2, prim: wp.vec2, rotMat: wp.mat22, radius: float
):
    """Detect contact between a point and a primitive rigid body.

    Args:
        point: Query point
        rigid_type: RigidType (BALL, BOX, CAPSULE)
        center: Rigid body center position
        prim: Primitive parameters (extent for BOX, line vector for CAPSULE)
        rotMat: Rotation Matrix (2x2 for 2D)
        radius: Radius for ball/capsule
    Returns:
        penetration: Signed distance (negative = penetration)
        normal: Contact normal (points from surface to point)
        cpoint: Contact point on rigid surface
    """
    penetration = 1e9
    normal = wp.vec2(0.0, 0.0)
    cpoint = point

    if rigid_type == RigidType.BALL:
        penetration, normal = sphereSDFQuery(point, center, radius)
        if penetration < 0.0:
            cpoint = point - normal * penetration

    elif rigid_type == RigidType.BOX:
        extent = prim
        penetration, normal = obbSDFQuery(point, center, extent, rotMat)
        if penetration < 0.0:
            cpoint = point - normal * penetration

    elif rigid_type == RigidType.CAPSULE:
        center_o = center
        lcdir = prim
        newlc = center + rotMat @ lcdir

        penetration, normal = capsuleSDFQuery(point, center_o, newlc, radius)

        if penetration < 0.0:
            cpoint = point - normal * penetration

    return penetration, normal, cpoint


@wp.func
def detectPointToAnalyticalPlane(point: wp.vec2, plane_point: wp.vec2, plane_normal: wp.vec2):
    """Detect contact between a point and an analytical plane.

    Args:
        point: Query point
        plane_point: A point on the plane
        plane_normal: Plane normal (unit vector)

    Returns:
        penetration: Signed distance (negative = behind plane)
        normal: Plane normal
        cpoint: Contact point on plane (projection)
    """
    penetration = wp.dot(point - plane_point, plane_normal)
    normal = plane_normal
    cpoint = point - normal * penetration

    return penetration, normal, cpoint


# ---------------------------------------------------------------------------
# Host NumPy helpers (Warp 1.14+ forbids host indexing inside @wp.func)
# ---------------------------------------------------------------------------

def detect_point_to_mesh_boundary_np(point, mesh_coords_np, conn, limit_penetration: float):
    point = np.asarray(point, dtype=np.float32).reshape(-1)[:2]
    n0 = np.asarray(mesh_coords_np[int(conn[0])], dtype=np.float32).reshape(-1)[:2]
    n1 = np.asarray(mesh_coords_np[int(conn[1])], dtype=np.float32).reshape(-1)[:2]
    edge = n1 - n0
    edge_len = float(np.linalg.norm(edge))
    if edge_len < 1e-9:
        return 1e9, np.array([0.0, 1.0], dtype=np.float32), point, np.array([0.5, 0.5], dtype=np.float32)
    edge_dir = edge / edge_len
    # Match pointToEdgeContact: normal = (ey, -ex) = right of directed edge
    n = np.array([edge_dir[1], -edge_dir[0]], dtype=np.float32)
    t = float(np.dot(point - n0, edge_dir) / edge_len)
    is_inside = 0.0 <= t <= 1.0
    t_clamped = min(max(t, 0.0), 1.0)
    cpoint = n0 + edge_dir * (t_clamped * edge_len)
    signed = float(np.dot(point - n0, n))
    if abs(signed) < limit_penetration and is_inside:
        weights = np.array([1.0 - t_clamped, t_clamped], dtype=np.float32)
        return signed, n, cpoint, weights
    return 1e9, np.array([0.0, 1.0], dtype=np.float32), point, np.array([0.5, 0.5], dtype=np.float32)


def detect_point_to_primitive_np(point, rigid_type: int, center, prim, rotMat, radius: float):
    point = np.asarray(point, dtype=np.float32).reshape(-1)[:2]
    center = np.asarray(center, dtype=np.float32).reshape(-1)[:2]
    prim = np.asarray(prim, dtype=np.float32).reshape(-1)[:2]
    R = np.asarray(rotMat, dtype=np.float32).reshape(2, 2)
    if rigid_type == RigidType.BALL:
        diff = point - center
        nlen = float(np.linalg.norm(diff))
        penetration = nlen - float(radius)
        n = diff / max(nlen, 1e-9)
        cpoint = point - n * penetration if penetration < 0.0 else point
        return penetration, n, cpoint
    if rigid_type == RigidType.BOX:
        local = R.T @ (point - center)
        half = 0.5 * prim
        q = np.abs(local) - half
        outside = np.maximum(q, 0.0)
        penetration = float(np.linalg.norm(outside)) + min(float(np.max(q)), 0.0)
        if float(np.max(q)) < 0.0:
            ax = 0 if abs(q[0]) > abs(q[1]) else 1
            n_local = np.zeros(2, dtype=np.float32)
            n_local[ax] = 1.0 if local[ax] >= 0 else -1.0
            n = R @ n_local
            return float(np.max(q)), n, point - n * float(np.max(q))
        n_local = outside / max(float(np.linalg.norm(outside)), 1e-9)
        n_local = n_local * np.sign(local + 1e-12)
        n = R @ n_local
        return penetration, n, point
    if rigid_type == RigidType.CAPSULE:
        lc = center + R @ prim
        uc = 2.0 * center - lc
        ab = uc - lc
        ab_len2 = float(np.dot(ab, ab))
        tt = 0.0 if ab_len2 < 1e-18 else float(np.dot(point - lc, ab) / ab_len2)
        tt = min(max(tt, 0.0), 1.0)
        closest = lc + ab * tt
        diff = point - closest
        nlen = float(np.linalg.norm(diff))
        penetration = nlen - float(radius)
        n = diff / max(nlen, 1e-9)
        cpoint = point - n * penetration if penetration < 0.0 else point
        return penetration, n, cpoint
    return 1e9, np.array([0.0, 1.0], dtype=np.float32), point
