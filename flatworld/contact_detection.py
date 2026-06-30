"""
Public contact detection functions for different object types.
These functions work directly with RigidManager data for maximum efficiency.
"""

from definitions import *
import taichi as ti
from utils import cal2DRotationMat, capsuleSDFQuery, obbSDFQuery, sphereSDFQuery


@ti.func
def pointToEdgeContact(point, edge_n0, edge_n1, d: ti.template()):
    """2D: Find closest point on edge and check penetration.

    Args:
        point: Query point
        edge_n0: Edge start point
        edge_n1: Edge end point
        d: Dimensionality (2 for 2D)

    Returns:
        penetration: Signed distance (negative = outside, positive = inside)
        normal: Contact normal (points from edge to point)
        cpoint: Closest point on edge
        is_inside: True if point projects inside edge
        weights: (w0, w1) nodal weights where cpoint = w0*edge_n0 + w1*edge_n1
    """
    edge = edge_n1 - edge_n0
    edge_len = edge.norm()

    penetration = 1e9
    normal = ti.Vector.zero(ti.f32, d)
    cpoint = point
    t = 0.0
    is_inside = False
    weights = ti.Vector([0.0, 0.0])

    if edge_len > 1e-9:
        edge_dir = edge / edge_len
        normal = ti.Vector([edge_dir[1], -edge_dir[0]])
        # Project point onto edge
        to_point = point - edge_n0
        t = to_point.dot(edge_dir)
        # Normalize t to [0,1]
        t = t / edge_len

        if 0.0 <= t <= 1.0:
            is_inside = True
            penetration = to_point.dot(normal)

            if penetration < 0.0:
                cpoint = edge_n0 + edge_dir * t * edge_len

            # Compute nodal weights: w0 = 1-t, w1 = t
            weights = ti.Vector([1.0 - t, t])

    return penetration, normal, cpoint, is_inside, weights


@ti.func
def detectPointToMeshBoundary(point, mesh_boundary_coords, elem_conn, limit_penetration: ti.f32):
    pen = 1e9
    best_penetration = 1e9
    best_normal = ti.Vector.zero(ti.f32, point.n)
    best_cpoint = point
    normal = ti.Vector.zero(ti.f32, point.n)
    weights = ti.Vector.zero(ti.f32, point.n)
    cp = point
    is_inside = False

    # 2D: edge contact
    n0_idx = elem_conn[0]
    n1_idx = elem_conn[1]

    n0 = mesh_boundary_coords[n0_idx]
    n1 = mesh_boundary_coords[n1_idx]

    pen, normal, cp, is_inside, weights = pointToEdgeContact(point, n0, n1, point.n)

    if ti.abs(pen) < limit_penetration and is_inside:
        best_penetration = pen
        best_normal = normal
        best_cpoint = cp

    return best_penetration, best_normal, best_cpoint, weights


@ti.func
def detectPointToMeshBoundaries(
    point, mesh_boundary_coords, mesh_boundary_elements, elem_offset, num_elems, limit_penetration: ti.f32
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
    best_normal = ti.Vector.zero(ti.f32, point.n)
    best_cpoint = point

    # Test all boundary elements
    for eid in range(num_elems):
        elem_conn = mesh_boundary_elements[elem_offset + eid]

        pen = 1e9
        normal = ti.Vector.zero(ti.f32, point.n)
        cp = point
        is_inside = False

        # 2D: edge contact
        n0_idx = elem_conn[0]
        n1_idx = elem_conn[1]

        n0 = mesh_boundary_coords[n0_idx]
        n1 = mesh_boundary_coords[n1_idx]

        pen, normal, cp, is_inside, weights = pointToEdgeContact(point, n0, n1, point.n)

        if ti.abs(pen) < best_penetration and ti.abs(pen) < limit_penetration and is_inside:
            best_penetration = pen
            best_normal = normal
            best_cpoint = cp

    return best_penetration, best_normal, best_cpoint


@ti.func
def detectPointToPrimitive(point, rigid_type, center, prim, rotMat, radius):
    """Detect contact between a point and a primitive rigid body.

    Args:
        point: Query point
        rigid_type: RigidType (BALL, BOX, CAPSULE)
        center: Rigid body center position
        prim: Primitive parameters (extent for BOX, line vector for CAPSULE)
        rot: Rotation Matrix (2x2 for 2D)
        radius: Radius for ball/capsule
    Returns:
        penetration: Signed distance (negative = penetration)
        normal: Contact normal (points from surface to point)
        cpoint: Contact point on rigid surface
    """
    penetration = 1e9
    normal = ti.Vector.zero(ti.f32, center.n)
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


@ti.func
def detectPointToAnalyticalPlane(point, plane_point, plane_normal):
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
    penetration = (point - plane_point).dot(plane_normal)
    normal = plane_normal
    cpoint = point - normal * penetration

    return penetration, normal, cpoint
