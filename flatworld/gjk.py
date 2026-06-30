import numpy as np
import taichi as ti

# ==================== Support Functions ====================


@ti.func
def support_box(
    center: ti.math.vec3, extent: ti.math.vec3, rotation: ti.math.mat3, direction: ti.math.vec3
) -> ti.math.vec3:
    """Support function for an oriented box (OBB).
    Args:
        center: Box center position
        extent: Box half-extents (half-width, half-height, half-depth)
        rotation: 3x3 rotation matrix
        direction: Direction to find support point
    Returns:
        The furthest point on the box in the given direction
    """
    # Transform direction to local space
    local_dir = rotation.transpose() @ direction

    # Find support point in local space (aligned box)
    local_support = ti.math.vec3(
        extent.x if local_dir.x > 0.0 else -extent.x,
        extent.y if local_dir.y > 0.0 else -extent.y,
        extent.z if local_dir.z > 0.0 else -extent.z,
    )

    # Transform back to world space
    return center + rotation @ local_support


@ti.func
def support_capsule(
    center: ti.math.vec3, axis: ti.math.vec3, radius: ti.f32, rotation: ti.math.mat3, direction: ti.math.vec3
) -> ti.math.vec3:
    """Support function for a capsule (cylinder with hemispherical caps).
    Args:
        center: Capsule center position
        axis: Capsule axis direction (will be normalized)
        radius: Capsule radius
        rotation: 3x3 rotation matrix
        direction: Direction to find support point
    Returns:
        The furthest point on the capsule in the given direction

    A capsule is essentially a cylinder where the flat caps are replaced with hemispheres.
    The support function is:
        center + sign(dir·axis) * half_height * axis + radius * normalized(direction)
    """
    # Ensure axis is normalized
    axis_normalized = axis
    half_height = axis.norm()
    if half_height > 1e-8:
        axis_normalized = axis / half_height

    axis_normalized = rotation @ axis_normalized

    # Decompose direction into axial component
    axial_component = direction.dot(axis_normalized)

    # Support along the axis (which end cap?)
    sign = 1.0 if axial_component >= 0.0 else -1.0
    axial_support = axis_normalized * half_height * sign

    # Support on the sphere (radial component)
    # The sphere extends in all directions by radius
    dir_norm = direction.norm()
    radial_support = ti.math.vec3(0.0)
    if dir_norm > 1e-8:
        radial_support = (direction / dir_norm) * radius

    return center + axial_support + radial_support


@ti.func
def support_ball(center: ti.math.vec3, radius: ti.f32, direction: ti.math.vec3) -> ti.math.vec3:
    """Support function for a ball (sphere).
    Args:
        center: Ball center position
        radius: Ball radius
        direction: Direction to find support point
    Returns:
        The furthest point on the ball in the given direction

    The support function for a sphere is simply:
        center + radius * normalized(direction)
    """
    dir = direction.normalized()
    # Arbitrary direction if direction is zero
    return center + dir * radius


@ti.func
def support_minkowski_diff(
    shape_a_type: ti.i32,
    center_a: ti.math.vec3,
    params_a: ti.math.vec4,
    rot_a: ti.math.mat3,
    shape_b_type: ti.i32,
    center_b: ti.math.vec3,
    params_b: ti.math.vec4,
    rot_b: ti.math.mat3,
    direction: ti.math.vec3,
) -> tuple:
    """Support function for Minkowski difference A - B.
    Shape types: 0=Box, 2=Capsule, 3=Ball
    For Box: params = (extent_x, extent_y, extent_z, 0)
    For Capsule: params = (radius, axis_x, axis_y, axis_z)
    For Ball: params = (radius, 0, 0, 0)

    Returns: (minkowski_point, witness_a, witness_b)
        - minkowski_point: support_a - support_b
        - witness_a: support point on shape A
        - witness_b: support point on shape B
    """
    support_a = ti.math.vec3(0.0)
    support_b = ti.math.vec3(0.0)

    # Get support point from shape A
    if shape_a_type == 0:  # Box
        support_a = support_box(center_a, ti.math.vec3(params_a.x, params_a.y, params_a.z), rot_a, direction)
    elif shape_a_type == 2:  # Capsule
        support_a = support_capsule(
            center_a, ti.math.vec3(params_a[1], params_a[2], params_a[3]), params_a[0], rot_a, direction
        )
    else:  # Ball (shape_a_type == 3)
        support_a = support_ball(center_a, params_a[0], direction)

    # Get support point from shape B in opposite direction
    if shape_b_type == 0:  # Box
        support_b = support_box(center_b, ti.math.vec3(params_b.x, params_b.y, params_b.z), rot_b, -direction)
    elif shape_b_type == 2:  # Capsule
        support_b = support_capsule(
            center_b, ti.math.vec3(params_b[1], params_b[2], params_b[3]), params_b[0], rot_b, -direction
        )
    else:  # Ball (shape_b_type == 3)
        support_b = support_ball(center_b, params_b[0], -direction)

    return support_a - support_b, support_a, support_b


@ti.func
def triple_product(a, b, c):
    """triple product：(a × b) × c"""
    return b * a.dot(c) - a * b.dot(c)


# ==================== Core GJK Logic ====================


@ti.func
def run_gjk(
    shape_a_type: ti.i32,
    center_a: ti.math.vec3,
    params_a: ti.math.vec4,
    rot_a: ti.math.mat3,
    shape_b_type: ti.i32,
    center_b: ti.math.vec3,
    params_b: ti.math.vec4,
    rot_b: ti.math.mat3,
    simplex: ti.template(),
    witness_a: ti.template(),
    witness_b: ti.template(),
    d: ti.i32,
) -> tuple:
    """
    Core GJK loop logic shared by gjk_collision and gjk_epa_collision.
    Updates simplex and witness points in place.
    Returns (has_collision, simplex_size)
    """
    # Initial direction
    direction = center_b - center_a
    if direction.norm() < 1e-10:
        direction = ti.math.vec3(1.0, 0.0, 0.0)
    else:
        direction = direction.normalized()

    simplex_size = 0
    max_iter = 64
    has_collision = 0
    should_continue = 1

    for iteration in range(max_iter):
        if should_continue == 1:
            p, sa, sb = support_minkowski_diff(
                shape_a_type, center_a, params_a, rot_a, shape_b_type, center_b, params_b, rot_b, direction
            )

            # CRITICAL: For first iteration, always add the point even if it doesn't pass origin
            # The initial direction might be wrong, so we need at least one point to start
            if simplex_size > 0 and p.dot(direction) < 0.0:
                # After first point, if new point doesn't pass origin, no collision
                should_continue = 0
            else:
                simplex[simplex_size, 0] = p.x
                simplex[simplex_size, 1] = p.y
                simplex[simplex_size, 2] = p.z
                witness_a[simplex_size, 0] = sa.x
                witness_a[simplex_size, 1] = sa.y
                witness_a[simplex_size, 2] = sa.z
                witness_b[simplex_size, 0] = sb.x
                witness_b[simplex_size, 1] = sb.y
                witness_b[simplex_size, 2] = sb.z

                simplex_size += 1

                if simplex_size == 1:
                    direction = -p.normalized()
                elif simplex_size == 2:
                    simplex_size, direction, contains = handle_line_gjk(simplex, witness_a, witness_b, direction)
                    if contains == 1:
                        has_collision = 1
                        should_continue = 0
                elif simplex_size == 3:
                    simplex_size, direction, contains = handle_triangle_gjk(simplex, witness_a, witness_b, direction, d)
                    if contains == 1:
                        has_collision = 1
                        should_continue = 0

                if should_continue == 1 and direction.norm() < 1e-10:
                    direction = ti.math.vec3(1.0, 0.0, 0.0)

    return has_collision, simplex_size


# ==================== GJK Algorithm ====================


@ti.func
def gjk_collision(
    shape_a_type: ti.i32,
    center_a: ti.math.vec3,
    params_a: ti.math.vec4,
    rot_a: ti.math.mat3,
    shape_b_type: ti.i32,
    center_b: ti.math.vec3,
    params_b: ti.math.vec4,
    rot_b: ti.math.mat3,
    d: ti.i32,
) -> ti.i32:
    """
    GJK collision detection algorithm.
    Args:
        d: Dimension (2 for 2D, 3 for 3D)
    Returns: 1 if collision detected, 0 otherwise
    """
    # Simplex storage (max 4 points for 3D)
    simplex = ti.Matrix.zero(ti.f32, 4, 3)
    # Dummy witness storage for GJK only
    witness_a = ti.Matrix.zero(ti.f32, 4, 3)
    witness_b = ti.Matrix.zero(ti.f32, 4, 3)

    collision_detected, _ = run_gjk(
        shape_a_type,
        center_a,
        params_a,
        rot_a,
        shape_b_type,
        center_b,
        params_b,
        rot_b,
        simplex,
        witness_a,
        witness_b,
        d,
    )

    return collision_detected


@ti.func
def handle_line_gjk(
    simplex: ti.template(), witness_a: ti.template(), witness_b: ti.template(), direction: ti.math.vec3
) -> tuple:
    """Handle line segment simplex for GJK. Returns (size, direction, contains_origin).

    CRITICAL: If origin is ON the line segment (degenerate case), this indicates collision!
    This happens when cylinder caps or flat surfaces create coplanar Minkowski difference.
    """
    a = ti.math.vec3(simplex[1, 0], simplex[1, 1], simplex[1, 2])
    b = ti.math.vec3(simplex[0, 0], simplex[0, 1], simplex[0, 2])

    ab = b - a
    ao = -a

    new_size = 2
    new_dir = direction
    contains = 0

    if ab.dot(ao) > 0.0:
        new_dir = triple_product(ab, ao, ab)

        # CRITICAL FIX: If triple product is nearly zero, origin is ON the line segment
        # This is a degenerate case indicating collision (e.g., cap-to-cap contact)
        if new_dir.norm() < 1e-6:
            # Origin is on or very close to the line segment
            # Check distance from origin to line
            ab_len = ab.norm()
            if ab_len > 1e-10:
                # Project origin onto line segment
                t = ti.max(0.0, ti.min(1.0, ao.dot(ab) / (ab_len * ab_len)))
                closest_point = a + ab * t
                dist_to_line = closest_point.norm()

                # If origin is very close to the line segment, we have collision
                if dist_to_line < 1e-6:
                    contains = 1
                else:
                    # Use ao as fallback direction
                    new_dir = ao
            else:
                # Degenerate line (a == b), check if origin is at this point
                if ao.norm() < 1e-6:
                    contains = 1
                else:
                    new_dir = ao
    else:
        # Keep only vertex a, move it to position 0
        simplex[0, 0] = a.x
        simplex[0, 1] = a.y
        simplex[0, 2] = a.z
        # Also move its witness points
        for k in range(3):
            witness_a[0, k] = witness_a[1, k]
            witness_b[0, k] = witness_b[1, k]
        new_size = 1
        new_dir = ao

    # Normalize direction if we're continuing
    if contains == 0 and new_dir.norm() > 1e-10:
        new_dir = new_dir.normalized()
    elif contains == 0:
        new_dir = ti.math.vec3(1.0, 0.0, 0.0)

    return new_size, new_dir, contains


@ti.func
def handle_triangle_gjk(
    simplex: ti.template(), witness_a: ti.template(), witness_b: ti.template(), direction: ti.math.vec3, d: ti.i32
) -> ti.template():
    """Handle triangle simplex for GJK. Returns (size, direction, contains_origin).

    Args:
        d: Dimension (2 for 2D, 3 for 3D)
            - In 2D: triangle can contain origin, returns contains=1 if origin is inside
            - In 3D: if origin is ON the triangle plane (degenerate), also returns contains=1

    CRITICAL: If origin is ON the triangle (coplanar, degenerate case), this indicates collision!
    This happens when cylinder caps or flat surfaces create coplanar Minkowski difference.
    """
    a = ti.math.vec3(simplex[2, 0], simplex[2, 1], simplex[2, 2])
    b = ti.math.vec3(simplex[1, 0], simplex[1, 1], simplex[1, 2])
    c = ti.math.vec3(simplex[0, 0], simplex[0, 1], simplex[0, 2])

    ab = b - a
    ac = c - a
    ao = -a

    abc = ab.cross(ac)
    abc_norm = abc.norm()

    new_size = 3
    new_dir = direction
    contains = 0

    # Check which region origin is in
    if abc.cross(ac).dot(ao) > 0.0:
        # Origin is on the outside of AC edge
        if ac.dot(ao) > 0.0:
            # Region AC: origin is closest to AC edge
            simplex[0, 0] = c.x
            simplex[0, 1] = c.y
            simplex[0, 2] = c.z
            simplex[1, 0] = a.x
            simplex[1, 1] = a.y
            simplex[1, 2] = a.z
            # Update witness points: keep indices 0 and 2, move them to 0 and 1
            for k in range(3):
                witness_a[0, k] = witness_a[0, k]  # c stays at 0
                witness_b[0, k] = witness_b[0, k]
                witness_a[1, k] = witness_a[2, k]  # a moves from 2 to 1
                witness_b[1, k] = witness_b[2, k]
            new_size = 2
            new_dir = triple_product(ac, ao, ac)
        else:
            # Region A: origin is closest to point A
            simplex[0, 0] = a.x
            simplex[0, 1] = a.y
            simplex[0, 2] = a.z
            # Move witness from index 2 to 0
            for k in range(3):
                witness_a[0, k] = witness_a[2, k]
                witness_b[0, k] = witness_b[2, k]
            new_size = 1
            new_dir = ao
    else:
        if ab.cross(abc).dot(ao) > 0.0:
            # Origin is on the outside of AB edge
            if ab.dot(ao) > 0.0:
                # Region AB: origin is closest to AB edge
                simplex[0, 0] = b.x
                simplex[0, 1] = b.y
                simplex[0, 2] = b.z
                simplex[1, 0] = a.x
                simplex[1, 1] = a.y
                simplex[1, 2] = a.z
                # Keep indices 1 and 2, move them to 0 and 1
                for k in range(3):
                    witness_a[0, k] = witness_a[1, k]  # b moves from 1 to 0
                    witness_b[0, k] = witness_b[1, k]
                    witness_a[1, k] = witness_a[2, k]  # a moves from 2 to 1
                    witness_b[1, k] = witness_b[2, k]
                new_size = 2
                new_dir = triple_product(ab, ao, ab)
            else:
                # Region A: origin is closest to point A
                simplex[0, 0] = a.x
                simplex[0, 1] = a.y
                simplex[0, 2] = a.z
                # Move witness from index 2 to 0
                for k in range(3):
                    witness_a[0, k] = witness_a[2, k]
                    witness_b[0, k] = witness_b[2, k]
                new_size = 1
                new_dir = ao
        else:
            # Origin is inside the triangle edges (in 2D projection)
            # In 2D, if origin is in the triangle projection, we have collision
            contains = 1

    # Normalize direction if continuing
    if contains == 0 and new_dir.norm() > 1e-10:
        new_dir = new_dir.normalized()
    elif contains == 0:
        new_dir = ti.math.vec3(1.0, 0.0, 0.0)

    return new_size, new_dir, contains


# ==================== EPA Algorithm ====================


@ti.func
def epa_penetration(
    shape_a_type: ti.i32,
    center_a: ti.math.vec3,
    params_a: ti.math.vec4,
    rot_a: ti.math.mat3,
    shape_b_type: ti.i32,
    center_b: ti.math.vec3,
    params_b: ti.math.vec4,
    rot_b: ti.math.mat3,
    simplex: ti.template(),
    simplex_witness_a: ti.template(),
    simplex_witness_b: ti.template(),
    simplex_size: ti.i32,
    d: ti.i32,
) -> ti.template():
    """
    EPA (Expanding Polytope Algorithm) for penetration depth calculation.
    Args:
        simplex: Minkowski difference vertices from GJK
        simplex_witness_a: Witness points on shape A (corresponding to simplex vertices)
        simplex_witness_b: Witness points on shape B (corresponding to simplex vertices)
        simplex_size: Number of vertices in simplex
        d: Dimension (2 for 2D, 3 for 3D)
    Returns: (penetration_depth, contact_normal, contact_point_on_A, contact_point_on_B, success)
    """
    # Use ti.static for compile-time constants
    max_vertices = ti.static(32)
    max_faces = ti.static(64)
    max_iter = ti.static(32)

    penetration_depth = 0.0
    contact_normal = ti.math.vec3(0.0, 0.0, 1.0)
    contact_point_a = center_a
    contact_point_b = center_b
    success = 1

    # ==================== 2D EPA: Polygon expansion ====================
    if d == 2:
        # In 2D, EPA works with edges of a polygon (not faces of a polyhedron)
        # Simplex is a triangle (3 points) in 2D
        vertices_2d = ti.Matrix.zero(ti.f32, max_vertices, 2)
        num_vertices = 0

        # Store witness points on A and B for each Minkowski vertex
        # This enables barycentric interpolation for accurate contact points
        witness_a = ti.Matrix.zero(ti.f32, max_vertices, 3)
        witness_b = ti.Matrix.zero(ti.f32, max_vertices, 3)

        # Initialize with simplex vertices (only x, y) and their witness points
        for i in range(simplex_size):
            vertices_2d[num_vertices, 0] = simplex[i, 0]
            vertices_2d[num_vertices, 1] = simplex[i, 1]

            # Use witness points from GJK phase (no need to recompute!)
            witness_a[num_vertices, 0] = simplex_witness_a[i, 0]
            witness_a[num_vertices, 1] = simplex_witness_a[i, 1]
            witness_a[num_vertices, 2] = simplex_witness_a[i, 2]
            witness_b[num_vertices, 0] = simplex_witness_b[i, 0]
            witness_b[num_vertices, 1] = simplex_witness_b[i, 1]
            witness_b[num_vertices, 2] = simplex_witness_b[i, 2]

            num_vertices += 1

        # Edges storage (each edge has 2 vertex indices)
        edges = ti.Matrix.zero(ti.i32, max_vertices, 2)
        edge_normals = ti.Matrix.zero(ti.f32, max_vertices, 2)
        edge_distances = ti.Vector.zero(ti.f32, max_vertices)
        num_edges = 0

        # Initialize triangle edges (in CCW order)
        if simplex_size == 3:
            for i in range(3):
                edges[i, 0] = i
                edges[i, 1] = (i + 1) % 3
            num_edges = 3

        # Compute edge normals and distances
        for i in range(num_edges):
            idx_a = edges[i, 0]
            idx_b = edges[i, 1]

            va = ti.math.vec2(vertices_2d[idx_a, 0], vertices_2d[idx_a, 1])
            vb = ti.math.vec2(vertices_2d[idx_b, 0], vertices_2d[idx_b, 1])

            edge = vb - va
            # 2D perpendicular: (x, y) -> (-y, x)
            normal_2d = ti.math.vec2(-edge.y, edge.x)
            normal_len = normal_2d.norm()

            if normal_len > 1e-10:
                normal_2d = normal_2d / normal_len

            # Ensure normal points away from origin (outward)
            # In EPA, origin is inside the polygon, so normal.dot(vertex) should be positive
            if normal_2d.dot(va) < 0.0:
                normal_2d = -normal_2d

            edge_normals[i, 0] = normal_2d.x
            edge_normals[i, 1] = normal_2d.y
            # Distance from origin to edge (should be positive)
            edge_distances[i] = normal_2d.dot(va)

        # EPA iterations for 2D
        epa_continue = 1
        for iteration in range(max_iter):
            if epa_continue == 1:
                # Find closest edge
                closest_edge = 0
                min_distance = edge_distances[0]

                for i in range(1, num_edges):
                    if edge_distances[i] < min_distance:
                        min_distance = edge_distances[i]
                        closest_edge = i

                # Get support point in direction of closest edge normal
                search_dir_2d = ti.math.vec2(edge_normals[closest_edge, 0], edge_normals[closest_edge, 1])
                search_dir_3d = ti.math.vec3(search_dir_2d.x, search_dir_2d.y, 0.0)

                support_point, support_a_new, support_b_new = support_minkowski_diff(
                    shape_a_type, center_a, params_a, rot_a, shape_b_type, center_b, params_b, rot_b, search_dir_3d
                )
                support_point_2d = ti.math.vec2(support_point.x, support_point.y)

                # Check if we've found the edge
                support_distance = support_point_2d.dot(search_dir_2d)

                if support_distance - min_distance < 1e-4:
                    # Converged
                    penetration_depth = min_distance
                    contact_normal = ti.math.vec3(search_dir_2d.x, search_dir_2d.y, 0.0)
                    epa_continue = 0
                elif num_vertices >= max_vertices:
                    # Vertex limit reached
                    success = 0
                    epa_continue = 0
                else:
                    # Add new vertex
                    vertices_2d[num_vertices, 0] = support_point_2d.x
                    vertices_2d[num_vertices, 1] = support_point_2d.y

                    # Store witness points for this new vertex
                    witness_a[num_vertices, 0] = support_a_new[0]
                    witness_a[num_vertices, 1] = support_a_new[1]
                    witness_a[num_vertices, 2] = support_a_new[2]
                    witness_b[num_vertices, 0] = support_b_new[0]
                    witness_b[num_vertices, 1] = support_b_new[1]
                    witness_b[num_vertices, 2] = support_b_new[2]

                    new_vertex_idx = num_vertices
                    num_vertices += 1

                    # Insert new vertex into the polygon by splitting the closest edge
                    # Remove closest edge and add two new edges
                    edge_start = edges[closest_edge, 0]
                    edge_end = edges[closest_edge, 1]

                    # Replace closest edge with first new edge
                    edges[closest_edge, 0] = edge_start
                    edges[closest_edge, 1] = new_vertex_idx

                    # Add second new edge at the end
                    if num_edges < max_vertices:
                        edges[num_edges, 0] = new_vertex_idx
                        edges[num_edges, 1] = edge_end
                        num_edges += 1

                    # Recompute edge normals and distances for modified edges
                    for i in range(num_edges):
                        idx_a = edges[i, 0]
                        idx_b = edges[i, 1]

                        va = ti.math.vec2(vertices_2d[idx_a, 0], vertices_2d[idx_a, 1])
                        vb = ti.math.vec2(vertices_2d[idx_b, 0], vertices_2d[idx_b, 1])

                        edge = vb - va
                        normal_2d = ti.math.vec2(-edge.y, edge.x)
                        normal_len = normal_2d.norm()

                        if normal_len > 1e-10:
                            normal_2d = normal_2d / normal_len

                        if normal_2d.dot(va) < 0.0:
                            normal_2d = -normal_2d

                        edge_normals[i, 0] = normal_2d.x
                        edge_normals[i, 1] = normal_2d.y
                        edge_distances[i] = ti.abs(normal_2d.dot(va))

        # Compute contact points for 2D using barycentric coordinates
        closest_edge = 0
        min_distance = edge_distances[0]
        for i in range(1, num_edges):
            if edge_distances[i] < min_distance:
                min_distance = edge_distances[i]
                closest_edge = i

        # Get closest edge vertices (indices in the Minkowski polytope)
        idx_a = edges[closest_edge, 0]
        idx_b = edges[closest_edge, 1]

        va_2d = ti.math.vec2(vertices_2d[idx_a, 0], vertices_2d[idx_a, 1])
        vb_2d = ti.math.vec2(vertices_2d[idx_b, 0], vertices_2d[idx_b, 1])

        # Project origin onto the closest edge to find barycentric coordinates
        # In 2D, we have an edge (line segment), so we need parameter t ∈ [0,1]
        # v = (1-t) * va + t * vb, where v is the closest point to origin
        edge_vec = vb_2d - va_2d
        edge_len_sq = edge_vec.dot(edge_vec)

        t = 0.0  # Barycentric coordinate
        if edge_len_sq > 1e-10:
            t = ti.max(0.0, ti.min(1.0, ((-va_2d).dot(edge_vec)) / edge_len_sq))

        # Barycentric weights for the edge
        c1 = 1.0 - t  # Weight for vertex idx_a
        c2 = t  # Weight for vertex idx_b

        # Get witness points on A and B for each Minkowski vertex
        wa1 = ti.math.vec3(witness_a[idx_a, 0], witness_a[idx_a, 1], witness_a[idx_a, 2])
        wb1 = ti.math.vec3(witness_b[idx_a, 0], witness_b[idx_a, 1], witness_b[idx_a, 2])
        wa2 = ti.math.vec3(witness_a[idx_b, 0], witness_a[idx_b, 1], witness_a[idx_b, 2])
        wb2 = ti.math.vec3(witness_b[idx_b, 0], witness_b[idx_b, 1], witness_b[idx_b, 2])

        # Compute contact points using barycentric interpolation
        # A* = c1*A1 + c2*A2
        # B* = c1*B1 + c2*B2
        contact_point_a = c1 * wa1 + c2 * wa2
        contact_point_b = c1 * wb1 + c2 * wb2

    # ==================== 3D EPA: Polyhedron expansion ====================
    else:
        # Polytope vertices
        vertices = ti.Matrix.zero(ti.f32, max_vertices, 3)
        num_vertices = 0

        # Store witness points on A and B for each Minkowski vertex
        witness_a = ti.Matrix.zero(ti.f32, max_vertices, 3)
        witness_b = ti.Matrix.zero(ti.f32, max_vertices, 3)

        # Initialize with simplex vertices and their witness points from GJK
        for i in range(simplex_size):
            vertices[num_vertices, 0] = simplex[i, 0]
            vertices[num_vertices, 1] = simplex[i, 1]
            vertices[num_vertices, 2] = simplex[i, 2]

            # Use witness points from GJK phase (no need to recompute!)
            witness_a[num_vertices, 0] = simplex_witness_a[i, 0]
            witness_a[num_vertices, 1] = simplex_witness_a[i, 1]
            witness_a[num_vertices, 2] = simplex_witness_a[i, 2]
            witness_b[num_vertices, 0] = simplex_witness_b[i, 0]
            witness_b[num_vertices, 1] = simplex_witness_b[i, 1]
            witness_b[num_vertices, 2] = simplex_witness_b[i, 2]

            num_vertices += 1

        # Faces storage
        faces = ti.Matrix.zero(ti.i32, max_faces, 3)  # Each face has 3 vertex indices
        face_normals = ti.Matrix.zero(ti.f32, max_faces, 3)
        face_distances = ti.Vector.zero(ti.f32, max_faces)
        num_faces = 0

        # Initialize tetrahedron faces
        if simplex_size == 4:
            # Face ABC
            faces[0, 0] = 0
            faces[0, 1] = 1
            faces[0, 2] = 2
            # Face ACD
            faces[1, 0] = 0
            faces[1, 1] = 2
            faces[1, 2] = 3
            # Face ADB
            faces[2, 0] = 0
            faces[2, 1] = 3
            faces[2, 2] = 1
            # Face BCD
            faces[3, 0] = 1
            faces[3, 1] = 3
            faces[3, 2] = 2
            num_faces = 4

        # Compute face normals and distances
        for i in range(num_faces):
            idx_a = faces[i, 0]
            idx_b = faces[i, 1]
            idx_c = faces[i, 2]

            va = ti.math.vec3(vertices[idx_a, 0], vertices[idx_a, 1], vertices[idx_a, 2])
            vb = ti.math.vec3(vertices[idx_b, 0], vertices[idx_b, 1], vertices[idx_b, 2])
            vc = ti.math.vec3(vertices[idx_c, 0], vertices[idx_c, 1], vertices[idx_c, 2])

            ab = vb - va
            ac = vc - va
            normal = ab.cross(ac)
            normal_len = normal.norm()

            if normal_len > 1e-10:
                normal = normal / normal_len

            # Ensure normal points outward (away from origin)
            if normal.dot(va) < 0.0:
                normal = -normal
                # Swap b and c to maintain winding
                faces[i, 1] = idx_c
                faces[i, 2] = idx_b

            face_normals[i, 0] = normal.x
            face_normals[i, 1] = normal.y
            face_normals[i, 2] = normal.z
            face_distances[i] = ti.abs(normal.dot(va))

        # EPA iterations for 3D
        epa_continue = 1
        for iteration in range(max_iter):
            if epa_continue == 1:
                # Find closest face
                closest_face = 0
                min_distance = face_distances[0]

                for i in range(1, num_faces):
                    if face_distances[i] < min_distance:
                        min_distance = face_distances[i]
                        closest_face = i

                # Get support point in direction of closest face normal
                search_dir = ti.math.vec3(
                    face_normals[closest_face, 0], face_normals[closest_face, 1], face_normals[closest_face, 2]
                )

                support_point, support_a_new, support_b_new = support_minkowski_diff(
                    shape_a_type, center_a, params_a, rot_a, shape_b_type, center_b, params_b, rot_b, search_dir
                )

                # Check if we've found the edge
                support_distance = support_point.dot(search_dir)

                if support_distance - min_distance < 1e-4:
                    # Converged
                    penetration_depth = min_distance
                    contact_normal = search_dir
                    epa_continue = 0
                elif num_vertices >= max_vertices:
                    # Vertex limit reached
                    success = 0
                    epa_continue = 0
                else:
                    # Add new vertex
                    vertices[num_vertices, 0] = support_point.x
                    vertices[num_vertices, 1] = support_point.y
                    vertices[num_vertices, 2] = support_point.z

                    # Store witness points for this new vertex (already computed above)
                    witness_a[num_vertices, 0] = support_a_new[0]
                    witness_a[num_vertices, 1] = support_a_new[1]
                    witness_a[num_vertices, 2] = support_a_new[2]
                    witness_b[num_vertices, 0] = support_b_new[0]
                    witness_b[num_vertices, 1] = support_b_new[1]
                    witness_b[num_vertices, 2] = support_b_new[2]

                    new_vertex_idx = num_vertices
                    num_vertices += 1

                    # Expand polytope: remove faces visible from new point and add new faces
                    # Mark visible faces
                    visible = ti.Vector.zero(ti.i32, max_faces)
                    for i in range(num_faces):
                        normal = ti.math.vec3(face_normals[i, 0], face_normals[i, 1], face_normals[i, 2])
                        va = ti.math.vec3(vertices[faces[i, 0], 0], vertices[faces[i, 0], 1], vertices[faces[i, 0], 2])
                        if normal.dot(support_point - va) > 0.0:
                            visible[i] = 1

                    # Find boundary edges and create new faces
                    # (Simplified: rebuild faces from scratch)
                    new_faces = ti.Matrix.zero(ti.i32, max_faces, 3)
                    new_num_faces = 0

                    # Keep non-visible faces
                    for i in range(num_faces):
                        if visible[i] == 0:
                            new_faces[new_num_faces, 0] = faces[i, 0]
                            new_faces[new_num_faces, 1] = faces[i, 1]
                            new_faces[new_num_faces, 2] = faces[i, 2]
                            new_num_faces += 1

                    # Add new faces connecting to new vertex
                    # (Simplified boundary edge detection)
                    for i in range(num_faces):
                        if visible[i] == 1:
                            # Add faces from each edge to new vertex
                            for j in range(3):
                                edge_start = faces[i, j]
                                edge_end = faces[i, (j + 1) % 3]

                                # Check if this edge is on the boundary (only in one visible face)
                                is_boundary = 1
                                for k in range(num_faces):
                                    if k != i and visible[k] == 1:
                                        # Check if edge is shared
                                        edge_found = 0
                                        for m in range(3):
                                            e_start = faces[k, m]
                                            e_end = faces[k, (m + 1) % 3]
                                            if (e_start == edge_start and e_end == edge_end) or (
                                                e_start == edge_end and e_end == edge_start
                                            ):
                                                is_boundary = 0
                                                edge_found = 1
                                                break
                                        if edge_found == 1:
                                            break

                                if is_boundary == 1 and new_num_faces < max_faces:
                                    new_faces[new_num_faces, 0] = edge_start
                                    new_faces[new_num_faces, 1] = edge_end
                                    new_faces[new_num_faces, 2] = new_vertex_idx
                                    new_num_faces += 1

                    # Update faces
                    num_faces = new_num_faces
                    for i in range(num_faces):
                        faces[i, 0] = new_faces[i, 0]
                        faces[i, 1] = new_faces[i, 1]
                        faces[i, 2] = new_faces[i, 2]

                    # Recompute face normals and distances
                    for i in range(num_faces):
                        idx_a = faces[i, 0]
                        idx_b = faces[i, 1]
                        idx_c = faces[i, 2]

                        va = ti.math.vec3(vertices[idx_a, 0], vertices[idx_a, 1], vertices[idx_a, 2])
                        vb = ti.math.vec3(vertices[idx_b, 0], vertices[idx_b, 1], vertices[idx_b, 2])
                        vc = ti.math.vec3(vertices[idx_c, 0], vertices[idx_c, 1], vertices[idx_c, 2])

                        ab = vb - va
                        ac = vc - va
                        normal = ab.cross(ac)
                        normal_len = normal.norm()

                        if normal_len > 1e-10:
                            normal = normal / normal_len

                        if normal.dot(va) < 0.0:
                            normal = -normal

                        face_normals[i, 0] = normal.x
                        face_normals[i, 1] = normal.y
                        face_normals[i, 2] = normal.z
                        face_distances[i] = ti.abs(normal.dot(va))
            # else:
            #     print("EPA terminated at iteration:", iteration)

        # Compute contact points for 3D using barycentric coordinates
        # Find closest face (already computed by EPA)
        closest_face = 0
        min_distance = face_distances[0]
        for i in range(1, num_faces):
            if face_distances[i] < min_distance:
                min_distance = face_distances[i]
                closest_face = i

        # Get closest face vertices (indices in the Minkowski polytope)
        idx_a = faces[closest_face, 0]
        idx_b = faces[closest_face, 1]
        idx_c = faces[closest_face, 2]

        pa = ti.math.vec3(vertices[idx_a, 0], vertices[idx_a, 1], vertices[idx_a, 2])
        pb = ti.math.vec3(vertices[idx_b, 0], vertices[idx_b, 1], vertices[idx_b, 2])
        pc = ti.math.vec3(vertices[idx_c, 0], vertices[idx_c, 1], vertices[idx_c, 2])

        # Project origin onto the closest face triangle to find barycentric coordinates
        # v = c1*pa + c2*pb + c3*pc, where c1 + c2 + c3 = 1
        ab = pb - pa
        ac = pc - pa
        ap = -pa  # Vector from pa to origin

        d1 = ab.dot(ap)
        d2 = ac.dot(ap)

        # Compute barycentric coordinates using Voronoi regions
        # Initialize with vertex A
        c1 = 1.0
        c2 = 0.0
        c3 = 0.0

        if d1 <= 0.0 and d2 <= 0.0:
            # Vertex region A
            c1 = 1.0
            c2 = 0.0
            c3 = 0.0
        else:
            bp = -pb
            d3 = ab.dot(bp)
            d4 = ac.dot(bp)

            if d3 >= 0.0 and d4 <= d3:
                # Vertex region B
                c1 = 0.0
                c2 = 1.0
                c3 = 0.0
            else:
                vc = d1 * d4 - d3 * d2
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    # Edge region AB
                    v = d1 / (d1 - d3)
                    c1 = 1.0 - v
                    c2 = v
                    c3 = 0.0
                else:
                    cp = -pc
                    d5 = ab.dot(cp)
                    d6 = ac.dot(cp)

                    if d6 >= 0.0 and d5 <= d6:
                        # Vertex region C
                        c1 = 0.0
                        c2 = 0.0
                        c3 = 1.0
                    else:
                        vb = d5 * d2 - d1 * d6
                        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                            # Edge region AC
                            w = d2 / (d2 - d6)
                            c1 = 1.0 - w
                            c2 = 0.0
                            c3 = w
                        else:
                            va = d3 * d6 - d5 * d4
                            if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                                # Edge region BC
                                w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                                c1 = 0.0
                                c2 = 1.0 - w
                                c3 = w
                            else:
                                # Inside face region
                                denom = 1.0 / (va + vb + vc)
                                v = vb * denom
                                w = vc * denom
                                c1 = 1.0 - v - w
                                c2 = v
                                c3 = w

        # Get witness points on A and B for each Minkowski vertex
        wa1 = ti.math.vec3(witness_a[idx_a, 0], witness_a[idx_a, 1], witness_a[idx_a, 2])
        wb1 = ti.math.vec3(witness_b[idx_a, 0], witness_b[idx_a, 1], witness_b[idx_a, 2])
        wa2 = ti.math.vec3(witness_a[idx_b, 0], witness_a[idx_b, 1], witness_a[idx_b, 2])
        wb2 = ti.math.vec3(witness_b[idx_b, 0], witness_b[idx_b, 1], witness_b[idx_b, 2])
        wa3 = ti.math.vec3(witness_a[idx_c, 0], witness_a[idx_c, 1], witness_a[idx_c, 2])
        wb3 = ti.math.vec3(witness_b[idx_c, 0], witness_b[idx_c, 1], witness_b[idx_c, 2])

        # Compute contact points using barycentric interpolation
        # Each Minkowski vertex M_i = A_i - B_i has witness points (A_i, B_i)
        # Contact points: A* = Σc_i·A_i, B* = Σc_i·B_i
        contact_point_a = c1 * wa1 + c2 * wa2 + c3 * wa3
        contact_point_b = c1 * wb1 + c2 * wb2 + c3 * wb3
    return penetration_depth, contact_normal, contact_point_a, contact_point_b, success


# ==================== Combined GJK+EPA ====================


@ti.func
def gjk_epa_collision(
    shape_a_type: ti.i32,
    center_a: ti.math.vec3,
    params_a: ti.math.vec4,
    rot_a: ti.math.mat3,
    shape_b_type: ti.i32,
    center_b: ti.math.vec3,
    params_b: ti.math.vec4,
    rot_b: ti.math.mat3,
    d: ti.i32,
) -> ti.template():
    """
    Combined GJK+EPA collision detection and penetration depth calculation.
    Args:
        d: Dimension (2 for 2D, 3 for 3D)
            - In 2D: triangle simplex (3 points) is sufficient to confirm collision
            - In 3D: tetrahedron simplex (4 points) is needed to confirm collision
    Returns: (has_collision, penetration_depth, contact_normal, contact_point_a, contact_point_b)
    """
    has_collision = 0
    penetration_depth = 0.0
    contact_normal = ti.math.vec3(0.0, 0.0, 1.0)
    contact_point_a = center_a
    contact_point_b = center_b

    simplex = ti.Matrix.zero(ti.f32, 4, 3)
    # CRITICAL: Store witness points during GJK phase
    simplex_witness_a = ti.Matrix.zero(ti.f32, 4, 3)
    simplex_witness_b = ti.Matrix.zero(ti.f32, 4, 3)

    has_collision, simplex_size = run_gjk(
        shape_a_type,
        center_a,
        params_a,
        rot_a,
        shape_b_type,
        center_b,
        params_b,
        rot_b,
        simplex,
        simplex_witness_a,
        simplex_witness_b,
        d,
    )
    # If collision detected, run EPA to get penetration info
    if has_collision == 1:
        penetration_depth, contact_normal, contact_point_a, contact_point_b, success = epa_penetration(
            shape_a_type,
            center_a,
            params_a,
            rot_a,
            shape_b_type,
            center_b,
            params_b,
            rot_b,
            simplex,
            simplex_witness_a,
            simplex_witness_b,
            simplex_size,
            d,
        )

    return has_collision, penetration_depth, contact_normal, contact_point_a, contact_point_b

