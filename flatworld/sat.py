import warp as wp


# ===== 2D SAT helpers (explicit coordinates) =====
@wp.func
def perp2(v: wp.vec2):
    return wp.vec2(-v[1], v[0])


@wp.func
def sat2d_project_quad_on_axis(v0: wp.vec2, v1: wp.vec2, v2: wp.vec2, v3: wp.vec2, axis: wp.vec2):
    p0 = wp.dot(v0, axis)
    p1 = wp.dot(v1, axis)
    p2 = wp.dot(v2, axis)
    p3 = wp.dot(v3, axis)
    mn = wp.min(wp.min(p0, p1), wp.min(p2, p3))
    mx = wp.max(wp.max(p0, p1), wp.max(p2, p3))
    return mn, mx


@wp.func
def sat2d_project_segment_on_axis(s0: wp.vec2, s1: wp.vec2, axis: wp.vec2):
    p0 = wp.dot(s0, axis)
    p1 = wp.dot(s1, axis)
    mn = wp.min(p0, p1)
    mx = wp.max(p0, p1)
    return mn, mx


@wp.func
def sat2d_project_triangle_on_axis(t0: wp.vec2, t1: wp.vec2, t2: wp.vec2, axis: wp.vec2):
    p0 = wp.dot(t0, axis)
    p1 = wp.dot(t1, axis)
    p2 = wp.dot(t2, axis)
    mn = wp.min(p0, wp.min(p1, p2))
    mx = wp.max(p0, wp.max(p1, p2))
    return mn, mx


@wp.func
def _sat2d_eval_intervals(amin: float, amax: float, bmin: float, bmax: float, axis: wp.vec2):
    # separation distances (positive if separated)
    d1 = amin - bmax
    d2 = bmin - amax
    sep = float(0.0)
    side = int(0)
    if d1 > 0.0 or d2 > 0.0:
        if d1 > d2:
            sep = d1
            side = +1
        else:
            sep = d2
            side = -1
    pen = float(0.0)
    if sep <= 0.0:
        pen_pos = bmax - amin
        pen_neg = amax - bmin
        pen = wp.min(pen_pos, pen_neg)
    # normalize axis
    ax = axis
    ln = wp.length(ax)
    if ln > 1e-12:
        ax = ax / ln
    return sep, pen, ax, side


@wp.func
def obb2d_signed_distance_quad_vs_quad(
    a0: wp.vec2, a1: wp.vec2, a2: wp.vec2, a3: wp.vec2, b0: wp.vec2, b1: wp.vec2, b2: wp.vec2, b3: wp.vec2
):
    """2D SAT collision detection between two quads (OBB vs OBB).

    For rectangles, we need to test the perpendicular directions of the edges.
    Vertices should be ordered counter-clockwise: a0, a1, a2, a3
    """
    # Candidate axes: normals of both quads' edges
    ca = (a0 + a1 + a2 + a3) * 0.25
    cb = (b0 + b1 + b2 + b3) * 0.25
    # Use a flag to record whether we found a separating axis.
    # Initialize best_sep to a very small value and best_pen to large.
    found_separating = int(0)
    best_sep = float(-1e9)
    best_sep_axis = wp.vec2(1.0, 0.0)
    best_sep_side = int(0)
    best_pen = float(1e9)
    best_pen_axis = wp.vec2(1.0, 0.0)

    # Test edge normals from quad A
    # For a rectangle, we test two perpendicular edges (a0->a1 and a1->a2)
    # This covers both normal directions
    for i in range(2):
        e = a1 - a0 if i == 0 else a2 - a1
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_quad_on_axis(b0, b1, b2, b3, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = 1
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # Test edge normals from quad B
    for i in range(2):
        e = b1 - b0 if i == 0 else b2 - b1
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_quad_on_axis(b0, b1, b2, b3, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = 1
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    signed = float(0.0)
    n_axis = wp.vec2(1.0, 0.0)
    if found_separating:
        # Separated
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        # Penetrating (choose minimum penetration axis)
        n_axis = best_pen_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen
    return signed, n_axis


@wp.func
def _obb2d_corners(center: wp.vec2, half: wp.vec2, rot: wp.mat22):
    ax = wp.vec2(rot[0, 0], rot[1, 0]) * half[0]
    ay = wp.vec2(rot[0, 1], rot[1, 1]) * half[1]
    a0 = center - ax - ay
    a1 = center + ax - ay
    a2 = center + ax + ay
    a3 = center - ax + ay
    return a0, a1, a2, a3


@wp.func
def _quad_support(a0: wp.vec2, a1: wp.vec2, a2: wp.vec2, a3: wp.vec2, direction: wp.vec2):
    best = a0
    best_d = wp.dot(a0, direction)
    d1 = wp.dot(a1, direction)
    if d1 > best_d:
        best = a1
        best_d = d1
    d2 = wp.dot(a2, direction)
    if d2 > best_d:
        best = a2
        best_d = d2
    d3 = wp.dot(a3, direction)
    if d3 > best_d:
        best = a3
    return best


@wp.func
def obb2d_contact_quad_vs_quad(
    center_a: wp.vec2, half_ext_a: wp.vec2, rot_a: wp.mat22, center_b: wp.vec2, half_ext_b: wp.vec2, rot_b: wp.mat22
):
    """Resolve penetrating 2D OBB-OBB contact via SAT (depth, normal A→B, midpoint)."""
    a0, a1, a2, a3 = _obb2d_corners(center_a, half_ext_a, rot_a)
    b0, b1, b2, b3 = _obb2d_corners(center_b, half_ext_b, rot_b)
    signed, n_axis = obb2d_signed_distance_quad_vs_quad(a0, a1, a2, a3, b0, b1, b2, b3)
    hit = int(0)
    penetration = float(0.0)
    normal = n_axis
    cpoint = (center_a + center_b) * 0.5
    if signed < 0.0:
        hit = 1
        nlen = wp.length(n_axis)
        if nlen > 1e-9:
            normal = n_axis / nlen
        else:
            diff = center_b - center_a
            dlen = wp.length(diff)
            if dlen > 1e-9:
                normal = diff / dlen
            else:
                normal = wp.vec2(1.0, 0.0)
        penetration = signed
        if wp.dot(center_b - center_a, normal) < 0.0:
            normal = -normal
        sa = _quad_support(a0, a1, a2, a3, normal)
        sb = _quad_support(b0, b1, b2, b3, -normal)
        cpoint = (sa + sb) * 0.5
    return hit, penetration, normal, cpoint


@wp.func
def obb2d_signed_distance_quad_vs_segment(
    a0: wp.vec2, a1: wp.vec2, a2: wp.vec2, a3: wp.vec2, s0: wp.vec2, s1: wp.vec2
):
    ca = (a0 + a1 + a2 + a3) * 0.25
    cb = (s0 + s1) * 0.5
    found_separating = int(0)
    best_sep = float(-1e9)
    best_sep_axis = wp.vec2(1.0, 0.0)
    best_sep_side = int(0)
    best_pen = float(1e9)
    best_pen_axis = wp.vec2(1.0, 0.0)

    # axes from quad
    for i in range(2):
        e = a1 - a0 if i == 0 else a3 - a0
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_segment_on_axis(s0, s1, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = 1
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # axis from segment (its normal)
    axis = perp2(s1 - s0)
    amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
    bmin, bmax = sat2d_project_segment_on_axis(s0, s1, axis)
    sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
    if sep > 0.0:
        found_separating = 1
        if sep > best_sep:
            best_sep = sep
            best_sep_axis = ax
            best_sep_side = side
    elif pen < best_pen:
        best_pen = pen
        best_pen_axis = ax

    signed = float(0.0)
    n_axis = wp.vec2(1.0, 0.0)
    if found_separating:
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        n_axis = best_pen_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen
    return signed, n_axis


@wp.func
def obb2d_signed_distance_quad_vs_triangle(
    a0: wp.vec2, a1: wp.vec2, a2: wp.vec2, a3: wp.vec2, t0: wp.vec2, t1: wp.vec2, t2: wp.vec2
):
    ca = (a0 + a1 + a2 + a3) * 0.25
    cb = (t0 + t1 + t2) / 3.0
    found_separating = int(0)
    best_sep = float(-1e9)
    best_sep_axis = wp.vec2(1.0, 0.0)
    best_sep_side = int(0)
    best_pen = float(1e9)
    best_pen_axis = wp.vec2(1.0, 0.0)

    # axes from quad
    for i in range(2):
        e = a1 - a0 if i == 0 else a3 - a0
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_triangle_on_axis(t0, t1, t2, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = 1
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # axes from triangle
    for i in range(3):
        e = t1 - t0 if i == 0 else (t2 - t1 if i == 1 else t0 - t2)
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_triangle_on_axis(t0, t1, t2, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = 1
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    signed = float(0.0)
    n_axis = wp.vec2(1.0, 0.0)
    if found_separating:
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        n_axis = best_pen_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen
    return signed, n_axis


@wp.func
def obb2d_signed_distance_quad_vs_circle(
    a0: wp.vec2, a1: wp.vec2, a2: wp.vec2, a3: wp.vec2, circle_center: wp.vec2, radius: float
):
    """2D SAT collision detection between an OBB (quad) and a circle.

    Args:
        a0, a1, a2, a3: quad vertices (counter-clockwise)
        circle_center: center of circle
        radius: radius of circle

    Returns:
        (signed_distance, normal):
        - signed_distance > 0: separated
        - signed_distance < 0: penetrating
        - normal: points from quad to circle
    """
    ca = (a0 + a1 + a2 + a3) * 0.25
    cb = circle_center

    found_separating = int(0)
    best_sep = float(-1e9)
    best_sep_axis = wp.vec2(1.0, 0.0)
    best_sep_side = int(0)
    best_pen = float(1e9)
    best_pen_axis = wp.vec2(1.0, 0.0)

    # Test quad edge normals (test two perpendicular edges: a0->a1 and a1->a2)
    for i in range(2):
        e = a1 - a0 if i == 0 else a2 - a1
        axis = perp2(e)

        # Project quad onto axis
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)

        # Project circle onto axis (center ± radius)
        c_proj = wp.dot(circle_center, axis)
        ax_norm = wp.length(axis)
        r_proj = radius * ax_norm if ax_norm > 1e-12 else 0.0
        bmin = c_proj - r_proj
        bmax = c_proj + r_proj

        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = 1
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # Find closest point on quad to circle center (Voronoi region test)
    # Test axis from circle center to closest point on quad
    closest_point = a0
    min_dist_sq = wp.length_sq(a0 - circle_center)

    for i in range(4):
        v = a0 if i == 0 else (a1 if i == 1 else (a2 if i == 2 else a3))
        dist_sq = wp.length_sq(v - circle_center)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_point = v

    # Check edges for closest point
    for i in range(4):
        v0 = a0 if i == 0 else (a1 if i == 1 else (a2 if i == 2 else a3))
        v1 = a1 if i == 0 else (a2 if i == 1 else (a3 if i == 2 else a0))

        # Project circle center onto edge
        edge = v1 - v0
        edge_len_sq = wp.length_sq(edge)
        if edge_len_sq > 1e-12:
            t = wp.dot(circle_center - v0, edge) / edge_len_sq
            t = wp.max(0.0, wp.min(1.0, t))  # Clamp to [0, 1]
            point_on_edge = v0 + t * edge
            dist_sq = wp.length_sq(point_on_edge - circle_center)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = point_on_edge

    # Test axis from circle center to closest point on quad
    axis = circle_center - closest_point
    axis_norm = wp.length(axis)
    if axis_norm > 1e-9:
        axis = axis / axis_norm

        # Project quad onto axis
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)

        # Project circle onto axis
        c_proj = wp.dot(circle_center, axis)
        bmin = c_proj - radius
        bmax = c_proj + radius

        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = 1
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # Determine signed distance and normal
    signed = float(0.0)
    n_axis = wp.vec2(1.0, 0.0)

    if found_separating:
        # Separated
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        # Penetrating
        n_axis = best_pen_axis
        if wp.dot(ca - cb, n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen

    return signed, n_axis
