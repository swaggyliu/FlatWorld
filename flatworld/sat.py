import taichi as ti


# ===== 2D SAT helpers (explicit coordinates) =====
@ti.func
def perp2(v):
    return ti.Vector([-v[1], v[0]])


@ti.func
def sat2d_project_quad_on_axis(v0, v1, v2, v3, axis):
    p0 = v0.dot(axis)
    p1 = v1.dot(axis)
    p2 = v2.dot(axis)
    p3 = v3.dot(axis)
    mn = ti.min(ti.min(p0, p1), ti.min(p2, p3))
    mx = ti.max(ti.max(p0, p1), ti.max(p2, p3))
    return mn, mx


@ti.func
def sat2d_project_segment_on_axis(s0, s1, axis):
    p0 = s0.dot(axis)
    p1 = s1.dot(axis)
    mn = ti.min(p0, p1)
    mx = ti.max(p0, p1)
    return mn, mx


@ti.func
def sat2d_project_triangle_on_axis(t0, t1, t2, axis):
    p0 = t0.dot(axis)
    p1 = t1.dot(axis)
    p2 = t2.dot(axis)
    mn = ti.min(p0, ti.min(p1, p2))
    mx = ti.max(p0, ti.max(p1, p2))
    return mn, mx


@ti.func
def _sat2d_eval_intervals(amin, amax, bmin, bmax, axis):
    # separation distances (positive if separated)
    d1 = amin - bmax
    d2 = bmin - amax
    sep = 0.0
    side = 0
    if d1 > 0.0 or d2 > 0.0:
        if d1 > d2:
            sep = d1
            side = +1
        else:
            sep = d2
            side = -1
    pen = 0.0
    if sep <= 0.0:
        pen_pos = bmax - amin
        pen_neg = amax - bmin
        pen = ti.min(pen_pos, pen_neg)
    # normalize axis
    ax = axis
    ln = ax.norm()
    if ln > 1e-12:
        ax = ax / ln
    return sep, pen, ax, side


@ti.func
def obb2d_signed_distance_quad_vs_quad(a0, a1, a2, a3, b0, b1, b2, b3):
    """2D SAT collision detection between two quads (OBB vs OBB).

    For rectangles, we need to test the perpendicular directions of the edges.
    Vertices should be ordered counter-clockwise: a0, a1, a2, a3
    """
    # Candidate axes: normals of both quads' edges
    ca = (a0 + a1 + a2 + a3) * 0.25
    cb = (b0 + b1 + b2 + b3) * 0.25
    # Use a flag to record whether we found a separating axis.
    # Initialize best_sep to a very small value and best_pen to large.
    found_separating = False
    best_sep = -1e9
    best_sep_axis = ti.Vector([1.0, 0.0])
    best_sep_side = 0
    best_pen = 1e9
    best_pen_axis = ti.Vector([1.0, 0.0])

    # Test edge normals from quad A
    # For a rectangle, we test two perpendicular edges (a0->a1 and a1->a2)
    # This covers both normal directions
    for i in ti.static(range(2)):
        e = a1 - a0 if i == 0 else a2 - a1
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_quad_on_axis(b0, b1, b2, b3, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = True
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # Test edge normals from quad B
    for i in ti.static(range(2)):
        e = b1 - b0 if i == 0 else b2 - b1
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_quad_on_axis(b0, b1, b2, b3, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = True
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    signed = 0.0
    n_axis = ti.Vector([1.0, 0.0])
    if found_separating:
        # Separated
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        # Penetrating (choose minimum penetration axis)
        n_axis = best_pen_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen
    return signed, n_axis


@ti.func
def obb2d_signed_distance_quad_vs_segment(a0, a1, a2, a3, s0, s1):
    ca = (a0 + a1 + a2 + a3) * 0.25
    cb = (s0 + s1) * 0.5
    found_separating = False
    best_sep = -1e9
    best_sep_axis = ti.Vector([1.0, 0.0])
    best_sep_side = 0
    best_pen = 1e9
    best_pen_axis = ti.Vector([1.0, 0.0])

    # axes from quad
    for i in ti.static(range(2)):
        e = a1 - a0 if i == 0 else a3 - a0
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_segment_on_axis(s0, s1, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = True
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
        found_separating = True
        if sep > best_sep:
            best_sep = sep
            best_sep_axis = ax
            best_sep_side = side
    elif pen < best_pen:
        best_pen = pen
        best_pen_axis = ax

    signed = 0.0
    n_axis = ti.Vector([1.0, 0.0])
    if found_separating:
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        n_axis = best_pen_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen
    return signed, n_axis


@ti.func
def obb2d_signed_distance_quad_vs_triangle(a0, a1, a2, a3, t0, t1, t2):
    ca = (a0 + a1 + a2 + a3) * 0.25
    cb = (t0 + t1 + t2) / 3.0
    found_separating = False
    best_sep = -1e9
    best_sep_axis = ti.Vector([1.0, 0.0])
    best_sep_side = 0
    best_pen = 1e9
    best_pen_axis = ti.Vector([1.0, 0.0])

    # axes from quad
    for i in ti.static(range(2)):
        e = a1 - a0 if i == 0 else a3 - a0
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_triangle_on_axis(t0, t1, t2, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = True
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # axes from triangle
    for i in ti.static(range(3)):
        e = t1 - t0 if i == 0 else (t2 - t1 if i == 1 else t0 - t2)
        axis = perp2(e)
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)
        bmin, bmax = sat2d_project_triangle_on_axis(t0, t1, t2, axis)
        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = True
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    signed = 0.0
    n_axis = ti.Vector([1.0, 0.0])
    if found_separating:
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        n_axis = best_pen_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen
    return signed, n_axis


@ti.func
def obb2d_signed_distance_quad_vs_circle(a0, a1, a2, a3, circle_center, radius):
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

    found_separating = False
    best_sep = -1e9
    best_sep_axis = ti.Vector([1.0, 0.0])
    best_sep_side = 0
    best_pen = 1e9
    best_pen_axis = ti.Vector([1.0, 0.0])

    # Test quad edge normals (test two perpendicular edges: a0->a1 and a1->a2)
    for i in ti.static(range(2)):
        e = a1 - a0 if i == 0 else a2 - a1
        axis = perp2(e)

        # Project quad onto axis
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)

        # Project circle onto axis (center ± radius)
        c_proj = circle_center.dot(axis)
        ax_norm = axis.norm()
        r_proj = radius * ax_norm if ax_norm > 1e-12 else 0.0
        bmin = c_proj - r_proj
        bmax = c_proj + r_proj

        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = True
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
    min_dist_sq = (a0 - circle_center).norm_sqr()

    for i in ti.static(range(4)):
        v = a0 if i == 0 else (a1 if i == 1 else (a2 if i == 2 else a3))
        dist_sq = (v - circle_center).norm_sqr()
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_point = v

    # Check edges for closest point
    for i in ti.static(range(4)):
        v0 = a0 if i == 0 else (a1 if i == 1 else (a2 if i == 2 else a3))
        v1 = a1 if i == 0 else (a2 if i == 1 else (a3 if i == 2 else a0))

        # Project circle center onto edge
        edge = v1 - v0
        edge_len_sq = edge.norm_sqr()
        if edge_len_sq > 1e-12:
            t = (circle_center - v0).dot(edge) / edge_len_sq
            t = ti.max(0.0, ti.min(1.0, t))  # Clamp to [0, 1]
            point_on_edge = v0 + t * edge
            dist_sq = (point_on_edge - circle_center).norm_sqr()
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = point_on_edge

    # Test axis from circle center to closest point on quad
    axis = circle_center - closest_point
    axis_norm = axis.norm()
    if axis_norm > 1e-9:
        axis = axis / axis_norm

        # Project quad onto axis
        amin, amax = sat2d_project_quad_on_axis(a0, a1, a2, a3, axis)

        # Project circle onto axis
        c_proj = circle_center.dot(axis)
        bmin = c_proj - radius
        bmax = c_proj + radius

        sep, pen, ax, side = _sat2d_eval_intervals(amin, amax, bmin, bmax, axis)
        if sep > 0.0:
            found_separating = True
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax
                best_sep_side = side
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # Determine signed distance and normal
    signed = 0.0
    n_axis = ti.Vector([1.0, 0.0])

    if found_separating:
        # Separated
        n_axis = best_sep_axis if best_sep_side >= 0 else -best_sep_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        # Penetrating
        n_axis = best_pen_axis
        if (ca - cb).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen

    return signed, n_axis


# ===== 3D OBB vs OBB SAT =====
@ti.func
def obb3d_signed_distance(centerA, extentsA, quatA, centerB, extentsB, quatB):
    """
    SAT-based OBB vs OBB collision detection for 3D.
    Returns: (signed_distance, normal)
    - signed_distance > 0: separated
    - signed_distance < 0: penetrating (negative penetration depth)
    - normal: points from A to B
    """
    heA = 0.5 * extentsA
    heB = 0.5 * extentsB
    a0, a1, a2 = obb_axes_from_quat(quatA)
    b0, b1, b2 = obb_axes_from_quat(quatB)

    best_sep = -1e9
    best_sep_axis = ti.Vector([1.0, 0.0, 0.0])
    best_pen = 1e9
    best_pen_axis = ti.Vector([1.0, 0.0, 0.0])
    found_separating = False

    # 3 face normals of A
    for i in ti.static(range(3)):
        axis = a0 if i == 0 else (a1 if i == 1 else a2)
        sep, pen, ax, side = _eval_axis_obb_vs_obb(centerA, heA, a0, a1, a2, centerB, heB, b0, b1, b2, axis)
        if sep > 0.0:
            found_separating = True
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax if side >= 0 else -ax
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # 3 face normals of B
    for i in ti.static(range(3)):
        axis = b0 if i == 0 else (b1 if i == 1 else b2)
        sep, pen, ax, side = _eval_axis_obb_vs_obb(centerA, heA, a0, a1, a2, centerB, heB, b0, b1, b2, axis)
        if sep > 0.0:
            found_separating = True
            if sep > best_sep:
                best_sep = sep
                best_sep_axis = ax if side >= 0 else -ax
        elif pen < best_pen:
            best_pen = pen
            best_pen_axis = ax

    # 9 cross-product axes (edge-edge)
    for i in ti.static(range(3)):
        ai = a0 if i == 0 else (a1 if i == 1 else a2)
        for j in ti.static(range(3)):
            bj = b0 if j == 0 else (b1 if j == 1 else b2)
            axis = ai.cross(bj)
            # Skip degenerate edge-edge cases (parallel edges)
            if axis.norm() > 1e-6:
                sep, pen, ax, side = _eval_axis_obb_vs_obb(centerA, heA, a0, a1, a2, centerB, heB, b0, b1, b2, axis)
                if sep > 0.0:
                    found_separating = True
                    if sep > best_sep:
                        best_sep = sep
                        best_sep_axis = ax if side >= 0 else -ax
                elif pen < best_pen:
                    best_pen = pen
                    best_pen_axis = ax

    signed = 0.0
    n_axis = ti.Vector([1.0, 0.0, 0.0])

    if found_separating:
        # Boxes are separated
        n_axis = best_sep_axis
        if (centerA - centerB).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = best_sep
    else:
        # Boxes are penetrating, use minimum penetration axis
        n_axis = best_pen_axis
        if (centerA - centerB).dot(n_axis) < 0.0:
            n_axis = -n_axis
        signed = -best_pen

    return signed, n_axis


@ti.func
def _eval_axis_obb_vs_obb(centerA, heA, a0, a1, a2, centerB, heB, b0, b1, b2, axis):
    """
    Evaluate separation/penetration along a given axis for OBB vs OBB.
    Returns: (separation, penetration, normalized_axis, side)
    """
    ax = axis
    ln = ax.norm()

    # Initialize return values
    sep = 0.0
    pen = 0.0
    side = 0

    # Handle degenerate axis case
    is_degenerate = 0
    if ln < 1e-12:
        is_degenerate = 1
        ax = ti.Vector([1.0, 0.0, 0.0])
    else:
        ax = ax / ln

    if is_degenerate == 0:
        amin, amax = sat_project_obb_on_axis(centerA, heA, a0, a1, a2, ax)
        bmin, bmax = sat_project_obb_on_axis(centerB, heB, b0, b1, b2, ax)

        # Check for separation
        d1 = amin - bmax  # A's min beyond B's max
        d2 = bmin - amax  # B's min beyond A's max

        if d1 > 0.0:
            # Separated: A is on positive side
            sep = d1
            side = +1
        elif d2 > 0.0:
            # Separated: B is on positive side
            sep = d2
            side = -1
        else:
            # Overlapping: compute penetration depth
            pen_pos = bmax - amin  # penetration if pushed along +axis
            pen_neg = amax - bmin  # penetration if pushed along -axis
            pen = ti.min(pen_pos, pen_neg)

    return sep, pen, ax, side
