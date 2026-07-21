from definitions import *
import numpy as np
import warp as wp


@wp.func
def cal2DRotationMat(theta: float):
    """Return 2D rotation matrix for angle theta."""
    c = wp.cos(theta)
    s = wp.sin(theta)
    return wp.mat22(c, -s, s, c)


@wp.func
def vectorCrossProduct(v1: wp.vec2, v2: wp.vec2):
    """2D cross product stored in the first component of a vec2 (z-component)."""
    return wp.vec2(v1[0] * v2[1] - v2[0] * v1[1], 0.0)


# ---------------------------------------------------------------------------------
# Geometry Queries
# ---------------------------------------------------------------------------------


@wp.func
def planeSDFQuery(pos: wp.vec2, origin: wp.vec2, normal: wp.vec2):
    """Signed distance from point to plane (positive outside along normal)."""
    p = pos - origin
    penetration = wp.dot(p, normal)
    return penetration


@wp.func
def sphereSDFQuery(pos: wp.vec2, origin: wp.vec2, radius: float):
    """Signed distance and outward normal from point to sphere surface."""
    p = pos - origin
    nlen = wp.length(p)
    penetration = nlen - radius
    n = p / wp.max(nlen, 1e-9)
    return penetration, n


@wp.func
def capsuleSDFQuery(pos: wp.vec2, origin: wp.vec2, lc: wp.vec2, radius: float):
    """Signed distance from point to capsule (rounded cylinder) surface."""
    # Transform point to capsule local space
    p = lc
    q = 2.0 * origin - lc
    dis, t, closest = calMinDisNode2Segment(pos, p, q)
    penetration = dis - radius
    normal_vec = pos - closest
    nlen = wp.length(normal_vec)
    n = normal_vec / wp.max(nlen, 1e-9)
    return penetration, n


@wp.func
def obbSDFQuery(point: wp.vec2, obbCenter: wp.vec2, obbExtents: wp.vec2, rotMat: wp.mat22):
    """Compute signed distance and outward normal from point to 2D OBB.

    Args:
        point: Query point position
        obbCenter: OBB center position
        obbExtents: Full extents of the box (will be halved internally)
        rotMat: 2x2 rotation matrix

    Returns:
        (signed_distance, normal):
            - signed_distance > 0: point outside, distance to surface
            - signed_distance < 0: point inside, negative distance to nearest surface
            - normal: always points FROM OBB TO point (outward from OBB perspective)
    """
    maxSignedDis = float(-1e9)
    normal = wp.vec2(0.0, 0.0)
    half_ext = 0.5 * obbExtents

    # Transform point to local space
    localPoint = point - obbCenter
    localPoint = wp.transpose(rotMat) @ localPoint

    # Signed distances to all four faces
    sd0 = localPoint[0] - half_ext[0]  # +x
    sd1 = -localPoint[0] - half_ext[0]  # -x
    sd2 = localPoint[1] - half_ext[1]  # +y
    sd3 = -localPoint[1] - half_ext[1]  # -y

    if sd0 > maxSignedDis:
        maxSignedDis = sd0
        normal = wp.vec2(1.0, 0.0)
    if sd1 > maxSignedDis:
        maxSignedDis = sd1
        normal = wp.vec2(-1.0, 0.0)
    if sd2 > maxSignedDis:
        maxSignedDis = sd2
        normal = wp.vec2(0.0, 1.0)
    if sd3 > maxSignedDis:
        maxSignedDis = sd3
        normal = wp.vec2(0.0, -1.0)

    is_outside = False
    if sd0 > 0.0 or sd1 > 0.0 or sd2 > 0.0 or sd3 > 0.0:
        is_outside = True

    if is_outside:
        # Clamp local point to box extents to find nearest point on box
        cx = localPoint[0]
        if localPoint[0] < -half_ext[0]:
            cx = -half_ext[0]
        elif localPoint[0] > half_ext[0]:
            cx = half_ext[0]

        cy = localPoint[1]
        if localPoint[1] < -half_ext[1]:
            cy = -half_ext[1]
        elif localPoint[1] > half_ext[1]:
            cy = half_ext[1]

        clamped = wp.vec2(cx, cy)
        delta = localPoint - clamped
        dist = wp.length(delta)

        if dist > 1e-9:
            normal = delta / dist

        maxSignedDis = dist

    # Transform normal back to world space
    normal = rotMat @ normal

    return maxSignedDis, normal


@wp.func
def calMinDisNode2Segment(pos: wp.vec2, p: wp.vec2, q: wp.vec2):
    """Return (distance, t, closest) from point to segment [p,q], with t in [0,1]."""
    ap = pos - p
    ab = q - p

    ab_sq = wp.dot(ab, ab)

    t = wp.dot(ap, ab) / ab_sq
    t = wp.max(0.0, wp.min(1.0, t))

    closest = p + t * ab
    return wp.length(pos - closest), t, closest


@wp.func
def calMinDisSegment2Segment(a: wp.vec2, b: wp.vec2, c: wp.vec2, d: wp.vec2):
    """Return closest points (p, q) and parameters (sc, tc) between two segments."""
    # Robust algorithm for closest points between two segments
    # Based on the algorithm in "Real-Time Collision Detection" (Ericson).
    u = b - a
    v = d - c
    w = a - c

    a_val = wp.dot(u, u)  # squared length of segment S1
    b_val = wp.dot(u, v)
    c_val = wp.dot(v, v)  # squared length of segment S2
    d_val = wp.dot(u, w)
    e_val = wp.dot(v, w)

    SMALL_NUM = float(1e-9)

    D = a_val * c_val - b_val * b_val  # denominator

    sN = float(0.0)
    tN = float(0.0)
    sD = D
    tD = D

    # compute the line parameters of the two closest points
    if D < SMALL_NUM:  # the lines are almost parallel
        sN = 0.0  # force using s = 0 on segment S1
        sD = 1.0  # to avoid divide by 0
        tN = e_val
        tD = c_val
    else:
        sN = b_val * e_val - c_val * d_val
        tN = a_val * e_val - b_val * d_val

    # clamp sN to [0, sD]
    if sN < 0.0:
        sN = 0.0
        tN = e_val
        tD = c_val
    elif sN > sD:
        sN = sD
        tN = e_val + b_val
        tD = c_val

    # clamp tN to [0, tD]
    if tN < 0.0:
        tN = 0.0
        # recompute sN for this tN
        if -d_val < 0.0:
            sN = 0.0
        elif -d_val > a_val:
            sN = sD
        else:
            sN = -d_val
            sD = a_val
    elif tN > tD:
        tN = tD
        # recompute sN for this tN
        if (-d_val + b_val) < 0.0:
            sN = 0.0
        elif (-d_val + b_val) > a_val:
            sN = sD
        else:
            sN = -d_val + b_val
            sD = a_val

    # finally compute the parameters sc and tc
    sc = float(0.0)
    if wp.abs(sN) >= SMALL_NUM:
        sc = sN / sD
    tc = float(0.0)
    if wp.abs(tN) >= SMALL_NUM:
        tc = tN / tD

    # compute the closest points
    p_closest = a + sc * u
    q_closest = c + tc * v

    return p_closest, q_closest, sc, tc


# ---------------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------------


@wp.func
def getBallBBox(center: wp.vec2, radius: float, rotMat: wp.mat22):
    """Return axis-aligned bbox (lb, up) for a ball."""
    lb = center - wp.vec2(radius, radius)
    up = center + wp.vec2(radius, radius)
    return lb, up


@wp.func
def getBoxBBox(center: wp.vec2, extent: wp.vec2, rotMat: wp.mat22):
    """Return AABB and four corner coords for a 2D OBB (lb, ub, shapeCoords 4x2)."""
    lr0 = -0.5 * (rotMat @ extent)
    lr1 = 0.5 * (rotMat @ wp.vec2(extent[0], -extent[1]))
    lr2 = 0.5 * (rotMat @ wp.vec2(extent[0], extent[1]))
    lr3 = 0.5 * (rotMat @ wp.vec2(-extent[0], extent[1]))

    newCoord0 = center + lr0
    newCoord1 = center + lr1
    newCoord2 = center + lr2
    newCoord3 = center + lr3

    aabb_min = wp.min(wp.min(newCoord0, newCoord1), wp.min(newCoord2, newCoord3))
    aabb_max = wp.max(wp.max(newCoord0, newCoord1), wp.max(newCoord2, newCoord3))

    shapeCoords = wp.matrix(
        newCoord0[0],
        newCoord0[1],
        newCoord1[0],
        newCoord1[1],
        newCoord2[0],
        newCoord2[1],
        newCoord3[0],
        newCoord3[1],
        shape=(4, 2),
        dtype=float,
    )

    return aabb_min, aabb_max, shapeCoords


@wp.func
def getCapsuleBBox(center: wp.vec2, lcdir: wp.vec2, radius: float, RotMat: wp.mat22):
    """Compute bbox for a capsule (rounded cylinder) shape."""
    axis = lcdir
    axis_len = wp.length(axis)
    dir = axis / axis_len  # half-axis direction before rotation

    # Apply rotation to get world-space axis direction
    dir = RotMat @ dir

    # Endpoints of the capsule axis (centers of hemispherical caps)
    p0 = center - dir * axis_len
    p1 = center + dir * axis_len

    # Perpendicular + axial radius contributions per axis
    perp0 = radius * wp.sqrt(wp.max(0.0, 1.0 - dir[0] * dir[0]))
    axial0 = radius * wp.abs(dir[0])
    perp1 = radius * wp.sqrt(wp.max(0.0, 1.0 - dir[1] * dir[1]))
    axial1 = radius * wp.abs(dir[1])

    coord_min0 = wp.min(p0[0], p1[0])
    coord_max0 = wp.max(p0[0], p1[0])
    coord_min1 = wp.min(p0[1], p1[1])
    coord_max1 = wp.max(p0[1], p1[1])

    lb = wp.vec2(coord_min0 - perp0 - axial0, coord_min1 - perp1 - axial1)
    ub = wp.vec2(coord_max0 + perp0 + axial0, coord_max1 + perp1 + axial1)

    return lb, ub


# ============================================================================================
# ============================================================================================
class Transform:
    def __init__(self, offset, scale, quat):
        self.offset = np.asarray(offset, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.quat = np.asarray(quat, dtype=np.float32)
