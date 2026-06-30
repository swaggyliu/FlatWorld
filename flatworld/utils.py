from definitions import *
import numpy as np
import taichi as ti

@ti.func
def cal2DRotationMat(theta):
    """Return 2D rotation matrix for angle theta."""
    return ti.Matrix([[ti.cos(theta), -ti.sin(theta)], [ti.sin(theta), ti.cos(theta)]])


@ti.func
def vectorCrossProduct(v1, v2):
    res = ti.Vector.zero(ti.f32, v1.n)
    res[0] = v1[0] * v2[1] - v2[0] * v1[1]

    return res

# ---------------------------------------------------------------------------------
# Geometry Queries
# ---------------------------------------------------------------------------------


@ti.func
def planeSDFQuery(pos, origin, normal):
    """Signed distance from point to plane (positive outside along normal)."""
    p = pos - origin
    penetration = p.dot(normal)
    return penetration


@ti.func
def sphereSDFQuery(pos, origin, radius):
    """Signed distance and outward normal from point to sphere surface."""
    p = pos - origin
    penetration = p.norm() - radius
    n = p.normalized(1e-9)
    return penetration, n


@ti.func
def capsuleSDFQuery(pos, origin, lc, radius):
    """Signed distance from point to capsule (rounded cylinder) surface."""
    # Transform point to capsule local space
    p = lc
    q = 2 * origin - lc
    dis, t, closest = calMinDisNode2Segment(pos, p, q)
    penetration = dis - radius
    normal_vec = pos - closest
    n = normal_vec.normalized(1e-9)
    return penetration, n


@ti.func
def obbSDFQuery(point, obbCenter, obbExtents, rotMat):
    """Compute signed distance and outward normal from point to OBB.

    Works for both 2D and 3D based on point.n dimension.

    Args:
        point: Query point position (2D or 3D vector)
        obbCenter: OBB center position
        obbExtents: Full extents of the box (will be halved internally)
        rotMat: Rotation matrix (2x2 for 2D, 3x3 for 3D)

    Returns:
        (signed_distance, normal):
            - signed_distance > 0: point outside, distance to surface
            - signed_distance < 0: point inside, negative distance to nearest surface
            - normal: always points FROM OBB TO point (outward from OBB perspective)
    """
    d = ti.static(point.n)  # Dimension: 2 or 3
    maxSignedDis = -1e9
    normal = ti.Vector.zero(ti.f32, d)
    obbExtents = 0.5 * obbExtents

    # Transform point to local space
    localPoint = point - obbCenter
    localPoint = rotMat.transpose() @ localPoint

    # Compute signed distances to all faces (2*d faces total)
    num_faces = ti.static(2 * d)
    signedDistances = ti.Vector.zero(ti.f32, num_faces)

    for i in ti.static(range(d)):
        # Distance to positive face (normal = +axis_i)
        signedDistances[2 * i] = localPoint[i] - obbExtents[i]
        # Distance to negative face (normal = -axis_i)
        signedDistances[2 * i + 1] = -localPoint[i] - obbExtents[i]

        # Track the face with maximum signed distance
        if signedDistances[2 * i] > maxSignedDis:
            maxSignedDis = signedDistances[2 * i]
            normal.fill(0.0)
            normal[i] = 1.0

        if signedDistances[2 * i + 1] > maxSignedDis:
            maxSignedDis = signedDistances[2 * i + 1]
            normal.fill(0.0)
            normal[i] = -1.0

    # Determine if point is outside the box
    is_outside = False
    for i in ti.static(range(d)):
        if signedDistances[2 * i] > 0 or signedDistances[2 * i + 1] > 0:
            is_outside = True

    # Compute final distance and normal
    if is_outside:
        # Point is outside: compute Euclidean distance to nearest surface point
        # Clamp local point to box extents to find nearest point on box
        clamped = ti.Vector.zero(ti.f32, d)
        for i in ti.static(range(d)):
            if localPoint[i] < -obbExtents[i]:
                clamped[i] = -obbExtents[i]
            elif localPoint[i] > obbExtents[i]:
                clamped[i] = obbExtents[i]
            else:
                clamped[i] = localPoint[i]

        # Distance vector from nearest point to actual point (in local space)
        delta = localPoint - clamped
        dist = delta.norm()

        # Normal points from box surface to point
        if dist > 1e-9:
            normal = delta / dist  # Local normal
        # else: point exactly on surface, use face normal from maxSignedDis

        maxSignedDis = dist
    # else: point is inside, maxSignedDis and normal already set correctly

    # Transform normal back to world space
    normal = rotMat @ normal

    return maxSignedDis, normal


@ti.func
def calMinDisNode2Segment(pos, p, q):
    """Return (distance, t, closest) from point to segment [p,q], with t in [0,1]."""
    # p, q is the start and end point of the segment
    ap = pos - p
    ab = q - p

    ab_sq = ab.dot(ab)
    # if ab_sq == 0:
    #     return ap.norm()

    t = ap.dot(ab) / ab_sq
    t = max(0, min(1, t))

    closest = p + t * ab
    return (pos - closest).norm(), t, closest


@ti.func
def calMinDisSegment2Segment(a, b, c, d):
    """Return concatenated nearest points (p followed by q) between two segments."""
    # Robust algorithm for closest points between two segments
    # Based on the algorithm in "Real-Time Collision Detection" (Ericson).
    u = b - a
    v = d - c
    w = a - c

    a_val = u.dot(u)  # squared length of segment S1
    b_val = u.dot(v)
    c_val = v.dot(v)  # squared length of segment S2
    d_val = u.dot(w)
    e_val = v.dot(w)

    SMALL_NUM = 1e-9

    D = a_val * c_val - b_val * b_val  # denominator

    sN = 0.0
    tN = 0.0
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
    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD

    # compute the closest points
    p_closest = a + sc * u
    q_closest = c + tc * v

    return p_closest, q_closest, sc, tc


# ---------------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------------

# Some functions
@ti.func
def getBallBBox(center, radius, rotMat):
    """Return axis-aligned bbox (lb, up) for a ball."""
    lb = center - radius
    up = center + radius
    return lb, up


@ti.func
def getBoxBBox(center, extent, rotMat):
    """Get box bounding box. For 2D, RotU_or_quat is a scalar angle. For 3D, it's a quaternion [w,x,y,z]."""
    aabb_min = ti.Vector([1e9 for i in ti.static(range(center.n))])
    aabb_max = ti.Vector([-1e9 for i in ti.static(range(center.n))])
    shapeCoords = ti.Matrix.zero(ti.f32, 2**center.n, center.n)

    """Return AABB and four corner coords for a 2D OBB (lb, ub, c0..c3)."""
    lr0 = -0.5 * (rotMat @ extent)
    lr1 = 0.5 * (rotMat @ ti.Vector([extent[0], -extent[1]]))
    lr2 = 0.5 * (rotMat @ ti.Vector([extent[0], extent[1]]))
    lr3 = 0.5 * (rotMat @ ti.Vector([-extent[0], extent[1]]))

    newCoord0 = center + lr0
    newCoord1 = center + lr1
    newCoord2 = center + lr2
    newCoord3 = center + lr3

    aabb_min = ti.min(aabb_min, newCoord0)
    aabb_max = ti.max(aabb_max, newCoord0)
    aabb_min = ti.min(aabb_min, newCoord1)
    aabb_max = ti.max(aabb_max, newCoord1)
    aabb_min = ti.min(aabb_min, newCoord2)
    aabb_max = ti.max(aabb_max, newCoord2)
    aabb_min = ti.min(aabb_min, newCoord3)
    aabb_max = ti.max(aabb_max, newCoord3)

    shapeCoords[0, 0] = newCoord0[0]
    shapeCoords[0, 1] = newCoord0[1]
    shapeCoords[1, 0] = newCoord1[0]
    shapeCoords[1, 1] = newCoord1[1]
    shapeCoords[2, 0] = newCoord2[0]
    shapeCoords[2, 1] = newCoord2[1]
    shapeCoords[3, 0] = newCoord3[0]
    shapeCoords[3, 1] = newCoord3[1]

    return aabb_min, aabb_max, shapeCoords


@ti.func
def getCapsuleBBox(center, lcdir, radius, RotMat):
    """Compute bbox for a capsule (rounded cylinder) shape."""
    axis = lcdir
    axis_len = axis.norm()
    dir = axis / axis_len  # This is the half of axis length

    # Apply rotation to get world-space axis direction
    dir = RotMat @ dir

    # Endpoints of the capsule axis (centers of hemispherical caps)
    p0 = center - dir * axis_len  # One endpoint
    p1 = center + dir * axis_len  # Other endpoint

    # For each coordinate axis, the extent is determined by:
    # - The axis endpoints
    # - The perpendicular radius contribution: radius * sqrt(1 - (axis_dir[i])^2)
    # - The axial radius contribution for hemispherical caps: radius * |axis_dir[i]|
    lb = ti.Vector.zero(ti.f32, lcdir.n)
    ub = ti.Vector.zero(ti.f32, lcdir.n)

    for i in ti.static(range(lcdir.n)):
        # Perpendicular component for this axis (cylindrical body)
        perp_contrib = radius * ti.sqrt(ti.max(0.0, 1.0 - dir[i] * dir[i]))

        # Axial component for hemispherical caps
        axial_contrib = radius * ti.abs(dir[i])

        # Min and max from endpoints
        coord_min = ti.min(p0[i], p1[i])
        coord_max = ti.max(p0[i], p1[i])

        # Expand by perpendicular radius AND axial cap radius
        lb[i] = coord_min - perp_contrib - axial_contrib
        ub[i] = coord_max + perp_contrib + axial_contrib

    return lb, ub


# ============================================================================================
# ============================================================================================
@ti.data_oriented
class Transform:
    def __init__(self, offset, scale, quat):
        self.offset = ti.Vector(offset)
        self.scale = ti.Vector(scale)
        self.quat = ti.Vector(quat)
