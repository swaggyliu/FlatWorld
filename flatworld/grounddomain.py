from definitions import *
from numericaldomain import DomainBase
import numpy as np
import taichi as ti
import time
from utils import *

@ti.data_oriented
class GroundDomain(DomainBase):
    def __init__(self, d, point, normal, bcs=[], considerContact=True, considerGroundContact=True, initials=[]):
        # now we assume analytical domain is a plane
        self.d = d
        self.point = ti.Vector(point)
        self.normal = ti.Vector(normal).normalized()
        self.nnodes = 1
        self.nelements = 0
        self.type = DomainType.ANALYTICAL
        self.considerContact = considerContact
        self.considerGroundContact = considerGroundContact
        self.initials = initials
        self.bcs = bcs
        self.category_bits = COLLISION_CATEGORY_GROUND
        self.collide_bits = COLLISION_MASK_ALL

    def attach(self, rigidManager, offset: ti.i32):
        # Data arrays
        self.ndOffset = int(offset)
        self.rigidManager = rigidManager

        for initialCondition in self.initials:
            initialCondition.update(self.rigidManager.V, self.d, offset, 1)

    def getCurrentRefPoint(self):
        return self.rigidManager.rigidParams[self.ndOffset, 0].to_numpy()

    def getCurrentNormal(self):
        return (
            self.rigidManager.cached_rotation_matrix[self.ndOffset] @ self.rigidManager.rigidParams[self.ndOffset, 1]
        ).to_numpy()

    def getBBox(
        self,
    ):
        # here I should give a buffer zone for the normal direction and a large zone for the tangent direction
        # Return a bounding box centered at the plane reference point. For the tangent directions
        # give a very large extent; for the normal direction give a small buffer.
        buffer = 0.1
        large_span = 10000.0
        p = self.point
        tangent = ti.Vector([-self.normal[1], self.normal[0]])
        # single tangent in 2D (perpendicular to normal)
        lo_raw = p - tangent * large_span - self.normal * buffer
        hi_raw = p + tangent * large_span + self.normal * buffer
        # Ensure component-wise min/max so lo <= hi even if tangent points negative
        lo = ti.Vector([min(float(lo_raw[i]), float(hi_raw[i])) for i in range(self.d)])
        hi = ti.Vector([max(float(lo_raw[i]), float(hi_raw[i])) for i in range(self.d)])
        return lo, hi

    def draw(self, gui, color=0xFFFFFF, resolution=10, leftlength=1e5, rightlength=1e5, linewidth=1):
        pos = self.getCurrentRefPoint()
        normal = self.getCurrentNormal()
        # in 2D
        tangent = np.array([-normal[1], normal[0]])
        gui.line(pos - tangent * rightlength, pos + tangent * leftlength, color=color, radius=linewidth)



