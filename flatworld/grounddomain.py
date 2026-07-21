from definitions import *
from numericaldomain import DomainBase
import numpy as np


def _to_np(x):
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        return x.numpy()
    if hasattr(x, "to_numpy") and callable(getattr(x, "to_numpy")):
        return x.to_numpy()
    return np.asarray(x)


def _vec_np(v):
    return np.asarray(_to_np(v), dtype=np.float32).reshape(-1)


class GroundDomain(DomainBase):
    def __init__(self, d, point, normal, bcs=[], considerContact=True, considerGroundContact=True, initials=[]):
        # now we assume analytical domain is a plane
        self.d = d
        self.point = np.asarray(point, dtype=np.float32)
        n = np.asarray(normal, dtype=np.float32)
        nlen = float(np.linalg.norm(n))
        self.normal = n / max(nlen, 1e-12)
        self.nnodes = 1
        self.nelements = 0
        self.type = DomainType.ANALYTICAL
        self.considerContact = considerContact
        self.considerGroundContact = considerGroundContact
        self.initials = initials
        self.bcs = bcs
        self.category_bits = COLLISION_CATEGORY_GROUND
        self.collide_bits = COLLISION_MASK_ALL

    def attach(self, rigidManager, offset: int):
        # Data arrays
        self.ndOffset = int(offset)
        self.rigidManager = rigidManager

        for initialCondition in self.initials:
            initialCondition.update(self.rigidManager.V, self.d, offset, 1)

    def getCurrentRefPoint(self):
        params = _to_np(self.rigidManager.rigidParams)
        return _vec_np(params[self.ndOffset, 0])

    def getCurrentNormal(self):
        rot = np.asarray(_to_np(self.rigidManager.cached_rotation_matrix)[self.ndOffset], dtype=np.float32)
        params = _to_np(self.rigidManager.rigidParams)
        n_param = _vec_np(params[self.ndOffset, 1])
        return (rot @ n_param).astype(np.float32)

    def getBBox(
        self,
    ):
        # here I should give a buffer zone for the normal direction and a large zone for the tangent direction
        # Return a bounding box centered at the plane reference point. For the tangent directions
        # give a very large extent; for the normal direction give a small buffer.
        buffer = 0.1
        large_span = 10000.0
        p = self.point
        tangent = np.array([-self.normal[1], self.normal[0]], dtype=np.float32)
        # single tangent in 2D (perpendicular to normal)
        lo_raw = p - tangent * large_span - self.normal * buffer
        hi_raw = p + tangent * large_span + self.normal * buffer
        # Ensure component-wise min/max so lo <= hi even if tangent points negative
        lo = np.minimum(lo_raw, hi_raw).astype(np.float32)
        hi = np.maximum(lo_raw, hi_raw).astype(np.float32)
        return lo, hi

    def draw(self, gui, color=0xFFFFFF, resolution=10, leftlength=1e5, rightlength=1e5, linewidth=1):
        pos = self.getCurrentRefPoint()
        normal = self.getCurrentNormal()
        # in 2D
        tangent = np.array([-normal[1], normal[0]], dtype=np.float32)
        gui.line(pos - tangent * rightlength, pos + tangent * leftlength, color=color, radius=linewidth)
