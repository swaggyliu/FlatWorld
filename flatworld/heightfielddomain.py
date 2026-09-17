from definitions import *
from numericaldomain import DomainBase
import numpy as np


class HeightFieldDomain(DomainBase):
    """Static height field ground domain (2D: z=h(x)).

    Notes:
    - Ground only: bcs is ignored (treated as None).
    - Supports d=2. Vertical axis is z.
    - The height array is provided as a numpy array:
        - d=2: shape (nx,)
    - lb/ub define the world-space lateral extents used for sampling (x and y ranges).
    """

    def __init__(self, d, height_np: np.ndarray, lb, ub, considerContact=True, reverse=False):
        self.d = d
        self.type = DomainType.HEIGHTFIELD
        self.considerContact = considerContact
        self.initials = []
        self.bcs = []
        self.reverse = reverse
        height_np = np.asarray(height_np, dtype=np.float32)
        self.maxHeight = float(np.max(height_np)) + 0.1
        self.minHeight = float(np.min(height_np)) - 0.1

        assert height_np.ndim == 1, "2D height field expects 1D numpy array of shape (nx,)"
        self.nx = int(height_np.shape[0])
        self.ny = 1
        self.height = height_np.copy()


        self.point = np.array([0.0, -1e6], dtype=np.float32)
        self.normal = np.array([0.0, 1.0], dtype=np.float32)

        self.nnodes = 1
        self.nelements = 0
        self.lb = np.asarray(lb, dtype=np.float32)
        self.ub = np.asarray(ub, dtype=np.float32)

        self.category_bits = COLLISION_CATEGORY_GROUND
        self.collide_bits = COLLISION_MASK_ALL

    def attach(self, rigidManager, offset: int):
        self.ndOffset = int(offset)
        self.rigidManager = rigidManager

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    def sample_height_2d(self, x):
        u = (x - self.lb[0]) / (self.ub[0] - self.lb[0])
        u = float(np.clip(u, 0.0, 1.0))
        s = u * (self.nx - 1)
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, self.nx - 1)
        t = s - float(i0)
        return float(self._lerp(self.height[i0], self.height[i1], t))

    def sample_dhdx_2d(self, x):
        u = (x - self.lb[0]) / (self.ub[0] - self.lb[0])
        u = float(np.clip(u, 0.0, 1.0))
        s = u * (self.nx - 1)
        i = int(np.round(s))
        i0 = max(0, i - 1)
        i1 = min(self.nx - 1, i + 1)
        h0 = float(self.height[i0])
        h1 = float(self.height[i1])
        dx_world = (self.ub[0] - self.lb[0]) / (self.nx - 1)
        return (h1 - h0) / max(dx_world * (i1 - i0), 1e-6)

    def sample_normal_2d(self, x):
        dhdx = self.sample_dhdx_2d(x)
        n = np.array([-dhdx, 1.0], dtype=np.float32)
        if self.reverse:
            n = -n
        nlen = float(np.linalg.norm(n))
        return n / max(nlen, 1e-12)

    def nearest_on_curve_2d(self, x, z):
        h = self.sample_height_2d(x)
        foot = np.array([x, h], dtype=np.float32)
        n = self.sample_normal_2d(x)
        p = np.array([x, z], dtype=np.float32)
        signed = float(np.dot(p - foot, n))
        return foot, n, signed

    def getBBox(self):
        return (
            np.array([self.lb[0], self.minHeight], dtype=np.float32),
            np.array([self.ub[0], self.maxHeight], dtype=np.float32),
        )

    def getBoundaryMesh(self):
        xs = np.linspace(float(self.lb[0]), float(self.ub[0]), self.nx, dtype=np.float32)
        vertices = []
        for i, x in enumerate(xs):
            z = float(self.height[i])
            vertices.append([x, 0.0, z])
        vertices = np.array(vertices, dtype=np.float32)

        faces = []
        for i in range(self.nx - 1):
            faces.append([i, i + 1])
        faces = np.array(faces, dtype=np.int32)

        return vertices, faces

    def draw(self, gui, color=0x888888, resolution=512, linewidth=1):
        xs = np.linspace(float(self.lb[0]), float(self.ub[0]), num=min(self.nx, resolution))
        pts = []
        for x in xs:
            idx = int((x - float(self.lb[0])) / (float(self.ub[0]) - float(self.lb[0]) + 1e-12) * (self.nx - 1))
            idx = max(0, min(idx, self.nx - 1))
            z = float(self.height[idx])
            pts.append([x, z])
        pts = np.array(pts, dtype=np.float32)
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                gui.line(pts[i], pts[i + 1], color=color, radius=linewidth)
