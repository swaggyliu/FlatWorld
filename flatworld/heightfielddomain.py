from definitions import *
from numericaldomain import DomainBase
import numpy as np


class HeightFieldDomain(DomainBase):
    """Static height field ground domain (2D: z=h(x), 3D: z=h(x,y)).

    Notes:
    - Ground only: bcs is ignored (treated as None).
    - Supports d=2 and d=3. Vertical axis is z.
    - The height array is provided as a numpy array:
        - d=2: shape (nx,)
        - d=3: shape (nx, ny)
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

        if d == 2:
            assert height_np.ndim == 1, "2D height field expects 1D numpy array of shape (nx,)"
            self.nx = int(height_np.shape[0])
            self.ny = 1
            self.height = height_np.copy()
        else:
            assert height_np.ndim == 2, "3D height field expects 2D numpy array of shape (nx, ny)"
            self.nx = int(height_np.shape[0])
            self.ny = int(height_np.shape[1])
            self.height = height_np.copy()

        if d == 2:
            self.point = np.array([0.0, -1e6], dtype=np.float32)
            self.normal = np.array([0.0, 1.0], dtype=np.float32)
        else:
            self.point = np.array([0.0, 0.0, -1e6], dtype=np.float32)
            self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

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

    def sample_height_3d(self, x, y):
        ux = float(np.clip((x - self.lb[0]) / (self.ub[0] - self.lb[0]), 0.0, 1.0))
        uy = float(np.clip((y - self.lb[1]) / (self.ub[1] - self.lb[1]), 0.0, 1.0))
        sx = ux * (self.nx - 1)
        sy = uy * (self.ny - 1)
        ix0 = int(np.floor(sx))
        iy0 = int(np.floor(sy))
        ix1 = min(ix0 + 1, self.nx - 1)
        iy1 = min(iy0 + 1, self.ny - 1)
        tx = sx - float(ix0)
        ty = sy - float(iy0)
        h00 = float(self.height[ix0, iy0])
        h10 = float(self.height[ix1, iy0])
        h01 = float(self.height[ix0, iy1])
        h11 = float(self.height[ix1, iy1])
        hx0 = self._lerp(h00, h10, tx)
        hx1 = self._lerp(h01, h11, tx)
        return float(self._lerp(hx0, hx1, ty))

    def sample_normal_3d(self, x, y):
        dx_world = (self.ub[0] - self.lb[0]) / (self.nx - 1)
        dy_world = (self.ub[1] - self.lb[1]) / (self.ny - 1)
        hx1 = self.sample_height_3d(x + dx_world, y)
        hx0 = self.sample_height_3d(x - dx_world, y)
        hy1 = self.sample_height_3d(x, y + dy_world)
        hy0 = self.sample_height_3d(x, y - dy_world)
        dhdx = (hx1 - hx0) / (2.0 * max(dx_world, 1e-6))
        dhdy = (hy1 - hy0) / (2.0 * max(dy_world, 1e-6))
        n = np.array([-dhdx, -dhdy, 1.0], dtype=np.float32)
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

    def nearest_on_surface_3d(self, x, y, z):
        h = self.sample_height_3d(x, y)
        foot = np.array([x, y, h], dtype=np.float32)
        n = self.sample_normal_3d(x, y)
        p = np.array([x, y, z], dtype=np.float32)
        signed = float(np.dot(p - foot, n))
        return foot, n, signed

    def get_maxmin_height_in_range_2d(self, x_min, x_max):
        x_min_clamp = float(np.clip(x_min, self.lb[0], self.ub[0]))
        x_max_clamp = float(np.clip(x_max, self.lb[0], self.ub[0]))
        u_min = (x_min_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        u_max = (x_max_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        i_min = int(np.floor(u_min * (self.nx - 1)))
        i_max = int(np.ceil(u_max * (self.nx - 1)))
        i_min = max(0, min(i_min, self.nx - 1))
        i_max = max(0, min(i_max, self.nx - 1))

        max_h = -1e9
        min_h = 1e9
        for i in range(i_min, i_max + 1):
            if i < self.nx:
                max_h = max(max_h, float(self.height[i]))
                min_h = min(min_h, float(self.height[i]))
        return max_h, min_h

    def get_maxmin_height_in_range_3d(self, x_min, x_max, y_min, y_max):
        x_min_clamp = float(np.clip(x_min, self.lb[0], self.ub[0]))
        x_max_clamp = float(np.clip(x_max, self.lb[0], self.ub[0]))
        y_min_clamp = float(np.clip(y_min, self.lb[1], self.ub[1]))
        y_max_clamp = float(np.clip(y_max, self.lb[1], self.ub[1]))

        ux_min = (x_min_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        ux_max = (x_max_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        uy_min = (y_min_clamp - self.lb[1]) / (self.ub[1] - self.lb[1])
        uy_max = (y_max_clamp - self.lb[1]) / (self.ub[1] - self.lb[1])

        ix_min = int(np.floor(ux_min * (self.nx - 1)))
        ix_max = int(np.ceil(ux_max * (self.nx - 1)))
        iy_min = int(np.floor(uy_min * (self.ny - 1)))
        iy_max = int(np.ceil(uy_max * (self.ny - 1)))

        ix_min = max(0, min(ix_min, self.nx - 1))
        ix_max = max(0, min(ix_max, self.nx - 1))
        iy_min = max(0, min(iy_min, self.ny - 1))
        iy_max = max(0, min(iy_max, self.ny - 1))

        max_h = -1e9
        min_h = 1e9
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                if ix < self.nx and iy < self.ny:
                    max_h = max(max_h, float(self.height[ix, iy]))
                    min_h = min(min_h, float(self.height[ix, iy]))
        return max_h, min_h

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
