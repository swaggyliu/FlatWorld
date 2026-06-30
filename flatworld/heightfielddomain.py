from definitions import *
from numericaldomain import DomainBase
import numpy as np
import taichi as ti


@ti.data_oriented
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
        self.maxHeight = np.max(height_np) + 0.1
        self.minHeight = np.min(height_np) - 0.1

        if d == 2:
            assert height_np.ndim == 1, "2D height field expects 1D numpy array of shape (nx,)"
            self.nx = int(height_np.shape[0])
            self.ny = 1
            self.height = ti.field(ti.f32, (self.nx,))
            self.height.from_numpy(height_np.astype(np.float32))
        else:
            assert height_np.ndim == 2, "3D height field expects 2D numpy array of shape (nx, ny)"
            self.nx = int(height_np.shape[0])
            self.ny = int(height_np.shape[1])
            self.height = ti.field(ti.f32, (self.nx, self.ny))
            self.height.from_numpy(height_np.astype(np.float32))

        if d == 2:
            self.point = ti.Vector([0.0, -1e6])
            self.normal = ti.Vector([0.0, 1.0])
        else:
            self.point = ti.Vector([0.0, 0.0, -1e6])
            self.normal = ti.Vector([0.0, 0.0, 1.0])

        self.nnodes = 1
        self.nelements = 0
        self.lb = ti.Vector(lb)
        self.ub = ti.Vector(ub)

        self.category_bits = COLLISION_CATEGORY_GROUND
        self.collide_bits = COLLISION_MASK_ALL

    def attach(self, rigidManager, offset: ti.i32):
        self.ndOffset = int(offset)
        self.rigidManager = rigidManager

    @ti.func
    def _lerp(self, a, b, t):
        return a + (b - a) * t

    @ti.func
    def sample_height_2d(self, x):
        u = (x - self.lb[0]) / (self.ub[0] - self.lb[0])
        u = ti.math.clamp(u, 0.0, 1.0)
        s = u * (self.nx - 1)
        i0 = ti.cast(ti.floor(s), ti.i32)
        i1 = ti.min(i0 + 1, self.nx - 1)
        t = s - ti.cast(i0, ti.f32)
        return self._lerp(self.height[i0], self.height[i1], t)

    @ti.func
    def sample_dhdx_2d(self, x):
        u = (x - self.lb[0]) / (self.ub[0] - self.lb[0])
        u = ti.math.clamp(u, 0.0, 1.0)
        s = u * (self.nx - 1)
        i = ti.cast(ti.round(s), ti.i32)
        i0 = ti.max(0, i - 1)
        i1 = ti.min(self.nx - 1, i + 1)
        h0 = self.height[i0]
        h1 = self.height[i1]
        dx_world = (self.ub[0] - self.lb[0]) / (self.nx - 1)
        return (h1 - h0) / (ti.max(dx_world * (i1 - i0), 1e-6))

    @ti.func
    def sample_normal_2d(self, x):
        dhdx = self.sample_dhdx_2d(x)
        n = ti.Vector([-dhdx, 1.0])
        if self.reverse:
            n = -n
        return n.normalized()

    @ti.func
    def sample_height_3d(self, x, y):
        ux = ti.math.clamp((x - self.lb[0]) / (self.ub[0] - self.lb[0]), 0.0, 1.0)
        uy = ti.math.clamp((y - self.lb[1]) / (self.ub[1] - self.lb[1]), 0.0, 1.0)
        sx = ux * (self.nx - 1)
        sy = uy * (self.ny - 1)
        ix0 = ti.cast(ti.floor(sx), ti.i32)
        iy0 = ti.cast(ti.floor(sy), ti.i32)
        ix1 = ti.min(ix0 + 1, self.nx - 1)
        iy1 = ti.min(iy0 + 1, self.ny - 1)
        tx = sx - ti.cast(ix0, ti.f32)
        ty = sy - ti.cast(iy0, ti.f32)
        h00 = self.height[ix0, iy0]
        h10 = self.height[ix1, iy0]
        h01 = self.height[ix0, iy1]
        h11 = self.height[ix1, iy1]
        hx0 = self._lerp(h00, h10, tx)
        hx1 = self._lerp(h01, h11, tx)
        return self._lerp(hx0, hx1, ty)

    @ti.func
    def sample_normal_3d(self, x, y):
        dx_world = (self.ub[0] - self.lb[0]) / (self.nx - 1)
        dy_world = (self.ub[1] - self.lb[1]) / (self.ny - 1)
        hx1 = self.sample_height_3d(x + dx_world, y)
        hx0 = self.sample_height_3d(x - dx_world, y)
        hy1 = self.sample_height_3d(x, y + dy_world)
        hy0 = self.sample_height_3d(x, y - dy_world)
        dhdx = (hx1 - hx0) / (2.0 * ti.max(dx_world, 1e-6))
        dhdy = (hy1 - hy0) / (2.0 * ti.max(dy_world, 1e-6))
        n = ti.Vector([-dhdx, -dhdy, 1.0])
        if self.reverse:
            n = -n
        return n.normalized()

    @ti.func
    def nearest_on_curve_2d(self, x, z):
        h = self.sample_height_2d(x)
        foot = ti.Vector([x, h])
        n = self.sample_normal_2d(x)
        p = ti.Vector([x, z])
        signed = (p - foot).dot(n)
        return foot, n, signed

    @ti.func
    def nearest_on_surface_3d(self, x, y, z):
        h = self.sample_height_3d(x, y)
        foot = ti.Vector([x, y, h])
        n = self.sample_normal_3d(x, y)
        p = ti.Vector([x, y, z])
        signed = (p - foot).dot(n)
        return foot, n, signed

    @ti.func
    def get_maxmin_height_in_range_2d(self, x_min, x_max):
        x_min_clamp = ti.math.clamp(x_min, self.lb[0], self.ub[0])
        x_max_clamp = ti.math.clamp(x_max, self.lb[0], self.ub[0])
        u_min = (x_min_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        u_max = (x_max_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        i_min = ti.cast(ti.floor(u_min * (self.nx - 1)), ti.i32)
        i_max = ti.cast(ti.ceil(u_max * (self.nx - 1)), ti.i32)
        i_min = ti.max(0, ti.min(i_min, self.nx - 1))
        i_max = ti.max(0, ti.min(i_max, self.nx - 1))

        max_h = -1e9
        min_h = 1e9
        for i in range(i_min, i_max + 1):
            if i < self.nx:
                max_h = ti.max(max_h, self.height[i])
                min_h = ti.min(min_h, self.height[i])
        return max_h, min_h

    @ti.func
    def get_maxmin_height_in_range_3d(self, x_min, x_max, y_min, y_max):
        x_min_clamp = ti.math.clamp(x_min, self.lb[0], self.ub[0])
        x_max_clamp = ti.math.clamp(x_max, self.lb[0], self.ub[0])
        y_min_clamp = ti.math.clamp(y_min, self.lb[1], self.ub[1])
        y_max_clamp = ti.math.clamp(y_max, self.lb[1], self.ub[1])

        ux_min = (x_min_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        ux_max = (x_max_clamp - self.lb[0]) / (self.ub[0] - self.lb[0])
        uy_min = (y_min_clamp - self.lb[1]) / (self.ub[1] - self.lb[1])
        uy_max = (y_max_clamp - self.lb[1]) / (self.ub[1] - self.lb[1])

        ix_min = ti.cast(ti.floor(ux_min * (self.nx - 1)), ti.i32)
        ix_max = ti.cast(ti.ceil(ux_max * (self.nx - 1)), ti.i32)
        iy_min = ti.cast(ti.floor(uy_min * (self.ny - 1)), ti.i32)
        iy_max = ti.cast(ti.ceil(uy_max * (self.ny - 1)), ti.i32)

        ix_min = ti.max(0, ti.min(ix_min, self.nx - 1))
        ix_max = ti.max(0, ti.min(ix_max, self.nx - 1))
        iy_min = ti.max(0, ti.min(iy_min, self.ny - 1))
        iy_max = ti.max(0, ti.min(iy_max, self.ny - 1))

        max_h = -1e9
        min_h = 1e9
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                if ix < self.nx and iy < self.ny:
                    max_h = ti.max(max_h, self.height[ix, iy])
                    min_h = ti.min(min_h, self.height[ix, iy])
        return max_h, min_h

    def getBBox(self):
        return ti.Vector([self.lb[0], self.minHeight]), ti.Vector([self.ub[0], self.maxHeight])

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
            z = float(
                self.height[
                    int((x - float(self.lb[0])) / (float(self.ub[0]) - float(self.lb[0]) + 1e-12) * (self.nx - 1))
                ]
            )
            pts.append([x, z])
        pts = np.array(pts, dtype=np.float32)
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                gui.line(pts[i], pts[i + 1], color=color, radius=linewidth)