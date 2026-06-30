from definitions import *
from numericaldomain import DomainBase
import numpy as np
import taichi as ti


@ti.data_oriented
class VoxelGridDomain(DomainBase):
    """Axis-aligned voxel grid domain used as static obstacle."""

    def __init__(self, d, nx, ny, lb, ub, nz=None, considerContact=True, occupancy_np=None):
        assert d in (2, 3), "VoxelGridDomain currently supports d=2 or 3"
        self.d = d
        self.type = DomainType.VOXELMAP
        self.considerContact = considerContact
        self.initials = []
        self.bcs = []

        self.nx = int(nx)
        self.ny = int(ny)
        self.lb = ti.Vector(lb)
        self.ub = ti.Vector(ub)

        if d == 2:
            self.dx = (self.ub[0] - self.lb[0]) / max(self.nx, 1)
            self.dz = (self.ub[1] - self.lb[1]) / max(self.ny, 1)

            self.occ = ti.field(dtype=ti.i32, shape=(self.nx, self.ny))
            self.max_edges = self.nx * self.ny * 4
            self.edge_p0 = ti.Vector.field(2, dtype=ti.f32, shape=self.max_edges)
            self.edge_p1 = ti.Vector.field(2, dtype=ti.f32, shape=self.max_edges)
            self.edge_n = ti.Vector.field(2, dtype=ti.f32, shape=self.max_edges)
            self.edge_count = ti.field(dtype=ti.i32, shape=())
        else:
            assert nz is not None, "VoxelGridDomain(d=3) requires nz to be specified"
            self.nz = int(nz)
            self.dx = (self.ub[0] - self.lb[0]) / max(self.nx, 1)
            self.dy = (self.ub[1] - self.lb[1]) / max(self.ny, 1)
            self.dz = (self.ub[2] - self.lb[2]) / max(self.nz, 1)

            self.occ = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            self.max_faces = self.nx * self.ny * self.nz * 6
            self.face_o = ti.Vector.field(3, dtype=ti.f32, shape=self.max_faces)
            self.face_u = ti.Vector.field(3, dtype=ti.f32, shape=self.max_faces)
            self.face_v = ti.Vector.field(3, dtype=ti.f32, shape=self.max_faces)
            self.face_n = ti.Vector.field(3, dtype=ti.f32, shape=self.max_faces)
            self.face_count = ti.field(dtype=ti.i32, shape=())

        if occupancy_np is not None:
            assert occupancy_np.shape == (self.nx, self.ny)
            self.occ.from_numpy(occupancy_np.astype(np.int32))
        else:
            self.occ.fill(0)

        self.nnodes = 1
        self.nelements = 0

        self.build_edges()

        self.category_bits = COLLISION_CATEGORY_GROUND
        self.collide_bits = COLLISION_MASK_ALL

    def attach(self, rigidManager, offset: ti.i32):
        self.ndOffset = int(offset)
        self.rigidManager = rigidManager

    @ti.func
    def _cell_bounds_2d(self, i, j):
        x0 = self.lb[0] + i * self.dx
        x1 = x0 + self.dx
        z0 = self.lb[1] + j * self.dz
        z1 = z0 + self.dz
        return ti.Vector([x0, z0]), ti.Vector([x1, z1])

    @ti.kernel
    def build_edges(self):
        self.edge_count[None] = 0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.occ[i, j] != 1:
                continue
            pmin, pmax = self._cell_bounds_2d(i, j)
            if i - 1 < 0 or self.occ[i - 1, j] == 0:
                eid = ti.atomic_add(self.edge_count[None], 1)
                if eid < self.max_edges:
                    self.edge_p0[eid] = ti.Vector([pmin[0], pmin[1]])
                    self.edge_p1[eid] = ti.Vector([pmin[0], pmax[1]])
                    self.edge_n[eid] = ti.Vector([-1.0, 0.0])
            if i + 1 >= self.nx or self.occ[i + 1, j] == 0:
                eid = ti.atomic_add(self.edge_count[None], 1)
                if eid < self.max_edges:
                    self.edge_p0[eid] = ti.Vector([pmax[0], pmin[1]])
                    self.edge_p1[eid] = ti.Vector([pmax[0], pmax[1]])
                    self.edge_n[eid] = ti.Vector([1.0, 0.0])
            if j - 1 < 0 or self.occ[i, j - 1] == 0:
                eid = ti.atomic_add(self.edge_count[None], 1)
                if eid < self.max_edges:
                    self.edge_p0[eid] = ti.Vector([pmin[0], pmin[1]])
                    self.edge_p1[eid] = ti.Vector([pmax[0], pmin[1]])
                    self.edge_n[eid] = ti.Vector([0.0, -1.0])
            if j + 1 >= self.ny or self.occ[i, j + 1] == 0:
                eid = ti.atomic_add(self.edge_count[None], 1)
                if eid < self.max_edges:
                    self.edge_p0[eid] = ti.Vector([pmin[0], pmax[1]])
                    self.edge_p1[eid] = ti.Vector([pmax[0], pmax[1]])
                    self.edge_n[eid] = ti.Vector([0.0, 1.0])

    @ti.func
    def _closest_on_edge_2d(self, p, a, b):
        ab = b - a
        ab2 = ab.dot(ab) + 1e-12
        t = (p - a).dot(ab) / ab2
        return a + t * ab, t

    @ti.func
    def signed_distance_to_edges_2d(self, p, limit_penetration):
        best_d = 1e9
        best_n = ti.Vector([0.0, 1.0])
        best_c = p

        i = ti.cast(ti.floor((p[0] - self.lb[0]) / self.dx), ti.i32)
        j = ti.cast(ti.floor((p[1] - self.lb[1]) / self.dz), ti.i32)

        if 0 <= i < self.nx and 0 <= j < self.ny:
            if self.occ[i, j] == 1:
                dist_neg_x = 1e9
                for step in range(self.nx):
                    idx = i - 1 - step
                    if idx < 0 or self.occ[idx, j] == 0:
                        boundary_x = self.lb[0] + (idx + 1) * self.dx
                        dist_neg_x = p[0] - boundary_x
                        break

                dist_pos_x = 1e9
                for step in range(self.nx):
                    idx = i + 1 + step
                    if idx >= self.nx or self.occ[idx, j] == 0:
                        boundary_x = self.lb[0] + idx * self.dx
                        dist_pos_x = boundary_x - p[0]
                        break

                dist_neg_y = 1e9
                for step in range(self.ny):
                    idx = j - 1 - step
                    if idx < 0 or self.occ[i, idx] == 0:
                        boundary_y = self.lb[1] + (idx + 1) * self.dz
                        dist_neg_y = p[1] - boundary_y
                        break

                dist_pos_y = 1e9
                for step in range(self.ny):
                    idx = j + 1 + step
                    if idx >= self.ny or self.occ[i, idx] == 0:
                        boundary_y = self.lb[1] + idx * self.dz
                        dist_pos_y = boundary_y - p[1]
                        break

                dist_to_faces = ti.Vector([dist_neg_x, dist_pos_x, dist_neg_y, dist_pos_y])

                min_face_idx = 0
                min_face_dist = dist_to_faces[0]
                for k in ti.static(range(4)):
                    if dist_to_faces[k] < min_face_dist:
                        min_face_dist = dist_to_faces[k]
                        min_face_idx = k

                if min_face_idx == 0:
                    best_d = -dist_neg_x
                    best_n = ti.Vector([-1.0, 0.0])
                    best_c = ti.Vector([p[0] - dist_neg_x, p[1]])
                elif min_face_idx == 1:
                    best_d = -dist_pos_x
                    best_n = ti.Vector([1.0, 0.0])
                    best_c = ti.Vector([p[0] + dist_pos_x, p[1]])
                elif min_face_idx == 2:
                    best_d = -dist_neg_y
                    best_n = ti.Vector([0.0, -1.0])
                    best_c = ti.Vector([p[0], p[1] - dist_neg_y])
                else:
                    best_d = -dist_pos_y
                    best_n = ti.Vector([0.0, 1.0])
                    best_c = ti.Vector([p[0], p[1] + dist_pos_y])
            else:
                ec = self.edge_count[None]
                for eid in range(ec):
                    a = self.edge_p0[eid]
                    b = self.edge_p1[eid]
                    n = self.edge_n[eid]
                    c, t = self._closest_on_edge_2d(p, a, b)
                    d = (p - c).dot(n)
                    limit_penetration = ti.max(limit_penetration, 0.1 * ((b - a).norm()))
                    if d < best_d and (0.0 <= t <= 1.0) and abs(d) < limit_penetration:
                        best_d = d
                        best_n = n
                        best_c = c
        return best_d, best_n, best_c

    def getBBox(self):
        return self.lb, self.ub

    def getBoundaryMesh(self):
        ec = int(self.edge_count[None])
        if ec == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

        p0 = self.edge_p0.to_numpy()[:ec]
        p1 = self.edge_p1.to_numpy()[:ec]

        vertices = []
        edges = []
        for k in range(ec):
            idx0 = len(vertices)
            vertices.extend([p0[k], p1[k]])
            edges.append([idx0, idx0 + 1])

        return np.asarray(vertices, dtype=np.float32), np.asarray(edges, dtype=np.int32)

    def draw(self, gui, color=0x55AAFF):
        ec = int(self.edge_count[None])
        for eid in range(ec):
            gui.line(self.edge_p0[eid], self.edge_p1[eid], color=color)