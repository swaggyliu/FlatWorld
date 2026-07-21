from definitions import *
from numericaldomain import DomainBase
import numpy as np


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
        self.lb = np.asarray(lb, dtype=np.float32)
        self.ub = np.asarray(ub, dtype=np.float32)

        if d == 2:
            self.dx = (self.ub[0] - self.lb[0]) / max(self.nx, 1)
            self.dz = (self.ub[1] - self.lb[1]) / max(self.ny, 1)

            self.occ = np.zeros((self.nx, self.ny), dtype=np.int32)
            self.max_edges = self.nx * self.ny * 4
            self.edge_p0 = np.zeros((self.max_edges, 2), dtype=np.float32)
            self.edge_p1 = np.zeros((self.max_edges, 2), dtype=np.float32)
            self.edge_n = np.zeros((self.max_edges, 2), dtype=np.float32)
            self.edge_count = 0
        else:
            assert nz is not None, "VoxelGridDomain(d=3) requires nz to be specified"
            self.nz = int(nz)
            self.dx = (self.ub[0] - self.lb[0]) / max(self.nx, 1)
            self.dy = (self.ub[1] - self.lb[1]) / max(self.ny, 1)
            self.dz = (self.ub[2] - self.lb[2]) / max(self.nz, 1)

            self.occ = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)
            self.max_faces = self.nx * self.ny * self.nz * 6
            self.face_o = np.zeros((self.max_faces, 3), dtype=np.float32)
            self.face_u = np.zeros((self.max_faces, 3), dtype=np.float32)
            self.face_v = np.zeros((self.max_faces, 3), dtype=np.float32)
            self.face_n = np.zeros((self.max_faces, 3), dtype=np.float32)
            self.face_count = 0

        if occupancy_np is not None:
            assert occupancy_np.shape == (self.nx, self.ny) or (
                d == 3 and occupancy_np.shape == (self.nx, self.ny, self.nz)
            )
            self.occ[:] = occupancy_np.astype(np.int32)

        self.nnodes = 1
        self.nelements = 0

        if d == 2:
            self.build_edges()

        self.category_bits = COLLISION_CATEGORY_GROUND
        self.collide_bits = COLLISION_MASK_ALL

    def attach(self, rigidManager, offset: int):
        self.ndOffset = int(offset)
        self.rigidManager = rigidManager

    def _cell_bounds_2d(self, i, j):
        x0 = self.lb[0] + i * self.dx
        x1 = x0 + self.dx
        z0 = self.lb[1] + j * self.dz
        z1 = z0 + self.dz
        return np.array([x0, z0], dtype=np.float32), np.array([x1, z1], dtype=np.float32)

    def build_edges(self):
        self.edge_count = 0
        for i in range(self.nx):
            for j in range(self.ny):
                if self.occ[i, j] != 1:
                    continue
                pmin, pmax = self._cell_bounds_2d(i, j)
                if i - 1 < 0 or self.occ[i - 1, j] == 0:
                    eid = self.edge_count
                    self.edge_count += 1
                    if eid < self.max_edges:
                        self.edge_p0[eid] = [pmin[0], pmin[1]]
                        self.edge_p1[eid] = [pmin[0], pmax[1]]
                        self.edge_n[eid] = [-1.0, 0.0]
                if i + 1 >= self.nx or self.occ[i + 1, j] == 0:
                    eid = self.edge_count
                    self.edge_count += 1
                    if eid < self.max_edges:
                        self.edge_p0[eid] = [pmax[0], pmin[1]]
                        self.edge_p1[eid] = [pmax[0], pmax[1]]
                        self.edge_n[eid] = [1.0, 0.0]
                if j - 1 < 0 or self.occ[i, j - 1] == 0:
                    eid = self.edge_count
                    self.edge_count += 1
                    if eid < self.max_edges:
                        self.edge_p0[eid] = [pmin[0], pmin[1]]
                        self.edge_p1[eid] = [pmax[0], pmin[1]]
                        self.edge_n[eid] = [0.0, -1.0]
                if j + 1 >= self.ny or self.occ[i, j + 1] == 0:
                    eid = self.edge_count
                    self.edge_count += 1
                    if eid < self.max_edges:
                        self.edge_p0[eid] = [pmin[0], pmax[1]]
                        self.edge_p1[eid] = [pmax[0], pmax[1]]
                        self.edge_n[eid] = [0.0, 1.0]

    @staticmethod
    def _closest_on_edge_2d(p, a, b):
        ab = b - a
        ab2 = float(np.dot(ab, ab)) + 1e-12
        t = float(np.dot(p - a, ab)) / ab2
        return a + t * ab, t

    def signed_distance_to_edges_2d(self, p, limit_penetration):
        p = np.asarray(p, dtype=np.float32)
        best_d = 1e9
        best_n = np.array([0.0, 1.0], dtype=np.float32)
        best_c = p.copy()

        i = int(np.floor((p[0] - self.lb[0]) / self.dx))
        j = int(np.floor((p[1] - self.lb[1]) / self.dz))

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

                dist_to_faces = np.array([dist_neg_x, dist_pos_x, dist_neg_y, dist_pos_y], dtype=np.float32)
                min_face_idx = int(np.argmin(dist_to_faces))

                if min_face_idx == 0:
                    best_d = -dist_neg_x
                    best_n = np.array([-1.0, 0.0], dtype=np.float32)
                    best_c = np.array([p[0] - dist_neg_x, p[1]], dtype=np.float32)
                elif min_face_idx == 1:
                    best_d = -dist_pos_x
                    best_n = np.array([1.0, 0.0], dtype=np.float32)
                    best_c = np.array([p[0] + dist_pos_x, p[1]], dtype=np.float32)
                elif min_face_idx == 2:
                    best_d = -dist_neg_y
                    best_n = np.array([0.0, -1.0], dtype=np.float32)
                    best_c = np.array([p[0], p[1] - dist_neg_y], dtype=np.float32)
                else:
                    best_d = -dist_pos_y
                    best_n = np.array([0.0, 1.0], dtype=np.float32)
                    best_c = np.array([p[0], p[1] + dist_pos_y], dtype=np.float32)
            else:
                ec = int(self.edge_count)
                limit_penetration = float(limit_penetration)
                for eid in range(ec):
                    a = self.edge_p0[eid]
                    b = self.edge_p1[eid]
                    n = self.edge_n[eid]
                    c, t = self._closest_on_edge_2d(p, a, b)
                    d = float(np.dot(p - c, n))
                    limit_penetration = max(limit_penetration, 0.1 * float(np.linalg.norm(b - a)))
                    if d < best_d and (0.0 <= t <= 1.0) and abs(d) < limit_penetration:
                        best_d = d
                        best_n = n.copy()
                        best_c = c
        return best_d, best_n, best_c

    def getBBox(self):
        return self.lb, self.ub

    def getBoundaryMesh(self):
        ec = int(self.edge_count)
        if ec == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

        p0 = self.edge_p0[:ec]
        p1 = self.edge_p1[:ec]

        vertices = []
        edges = []
        for k in range(ec):
            idx0 = len(vertices)
            vertices.extend([p0[k], p1[k]])
            edges.append([idx0, idx0 + 1])

        return np.asarray(vertices, dtype=np.float32), np.asarray(edges, dtype=np.int32)

    def draw(self, gui, color=0x55AAFF):
        ec = int(self.edge_count)
        for eid in range(ec):
            gui.line(self.edge_p0[eid], self.edge_p1[eid], color=color)
