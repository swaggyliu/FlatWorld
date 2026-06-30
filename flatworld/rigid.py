from definitions import *
from mesh_utils import calculate_mesh_inertia
from numericaldomain import DomainBase
import numpy as np
import taichi as ti
import time
from utils import *


# ============================================================================================
@ti.data_oriented
class RigidBase:
    def __init__(self, d):
        self.d = d
        self.numNodes = 1
        self.mesh = None
        self.initial_quat = None
        # rotational inertia in body frame. For 2D this is a scalar (around z),
        # for 3D this is a 3x3 inertia matrix stored as a Taichi Matrix.
        self.inertia_body = None
        # world-space inertia (scalar for 2D, 3x3 matrix for 3D)

        # Transform for rendering/export (offset, scale, rotation)
        self.transform = None

    def getRefPoint(self):
        """Return rigid reference point (override in subclasses)."""
        pass

    def getRadius(self):
        """Return approximate radius for bounding volume (override in subclasses)."""
        return 0.0

    def getPrimary(self):
        return [0.0] * self.d


# ============================================================================================
# ============================================================================================
# ============================================================================================
# ============================================================================================


@ti.data_oriented
class BallRigid(RigidBase):
    def __init__(self, d, origin, radius, mass, angle=None, inertia=None):
        super().__init__(d)
        self.origin = ti.Vector(origin)
        self.radius = radius
        self.mass = mass
        self.rtype = RigidType.BALL
        self.minSize = radius * 2.0
        self.angle = ti.Vector(angle) if angle is not None else ti.Vector([0.0 for _ in range(d)])
        if inertia is not None:
            self.inertia_body = inertia
        else:
            self.inertia_body = 0.5 * self.mass * (self.radius**2)
       
    def getRefPoint(self):
        """Return ball reference point (center)."""
        return self.origin

    def getRadius(self):
        return self.radius

    def draw(self, gui, rigidManager, ndOffset, color=0xFFFFFF, resolution=10):
        center = rigidManager.rigidParams[ndOffset, 0]
        gui.circle(center.to_numpy()[:2], radius=self.radius * resolution, color=color)


@ti.data_oriented
class BoxRigid(RigidBase):
    def __init__(self, d, origin, ext, angle, mass, inertia=None):
        super().__init__(d)
        self.numShapeNodes = 4
        self.ext = ti.Vector(ext)
        self.angle = ti.Vector(angle)
        self.origin = ti.Vector(origin)
        self.mass = mass
        self.rtype = RigidType.BOX
        if inertia is not None:
            self.inertia_body = inertia
        else:
            w = float(self.ext[0])
            h = float(self.ext[1])
            self.inertia_body = (1.0 / 12.0) * self.mass * (w * w + h * h)


        self.minSize = min(self.ext[0], self.ext[1])

    def getRefPoint(self):
        """Return box reference point (center)."""
        return self.origin

    def getPrimary(self):
        return self.ext

    def draw(self, gui, rigidManager, ndOffset, color=0xFFFFFF, resolution=10):
        # Fetch current state from manager
        center = rigidManager.rigidParams[ndOffset, 0].to_numpy()
        extent = rigidManager.rigidParams[ndOffset, 1].to_numpy()

        rotMat = rigidManager.cached_rotation_matrix[ndOffset].to_numpy()

        half_ext = 0.5 * extent
        corners_local = np.array(
            [
                [-half_ext[0], -half_ext[1]],
                [half_ext[0], -half_ext[1]],
                [half_ext[0], half_ext[1]],
                [-half_ext[0], half_ext[1]],
            ]
        )
        vertices = (rotMat @ corners_local.T).T + center

        for j in range(4):
            gui.line(vertices[j][:2], vertices[(j + 1) % 4][:2], radius=3, color=color)


@ti.data_oriented
class CapsuleRigid(RigidBase):
    def __init__(self, d, lc, uc, angle, radius, mass, inertia=None):
        super().__init__(d)

        self.lc = ti.Vector(lc)
        self.uc = ti.Vector(uc)
        self.radius = radius
        self.mass = mass
        self.rtype = RigidType.CAPSULE
        self.origin = 0.5 * (self.lc + self.uc)
        self.angle = ti.Vector(angle)

        axis = self.uc - self.lc
        axis_len = axis.norm()
        if inertia is not None:
            self.inertia_body = inertia
        else:
            self.inertia_body = (1.0 / 12.0) * self.mass * (axis_len**2) + 0.5 * self.mass * (self.radius**2)


        self.minSize = min(self.radius * 2.0, axis_len)

    def getPrimary(self):
        return self.lc

    def getRadius(self):
        return self.radius

    def getRefPoint(self):
        """Return capsule reference point (midpoint of axis)."""
        return self.origin

    def draw(self, gui, rigidManager, ndOffset, color=0xFFFFFF, resolution=10):
        center = rigidManager.rigidParams[ndOffset, 0].to_numpy()
        lcdir = rigidManager.rigidParams[ndOffset, 1].to_numpy()
        rotMat = rigidManager.cached_rotation_matrix[ndOffset].to_numpy()
        newlc = rotMat @ lcdir + center
        newuc = 2 * center - newlc
        axis = newuc - newlc
        axis_len = np.linalg.norm(axis)
        axis_dir = axis / axis_len
        t_dir = np.array([axis_dir[1], -axis_dir[0]])
        pos_lc0 = newlc + t_dir * self.radius
        pos_lc1 = newlc - t_dir * self.radius
        pos_uc0 = newuc + t_dir * self.radius
        pos_uc1 = newuc - t_dir * self.radius
        gui.triangle(pos_lc0[:2], pos_uc0[:2], pos_uc1[:2], color=color)
        gui.triangle(pos_lc0[:2], pos_lc1[:2], pos_uc1[:2], color=color)
        gui.circle(newlc[:2], radius=self.radius * resolution, color=color)
        gui.circle(newuc[:2], radius=self.radius * resolution, color=color)


@ti.data_oriented
class MeshRigid(RigidBase):
    def __init__(self, d, mesh, angle, mass, origin=None, inertia=None, transform=None):
        assert d == mesh.d
        super().__init__(d)

        self.mesh = mesh
        self.mass = mass
        if origin is not None:
            self.origin = ti.Vector(origin)
        else:
            self.origin = ti.Vector(self.mesh.getCenterPoint())

        self.rtype = RigidType.MESH
        self.angle = ti.Vector(angle)

        # Compute AABB from mesh coordinates to determine minSize
        coords = self.mesh.coords
        lb = np.min(coords, axis=0)
        ub = np.max(coords, axis=0)
        extents = ub - lb
        self.minSize = np.min(extents)

        if inertia is not None:
            self.inertia_body = inertia
        else:
            inertia_val = calculate_mesh_inertia(self.mesh, self.mass)
            self.inertia_body = inertia_val

        self.transform = transform

    def getRefPoint(self):
        """Return mesh center point as reference point."""
        return self.origin

    def draw(self, gui, rigidManager, ndOffset, color=0xFFFFFF, resolution=10):
        mesh_local_id = rigidManager.rigid2MeshIndices[ndOffset]
        node_offset = rigidManager.meshBoundaryNodeOffset[mesh_local_id]
        num_nodes = rigidManager.meshBoundaryNodeCount[mesh_local_id]
        elem_offset = rigidManager.meshBoundaryElementOffset[mesh_local_id]
        num_elems = rigidManager.meshBoundaryElementCount[mesh_local_id]

        # Get boundary node positions
        pos = rigidManager.meshBoundaryCoords.to_numpy()[node_offset : node_offset + num_nodes, :2]

        # Get boundary element connectivity (local indices)
        elements = rigidManager.meshBoundaryElements.to_numpy()[elem_offset : elem_offset + num_elems]

        # 2D: draw edges (stored as [n0, n1, -1])
        a, b = elements[:, 0], elements[:, 1]
        gui.lines(pos[a] - node_offset, pos[b] - node_offset, radius=2, color=color)
