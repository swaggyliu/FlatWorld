from definitions import *
from mesh_utils import calculate_mesh_inertia
import numpy as np
from utils import Transform  # noqa: F401 — re-exported for flatworld.__init__


def _to_np(x):
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        return x.numpy()
    if hasattr(x, "to_numpy") and callable(getattr(x, "to_numpy")):
        return x.to_numpy()
    return np.asarray(x)


def _vec_np(v):
    return np.asarray(_to_np(v), dtype=np.float32).reshape(-1)


# ============================================================================================
class RigidBase:
    def __init__(self, d):
        self.d = d
        self.numNodes = 1
        self.mesh = None
        self.initial_quat = None
        # rotational inertia in body frame. For 2D this is a scalar (around z),
        # for 3D this is a 3x3 inertia matrix stored as a numpy array.
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
        return np.zeros((self.d,), dtype=np.float32)


# ============================================================================================
# ============================================================================================
# ============================================================================================
# ============================================================================================


class BallRigid(RigidBase):
    def __init__(self, d, origin, radius, mass, angle=None, inertia=None):
        super().__init__(d)
        self.origin = np.asarray(origin, dtype=np.float32)
        self.radius = radius
        self.mass = mass
        self.rtype = RigidType.BALL
        self.minSize = radius * 2.0
        self.angle = (
            np.asarray(angle, dtype=np.float32)
            if angle is not None
            else np.zeros((d,), dtype=np.float32)
        )
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
        params = _to_np(rigidManager.rigidParams)
        center = _vec_np(params[ndOffset, 0])
        if hasattr(gui, "circle_world"):
            gui.circle_world(center[:2], radius=float(self.radius), color=color)
        else:
            gui.circle(center[:2], radius=self.radius * resolution, color=color)


class BoxRigid(RigidBase):
    def __init__(self, d, origin, ext, angle, mass, inertia=None):
        super().__init__(d)
        self.numShapeNodes = 4
        self.ext = np.asarray(ext, dtype=np.float32)
        self.angle = np.asarray(angle, dtype=np.float32)
        self.origin = np.asarray(origin, dtype=np.float32)
        self.mass = mass
        self.rtype = RigidType.BOX
        if inertia is not None:
            self.inertia_body = inertia
        else:
            w = float(self.ext[0])
            h = float(self.ext[1])
            self.inertia_body = (1.0 / 12.0) * self.mass * (w * w + h * h)

        self.minSize = min(float(self.ext[0]), float(self.ext[1]))

    def getRefPoint(self):
        """Return box reference point (center)."""
        return self.origin

    def getPrimary(self):
        return self.ext

    def draw(self, gui, rigidManager, ndOffset, color=0xFFFFFF, resolution=10):
        # Fetch current state from manager (use .numpy() when manager is Warp)
        params = _to_np(rigidManager.rigidParams)
        center = _vec_np(params[ndOffset, 0])
        extent = _vec_np(params[ndOffset, 1])

        rot_cache = _to_np(rigidManager.cached_rotation_matrix)
        rotMat = np.asarray(rot_cache[ndOffset], dtype=np.float32)

        half_ext = 0.5 * extent
        corners_local = np.array(
            [
                [-half_ext[0], -half_ext[1]],
                [half_ext[0], -half_ext[1]],
                [half_ext[0], half_ext[1]],
                [-half_ext[0], half_ext[1]],
            ],
            dtype=np.float32,
        )
        vertices = (rotMat @ corners_local.T).T + center

        for j in range(4):
            gui.line(vertices[j][:2], vertices[(j + 1) % 4][:2], radius=3, color=color)


class CapsuleRigid(RigidBase):
    def __init__(self, d, lc, uc, angle, radius, mass, inertia=None):
        super().__init__(d)

        self.lc = np.asarray(lc, dtype=np.float32)
        self.uc = np.asarray(uc, dtype=np.float32)
        self.radius = radius
        self.mass = mass
        self.rtype = RigidType.CAPSULE
        self.origin = 0.5 * (self.lc + self.uc)
        self.angle = np.asarray(angle, dtype=np.float32)

        axis = self.uc - self.lc
        axis_len = float(np.linalg.norm(axis))
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
        params = _to_np(rigidManager.rigidParams)
        center = _vec_np(params[ndOffset, 0])
        lcdir = _vec_np(params[ndOffset, 1])
        rotMat = np.asarray(_to_np(rigidManager.cached_rotation_matrix)[ndOffset], dtype=np.float32)
        newlc = rotMat @ lcdir + center
        newuc = 2 * center - newlc
        axis = newuc - newlc
        axis_len = float(np.linalg.norm(axis))
        axis_dir = axis / max(axis_len, 1e-12)
        t_dir = np.array([axis_dir[1], -axis_dir[0]], dtype=np.float32)
        pos_lc0 = newlc + t_dir * self.radius
        pos_lc1 = newlc - t_dir * self.radius
        pos_uc0 = newuc + t_dir * self.radius
        pos_uc1 = newuc - t_dir * self.radius
        # Viewer has no triangle — outline the capsule body with lines
        gui.line(pos_lc0[:2], pos_uc0[:2], radius=2, color=color)
        gui.line(pos_uc0[:2], pos_uc1[:2], radius=2, color=color)
        gui.line(pos_uc1[:2], pos_lc1[:2], radius=2, color=color)
        gui.line(pos_lc1[:2], pos_lc0[:2], radius=2, color=color)
        if hasattr(gui, "circle_world"):
            gui.circle_world(newlc[:2], radius=float(self.radius), color=color)
            gui.circle_world(newuc[:2], radius=float(self.radius), color=color)
        else:
            gui.circle(newlc[:2], radius=self.radius * resolution, color=color)
            gui.circle(newuc[:2], radius=self.radius * resolution, color=color)


class MeshRigid(RigidBase):
    def __init__(self, d, mesh, angle, mass, origin=None, inertia=None, transform=None):
        assert d == mesh.d
        super().__init__(d)

        self.mesh = mesh
        self.mass = mass
        if origin is not None:
            self.origin = np.asarray(origin, dtype=np.float32)
        else:
            self.origin = np.asarray(self.mesh.getCenterPoint(), dtype=np.float32)

        self.rtype = RigidType.MESH
        self.angle = np.asarray(angle, dtype=np.float32)

        # Compute AABB from mesh coordinates to determine minSize
        coords = self.mesh.coords
        lb = np.min(coords, axis=0)
        ub = np.max(coords, axis=0)
        extents = ub - lb
        self.minSize = float(np.min(extents))

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
        mesh_local_id = int(_to_np(rigidManager.rigid2MeshIndices)[ndOffset])
        node_offset = int(_to_np(rigidManager.meshBoundaryNodeOffset)[mesh_local_id])
        num_nodes = int(_to_np(rigidManager.meshBoundaryNodeCount)[mesh_local_id])
        elem_offset = int(_to_np(rigidManager.meshBoundaryElementOffset)[mesh_local_id])
        num_elems = int(_to_np(rigidManager.meshBoundaryElementCount)[mesh_local_id])

        if num_nodes <= 0 or num_elems <= 0:
            return

        # Get boundary node positions (use .numpy() when manager is Warp)
        pos = _to_np(rigidManager.meshBoundaryCoords)[node_offset : node_offset + num_nodes, :2]

        # Get boundary element connectivity (local indices)
        elements = np.asarray(
            _to_np(rigidManager.meshBoundaryElements)[elem_offset : elem_offset + num_elems],
            dtype=np.int32,
        )
        if elements.ndim == 1 or elements.shape[0] == 0:
            return

        # 2D: draw edges (stored as [n0, n1, -1]) — safe local/global indexing
        a = np.asarray(elements[:, 0], dtype=np.int32)
        b = np.asarray(elements[:, 1], dtype=np.int32)
        if a.size == 0:
            return
        # Connectivity may be global or already local relative to the sliced pos array
        if int(a.max()) >= num_nodes or int(b.max()) >= num_nodes:
            a = a - node_offset
            b = b - node_offset
        valid = (a >= 0) & (a < num_nodes) & (b >= 0) & (b < num_nodes)
        if np.any(valid):
            gui.lines(pos[a[valid]], pos[b[valid]], radius=2, color=color)
