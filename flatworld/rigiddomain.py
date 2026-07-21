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


def _patch_array(arr, index, value):
    """Host write for Warp 1.14+ (no ``arr[i] =`` from Python)."""
    if hasattr(arr, "numpy") and hasattr(arr, "assign"):
        np_arr = arr.numpy()
        np_arr[index] = value
        arr.assign(np_arr)
    else:
        arr[index] = value


class RigidBodyDomain(DomainBase):
    def __init__(
        self,
        rigid,
        bcs=[],
        considerContact=True,
        considerGroundContact=True,
        initials=[],
        friction: float = 0.0,
        restitution=0.9,
        name="",
        visual_mesh=None,
        collision_shapes=None,
        category_bits=COLLISION_CATEGORY_ORDINARY_RIGID,
        collide_bits=COLLISION_MASK_ALL,
    ):
        self.rigid = rigid
        self.d = rigid.d
        self.bcs = bcs
        self.considerContact = considerContact
        self.considerGroundContact = considerGroundContact
        self.type = DomainType.RIGID
        self.ref = rigid.getRefPoint()
        self.nnodes = rigid.numNodes
        self.nelements = 0
        self.initials = initials
        # Per-rigid Coulomb friction coefficient (0.0 = frictionless)
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.name = name
        # Optional visual mesh for rendering (separate from collision rigid).
        # Expected format: {'rest_vertices': np.ndarray (N,3), 'elements': np.ndarray (M,3)}
        self.visual_mesh = visual_mesh
        # Optional list of compound collision sub-shapes (e.g. 4 spheres on a foot).
        # Each entry: {'type': RigidType, 'local_pos': [x,y,z], 'radius': float}
        self.collision_shapes = collision_shapes or []
        # Optional ODE-style collision masks. If None, resolved from
        # considerContact/considerGroundContact at attach time.
        self.category_bits = category_bits
        self.collide_bits = collide_bits

    def attach(self, rigidManager, offset: int, domain_idx: int):
        """Attach this rigid domain to the RigidManager at the given offset.

        Args:
            rigidManager: RigidManager instance
            offset: Rigid index offset
            domain_idx: Domain index in the global domains list
        """
        # Data arrays
        self.ndOffset = int(offset)
        self.rigidManager = rigidManager

        # Set rigid domain ID with domain index, rigid type, and contact consideration flag
        # Flag meanings: 0 = no contact, 1 = all contact, 2 = ground contact only
        consider_flag = 1 if self.considerContact and rigidManager.considerRigidRigidContact else 0
        if not self.considerContact and self.considerGroundContact:
            consider_flag = 2  # Special flag: ground contact only, no rigid-rigid contact

        category_bits = self.category_bits
        if category_bits is None:
            category_bits = COLLISION_CATEGORY_ROBOT
        category_bits = int(category_bits) & 0b11111111

        collide_bits = self.collide_bits
        if collide_bits is None:
            if self.considerContact:
                collide_bits = COLLISION_MASK_ALL
            elif self.considerGroundContact:
                collide_bits = COLLISION_CATEGORY_GROUND
            else:
                collide_bits = 0
            if not self.considerGroundContact:
                collide_bits = collide_bits & (~COLLISION_CATEGORY_GROUND)
        collide_bits = int(collide_bits) & 0b11111111

        _patch_array(
            rigidManager.rigidDomainIds,
            self.ndOffset,
            [domain_idx, self.rigid.rtype, consider_flag],
        )
        print(
            f"Attaching RigidBodyDomain '{self.name}' at offset {self.ndOffset} with category_bits={bin(category_bits)} and collide_bits={bin(collide_bits)}"
        )
        _patch_array(rigidManager.category_bits, self.ndOffset, category_bits)
        _patch_array(rigidManager.collide_bits, self.ndOffset, collide_bits)

        # Store reverse mapping (domain_idx -> rigid_idx) for fast lookup
        _patch_array(rigidManager.domainToRigid, domain_idx, self.ndOffset)
        # Set per-rigid friction coefficient (default 0.0)
        _patch_array(rigidManager.contactParams, self.ndOffset, [self.friction, self.restitution])

        # Register visual mesh data for VTU export (if provided)
        if self.visual_mesh is not None:
            rigidManager.visual_mesh_data[self.ndOffset] = self.visual_mesh

        # Apply initial conditions (velocity, acceleration, etc.)
        for initialCondition in self.initials:
            if initialCondition.type in (VTYPE, ATYPE):
                initialCondition.update(self.rigidManager.V, self.rigidManager.d, offset, 1)
            else:
                initialCondition.update(self.rigidManager.RotV, self.rigidManager.d, offset, 1)

    def getBBox(
        self,
    ):
        # Get bounding box using global domain index
        domain_ids = _to_np(self.rigidManager.rigidDomainIds)
        domain_idx = int(np.asarray(domain_ids[self.ndOffset]).reshape(-1)[0])
        aabb = _to_np(self.rigidManager.aabb)
        return aabb[domain_idx, 0], aabb[domain_idx, 1]

    def getCurrentRefPoint(self):
        params = _to_np(self.rigidManager.rigidParams)
        return _vec_np(params[self.ndOffset, 0])

    def getCurrentRefAngles(self):
        """Return the current reference orientation angles (Euler angles) of the rigid body."""
        quat = _to_np(self.rigidManager.quat)
        angle = float(np.asarray(quat[self.ndOffset]).reshape(-1)[0])
        if hasattr(self.rigidManager, "visual_angle"):
            vis = _to_np(self.rigidManager.visual_angle)
            angle += float(np.asarray(vis[self.ndOffset]).reshape(-1)[0])
        return np.array([angle], dtype=np.float32)

    def getCurrentVelocity(self):
        return _vec_np(_to_np(self.rigidManager.V)[self.ndOffset])

    def getCurrentAngularVelocity(self):
        return _vec_np(_to_np(self.rigidManager.RotV)[self.ndOffset])

    def draw(self, gui, color=0xFFFFFF, resolution=10):
        self.rigid.draw(gui, self.rigidManager, self.ndOffset, color, resolution)
