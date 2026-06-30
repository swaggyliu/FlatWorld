from definitions import *
from numericaldomain import DomainBase
import taichi as ti


@ti.data_oriented
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

    def attach(self, rigidManager, offset: ti.i32, domain_idx: int):
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

        rigidManager.rigidDomainIds[self.ndOffset] = ti.Vector(
            [
                domain_idx,
                self.rigid.rtype,
                consider_flag,
            ]
        )
        print(
            f"Attaching RigidBodyDomain '{self.name}' at offset {self.ndOffset} with category_bits={bin(category_bits)} and collide_bits={bin(collide_bits)}"
        )
        rigidManager.category_bits[self.ndOffset] = category_bits
        rigidManager.collide_bits[self.ndOffset] = collide_bits

        # Store reverse mapping (domain_idx -> rigid_idx) for fast lookup
        rigidManager.domainToRigid[domain_idx] = self.ndOffset
        # Set per-rigid friction coefficient (default 0.0)
        rigidManager.contactParams[self.ndOffset] = ti.Vector([self.friction, self.restitution])

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
        domain_idx = int(self.rigidManager.rigidDomainIds[self.ndOffset][0])
        return self.rigidManager.aabb[domain_idx, 0], self.rigidManager.aabb[domain_idx, 1]

    def getCurrentRefPoint(self):
        return self.rigidManager.rigidParams[self.ndOffset, 0].to_numpy()

    def getCurrentRefAngles(self):
        """Return the current reference orientation angles (Euler angles) of the rigid body."""
        angle = float(self.rigidManager.quat[self.ndOffset][0])
        if hasattr(self.rigidManager, "visual_angle"):
            angle += float(self.rigidManager.visual_angle[self.ndOffset])
        return ti.Vector([angle]).to_numpy()

    def getCurrentVelocity(self):
        return self.rigidManager.V[self.ndOffset].to_numpy()

    def getCurrentAngularVelocity(self):
        return self.rigidManager.RotV[self.ndOffset].to_numpy()

    def draw(self, gui, color=0xFFFFFF, resolution=10):
        self.rigid.draw(gui, self.rigidManager, self.ndOffset, color, resolution)
