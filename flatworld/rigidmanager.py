# Here we assume all the domain the nodeIds and elementIds are arrange as 1...n
from bvh import CollisionDetector
from contact_detection import (
    detectPointToAnalyticalPlane,
    detectPointToMeshBoundary,
    detectPointToPrimitive,
)
from definitions import *
from mesh import *
from gjk import gjk_epa_collision
from joint_kernels import assemble_single_joint_rows
import numpy as np
from operator import pos
from rigid import *
from sat import *
from spatialmanager import SpatialHashManager
import taichi as ti
from utils import *
import time


@ti.data_oriented
class RigidManager:
    def __init__(self, d, domains, joints, bvh=None, skip_spatial_hash=False, considerRigidRigidContact=True, use_pd=0):
        """Initialize RigidManager and allocate Taichi fields for rigids and state.

        Args:
            d: Dimension (2)
            domains: List of domain objects
            joints: List of joint objects
            bvh: Shared BVH instance from ExplicitLoop (if None, creates local one for standalone use)
            skip_spatial_hash: If True, skip SpatialHashManager creation (saves ~8GB VRAM)
            considerRigidRigidContact: If True, consider rigid-rigid contact
            use_pd: 0 = no PD, 1 = velocity PD, 2 = torque PD
        """

        self.skip_spatial_hash = skip_spatial_hash
        self.use_pd = use_pd
        self.considerRigidRigidContact = considerRigidRigidContact
        # --- OPTIMIZATION: Pre-scan domains to allocate only needed memory ---
        count_rigid = 0
        count_anal = 0
        count_mesh_rigids = 0
        count_mesh_nodes = 0
        count_mesh_elems = 0
        count_compound_shapes = 0

        for dom in domains:
            if dom.type == DomainType.RIGID:
                count_rigid += 1
                if dom.rigid.rtype == RigidType.MESH:
                    count_mesh_rigids += 1
                    count_mesh_nodes += dom.rigid.mesh.numBoundNodes
                    count_mesh_elems += dom.rigid.mesh.numBoundElements
                # Count compound collision sub-shapes
                if hasattr(dom, "collision_shapes") and dom.collision_shapes:
                    count_compound_shapes += len(dom.collision_shapes)
            elif dom.type in [DomainType.ANALYTICAL, DomainType.HEIGHTFIELD, DomainType.VOXELMAP]:
                count_anal += 1

        # Dynamic allocation (count + buffer) - drastically reduces memory when few rigids/meshes are used
        num_joints = 0 if joints is None else len(joints)
        self.MAX_NODES = max(count_rigid + count_anal + 64, 128)

        print("Number of maxNodes allocated:", self.MAX_NODES)
        self.MAX_ANAL = max(count_anal + 8, 16)
        self.MAX_MESH = max(count_mesh_rigids + 18, 16)
        self.MAX_JOINTS = max(num_joints + 8, 128)

        self.d = d
        self.numDomains = len(domains)
        # keep a reference to domain objects for host-side operations (mesh handling)
        self.domains = domains
        self.rigidDomainIds = ti.Vector.field(3, ti.i32, self.MAX_NODES)
        self.category_bits = ti.field(ti.u32, self.MAX_NODES)
        self.collide_bits = ti.field(ti.u32, self.MAX_NODES)
        self.category_bits.fill(COLLISION_CATEGORY_ROBOT)
        self.collide_bits.fill(COLLISION_MASK_ALL)
        maxDomains = max(len(domains), self.MAX_NODES)
        self.domainToRigid = ti.field(ti.i32, maxDomains)
        self.domainToRigid.fill(-1)
        # Packed per-rigid parameters: rows hold different vector groups per-rigid.
        # row 0: reference point (current coords)
        # row 1: primary shape params (extents, endpoint1 (for capsule), normal for analytical domain etc.)
        self.rigidParams = ti.Vector.field(self.d, ti.f32, (self.MAX_NODES, 2))

        self.numRigids = 0
        self.numAnalytical = 0
        self.numMesh = 0
        self.numMeshRigidInContact = 0
        self.numRigidInContact = 0
        self.numRigidGroundContact = 0
        # When 0, ground narrow-phase skips AABB early-out so stale AABBs
        # cannot suppress first-contact detection in no-collision fast path.
        self._ground_use_aabb_early_out = ti.field(ti.i32, shape=())
        self._ground_use_aabb_early_out[None] = 1
        self.hasHeightFieldOrVoxel = False  # Set True if any HeightField/Voxel domains exist

        self.contact_erp = 0.2  # Baumgarte error reduction parameter for ground contacts
        self.restitution_velocity_threshold = 1.0  # Ignore restitution for low-speed contacts
        self.skip_bvh = False  # When True, skip BVH broadphase and use direct ground pair generation
        self.control_dt = 1.0 / 60.0  # default: 60 Hz control
        # These are nodal data
        self.bcNodes = ti.field(ti.i32, self.MAX_NODES)
        self.bcGValues = ti.Vector.field(self.d, ti.float32, self.MAX_NODES)
        self.bcTValues = ti.Vector.field(self.d, ti.float32, self.MAX_NODES)
        self.bcRValues = ti.Vector.field(1, ti.float32, self.MAX_NODES)

        # Mesh rigid storage - boundary elements only for contact detection
        # Dynamic resizing based on pre-scan results (count_mesh_nodes is calculated above)
        self.meshRigidContactMarginRatio = 0.02
        if count_mesh_nodes > 0:
            self.MAX_BOUNDARY_NODES = max(count_mesh_nodes + 4096, 4096)
            self.MAX_BOUNDARY_ELEMENTS = max(count_mesh_elems + 8192, 8192)

            # Boundary node coordinates (world space and local)
            self.meshBoundaryCoords = ti.Vector.field(self.d, ti.float32, self.MAX_BOUNDARY_NODES)
            # Boundary element connectivity (edges for 2D, triangles for 3D)
            # For 2D: each element has 2 node indices
            # For 3D: each element has 3 node indices
            self.meshBoundaryElements = ti.Vector.field(3, ti.i32, self.MAX_BOUNDARY_ELEMENTS)
            # Cached per-element AABBs (updated once per substep)
            self.meshElemLB = ti.Vector.field(self.d, ti.f32, self.MAX_BOUNDARY_ELEMENTS)
            self.meshElemUB = ti.Vector.field(self.d, ti.f32, self.MAX_BOUNDARY_ELEMENTS)
            self.meshElemMarginBase = ti.field(ti.f32, self.MAX_BOUNDARY_ELEMENTS)

            # Per-mesh bookkeeping
            self.meshBoundaryNodeCount = ti.field(ti.i32, self.MAX_MESH)  # number of boundary nodes per mesh
            self.meshBoundaryNodeOffset = ti.field(ti.i32, self.MAX_MESH)  # starting index in meshBoundaryCoords
            self.meshBoundaryElementCount = ti.field(ti.i32, self.MAX_MESH)  # number of boundary elements per mesh
            self.meshBoundaryElementOffset = ti.field(ti.i32, self.MAX_MESH)  # starting index in meshBoundaryElements

            # Index mappings
            self.mesh2RigidIndices = ti.field(ti.i32, self.MAX_MESH)
            self.mesh2RigidIndices.fill(-1)
            self.rigid2MeshIndices = ti.field(ti.i32, self.MAX_NODES)
            self.rigid2MeshIndices.fill(-1)

            # ==== MESH INSTANCING: Geometry Pool (shared boundary data) ====
            # Pool storage for unique mesh geometries
            self.MAX_POOL_GEOMETRIES = 256  # Max unique mesh shapes
            self.num_pool_geometries = 0

            # Pool only stores UNIQUE geometries; size matches total to avoid over-capacity
            # (when many clones exist, pool will naturally stop growing after unique meshes are added)
            # Falls back to legacy path gracefully if pool capacity is exceeded
            self.MAX_POOL_NODES = count_mesh_nodes + 4096
            self.MAX_POOL_ELEMENTS = count_mesh_elems + 8192

            # Pool boundary data (same structure as existing, but deduplicated)
            self.pool_boundary_lrs = ti.Vector.field(self.d, ti.float32, self.MAX_POOL_NODES)
            self.pool_boundary_elements = ti.Vector.field(3, ti.i32, self.MAX_POOL_ELEMENTS)

            # Per-geometry bookkeeping in pool
            self.pool_node_count = ti.field(ti.i32, self.MAX_POOL_GEOMETRIES)
            self.pool_node_offset = ti.field(ti.i32, self.MAX_POOL_GEOMETRIES)
            self.pool_elem_count = ti.field(ti.i32, self.MAX_POOL_GEOMETRIES)
            self.pool_elem_offset = ti.field(ti.i32, self.MAX_POOL_GEOMETRIES)

            # Hash-based deduplication lookup (Python-side dict for Phase 1)
            self.pool_hash_to_id = {}  # maps mesh_hash -> pool_geometry_id

            # ==== MESH INSTANCING: Instance Manager (per-rigid transforms) ====
            # Maps each mesh rigid to its pool geometry + transform
            self.instance_pool_id = ti.field(ti.i32, self.MAX_NODES)  # rigid_idx -> pool_geom_id
            self.instance_pool_id.fill(-1)  # -1 = not using pool (legacy path)

            self.total_pool_nodes = 0
            self.total_pool_elements = 0

            # Transform storage for mesh rigids (scale component)
            self.meshRigidScale = ti.Vector.field(self.d, ti.float32, self.MAX_NODES)
            # Initialize scale to [1,1,1] for all rigids
            for i in range(self.MAX_NODES):
                self.meshRigidScale[i] = ti.Vector([1.0 for _ in range(self.d)])

            # Transform storage for mesh rigids (offset component)
            self.meshRigidOffset = ti.Vector.field(self.d, ti.float32, self.MAX_NODES)
            # Initialize offset to [0,0,0] for all rigids
            for i in range(self.MAX_NODES):
                self.meshRigidOffset[i] = ti.Vector([0.0 for _ in range(self.d)])

            # Active mesh mask for spatial hash population optimization
            self.mesh_active = ti.field(ti.i32, self.MAX_MESH)
            self.mesh_active.fill(0)

        else:
            self.MAX_BOUNDARY_NODES = 1
            self.MAX_BOUNDARY_ELEMENTS = 1

            # Use 1-element fields to prevent compilation errors when no mesh rigids exist
            self.meshBoundaryCoords = ti.Vector.field(self.d, ti.float32, 1)
            self.meshBoundaryElements = ti.Vector.field(3, ti.i32, 1)
            self.meshElemLB = ti.Vector.field(self.d, ti.f32, 1)
            self.meshElemUB = ti.Vector.field(self.d, ti.f32, 1)
            self.meshElemMarginBase = ti.field(ti.f32, 1)
            self.meshBoundaryNodeCount = ti.field(ti.i32, 1)
            self.meshBoundaryNodeOffset = ti.field(ti.i32, 1)
            self.meshBoundaryElementCount = ti.field(ti.i32, 1)
            self.meshBoundaryElementOffset = ti.field(ti.i32, 1)
            self.mesh2RigidIndices = ti.field(ti.i32, 1)
            self.meshRigidScale = ti.Vector.field(self.d, ti.float32, 1)
            self.meshRigidOffset = ti.Vector.field(self.d, ti.float32, 1)
            self.mesh_active = ti.field(ti.i32, 1)

            self.rigid2MeshIndices = ti.field(ti.i32, 1)

            # Mesh instancing fields (even when no mesh rigids exist)
            self.MAX_POOL_GEOMETRIES = 1
            self.num_pool_geometries = 0

            self.pool_boundary_lrs = ti.Vector.field(self.d, ti.float32, 1)
            self.pool_boundary_elements = ti.Vector.field(3, ti.i32, 1)

            self.pool_node_count = ti.field(ti.i32, 1)
            self.pool_node_offset = ti.field(ti.i32, 1)
            self.pool_elem_count = ti.field(ti.i32, 1)
            self.pool_elem_offset = ti.field(ti.i32, 1)

            self.pool_hash_to_id = {}

            self.instance_pool_id = ti.field(ti.i32, self.MAX_NODES)
            self.instance_pool_id.fill(-1)

            self.total_pool_nodes = 0
            self.total_pool_elements = 0

        print("Allocating RigidManager with MAX_NODES =", self.MAX_NODES, "MAX_JOINTS =", self.MAX_JOINTS)

        # Visual mesh data for VTU export (indexed by rigid body index).
        # Populated by RigidBodyDomain.attach() when the domain carries
        # a visual_mesh dict (typically from URDF import with separate
        # collision / visual geometry).
        # Each entry: {'rest_vertices': np.ndarray (N,3), 'elements': np.ndarray (M,3)}
        self.visual_mesh_data = {}

        # Environment ID for collision filtering (batched training)
        # -1 means no env (e.g., ground, single robot), >= 0 means belongs to env_id
        self.rigid_env_id = ti.field(ti.i32, self.MAX_NODES)
        self.rigid_env_id.fill(-1)  # Default: no env filtering

        self.num_envs = 0
        self.joints_per_env = 0

        self.totalBoundaryNodes = 0
        self.totalBoundaryElements = 0

        self.U = ti.Vector.field(self.d, ti.float32, self.MAX_NODES)
        self.V = ti.Vector.field(self.d, ti.f32, self.MAX_NODES)
        self.accumulated_impulse = ti.Vector.field(self.d, ti.float32, self.MAX_NODES)
        # Rotation representation: 2D uses scalar angle, 3D uses quaternion
        self.quat = ti.Vector.field(1, ti.f32, self.MAX_NODES)  # angle for 2D
        self.quat_initial = ti.Vector.field(1, ti.f32, self.MAX_NODES)  # initial orientation snapshot
        self.RotV = ti.Vector.field(1, ti.f32, self.MAX_NODES)
        self.accumulated_rotational_impulse = ti.Vector.field(1, ti.float32, self.MAX_NODES)

        self.V.fill(0.0)
        self.RotV.fill(0.0)

        # AABB is now stored in ExplicitLoop using global domain indices
        # RigidManager will receive a reference to the global aabb field
        self.aabb = None  # Will be set by ExplicitLoop after initialization

        # Pre-allocated fields for collision pair processing (avoid dynamic field allocation)
        self.MAX_COLLISION_PAIRS = 10000  # Maximum collision pairs per frame

        # Buffers for different collision pair types
        self.primitive_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        self.ball_ball_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        self.box_box_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        self.box_ball_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        self.seg_point_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        self.seg_ball_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        self.seg_seg_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)

        self.mesh_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        self.mixed_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_COLLISION_PAIRS)
        # Ground collision pairs need larger buffer for batched environments
        # Each rigid can collide with ground, so need at least MAX_NODES capacity
        self.MAX_GROUND_PAIRS = max(self.MAX_NODES * 2, 4096)
        self.groundprim_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_GROUND_PAIRS)
        self.groundmesh_pairs_buffer = ti.Vector.field(2, ti.i32, self.MAX_GROUND_PAIRS)

        # Counters for each pair type
        self.num_primitive_pairs = ti.field(ti.i32, shape=())
        self.num_ball_ball_pairs = ti.field(ti.i32, shape=())
        self.num_box_box_pairs = ti.field(ti.i32, shape=())
        self.num_box_ball_pairs = ti.field(ti.i32, shape=())
        self.num_seg_point_pairs = ti.field(ti.i32, shape=())
        self.num_seg_ball_pairs = ti.field(ti.i32, shape=())
        self.num_seg_seg_pairs = ti.field(ti.i32, shape=())

        self.num_mesh_pairs = ti.field(ti.i32, shape=())
        self.num_mixed_pairs = ti.field(ti.i32, shape=())
        self.num_groundprim_pairs = ti.field(ti.i32, shape=())
        self.num_groundmesh_pairs = ti.field(ti.i32, shape=())

        # ==== Contact Cache: Store detected contacts to avoid redundant detection in PGS iterations ====
        self.MAX_CONTACTS = max(self.MAX_NODES * 16, 10000)
        # For mesh rigids, ground contacts scale with boundary nodes (each below-ground node = 1 contact)
        # Use boundary-aware scaling: typically only ~25% of mesh nodes contact ground simultaneously
        if count_mesh_nodes > 0:
            self.MAX_GROUND_CONTACTS = max(min(count_mesh_nodes // 4, self.MAX_NODES * 500), 50000)
        else:
            self.MAX_GROUND_CONTACTS = max(self.MAX_NODES * 200, 50000)

        # Rigid-Rigid contacts (use applyImpulsePair)
        self.num_contacts = ti.field(ti.i32, shape=())
        self.num_contacts[None] = 0
        self.contact_rigid_a = ti.field(ti.i32, self.MAX_CONTACTS)
        self.contact_rigid_b = ti.field(ti.i32, self.MAX_CONTACTS)
        self.contact_point = ti.Vector.field(self.d, ti.f32, self.MAX_CONTACTS)
        self.contact_normal = ti.Vector.field(self.d, ti.f32, self.MAX_CONTACTS)
        # PGS row indices for each contact (to map pgs_lambda back to contact forces)
        # Using single field to avoid LLVM memory layout issues on Windows
        self.contact_pgs_indices = ti.Vector.field(3, ti.i32, self.MAX_CONTACTS)  # [normal, tangent1, tangent2]
        self.contact_depth = ti.field(ti.f32, self.MAX_CONTACTS)
        self.contact_bounce_vel = ti.field(
            ti.f32, self.MAX_CONTACTS
        )  # Restitution bounce target velocity (computed once at cache time)
        self.contact_tangent1 = ti.Vector.field(self.d, ti.f32, self.MAX_CONTACTS)
        self.contact_force = ti.Vector.field(self.d, ti.f32, self.MAX_CONTACTS)
        self.contact_count_per_rigid = ti.field(ti.i32, self.MAX_NODES)

        # Ground-Rigid contacts (use applyImpulseAtPoint)
        self.num_ground_contacts = ti.field(ti.i32, shape=())
        self.num_ground_contacts[None] = 0
        # Track previous frame's contact counts to bound reset loops
        self.prev_num_contacts = ti.field(ti.i32, shape=())
        self.prev_num_contacts[None] = 0
        self.prev_num_ground_contacts = ti.field(ti.i32, shape=())
        self.prev_num_ground_contacts[None] = 0
        self.ground_contact_rigid = ti.field(ti.i32, self.MAX_GROUND_CONTACTS)
        self.ground_contact_point = ti.Vector.field(self.d, ti.f32, self.MAX_GROUND_CONTACTS)
        self.ground_contact_normal = ti.Vector.field(self.d, ti.f32, self.MAX_GROUND_CONTACTS)
        self.ground_contact_vel = ti.Vector.field(self.d, ti.f32, self.MAX_GROUND_CONTACTS)
        # PGS row indices for each ground contact
        # Using single field to avoid LLVM memory layout issues on Windows
        self.ground_contact_pgs_indices = ti.Vector.field(
            3, ti.i32, self.MAX_GROUND_CONTACTS
        )  # [normal, tangent1, tangent2]
        self.ground_contact_force = ti.Vector.field(self.d, ti.f32, self.MAX_GROUND_CONTACTS)
        self.ground_contact_depth = ti.field(
            ti.f32, self.MAX_GROUND_CONTACTS
        )  # Signed penetration depth (negative = penetrating)
        self.ground_contact_bounce_vel = ti.field(
            ti.f32, self.MAX_GROUND_CONTACTS
        )  # Restitution bounce target velocity (computed once at cache time)
        # Fixed tangent basis per contact (computed once at contact creation, stable across PGS iterations)
        self.ground_contact_tangent1 = ti.Vector.field(self.d, ti.f32, self.MAX_GROUND_CONTACTS)
        self.ground_contact_tangent2 = ti.Vector.field(self.d, ti.f32, self.MAX_GROUND_CONTACTS)  # only used in 3D

        # Per-env ground contact indexing for efficient PGS parallel scanning.
        # Without this, each of N env-threads scans ALL contacts → O(N × total_contacts).
        # With per-env indices, each thread accesses only its own contacts → O(contacts_per_env).
        self.MAX_ENVS_ALLOC = (
            4096  # Max envs we allocate for (can be set based on expected batch size, but keep reasonable upper bound)
        )
        self.MAX_GC_PER_ENV = max(self.MAX_GROUND_CONTACTS // self.MAX_ENVS_ALLOC, 1024)
        self.ground_contact_env_count = ti.field(ti.i32, self.MAX_ENVS_ALLOC)
        # Per-env index lists: ground_contact_env_idx[env_id * MAX_GC_PER_ENV + local_i] = global contact idx
        self.ground_contact_env_idx = ti.field(ti.i32, self.MAX_ENVS_ALLOC * self.MAX_GC_PER_ENV)
        # Same for rigid-rigid contacts
        self.MAX_CC_PER_ENV = max(self.MAX_CONTACTS // self.MAX_ENVS_ALLOC, 256)
        self.contact_env_count = ti.field(ti.i32, self.MAX_ENVS_ALLOC)
        self.contact_env_idx = ti.field(ti.i32, self.MAX_ENVS_ALLOC * self.MAX_CC_PER_ENV)

        print("Finished allocating contact cached ======")

        # Per-rigid friction coefficient (Coulomb, 0.0 = frictionless), first friction, second restitution
        self.contactParams = ti.Vector.field(2, ti.f32, self.MAX_NODES)

        # Inertia storage: 2D uses scalar, 3D uses 3x3 matrix
        self.inertia = ti.field(ti.f32, self.MAX_NODES)
        # OPTIMIZATION: Cached values for 2D joint solving
        self.cached_rotation_matrix = ti.Matrix.field(2, 2, ti.f32, self.MAX_NODES)
        self.cached_inertia_inv_2d = ti.field(ti.f32, self.MAX_NODES)
        self.visual_angle = ti.field(ti.f32, self.MAX_NODES)

        self.mass = ti.field(ti.f32, self.MAX_NODES)

        # per-rigid radius for capsule/ball where applicable
        self.radius = ti.field(ti.f32, self.MAX_NODES)

        # ── Compound collision shapes ────────────────────────────────────
        # Multiple collision primitives (e.g. 4 spheres on a foot) attached
        # to a single parent rigid body. Sub-colliders are stored in a flat
        # pool; per-rigid (count, offset) index into it.
        self.MAX_COMPOUND_SHAPES = max(count_compound_shapes + 64, 128)
        self.compound_count = ti.field(ti.i32, self.MAX_NODES)  # num sub-colliders per rigid (0 = use main shape)
        self.compound_offset = ti.field(ti.i32, self.MAX_NODES)  # start index in pool
        self.compound_local_pos = ti.Vector.field(d, ti.f32, self.MAX_COMPOUND_SHAPES)  # body-local offset
        self.compound_radius = ti.field(ti.f32, self.MAX_COMPOUND_SHAPES)  # sub-collider radius
        self.compound_type = ti.field(ti.i32, self.MAX_COMPOUND_SHAPES)  # RigidType (BALL for now)
        self.num_compound_shapes = 0  # total allocated sub-colliders

        self.needUpdate = False

        self.stableTime = 1.0 / 1000.0  # Default stable time step
        # Process rigid domains and allocate data
        print("Processing domains for RigidManager...")
        self.processDomains_(domains)
        print("Processed", self.numRigids, "rigids and", self.numAnalytical, "analytical planes.")

        self.spatialHash = None
        self._sh_contact_margin = ti.field(ti.f32, shape=())
        self._sh_contact_margin[None] = 0.0
        self._sh_mesh_elapsed = 0.0
        self._sh_mesh_rebuild_interval = 1.0 / 500.0  # rebuild at most every ~2ms
        self._sh_mesh_needs_rebuild = True  # first call always rebuilds
        self._sh_mesh_max_v = 0.0  # max rigid velocity at last rebuild
        self._sh_unbounded_lb = ti.Vector.field(self.d, ti.f32, shape=())
        self._sh_unbounded_ub = ti.Vector.field(self.d, ti.f32, shape=())
        self._sh_unbounded_lb[None] = ti.Vector([-1e9 for _ in range(self.d)])
        self._sh_unbounded_ub[None] = ti.Vector([1e9 for _ in range(self.d)])
        if (self.numMeshRigidInContact > 0) and not self.skip_spatial_hash:
            # Size spatial hash based on actual mesh boundary element count (not hardcoded)
            total_elems = self.totalBoundaryElements if self.totalBoundaryElements > 0 else count_mesh_elems
            sh_max_elements = max(total_elems + 4096, 10000)
            sh_max_cells = min(sh_max_elements, 100000)
            # Query capacity: avoid frequent candidate truncation in dense mesh
            # contacts (e.g. hand links). Keep bounded to control stack usage.
            sh_max_query = min(max(total_elems // 8, 2048), 4096)
            # Per-cell cap: assume ~100-500 actual cells; allow generous headroom
            # so no elements are silently dropped due to overflow.
            sh_max_per_cell = min(max(total_elems // 50 + 10, 200), 2000)
            self.spatialHash = SpatialHashManager(
                self.d,
                max_elements=sh_max_elements,
                max_cells=sh_max_cells,
                max_elements_per_cell=sh_max_per_cell,
                max_query_results=sh_max_query,
            )
            print(
                f"Initialized SpatialHashManager for mesh-rigid contacts "
                f"({sh_max_elements} elements, {sh_max_cells} cells)."
            )
        elif self.skip_spatial_hash and self.numMeshRigidInContact > 0:
            print(f"  [SpatialHash] SKIPPED (skip_spatial_hash=True, saving ~GB of VRAM)")

        # ==== Dynamic joint anchors (as special points using main rigid arrays) ====
        self.jointDict = dict()
        for joint in joints:
            self.jointDict[joint.name] = joint
        self.joints = joints

        self.numAnchors = 0

        # ==== Joint storage fields for unified kernel processing ====
        self.joint_type = ti.field(ti.i32, self.MAX_JOINTS)
        self.joint_anchor = ti.Vector.field(
            self.d, ti.f32, self.MAX_JOINTS
        )  # world-space anchor point for joint constraint
        self.joint_id_a = ti.field(ti.i32, self.MAX_JOINTS)
        self.joint_id_b = ti.field(ti.i32, self.MAX_JOINTS)

        # Local offset vectors (body-local coordinates)
        self.joint_l1 = ti.Vector.field(self.d, ti.f32, self.MAX_JOINTS)
        self.joint_l2 = ti.Vector.field(self.d, ti.f32, self.MAX_JOINTS)

        # Initial orientations for computing relative rotation
        self.joint_axis = ti.Vector.field(2, ti.f32, self.MAX_JOINTS)
        self.joint_q0_rel_inv = ti.Vector.field(1, ti.f32, self.MAX_JOINTS)

        # Joint parameters: [position_bias, angular_bias, lower_limit, upper_limit, velocity_limit, effort_limit]
        self.joint_params = ti.Vector.field(6, ti.f32, self.MAX_JOINTS)

        # Motor flag
        self.joint_has_motor = ti.field(ti.i32, self.MAX_JOINTS)
        self.joint_control_target = ti.field(
            ti.f32, self.MAX_JOINTS
        )  # target position for motor (angle for revolute, length for prismatic)
        # Motor command semantics:
        # 0 = velocity command (rad/s or m/s), 1 = acceleration command (rad/s^2 or m/s^2), 2 = torque motor
        self.joint_motor_target_mode = ti.field(ti.i32, self.MAX_JOINTS)
        # Runtime velocity target consumed by unified PGS motor rows.
        # This is updated per-substep from joint_control_target according to mode.
        self.joint_motor_target_vel = ti.field(ti.f32, self.MAX_JOINTS)
        self.kpd_field = ti.Vector.field(2, ti.f32, self.MAX_JOINTS)  # [kp_pos, kp_rot] for PD control

        print("Processing joints for RigidManager...")
        self.processJoints(joints)
        print("Processed", len(joints), "joints.")

        # World-frame PD flag: when set to 1, the PD controller measures the
        # child body's absolute orientation (world frame) projected onto the
        # joint axis, instead of the parent-child relative angle.  This lets
        # ankle/hip joints sense the full body tilt even when the whole chain
        # tips as a unit.
        self.joint_pd_world_frame = ti.field(ti.i32, self.MAX_JOINTS)

        self.pgs_iterations = 200  # Max PGS iterations (with early convergence exit)
        self.pgs_tol = 1e-5  # Convergence tolerance for PGS early exit

        self.MAX_CONSTRAINTS = 16 * self.MAX_CONTACTS * 3 + 7 * self.MAX_JOINTS  # 3 per contact and 7 per joints
        self.numConstraints = ti.field(ti.i32, shape=())
        self.numConstraints[None] = 0
        # Per-constraint Jacobians for body A and body B:
        #   - ground-vs-rigid uses only pgs_Jac_a (B is zero)
        #   - rigid-vs-rigid uses both pgs_Jac_a and pgs_Jac_b
        self.pgs_Jac_a = ti.Vector.field(6, ti.f32, self.MAX_CONSTRAINTS)
        self.pgs_Jac_b = ti.Vector.field(6, ti.f32, self.MAX_CONSTRAINTS)
        self.pgs_rhs = ti.field(ti.f32, self.MAX_CONSTRAINTS)
        self.pgs_limits = ti.Vector.field(2, ti.f32, self.MAX_CONSTRAINTS)  # [lower, upper]
        self.pgs_bodypair = ti.Vector.field(2, ti.i32, self.MAX_CONSTRAINTS)
        # Per-row accumulated impulse for projected GS and friction clamping.
        self.pgs_lambda = ti.field(ti.f32, self.MAX_CONSTRAINTS)
        # For friction rows, parent normal row index; -1 otherwise.
        self.pgs_parent_row = ti.field(ti.i32, self.MAX_CONSTRAINTS)
        self._pgs_check_interval = 5  # Check convergence every N iterations (after warm-up of 10)
        # Allocate PGS RotV snapshot with matching dimension
        self.pgsErrorNone = ti.field(ti.f32, shape=())
        self.pgsErrorNone[None] = 0.0

        # Snapshot fields for PGS convergence check
        self.V_prev = ti.Vector.field(self.d, ti.f32, self.MAX_NODES)
        self.RotV_prev = None  # allocated below after RotV
        self.RotV_prev = ti.Vector.field(1, ti.f32, self.MAX_NODES)

        # Optional Python-side PGS kernel timing breakdown.
        self.pgs_profile_enabled = False
        self.pgs_profile_sync_kernels = False
        self.pgs_profile_print_every = 100
        self._pgs_profile_calls = 0
        self._pgs_profile_iters = 0
        self._pgs_profile_breaks = 0
        self._pgs_profile_time_total = 0.0
        self._pgs_profile_time_snapshot = 0.0
        self._pgs_profile_time_joints_fwd = 0.0
        self._pgs_profile_time_contacts = 0.0
        self._pgs_profile_time_joints_bwd = 0.0
        self._pgs_profile_time_delta = 0.0

        # Adjust rigid poses to satisfy joint limits
        # self.adjust_rigid_poses_for_joint_limits()

        # Process boundary conditions from domains and joints
        self.movingAnalytical = False
        print("Processing boundary conditions for RigidManager...")
        self.processConditions()

        # Normal mode: use shared BVH from ExplicitLoop
        self.bvh = bvh

        print(
            "RigidManager initialized with",
            self.numRigids,
            "rigids,",
            self.numAnalytical,
            "analytical planes,",
            self.numAnchors,
            "anchors.",
        )

        print(f"In total {self.numRigidInContact} rigids are considered for rigid-rigid contact detection.")
        print(f"In total {self.numRigidGroundContact} rigids are considered for rigid-ground contact detection.")

    def _compute_mesh_hash(self, mesh) -> int:
        """Compute hash of mesh geometry for deduplication.

        Hash is based on boundary node count, element count, and a sample of coordinates.
        This is a fast approximate hash - collisions will be rare for typical robot geometries.

        Args:
            mesh: Mesh object with boundaryNodes, boundaryElements, coords

        Returns:
            Integer hash value
        """
        import hashlib

        # Use boundary topology as primary hash component
        num_nodes = mesh.numBoundNodes
        num_elems = mesh.numBoundElements

        # Sample first/last boundary node coordinates (rounded to avoid float precision issues)
        sample_coords = []
        for i in [0, min(5, num_nodes - 1), num_nodes - 1]:  # sample 3 nodes
            if i < num_nodes:
                node_id = int(mesh.boundaryNodes[i])
                coord = mesh.coords[node_id]
                # Round to 6 decimal places to avoid floating point precision differences
                sample_coords.extend([round(float(coord[j]), 6) for j in range(self.d)])

        # Sample first/last boundary element connectivity
        sample_elems = []
        for i in [0, num_elems - 1]:
            if i < num_elems:
                elem = mesh.boundaryElements[i]
                sample_elems.extend([int(elem[j]) for j in range(2)])

        # Combine into hash string
        hash_str = f"{num_nodes}_{num_elems}_{sample_coords}_{sample_elems}"
        return int(hashlib.md5(hash_str.encode()).hexdigest()[:16], 16)

    def _add_mesh_to_pool(self, mesh, rigid_idx: int) -> int:
        """Add a unique mesh geometry to the pool and return its pool ID.

        Args:
            mesh: Mesh object to add
            rigid_idx: Index of the rigid using this mesh

        Returns:
            pool_geometry_id: Index in the pool where this geometry is stored
        """
        pool_id = self.num_pool_geometries

        if pool_id >= self.MAX_POOL_GEOMETRIES:
            print(
                f"\033[91mWarning: Exceeded MAX_POOL_GEOMETRIES ({self.MAX_POOL_GEOMETRIES}), falling back to legacy storage\033[0m"
            )
            return -1

        num_nodes = mesh.numBoundNodes
        num_elems = mesh.numBoundElements

        node_offset = self.total_pool_nodes
        elem_offset = self.total_pool_elements

        # Safety checks
        if (node_offset + num_nodes) > self.MAX_POOL_NODES:
            print(
                f"\033[91mWarning: Pool boundary nodes exceeded capacity ({node_offset + num_nodes} > {self.MAX_POOL_NODES}), falling back to legacy\033[0m"
            )
            return -1

        if (elem_offset + num_elems) > self.MAX_POOL_ELEMENTS:
            print(f"\033[91mWarning: Pool boundary elements exceeded capacity, falling back to legacy\033[0m")
            return -1

        # Record pool geometry metadata
        self.pool_node_offset[pool_id] = node_offset
        self.pool_node_count[pool_id] = num_nodes
        self.pool_elem_offset[pool_id] = elem_offset
        self.pool_elem_count[pool_id] = num_elems

        # Get reference point from rigid
        ref = self.rigidParams[rigid_idx, 0].to_numpy()

        # Store boundary node coordinates in pool
        boundary_node_map = {}
        for local_bid in range(num_nodes):
            global_nid = int(mesh.boundaryNodes[local_bid])
            boundary_node_map[global_nid] = local_bid

            coord = mesh.coords[global_nid]
            lr = coord - ref

            self.pool_boundary_lrs[node_offset + local_bid] = lr

        # Store boundary element connectivity (remapped to local indices)
        for eid in range(num_elems):
            elem_conn = mesh.boundaryElements[eid]
            local_n0 = boundary_node_map[int(elem_conn[0])]
            local_n1 = boundary_node_map[int(elem_conn[1])]
            self.pool_boundary_elements[elem_offset + eid] = ti.Vector([local_n0, local_n1, -1])
          

        # Update pool counters
        self.total_pool_nodes += num_nodes
        self.total_pool_elements += num_elems
        self.num_pool_geometries += 1

        # print(
        #     f"[Pool] Added geometry #{pool_id}: {num_nodes} nodes, {num_elems} elements (pool total: {self.total_pool_nodes} nodes)"
        # )

        return pool_id

    def setGlobalAABB(self, aabb):
        """Set reference to ExplicitLoop's global AABB field.

        Args:
            aabb: Global AABB field indexed by domain index
        """
        self.aabb = aabb
        self.precompute_rigid_transforms()
        self.updateBBox()

    @ti.kernel
    def calInertiaInv(self):
        for i in range(self.numRigids):
            # Compute inverse using Taichi's built-in matrix inverse
            det = self.inertia[i].determinant()  # Ensure determinant is computed for validation
            if det > 1e-30:
                I_inv = self.inertia[i].inverse()
                self.inertiaInv[i] = I_inv
            else:
                self.inertiaInv[i] = ti.Matrix.zero(ti.f32, 3, 3)  # zero inverse = infinite inertia (fixed body)

    @ti.kernel
    def _copy_elements_from_pool_kernel(
        self, dst_offset: ti.i32, src_offset: ti.i32, count: ti.i32, node_offset: ti.i32
    ):
        """Fast kernel to copy element connectivity from pool to legacy array.

        OPTIMIZATION: GPU-parallel copy is 10-100x faster than Python loop.
        Used when multiple mesh rigids share the same geometry (instancing).
        """
        for i in range(count):
            conn = self.pool_boundary_elements[src_offset + i]
            out_conn = ti.Vector([conn[0], conn[1], conn[2]])
            if conn[0] >= 0:
                out_conn[0] = conn[0] + node_offset
            if conn[1] >= 0:
                out_conn[1] = conn[1] + node_offset
            if conn[2] >= 0:
                out_conn[2] = conn[2] + node_offset
            self.meshBoundaryElements[dst_offset + i] = out_conn

    @ti.kernel
    def _batch_copy_elements_kernel(self, dst_offset: ti.i32, node_offset: ti.i32, elem_data: ti.types.ndarray()):
        """Fast kernel to copy element array from numpy to Taichi field.

        OPTIMIZATION: Direct ndarray access in Taichi kernel avoids Python loop overhead.
        """
        for i in range(elem_data.shape[0]):
            c0 = elem_data[i, 0]
            c1 = elem_data[i, 1]
            c2 = elem_data[i, 2]
            if c0 >= 0:
                c0 += node_offset
            if c1 >= 0:
                c1 += node_offset
            if c2 >= 0:
                c2 += node_offset
            self.meshBoundaryElements[dst_offset + i] = ti.Vector([c0, c1, c2])

    def substep(self, collision_pairs_field, num_pairs, dt, damping, fem_lb=None, fem_ub=None, fem_margin=0.0):
        """High-level rigid update: integrate velocities, solve constraints, update positions.

        PERFORMANCE-CRITICAL: Fused kernels minimize Python↔Taichi round trips.
        Each kernel launch costs ~0.08-0.15ms of pure overhead.

        Fast path (no HeightField/Voxel): multiple small kernel launches instead of fewer large ones.
            1. classify_collision_pairs_kernel  (conditional, if broadphase pairs exist)
            1b. apply_motor_torques_kernel      (adds torque-mode motor forces to accumulated_rotational_impulse)
            2. _rigidStep_and_precompute_kernel (fused velocity integration + transform cache)
            3. reset + per-type contact detection + PGS split kernels
            4. _updateU_and_BBox_kernel         (fused position integration + AABB update)

        Slow path (HeightField/Voxel): original separate kernels for compatibility.
        """
        if not self.needUpdate:
            return

        # Accumulate time for lazy spatial hash rebuild
        self._sh_mesh_elapsed += dt
        if self._sh_mesh_elapsed >= self._sh_mesh_rebuild_interval:
            self._sh_mesh_needs_rebuild = True
        # Displacement-based trigger: rebuild if max displacement > 0.5 * cell_size
        if not self._sh_mesh_needs_rebuild:
            max_disp = self._sh_mesh_max_v * self._sh_mesh_elapsed
            if max_disp > 0.5 * self._sh_mesh_cell_size:
                self._sh_mesh_needs_rebuild = True
        # Adaptive query buffer: cell_size + estimated displacement, capped at 3x cell
        # Only update GPU field when rigid-rigid SH is actually used (avoids GPU sync).
        if self.considerRigidRigidContact:
            margin = self._sh_mesh_max_v * self._sh_mesh_elapsed
            self._sh_contact_margin[None] = margin

        self.update_joint_motor_velocity_targets_kernel(dt)
        if self.use_pd == 1:
            self.apply_joint_pd_velocity_kernel(dt)
        elif self.use_pd == 2:
            self.apply_joint_pd_torque_kernel(dt)

        if self.considerRigidRigidContact and num_pairs > 0:
            self.classify_collision_pairs_kernel(collision_pairs_field, num_pairs)
        elif self.numRigidGroundContact > 0:
            # Generate them directly to avoid scanning all BVH pairs.
            self._generate_ground_pairs_direct_kernel()

        # if (self.num_box_box_pairs[None] > 0):
        #     print("Num pairs:", num_pairs, "Ball-Ball:", self.num_ball_ball_pairs[None],
        #         "Box-Box:", self.num_box_box_pairs[None], "Box-Ball:", self.num_box_ball_pairs[None],
        #         "Seg-Point:", self.num_seg_point_pairs[None], "Seg-Ball:", self.num_seg_ball_pairs[None], "Seg-Seg:", self.num_seg_seg_pairs[None],
        # "Mesh-Mesh:", self.num_mesh_pairs[None], "Mixed:", self.num_mixed_pairs[None], "Ground-Prim:", self.num_groundprim_pairs[None], "Ground-Mesh:", self.num_groundmesh_pairs[None])
        has_rigid_rigid_contact = (
            self.num_ball_ball_pairs[None]
            + self.num_box_box_pairs[None]
            + self.num_box_ball_pairs[None]
            + self.num_seg_point_pairs[None]
            + self.num_seg_ball_pairs[None]
            + self.num_seg_seg_pairs[None]
            + self.num_mesh_pairs[None]
            + self.num_mixed_pairs[None]
        ) > 0
        has_rigid_ground_contact = (self.num_groundprim_pairs[None] + self.num_groundmesh_pairs[None]) > 0
        has_contact = has_rigid_rigid_contact or has_rigid_ground_contact
        has_solve = self.numAnchors > 0 or has_contact

        # K1: velocity integration + rotation matrix cache
        self._rigidStep_and_precompute_kernel(dt, damping)
        if has_solve:
            if has_contact:
                # K2: reset + detect contacts (split into small per-type kernels for fast JIT)
                self.reset_contact_caches_kernel()

                if self.considerRigidRigidContact and has_rigid_rigid_contact:
                    self.detect_all_contacts()

                if has_rigid_ground_contact:
                    self.detectRigidGroundContact()

            has_joints = self.numAnchors > 0
            # K3: unified PGS solve path.
            # Joint/contact rows are assembled first, then solved by unified PGS.
            if has_joints:
                self._assemble_joint_constraints_kernel(dt)
            if has_contact:
                self._assemble_contact_constraints_kernel(dt)
            self.solve_pgs(self.pgs_iterations)
            # Compute contact forces from impulses (force = impulse / dt)
            if has_contact:
                self._compute_contact_forces_kernel(dt)
            self.numConstraints[None] = 0
        # K4: position integration + AABB update (2→1 kernel)
        update_bbox_coords = 1
        self._updateU_and_BBox_kernel(dt, update_bbox_coords)

        if not has_contact and self.spatialHash is not None:
            # Meaning FEM-rigid contact need this SH rebuild
            self.maybe_rebuild_spatial_hash()

    def reset(self):
        """
        Reset all rigid bodies to their initial state.

        This method:
        1. Resets positions by re-packing rigid parameters from initial origins
        2. Resets velocities and angular velocities to zero
        3. Resets rotations to identity (zero angle)
        4. Resets accumulated displacements
        5. Resets joint anchors to initial positions
        6. Updates bounding boxes and shape coordinates

        Note: Does NOT reset boundary conditions - those should be managed by the caller.
        """
        # Reset all dynamic state variables to zero
        self.V.fill(0.0)
        self.RotV.fill(0.0)
        self.U.fill(0.0)  # CRITICAL: Reset accumulated displacement

        # Re-pack rigid parameters from initial origins stored in rigid objects
        # This restores initial positions (and will set initial rotations)
        for i in range(self.numRigids):
            domain_idx = int(self.rigidDomainIds[i][0])
            domain = self.domains[domain_idx]
            if domain.type == DomainType.RIGID:
                # Re-pack parameters from the rigid object (which stores initial origin)
                self.resetRigidParams(domain.rigid, i)

        # Update the cached rotation matrices and inertia tensors based on reset positions and zero rotations
        self.precompute_rigid_transforms()
        # Update bounding boxes and shape coordinates after reset
        # This will recompute shapeCoords for boxes based on new positions and zero rotation
        self.updateBBox()

    @staticmethod
    def _resolve_initial_quat(rigid):
        if rigid.initial_quat is not None:
            return np.asarray(rigid.initial_quat, dtype=np.float32)

        euler = rigid.angle
        mag = euler.norm()
        if mag < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _resolve_visual_quat(rigid):
        if hasattr(rigid, "visual_quat") and rigid.visual_quat is not None:
            return np.asarray(rigid.visual_quat, dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def packRigidParams(self, rigid, idx: int):
        """Pack a rigid object's key parameters into the compact `rigidParams` layout.

        Layout (per-rigid index `idx`):
        - rigidParams[idx, 0] : reference point (vector, dim d)
        - rigidParams[idx, 1] : primary params (vector, dim d) e.g. extents or radius in component 0
        """
        # Reference point
        ref = rigid.getRefPoint()
        self.rigidParams[idx, 0] = ref
        self.mass[idx] = rigid.mass

        # default zero for primary / aux
        zero_vec = ti.Vector([0.0 for _ in range(self.d)])
        self.rigidParams[idx, 1] = zero_vec

        # fill according to rigid type if available
        if rigid.rtype == RigidType.BALL:
            # also keep scalar radius field
            self.radius[idx] = rigid.getRadius()

        elif rigid.rtype == RigidType.BOX:
            # extents -> primary, angle -> aux
            self.rigidParams[idx, 1] = rigid.getPrimary()

        elif rigid.rtype == RigidType.CAPSULE:
            # store segment endpoints relative to the reference point (local offsets)
            # so that they can be rotated/translated in-kernel correctly.
            self.rigidParams[idx, 1] = rigid.getPrimary() - ref
            self.radius[idx] = rigid.getRadius()

        elif rigid.rtype == RigidType.MESH:
            # ==== MESH INSTANCING: Use geometry pool with deduplication ====
            mesh = rigid.mesh
            mesh_hash = self._compute_mesh_hash(mesh)

            # Check if this geometry already exists in pool
            if mesh_hash in self.pool_hash_to_id:
                pool_id = self.pool_hash_to_id[mesh_hash]
                print(f"Packing mesh rigid idx {idx}: REUSING pool geometry #{pool_id} (hash collision = efficient!)")
            else:
                # Add new unique geometry to pool
                pool_id = self._add_mesh_to_pool(mesh, idx)
                if pool_id >= 0:
                    self.pool_hash_to_id[mesh_hash] = pool_id
                    print(f"Packing mesh rigid idx {idx}: NEW pool geometry #{pool_id}")

            # Link this rigid instance to its pool geometry
            if pool_id >= 0:
                self.instance_pool_id[idx] = pool_id

            # Store scale and offset (needed for transform operations)
            if rigid.transform is not None:
                scale_vec = rigid.transform.scale
                self.meshRigidScale[idx] = ti.Vector([float(scale_vec[i]) for i in range(self.d)])
                # Absorb offset into the body center so that the physics
                # pivot (rigidParams[idx,0]) coincides with the geometry
                # center.  This ensures correct lever arms in impulse
                # computations (applyImpulseAtPoint).  Pool boundary lrs
                # remain relative to the original mesh center, and the
                # updateBBox formula  center + offset + R @ (lr*scale)
                # still produces the correct world coords because offset
                # is now zero.
                offset_vec = rigid.transform.offset
                offset_ti = ti.Vector([float(offset_vec[i]) for i in range(self.d)])
                self.rigidParams[idx, 0] += offset_ti
                self.meshRigidOffset[idx] = ti.Vector([0.0 for _ in range(self.d)])
            else:
                # Default to uniform scale of 1.0
                self.meshRigidScale[idx] = ti.Vector([1.0 for _ in range(self.d)])
                self.meshRigidOffset[idx] = ti.Vector([0.0 for _ in range(self.d)])

            # Maintain index mappings and metadata for backward compatibility
            mesh_local_idx = self.numMesh
            self.rigid2MeshIndices[idx] = mesh_local_idx
            self.mesh2RigidIndices[mesh_local_idx] = idx

            # ==== OPTIMIZATION: Fast element copy using mesh instancing ====
            # If this mesh reuses pool geometry, directly copy elements from pool
            # Otherwise, build new element array
            num_boundary_nodes = rigid.mesh.numBoundNodes
            num_boundary_elements = rigid.mesh.numBoundElements

            # Allocate space in legacy arrays (still needed for contact detection)
            node_offset = self.totalBoundaryNodes
            elem_offset = self.totalBoundaryElements

            # Safety checks
            if (node_offset + num_boundary_nodes) > self.MAX_BOUNDARY_NODES:
                print(
                    f"\033[91mError: Exceeded MAX_BOUNDARY_NODES ({self.MAX_BOUNDARY_NODES})! Current total: {node_offset}, attempting to add: {num_boundary_nodes}\033[0m"
                )
                raise RuntimeError("Exceeded MAX_BOUNDARY_NODES")

            if (elem_offset + num_boundary_elements) > self.MAX_BOUNDARY_ELEMENTS:
                print(
                    f"\033[91mError: Exceeded MAX_BOUNDARY_ELEMENTS ({self.MAX_BOUNDARY_ELEMENTS})! Current total: {elem_offset}, attempting to add: {num_boundary_elements}\033[0m"
                )
                raise RuntimeError("Exceeded MAX_BOUNDARY_ELEMENTS")

            # Record metadata
            self.meshBoundaryNodeOffset[mesh_local_idx] = node_offset
            self.meshBoundaryNodeCount[mesh_local_idx] = num_boundary_nodes
            self.meshBoundaryElementOffset[mesh_local_idx] = elem_offset
            self.meshBoundaryElementCount[mesh_local_idx] = num_boundary_elements

            # OPTIMIZATION: Reuse element connectivity from pool if available
            if pool_id >= 0:
                # Fast path: copy from pool using Taichi kernel
                pool_elem_offset = self.pool_elem_offset[pool_id]
                self._copy_elements_from_pool_kernel(elem_offset, pool_elem_offset, num_boundary_elements, node_offset)
            else:
                # Slow path: build element array (only for first instance of unique geometry)
                boundary_node_map = {}
                for local_bid in range(num_boundary_nodes):
                    global_nid = int(mesh.boundaryNodes[local_bid])
                    boundary_node_map[global_nid] = local_bid

                # Build remapped element array in numpy
                elem_array = np.zeros((num_boundary_elements, 3), dtype=np.int32)
                for eid in range(num_boundary_elements):
                    elem_conn = mesh.boundaryElements[eid]
                    elem_array[eid] = [
                        boundary_node_map[int(elem_conn[0])],
                        boundary_node_map[int(elem_conn[1])],
                        -1,
                    ]


                # Batch copy using Taichi kernel (much faster than Python loop)
                self._batch_copy_elements_kernel(elem_offset, node_offset, elem_array)

            self.totalBoundaryNodes += num_boundary_nodes
            self.totalBoundaryElements += num_boundary_elements
            self.numMesh += 1

        # consider angle for mesh rigid, box, capsule
        # 2D: just store the scalar angle
        self.quat[idx] = rigid.angle
        self.quat_initial[idx] = rigid.angle
        self.visual_angle[idx] = 0.0

        # Store inertia if available on python-side rigid object
        ib = rigid.inertia_body
        self.inertia[idx] = float(ib)
      

    def resetRigidParams(self, rigid, idx: int):
        # Reference point (absorb transform offset so physics center = geometry center)
        ref = rigid.getRefPoint()
        self.rigidParams[idx, 0] = ref
        if hasattr(rigid, "transform") and rigid.transform is not None:
            offset_vec = rigid.transform.offset
            for i in range(self.d):
                self.rigidParams[idx, 0][i] += float(offset_vec[i])

        # consider angle for mesh rigid, box, capsule
        if hasattr(rigid, "angle"):
            # 2D: just store the scalar angle
            self.quat[idx] = rigid.angle
            self.quat_initial[idx] = rigid.angle
            self.visual_angle[idx] = 0.0
          
        else:
            self.quat[idx] = ti.Vector([0.0])
            self.quat_initial[idx] = ti.Vector([0.0])
            self.visual_angle[idx] = 0.0

    def _register_compound_shapes(self, rigid_idx, collision_shapes):
        """Register compound sub-colliders for a rigid body.

        Args:
            rigid_idx: index of the parent rigid body
            collision_shapes: list of dicts with keys 'type' (RigidType), 'local_pos' (list/tuple), 'radius' (float)
        """
        n = len(collision_shapes)
        offset = self.num_compound_shapes
        if offset + n > self.MAX_COMPOUND_SHAPES:
            print(f"\033[91mError: Compound shape pool exhausted ({offset + n} > {self.MAX_COMPOUND_SHAPES})\033[0m")
            raise RuntimeError("Exceeded MAX_COMPOUND_SHAPES")
        self.compound_count[rigid_idx] = n
        self.compound_offset[rigid_idx] = offset
        for k, shape in enumerate(collision_shapes):
            idx = offset + k
            pos = shape["local_pos"]

            self.compound_local_pos[idx] = ti.Vector([float(pos[0]), float(pos[1])])
            self.compound_radius[idx] = float(shape["radius"])
            self.compound_type[idx] = int(shape["type"])
        self.num_compound_shapes = offset + n

    def processDomains_(self, domains):
        # Process rigid domains - attach logic moved to RigidBodyDomain.attach()
        for i, domain in enumerate(domains):
            if domain.type == DomainType.RIGID:
                if self.numRigids >= self.MAX_NODES:
                    print(
                        f"\033[91mError: Exceeded MAX_NODES ({self.MAX_NODES}) for rigid bodies! Cannot add more rigids.\033[0m"
                    )
                    raise RuntimeError(f"Exceeded maximum rigid bodies ({self.MAX_NODES})")

                rigid_idx = self.numRigids
                domain.attach(self, rigid_idx, i)
                self.packRigidParams(domain.rigid, rigid_idx)

                # Register compound collision sub-shapes if present
                collision_shapes = getattr(domain, "collision_shapes", None)
                if collision_shapes and len(collision_shapes) > 0:
                    self._register_compound_shapes(rigid_idx, collision_shapes)

                # Set environment ID for collision filtering (batched training)
                if hasattr(domain, "env_id"):
                    self.rigid_env_id[rigid_idx] = domain.env_id

                self.numRigids += 1
                self.needUpdate = True
                # Here I assume the maximum velocity is 100.0
                # so the stable time should be at least 0.1 * rigid.minSize / 100.0
                # self.stableTime = min(self.stableTime, domain.rigid.minSize / 100.0)  # too large time step will lead to inaccurate results although stable
                if domain.considerContact:
                    self.numRigidGroundContact += 1
                    if self.considerRigidRigidContact:
                        self.numRigidInContact += 1
                    if domain.rigid.rtype == RigidType.MESH:
                        self.numMeshRigidInContact += 1
                elif domain.considerGroundContact:
                    self.numRigidGroundContact += 1

        # Process analytical domains (planes, heightfields, voxel maps)
        for i, domain in enumerate(domains):
            if (
                domain.type == DomainType.ANALYTICAL
                or domain.type == DomainType.HEIGHTFIELD
                or domain.type == DomainType.VOXELMAP
            ):
                if self.numAnalytical >= self.MAX_ANAL:
                    print(
                        f"\033[91mError: Exceeded MAX_ANAL ({self.MAX_ANAL}) for analytical domains! Cannot add more analytical domains.\033[0m"
                    )
                    raise RuntimeError(f"Exceeded maximum analytical domains ({self.MAX_ANAL})")

                self.rigidDomainIds[self.numAnalytical + self.numRigids] = [
                    i,
                    0,
                    1,
                ]  # type 0 for ground, type 1 for considerContact as default
                anal_idx = self.numAnalytical + self.numRigids
                category_bits = int(getattr(domain, "category_bits", COLLISION_CATEGORY_GROUND)) & 0b11111111
                collide_bits = int(getattr(domain, "collide_bits", COLLISION_MASK_ALL)) & 0b11111111
                self.category_bits[anal_idx] = category_bits
                self.collide_bits[anal_idx] = collide_bits
                self.mass[self.numAnalytical + self.numRigids] = 1e10  # very large mass to simulate immovable object
                domain.attach(self, self.numAnalytical + self.numRigids)
                self.domainToRigid[i] = self.numAnalytical + self.numRigids
                if domain.type == DomainType.ANALYTICAL:
                    self.rigidParams[self.numRigids + self.numAnalytical, 0] = domain.point
                    self.rigidParams[self.numRigids + self.numAnalytical, 1] = domain.normal
                    if len(domain.bcs) > 0:
                        self.needUpdate = True
                elif domain.type in (DomainType.HEIGHTFIELD, DomainType.VOXELMAP):
                    self.hasHeightFieldOrVoxel = True
                self.numAnalytical += 1


    def processJoints(self, joints):
        for i in range(len(joints)):
            joints[i].attach(self)
            self.numAnchors += 1
            self.needUpdate = True

    def _addBcValue(self, idx, type, value):
        self.bcNodes[idx] |= type

        if type == UTYPE:
            self.bcTValues[idx] = [0.0 for j in range(self.d)]
            self.mass[idx] = 1e10  # large mass to fix in space
        elif type == RTYPE:
            self.bcTValues[idx] = [0.0 for j in range(self.d)]
            self.mass[idx] = 1e10  # large mass to fix in space
            self.inertia[idx] = 1e10
            self.bcRValues[idx] = [0.0]
        elif type == VTYPE:
            self.bcTValues[idx] = value
            self.mass[idx] = 1e10

        elif type == ROTVTYPE:
            self.bcRValues[idx] = value
            self.inertia[idx] = 1e10

        elif type == ROTATYPE or type == TORQUETYPE:
            self.bcRValues[idx] = value

        elif type == GRAVITY:
            self.bcGValues[idx] = value
        else:
            self.bcTValues[idx] = value

    def processConditions(self):
        for i in range(self.numRigids):
            idx = int(self.rigidDomainIds[i][0])
            domain = self.domains[idx]
            for bc in domain.bcs:
                type, nodes, value = bc.processData()
                self._addBcValue(i, type, value)

        for i in range(self.numAnalytical):
            idx = int(self.rigidDomainIds[i + self.numRigids][0])
            domain = self.domains[idx]
            for bc in domain.bcs:
                type, nodes, value = bc.processData()
                self._addBcValue(i + self.numRigids, type, value)
                self.movingAnalytical = True

        for i in range(self.numAnchors):
            joint = self.joints[i]
            for bc in joint.bcs:
                type, nodes, value = bc.processData()
                target_value = 0.0
                if type == ROTVTYPE or type == VTYPE:
                    self.joint_motor_target_mode[i] = 0
                elif type == ROTATYPE or type == ATYPE:
                    self.joint_motor_target_mode[i] = 1

                if isinstance(value, (list, tuple, np.ndarray)):
                    axis_np = self.joint_axis[i].to_numpy()
                    axis_norm = np.linalg.norm(axis_np)
                    if axis_norm > 1e-9:
                        axis_np = axis_np / axis_norm
                        target_value = float(np.dot(np.asarray(value[: self.d], dtype=np.float32), axis_np))
                    else:
                        target_value = float(value[0])  # fallback to first component if axis is degenerate
                else:
                    target_value = float(value)

                self.joint_control_target[i] = target_value

    # -------  Rigid-Rigid contact related functions ----------------------------
    # ---------------------------------------------------------------------------

    def detect_all_contacts(self):
        """Python-side dispatcher for contact detection.

        Primitive rigid-rigid contacts now share one kernel entry and two
        bottom-layer geometry families:
        - convex-convex via shared GJK/EPA helpers
        - point-vs-primitive via shared SDF helpers

        Mesh kernels stay separately gated because they depend on spatial-hash
        infrastructure and should not be compiled when a scene has no meshes.
        """
        # Primitive contacts: use a single dispatch kernel so first-frame JIT
        # only needs one kernel entry for primitive rigid-rigid contacts.
        if self.num_primitive_pairs[None] > 0:
            self._detect_primitive_contacts_kernel()
        # Mesh and mixed: only compile/launch when spatialHash exists.
        # When spatialHash is None (no mesh rigids), these kernels reference
        # spatialHash.queryPointWithBuffer which Taichi cannot compile.
        has_mesh_related_pairs = (self.num_mesh_pairs[None] + self.num_mixed_pairs[None]) > 0
        if self.spatialHash is not None and has_mesh_related_pairs:
            self.maybe_rebuild_spatial_hash()
            if self.num_mesh_pairs[None] > 0:
                self.detect_mesh_mesh_contacts_kernel()
            if self.num_mixed_pairs[None] > 0:
                self.detect_mixed_contacts_kernel()

    # ── Primitive contact dispatch ──
    # A single kernel entry reduces first-frame JIT overhead for scenes that
    # only need a small subset of primitive contact types.

    @ti.func
    def _dispatch_primitive_contact(self, rigid_a: ti.i32, rigid_b: ti.i32):
        type_a = self.rigidDomainIds[rigid_a][1]
        type_b = self.rigidDomainIds[rigid_b][1]
        contact_type = type_a | type_b

        if contact_type == RigidContactType.BALLBALL:
            self.detectBallBallContact_(rigid_a, rigid_b)

        elif contact_type == RigidContactType.BOXBOX:
            self.detectBoxBoxContact_(rigid_a, rigid_b)

        elif contact_type == RigidContactType.BOXBALL:
            box_idx = rigid_a
            ball_idx = rigid_b
            if type_a == RigidType.BALL:
                box_idx = rigid_b
                ball_idx = rigid_a
            self.detectBoxBallContact_(box_idx, ball_idx)

        elif contact_type == RigidContactType.CAPSULEBOX:
            seg_idx = rigid_a
            box_idx = rigid_b
            if type_a == RigidType.BOX:
                seg_idx = rigid_b
                box_idx = rigid_a
            self.detectSegmentBoxContact_(seg_idx, box_idx)

        elif contact_type == RigidContactType.CAPSULEBALL:
            seg_idx = rigid_a
            ball_idx = rigid_b
            if type_a == RigidType.BALL:
                seg_idx = rigid_b
                ball_idx = rigid_a
            self.detectSegmentBallContact_(seg_idx, ball_idx)

        elif contact_type == RigidContactType.CAPSULECAPSULE:
            self.detectSegmentSegmentContact_(rigid_a, rigid_b)

    @ti.kernel
    def _detect_primitive_contacts_kernel(self):
        for i in range(self.num_primitive_pairs[None]):
            pair = self.primitive_pairs_buffer[i]
            self._dispatch_primitive_contact(pair[0], pair[1])

    @ti.kernel
    def detect_analyticalprim_contacts_kernel(self):
        """Detect and resolve collisions between analytical planes and rigids."""
        for i in range(self.num_groundprim_pairs[None]):
            analIdx, rigidIdx = self.groundprim_pairs_buffer[i]
            self.detectAnalaytical2Rigid(analIdx, rigidIdx)

    @ti.kernel
    def detect_mesh_mesh_contacts_kernel(self):
        """Detect and resolve collisions between two mesh rigids."""
        for i in range(self.num_mesh_pairs[None]):
            pair = self.mesh_pairs_buffer[i]
            ic = int(pair[0])
            jc = int(pair[1])
            self.detectMeshMeshContact_(ic, jc)

    @ti.kernel
    def detect_mixed_contacts_kernel(self):
        """Detect and resolve collisions between a mesh rigid and a primitive rigid."""
        for i in range(self.num_mixed_pairs[None]):
            pair = self.mixed_pairs_buffer[i]
            ic = int(pair[0])
            jc = int(pair[1])
            rigid1Type = int(self.rigidDomainIds[ic][1])

            # Determine which index is the mesh and which is primitive
            mesh_idx = jc
            other_idx = ic
            if rigid1Type == RigidType.MESH:
                mesh_idx = ic
                other_idx = jc

            self.detectMeshPrimitiveContact_(mesh_idx, other_idx)

    # ===========================================================================
    # ===== All the followings are rigid-rigid contact detection functions ======
    # ===========================================================================

    @ti.func
    def _rigid_center_3d(self, rigid_id: ti.i32):
        center = self.rigidParams[rigid_id, 0]
        return ti.math.vec3(center[0], center[1], 0.0)

    @ti.func
    def _rigid_rotation_3d(self, rigid_id: ti.i32):
        rot = self.cached_rotation_matrix[rigid_id]
        return ti.math.mat3([[rot[0, 0], rot[0, 1], 0.0], [rot[1, 0], rot[1, 1], 0.0], [0.0, 0.0, 1.0]])

    @ti.func
    def _vec3_to_sim_dim(self, value):
        return ti.Vector([value[0], value[1]])

    @ti.func
    def _box_convex_params(self, rigid_id: ti.i32):
        extent = self.rigidParams[rigid_id, 1]
        return ti.math.vec4(
            extent[0] * 0.5,
            extent[1] * 0.5,
            0.0,
            0.0,
        )

    @ti.func
    def _segment_convex_shape(self, rigid_id: ti.i32):
        rigid_type = self.rigidDomainIds[rigid_id][1]
        lcdir = self.rigidParams[rigid_id, 1]
        shape_type = 1
        if rigid_type == RigidType.CAPSULE:
            shape_type = 2
        params = ti.math.vec4(
            self.radius[rigid_id],
            lcdir[0],
            lcdir[1],
            0.0,
        )
        return shape_type, params

    @ti.func
    def _run_convex_contact_query(self, rigid_a: ti.i32, shape_type_a: ti.i32, params_a, rigid_b: ti.i32, shape_type_b: ti.i32, params_b):
        has_collision, penetration_depth, contact_normal, contact_point_a, contact_point_b = gjk_epa_collision(
            shape_type_a,
            self._rigid_center_3d(rigid_a),
            params_a,
            self._rigid_rotation_3d(rigid_a),
            shape_type_b,
            self._rigid_center_3d(rigid_b),
            params_b,
            self._rigid_rotation_3d(rigid_b),
            self.d,
        )

        hit = 0
        penetration = 0.0
        normal = ti.Vector.zero(ti.f32, self.d)
        cpoint = ti.Vector.zero(ti.f32, self.d)
        if has_collision == 1:
            normal = self._vec3_to_sim_dim(contact_normal)
            if normal.norm() > 1e-9:
                normal = normal.normalized()
            else:
                normal = (self.rigidParams[rigid_b, 0] - self.rigidParams[rigid_a, 0]).normalized(1e-9)
            cpoint = (self._vec3_to_sim_dim(contact_point_a) + self._vec3_to_sim_dim(contact_point_b)) * 0.5
            penetration = -penetration_depth
            hit = 1
        return hit, penetration, normal, cpoint

    @ti.func
    def detectBallBallContact_(self, ic, jc):
        """Handle ball-ball instantaneous collision response by velocity impulse."""

        # radii stored in row 1 component 0
        radius = self.radius[ic] + self.radius[jc]
        p = self.rigidParams[ic, 0] - self.rigidParams[jc, 0]
        l = p.norm()
        if l < radius:
            # contact happens -> use symmetric impulse pair
            n = p / l
            # contact point: midpoint between sphere centers
            cpoint_mid = (self.rigidParams[ic, 0] + self.rigidParams[jc, 0]) * 0.5
            self.cacheContact(ic, jc, cpoint_mid, n, l - radius)

    @ti.func
    def detectSegmentSegmentContact_(self, id1, id2):
        """Handle collision between two segment-like rigids (capsule) using GJK+EPA.

        - Capsule-capsule : use closest segment method
        """
        # Get rigid types to determine if using capsule
        type1 = self.rigidDomainIds[id1][1]
        type2 = self.rigidDomainIds[id2][1]

        self.detectCapsuleCapsuleContact_(id1, id2)


    @ti.func
    def detectCapsuleCapsuleContact_(self, id1, id2):
        """
        Check capsule-capsule contact using closest points on segments.
        """
        center1 = self.rigidParams[id1, 0]
        lcdir1 = self.rigidParams[id1, 1]
        lc1 = ti.Vector.zero(ti.f32, self.d)
        lc1 = self.cached_rotation_matrix[id1] @ lcdir1 + center1
        uc1 = center1 * 2 - lc1
        r1 = self.radius[id1]

        center2 = self.rigidParams[id2, 0]
        lcdir2 = self.rigidParams[id2, 1]
        lc2 = ti.Vector.zero(ti.f32, self.d)
        lc2 = self.cached_rotation_matrix[id2] @ lcdir2 + center2
        uc2 = center2 * 2 - lc2
        r2 = self.radius[id2]

        p, q, t1, t2 = calMinDisSegment2Segment(lc1, uc1, lc2, uc2)
        pq = q - p
        dis = pq.norm()

        # Compute normal direction from p to q (fallback unit-x)
        normal = ti.Vector([1.0 if i == 0 else 0.0 for i in ti.static(range(self.d))])
        if dis > 1e-9:
            normal = pq / dis
        penetration = 1.0

        # For capsule-vs-capsule the classic formula applies (point-sphere ends)
        penetration = dis - (r1 + r2)

        if penetration < 0.0:
            # apply symmetric impulses at the closest points p (on id1) and q (on id2)
            cpoint1 = p
            cpoint2 = q

            # apply a symmetric impulse pair at the midpoint between the segments
            cpoint_mid = (cpoint1 + cpoint2) * 0.5
            self.cacheContact(id1, id2, cpoint_mid, -normal, penetration)

    @ti.func
    def detectBoxBoxContact_(self, ic, jc):
        """Detect contacts from box `ic` vs box `jc` using GJK+EPA algorithm.
        GJK+EPA provides accurate penetration depth and contact normal for OBB-OBB collision.
        """
        hit, penetration, normal_ij, cpoint = self._run_convex_contact_query(
            ic, 0, self._box_convex_params(ic), jc, 0, self._box_convex_params(jc)
        )
        if hit == 1:
            self.cacheContact(ic, jc, cpoint, -normal_ij, penetration)

    @ti.func
    def detectBoxBallContact_(self, ic, jc):
        """Test box representative vertices vs a sphere and apply forces/torques."""
        pos = self.rigidParams[jc, 0]  # jc is ball, get center
        l, n, _ = detectPointToPrimitive(
            pos,
            self.rigidDomainIds[ic][1],
            self.rigidParams[ic, 0],
            self.rigidParams[ic, 1],
            self.cached_rotation_matrix[ic],
            self.radius[ic],
        )
        l -= self.radius[jc]
        if l < 0:
            # contact happens -> apply symmetric impulse between box (ic) and ball (jc)
            n = n.normalized() if n.norm() > 1e-9 else (pos - self.rigidParams[ic, 0]).normalized(1e-9)
            # contact point on sphere surface
            cpoint = pos - n * self.radius[jc]
            self.cacheContact(jc, ic, cpoint, n, l)

    @ti.func
    def detectSegmentBoxContact_(self, seg_id, other_id):
        """Detect contacts between a segment-like rigid (capsule) and
        a box/ball using GJK+EPA for accurate collision detection.

        This function handles:
        - Capsule vs Box (gjk method)
        """
        seg_shape_type, params_seg = self._segment_convex_shape(seg_id)
        hit, penetration, normal_seg_box, cpoint = self._run_convex_contact_query(
            seg_id, seg_shape_type, params_seg, other_id, 0, self._box_convex_params(other_id)
        )
        if hit == 1:
            self.cacheContact(other_id, seg_id, cpoint, normal_seg_box, penetration)

    @ti.func
    def detectSegmentBallContact_(self, seg_id, other_id):
        """segment-point contact using SDF queries (kept for ball interactions).

        - Capsule vs Ball (SDF method)
        """
        pos = self.rigidParams[other_id, 0]
        dis, normal, _ = detectPointToPrimitive(
            pos,
            self.rigidDomainIds[seg_id][1],
            self.rigidParams[seg_id, 0],
            self.rigidParams[seg_id, 1],
            self.cached_rotation_matrix[seg_id],
            self.radius[seg_id],
        )

        penetration = dis
        penetration -= self.radius[other_id]

        if penetration < 0.0:
            # compute normalized contact normal (points from segment -> sample point)
            nrm = normal.norm()
            n = ti.Vector.zero(ti.f32, self.d)
            if nrm > 1e-9:
                n = normal / nrm
            else:
                n = ti.Vector([1.0 if k == 0 else 0.0 for k in ti.static(range(self.d))])

            # contact point on the segment primitive
            cpoint = pos - normal * dis
            # apply symmetric impulse (normal points from segment -> other)
            self.cacheContact(other_id, seg_id, cpoint, n, penetration)

    @ti.func
    def detectMeshPrimitiveContact_(self, mesh_idx: ti.i32, other_idx: ti.i32):
        """Handle mesh-primitive contacts (Optimized with Spatial Hash).

        Uses Spatial Hash for primitive-sample-vs-mesh checks.
        Uses direct node iteration for mesh-node-vs-primitive checks.
        """
        mesh_local_idx = self.rigid2MeshIndices[mesh_idx]
        node_offset = self.meshBoundaryNodeOffset[mesh_local_idx]
        num_nodes = self.meshBoundaryNodeCount[mesh_local_idx]

        # Intersected bounding box (for node culling)
        intersect_lb = ti.max(
            self.aabb[self.rigidDomainIds[mesh_idx][0], 0], self.aabb[self.rigidDomainIds[other_idx][0], 0]
        )
        intersect_ub = ti.min(
            self.aabb[self.rigidDomainIds[mesh_idx][0], 1], self.aabb[self.rigidDomainIds[other_idx][0], 1]
        )

        # Expand intersection box slightly
        limit_penetration = (
            self.aabb[self.rigidDomainIds[mesh_idx][0], 1] - self.aabb[self.rigidDomainIds[mesh_idx][0], 0]
        ).min() * 0.1
        intersect_lb -= limit_penetration
        intersect_ub += limit_penetration

        # --- Optimization: Pre-calculate primitive sample points ---
        other_type = self.rigidDomainIds[other_idx][1]

        # Buffer for sample points (max 8 for Box in 3D)
        test_points = ti.Matrix.zero(ti.f32, self.d, 8)
        num_samples = 0
        p_radius = 0.0

        if other_type == RigidType.BALL:
            test_points[:, 0] = self.rigidParams[other_idx, 0]
            p_radius = self.radius[other_idx]
            num_samples = 1

        elif other_type == RigidType.BOX:
            center = self.rigidParams[other_idx, 0]
            extent = self.rigidParams[other_idx, 1]
            half_ext = extent * 0.5
            rot = self.cached_rotation_matrix[other_idx]

            num_cnt = 2**self.d
            num_samples = num_cnt
            for nid in range(num_cnt):
                local_pos = ti.Vector.zero(ti.f32, self.d)
                for k in ti.static(range(self.d)):
                    if (nid >> k) & 1:
                        local_pos[k] = half_ext[k]
                    else:
                        local_pos[k] = -half_ext[k]
                test_points[:, nid] = center + rot @ local_pos
            p_radius = 0.0

        elif other_type == RigidType.CAPSULE:
            center = self.rigidParams[other_idx, 0]
            lcdir = self.rigidParams[other_idx, 1]
            lc = self.cached_rotation_matrix[other_idx] @ lcdir + center
            uc = center * 2 - lc
            p_radius = self.radius[other_idx]

            num_s = 2
            num_samples = num_s
            for k in range(num_s):
                t = k / (num_s - 1) if num_s > 1 else 0.5
                test_points[:, k] = lc * (1 - t) + uc * t

        # Part 1: Mesh nodes vs Primitive (Iterate nodes directly)
        for nid_local in range(num_nodes):
            coord = self.meshBoundaryCoords[node_offset + nid_local]

            # Simple AABB Cull
            if (coord - intersect_ub).max() <= 0.0 and (intersect_lb - coord).max() <= 0.0:
                self._mesh_node_vs_primitive(mesh_idx, other_idx, coord)

        # Part 2: Primitive Samples vs Mesh (Using Spatial Hash)
        for k in range(num_samples):
            # Extract point from matrix column
            p = ti.Vector([test_points[d_i, k] for d_i in ti.static(range(self.d))])

            # Use spatial hash to find nearby triangles
            self._prim_point_vs_mesh_with_sh(p, p_radius, mesh_idx, other_idx, limit_penetration)

    @ti.func
    def _prim_point_vs_mesh_with_sh(
        self, point, radius: ti.f32, mesh_idx: ti.i32, other_idx: ti.i32, limit_penetration: ti.f32
    ):
        """Test a point (with optional radius) against mesh using spatial hash.

        Args:
            point: Test point position
            radius: Radius to offset (0 for vertices, >0 for balls/capsules)
            mesh_idx: Mesh rigid index
            other_idx: Primitive rigid index
            limit_penetration: Penetration tolerance

        """
        # Skip the hash query for large primitives to avoid MAX_QUERY overflow.
        # Part 1 (mesh-node vs primitive SDF) covers all contacts above this size.
        cell_size = self.spatialHash.gridSize[None].min()
        if radius < cell_size * 4.0:

            # Query spatial hash with radius to get candidate triangles
            potentialEls, dids, numPotentials = self.spatialHash.queryPointWithBuffer(point, radius, mesh_idx)

            mesh_local_idx = self.rigid2MeshIndices[mesh_idx]

            # Test against each candidate triangle
            for pot_idx in range(numPotentials):
                eidx = potentialEls[pot_idx]  # Local element ID
                if eidx >= 0:
                    conn = self.meshBoundaryElements[eidx]

                    # Compute detailed point-to-triangle distance
                    penetration, normal, cpoint, _ = detectPointToMeshBoundary(
                        point, self.meshBoundaryCoords, conn, limit_penetration=limit_penetration + radius
                    )

                    # Account for radius offset
                    l = penetration - radius
                    if l < 0.0:
                        self.cacheContact(other_idx, mesh_idx, cpoint, normal, l)

    @ti.func
    def _mesh_node_vs_primitive(self, mesh_idx: ti.i32, other_idx: ti.i32, pos):
        """Thin dispatch over shared point-vs-primitive SDF family helpers."""
        l, n, _ = detectPointToPrimitive(
            pos,
            self.rigidDomainIds[other_idx][1],
            self.rigidParams[other_idx, 0],
            self.rigidParams[other_idx, 1],
            self.cached_rotation_matrix[other_idx],
            self.radius[other_idx],
        )
        if l < 0.0:
            cpoint = pos - n * l
            self.cacheContact(mesh_idx, other_idx, cpoint, n, l)

    @ti.func
    def detectMeshMeshContact_(self, ic: ti.i32, jc: ti.i32):
        """Test mesh ic boundary nodes against mesh jc boundary elements using spatial hash acceleration.

        OPTIMIZED: Split into two separate passes to reduce JIT compilation complexity.
        Each pass is now a separate function to reduce branching and improve compilation time.

        TODO: if several nodes are in contact, we need a manifold reduction strategy to avoid over-correction during PGS iteration. 
        We can select representative contacts based on penetration depth and spatial distribution across the contact patch.
        """
        # First pass: ic nodes vs jc triangles
        self._mesh_nodes_vs_mesh_elements(ic, jc)

        # Second pass: jc nodes vs ic triangles
        self._mesh_nodes_vs_mesh_elements(jc, ic)

    @ti.func
    def _mesh_nodes_vs_mesh_elements(self, node_mesh_rigid_idx: ti.i32, elem_mesh_rigid_idx: ti.i32):
        """Test boundary nodes of node_mesh against boundary elements of elem_mesh.

        This is a specialized helper that reduces compilation complexity by handling
        only one direction of mesh-mesh collision detection.
        """
        # Get mesh indices
        node_mesh_idx = self.rigid2MeshIndices[node_mesh_rigid_idx]
        elem_mesh_idx = self.rigid2MeshIndices[elem_mesh_rigid_idx]

        # Get node mesh info
        node_offset = self.meshBoundaryNodeOffset[node_mesh_idx]
        num_nodes = self.meshBoundaryNodeCount[node_mesh_idx]

        # Get intersection bounding box for broad phase filtering
        intersect_lb = ti.max(
            self.aabb[self.rigidDomainIds[node_mesh_rigid_idx][0], 0],
            self.aabb[self.rigidDomainIds[elem_mesh_rigid_idx][0], 0],
        )
        intersect_ub = ti.min(
            self.aabb[self.rigidDomainIds[node_mesh_rigid_idx][0], 1],
            self.aabb[self.rigidDomainIds[elem_mesh_rigid_idx][0], 1],
        )
        limit_penetration = (
            0.1
            * (
                self.aabb[self.rigidDomainIds[elem_mesh_rigid_idx][0], 1]
                - self.aabb[self.rigidDomainIds[elem_mesh_rigid_idx][0], 0]
            ).min()
        )

        # Expand intersection box slightly
        intersect_lb -= limit_penetration
        intersect_ub += limit_penetration

        # Test each boundary node against elements
        for nidx in range(num_nodes):
            coord = self.meshBoundaryCoords[node_offset + nidx]

            # Broad phase: check if node is in intersection region
            if ((coord - intersect_lb).min() > 0.0) and ((coord - intersect_ub).max() < 0.0):
                # Lazy hash: rebuilt adaptively based on displacement.
                # Query buffer = cell_size + estimated displacement since rebuild.
                query_buf = self._sh_contact_margin[None]
                potentialEls, dids, numPotentials = self.spatialHash.queryPointWithBuffer(
                    coord, query_buf, elem_mesh_rigid_idx
                )

                # Test against each potential triangle
                for pot_idx in range(numPotentials):
                    elem_idx = potentialEls[pot_idx]  # Local element ID
                    if elem_idx >= 0:
                        conn = self.meshBoundaryElements[elem_idx]

                        # Triangle-level AABB check
                        lb = ti.Vector([1e30 for k in range(self.d)])
                        ub = ti.Vector([-1e30 for k in range(self.d)])
                        for j in ti.static(range(self.d)):
                            if conn[j] >= 0:
                                tri_coord = self.meshBoundaryCoords[conn[j]]
                                lb = ti.min(lb, tri_coord)
                                ub = ti.max(ub, tri_coord)

                        # Expand triangle bbox slightly
                        lb -= limit_penetration
                        ub += limit_penetration

                        # Skip if node is outside triangle's bbox
                        if (lb - coord).max() > 0.0 or (coord - ub).max() > 0.0:
                            continue

                        # Perform detailed point-to-triangle distance check
                        penetration, normal, cpoint, _ = detectPointToMeshBoundary(
                            coord, self.meshBoundaryCoords, conn, limit_penetration=limit_penetration
                        )

                        if penetration < 0.0:
                            # Cache contact (node_mesh collides with elem_mesh)
                            self.cacheContact(node_mesh_rigid_idx, elem_mesh_rigid_idx, cpoint, normal, penetration)

    # ================== The end of contact detection functions ==========================

    # -------  End of Rigid-Rigid contact related functions -----------------------------
    # -----------------------------------------------------------------------------------

    # -------  Rigid-Ground contact related functions -----------------------------
    # ----------------------------------------------------------------------------------

    def detectRigidGroundContact(self):
        """Run analytical-vs-rigid and analytical-vs-mesh contact detectors.

        OPTIMIZATION: Analytical plane contacts use kernelized dispatch.
        HeightField/Voxel contacts require Python loops (ti.template() args),
        but are skipped entirely when hasHeightFieldOrVoxel is False.

        When heightfield/voxel domains exist, we must NOT run the blanket
        analytical-plane kernel for ALL groundprim pairs, because it would
        treat heightfield/voxel domains as flat planes and generate incorrect
        duplicate contacts.  Instead, we dispatch each pair to the correct
        handler in the Python loop.
        """
        if not self.hasHeightFieldOrVoxel:
            # ── Fast path: all analytical domains are simple planes ──
            # Always launch both kernels — they read pair counts from GPU
            # memory internally and exit immediately when count == 0.
            # This eliminates two GPU→CPU syncs per rigid substep.
            self.detect_analyticalprim_contacts_kernel()
            self.detect_analyticalmesh_contacts_kernel()
            return

        # ── Slow path: mix of analytical planes, heightfields, and voxelmaps ──
        # Primitive rigids vs ground domains
        for i in range(self.num_groundprim_pairs[None]):
            rigid_i, rigid_j = self.groundprim_pairs_buffer[i]  # rigid_i : ground, rigid_j : rigid
            anlDomain = self.domains[self.rigidDomainIds[rigid_i][0]]
            if anlDomain.type == DomainType.HEIGHTFIELD and anlDomain.considerContact:
                self.detectHeightField2Rigids_(rigid_j, anlDomain)
            elif anlDomain.type == DomainType.VOXELMAP and anlDomain.considerContact:
                self.detectVoxel2Rigids_(rigid_j, anlDomain)
            elif anlDomain.considerContact:
                # True analytical plane — single-pair kernel dispatch
                self._detect_single_analyticalprim_kernel(int(rigid_i), int(rigid_j))

        # Mesh rigids vs ground domains
        for i in range(self.num_groundmesh_pairs[None]):
            rigid_i, rigid_j = self.groundmesh_pairs_buffer[i]  # rigid_i : ground, rigid_j : rigid
            anlDomain = self.domains[self.rigidDomainIds[rigid_i][0]]
            if anlDomain.type == DomainType.HEIGHTFIELD and anlDomain.considerContact:
                self.detectHeightField2MeshContacts_(rigid_j, anlDomain)
            elif anlDomain.type == DomainType.VOXELMAP and anlDomain.considerContact:
                self.detectVoxel2MeshContacts_(rigid_j, anlDomain)
            elif anlDomain.considerContact:
                # True analytical plane — single-pair kernel dispatch
                self._detect_single_analyticalmesh_kernel(int(rigid_i), int(rigid_j))

    @ti.kernel
    def _detect_single_analyticalprim_kernel(self, analIdx: ti.i32, rigidIdx: ti.i32):
        """Dispatch a single analytical-plane vs primitive-rigid contact check."""
        self.detectAnalaytical2Rigid(analIdx, rigidIdx)

    @ti.kernel
    def _detect_single_analyticalmesh_kernel(self, analIdx: ti.i32, rigidIdx: ti.i32):
        """Dispatch a single analytical-plane vs mesh-rigid contact check."""
        self.detectAnalytical2MeshPair(analIdx, rigidIdx)

    @ti.kernel
    def detect_analyticalmesh_contacts_kernel(self):
        """Detect analytical-plane vs mesh-rigid contacts, parallelized over ground-mesh pairs.

        Each GPU thread handles one (analIdx, rigidIdx) pair and iterates over its
        boundary nodes internally. This pair-level dispatch avoids the global atomic
        contention that node-level expansion would cause (2.6M atomics for 1024 envs).
        """
        for i in range(self.num_groundmesh_pairs[None]):
            analIdx = self.groundmesh_pairs_buffer[i][0]
            rigidIdx = self.groundmesh_pairs_buffer[i][1]
            self.detectAnalytical2MeshPair(analIdx, rigidIdx)

    @ti.func
    def detectAnalaytical2Rigid(self, analIdx, rigidIdx):
        """
          Detect and resolve collisions between analytical plane and rigids.
        """
        # Cache plane parameters (read once instead of per-rigid)
        planepoint = self.rigidParams[analIdx, 0]
        normal = self.rigidParams[analIdx, 1]
        anal_vel = self.V[analIdx]

        # Contact margin: detect contacts slightly before penetration for stable resting.
        contact_margin = 0.0005
        run_narrow_phase = True

        if self._ground_use_aabb_early_out[None] == 1:
            # Get global domain index from rigid domain ID
            domain_idx = self.rigidDomainIds[rigidIdx][0]

            # OPTIMIZED AABB-plane test: project AABB extent onto normal
            bbox_min = self.aabb[domain_idx, 0]  # Lower bound
            bbox_max = self.aabb[domain_idx, 1]  # Upper bound

            support_point = ti.Vector.zero(ti.f32, self.d)
            for dim in ti.static(range(self.d)):
                support_point[dim] = bbox_max[dim] if normal[dim] < 0 else bbox_min[dim]

            min_dist = (support_point - planepoint).dot(normal)
            run_narrow_phase = min_dist <= contact_margin

        if run_narrow_phase:
            # ── Compound sub-colliders ───────────────────────────────
            # If this rigid has compound shapes, test each sub-collider
            # against the analytical plane instead of the main shape.
            n_sub = self.compound_count[rigidIdx]
            if n_sub > 0:
                # print(
                #     "\033[93m[Debug] Rigid {} has {} compound sub-colliders, testing each against plane\033[0m".format(
                #         rigidIdx, n_sub
                #     )
                # )
                base = self.compound_offset[rigidIdx]
                parent_center = self.rigidParams[rigidIdx, 0]
                R = self.cached_rotation_matrix[rigidIdx]
                for k in range(n_sub):
                    idx = base + k
                    local_p = self.compound_local_pos[idx]
                    r_sub = self.compound_radius[idx]
                    world_p = R @ local_p + parent_center
                    d_sub, _, _ = detectPointToAnalyticalPlane(world_p, planepoint, normal)
                    if d_sub < r_sub + contact_margin:
                        cpoint = world_p - normal * r_sub
                        depth = d_sub - r_sub
                        self.cacheGroundContact(rigidIdx, cpoint, normal, anal_vel, depth)
            else:
                # ── Single-shape path (original logic) ───────────────────
                # Cache rigid type to avoid double indexing
                type = self.rigidDomainIds[rigidIdx][1]

                # BOX: Check vertices
                if type == RigidType.BOX:
                    num_verts = 4
                    for i in range(num_verts):
                        pos = self.get_box_vertex(rigidIdx, i)
                        d, _, _ = detectPointToAnalyticalPlane(pos, planepoint, normal)
                        if d < contact_margin:
                            self.cacheGroundContact(rigidIdx, pos, normal, anal_vel, d)

                # BALL: Check sphere center with radius
                elif type == RigidType.BALL:
                    center = self.rigidParams[rigidIdx, 0]
                    radius = self.radius[rigidIdx]
                    d, _, _ = detectPointToAnalyticalPlane(center, planepoint, normal)
                    if d < radius + contact_margin:
                        cpoint = center - normal * radius
                        depth = d - radius  # negative when penetrating
                        self.cacheGroundContact(rigidIdx, cpoint, normal, anal_vel, depth)


                # CAPSULE: Check endpoints (branchless)
                elif type == RigidType.CAPSULE:
                    center = self.rigidParams[rigidIdx, 0]
                    lcdir = self.rigidParams[rigidIdx, 1]
                    lc = ti.Vector.zero(ti.f32, self.d)
                    lc = self.cached_rotation_matrix[rigidIdx] @ lcdir + center
                    uc = center * 2.0 - lc
                    radius = self.radius[rigidIdx]

                    # Check both endpoints in unrolled loop (better for GPU)
                    for ep in ti.static(range(2)):
                        test_p = lc if ep == 0 else uc
                        d_ep, _, _ = detectPointToAnalyticalPlane(test_p, planepoint, normal)
                        if d_ep < radius + contact_margin:
                            cpoint = test_p - normal * radius
                            depth = d_ep - radius
                            self.cacheGroundContact(rigidIdx, cpoint, normal, anal_vel, depth)

    @ti.func
    def detectAnalytical2MeshPair(self, analIdx, rigidIdx):
        """Detect representative boundary-node contacts between an analytical plane
        and a mesh rigid.

        Instead of caching every penetrating boundary node (which can be dozens for
        mesh bodies and causes angular over-correction during PGS iteration), this
        selects up to 5 representative contacts that span the contact patch:
          - 1 deepest penetrating node
          - 2 extreme nodes along first tangent direction (max / min t1)
          - 2 extreme nodes along second tangent direction (max / min t2, 3D only)
        This matches the ~4-8 contacts that primitive box bodies generate.
        """
        planepoint = self.rigidParams[analIdx, 0]
        normal = self.rigidParams[analIdx, 1]
        domain_idx = self.rigidDomainIds[rigidIdx][0]
        bbox_min = self.aabb[domain_idx, 0]
        bbox_max = self.aabb[domain_idx, 1]

        # AABB-plane early exit: compute support point closest to plane
        support_point = ti.Vector.zero(ti.f32, self.d)
        for dim in ti.static(range(self.d)):
            support_point[dim] = bbox_max[dim] if normal[dim] < 0 else bbox_min[dim]
        min_dist = (support_point - planepoint).dot(normal)

        if min_dist < 0.0:
            mesh_local_idx = self.rigid2MeshIndices[rigidIdx]
            if mesh_local_idx >= 0:
                node_offset = self.meshBoundaryNodeOffset[mesh_local_idx]
                num_nodes = self.meshBoundaryNodeCount[mesh_local_idx]
                anal_vel = self.V[analIdx]

                # --- Build tangent frame for manifold extremes ---
                t1 = ti.Vector.zero(ti.f32, self.d)
                t2 = ti.Vector.zero(ti.f32, self.d)

                t1 = ti.Vector([-normal[1], normal[0]])

                # --- Representative contact tracking (single pass) ---
                has_contact = False

                # Slot 0: deepest penetrating node
                deep_d = 0.0
                deep_pos = ti.Vector.zero(ti.f32, self.d)

                # Slot 1/2: t1-direction extremes
                max_t1_proj = ti.cast(-1e30, ti.f32)
                max_t1_d = 0.0
                max_t1_pos = ti.Vector.zero(ti.f32, self.d)
                min_t1_proj = ti.cast(1e30, ti.f32)
                min_t1_d = 0.0
                min_t1_pos = ti.Vector.zero(ti.f32, self.d)

                # Slot 3/4: t2-direction extremes (3D only)
                max_t2_proj = ti.cast(-1e30, ti.f32)
                max_t2_d = 0.0
                max_t2_pos = ti.Vector.zero(ti.f32, self.d)
                min_t2_proj = ti.cast(1e30, ti.f32)
                min_t2_d = 0.0
                min_t2_pos = ti.Vector.zero(ti.f32, self.d)

                for nidx in range(num_nodes):
                    pos = self.meshBoundaryCoords[node_offset + nidx]
                    d = (pos - planepoint).dot(normal)
                    if d < 0.0:
                        has_contact = True

                        # Track deepest
                        if d < deep_d:
                            deep_d = d
                            deep_pos = pos

                        # Track t1 extremes
                        p1 = pos.dot(t1)
                        if p1 > max_t1_proj:
                            max_t1_proj = p1
                            max_t1_d = d
                            max_t1_pos = pos
                        if p1 < min_t1_proj:
                            min_t1_proj = p1
                            min_t1_d = d
                            min_t1_pos = pos

                if has_contact:
                    # Use actual node positions as contact points (consistent
                    # with detectAnalaytical2Rigid / BOX which passes the vertex
                    # position directly, not the projected-onto-plane point).
                    self.cacheGroundContact(rigidIdx, deep_pos, normal, anal_vel, deep_d)

                    # Minimum separation to consider a contact as distinct
                    sep = 0.005

                    # t1 extremes
                    if (max_t1_pos - deep_pos).norm() > sep:
                        self.cacheGroundContact(rigidIdx, max_t1_pos, normal, anal_vel, max_t1_d)
                    if (min_t1_pos - deep_pos).norm() > sep and (min_t1_pos - max_t1_pos).norm() > sep:
                        self.cacheGroundContact(rigidIdx, min_t1_pos, normal, anal_vel, min_t1_d)


    @ti.kernel
    def detectHeightField2Rigids_(self, rigidIdx: ti.i32, hf: ti.template()):
        """Detect and resolve collisions between a heightfield ground (z = h(x[,y])) and rigids.

        Optimized to iterate the collision pairs buffer instead of all rigids.
        """
        if rigidIdx != -1:
            j = rigidIdx
            # Rigid logic follows...
            # Early exit optimization: check rigid AABB against heightfield
            domain_idx = self.rigidDomainIds[j][0]
            rigid_min_x = self.aabb[domain_idx, 0][0]
            rigid_max_x = self.aabb[domain_idx, 1][0]
            rigid_min_z = self.aabb[domain_idx, 0][self.d - 1]
            rigid_max_z = self.aabb[domain_idx, 1][self.d - 1]

            # Get max/min height in the rigid's x (and y for 3D) range
            max_height_in_range = 0.0
            min_height_in_range = 0.0
            skip_rigid = False

            max_height_in_range, min_height_in_range = hf.get_maxmin_height_in_range_2d(rigid_min_x, rigid_max_x)

            # Early exit: if max heightfield height < min rigid z, no contact possible
            if max_height_in_range < rigid_min_z or (hf.reverse and (min_height_in_range > rigid_max_z)):
                skip_rigid = True

            if not skip_rigid:
                # ── Compound sub-colliders for heightfield ──────────────
                n_sub = self.compound_count[j]
                if n_sub > 0:
                    base = self.compound_offset[j]
                    parent_center = self.rigidParams[j, 0]
                    R = self.cached_rotation_matrix[j]
                    for k in range(n_sub):
                        sidx = base + k
                        local_p = self.compound_local_pos[sidx]
                        r_sub = self.compound_radius[sidx]
                        world_p = R @ local_p + parent_center
                        x = world_p[0]
                        z = world_p[1]
                        foot, n, signed = hf.nearest_on_curve_2d(x, z)
                        penetration = r_sub - signed
                        if penetration > 0.0:
                            cpoint = world_p - n * r_sub
                            self.cacheGroundContact(j, cpoint, n, ti.Vector.zero(ti.f32, self.d), signed - r_sub)
                else:
                    # ── Single-shape heightfield path ──────────────────────
                    type = self.rigidDomainIds[j][1]
                    # BOX: calculate vertices on-the-fly from center, extents, and rotation
                    if type == RigidType.BOX:
                        center = self.rigidParams[j, 0]
                        ext = self.rigidParams[j, 1]
                        num_verts = 4
                        for vi in range(num_verts):
                            # Generate local vertex position (corners of box in local space)
                            local_pos = ti.Vector.zero(ti.f32, self.d)
                            # 2D: 4 corners (±ext_x, ±ext_y)
                            local_pos[0] = ext[0] if (vi & 1) else -ext[0]
                            local_pos[1] = ext[1] if (vi & 2) else -ext[1]
                            # Rotate and translate
                            pos = self.cached_rotation_matrix[j] @ local_pos + center
                            x = pos[0]
                            z = pos[1]
                            foot, n, signed = hf.nearest_on_curve_2d(x, z)
                            if signed < 0.0:  # vertex penetrating surface
                                cpoint = pos  # contact point is the vertex itself
                                self.cacheGroundContact(j, cpoint, n, ti.Vector.zero(ti.f32, self.d), signed)

                    # BALL: test center against surface + radius
                    elif type == RigidType.BALL:
                        center = self.rigidParams[j, 0]
                        radius = self.radius[j]
                        x = center[0]
                        z = center[1]
                        foot, n, signed = hf.nearest_on_curve_2d(x, z)
                        # Ball is penetrating if signed distance < radius
                        penetration = radius - signed
                        if penetration > 0.0:
                            # Contact point is on ball surface along normal direction
                            cpoint = center - n * radius
                            self.cacheGroundContact(j, cpoint, n, ti.Vector.zero(ti.f32, self.d), signed - radius)

                    # CAPSULE: test endpoints + radius
                    elif type == RigidType.CAPSULE:
                        center = self.rigidParams[j, 0]
                        lcdir = self.rigidParams[j, 1]
                        lc = self.cached_rotation_matrix[j] @ lcdir + center
                        uc = center * 2.0 - lc
                        radius = self.radius[j]
                        for ep in ti.static(range(2)):
                            test_p = lc if ep == 0 else uc
                            x = test_p[0]
                            z = test_p[1]
                            foot, n, signed = hf.nearest_on_curve_2d(x, z)
                            penetration = radius - signed
                            if penetration > 0.0:
                                # Contact point is on capsule surface along normal
                                cpoint = test_p - n * radius
                                self.cacheGroundContact(
                                    j, cpoint, n, ti.Vector.zero(ti.f32, self.d), signed - radius
                                )

    @ti.kernel
    def detectHeightField2MeshContacts_(self, rigidIdx: ti.i32, hf: ti.template()):
        """Detect and resolve contacts between heightfield ground and mesh rigids.

        Iterates over ground-mesh pairs buffer to support multiple HFs and avoid O(N) scan.

        TODO: Similar to analytical-plane vs mesh, we can apply contact manifold reduction here to avoid over-correction from too many boundary nodes.
          We can select representative contacts based on penetration depth and spatial distribution along the heightfield surface.
        """

        if rigidIdx != -1:
            meshid = rigidIdx

            # Early exit optimization: check mesh rigid AABB against heightfield
            domain_idx = self.rigidDomainIds[meshid][0]
            mesh_min_x = self.aabb[domain_idx, 0][0]
            mesh_max_x = self.aabb[domain_idx, 1][0]
            mesh_min_z = self.aabb[domain_idx, 0][self.d - 1]
            mesh_max_z = self.aabb[domain_idx, 1][self.d - 1]

            # Get max/min height in the mesh's x (and y for 3D) range
            max_height_in_range = 0.0
            min_height_in_range = 0.0
            skip_mesh = False

            max_height_in_range, min_height_in_range = hf.get_maxmin_height_in_range_2d(mesh_min_x, mesh_max_x)

            # Early exit: if max heightfield height < min mesh z, no contact possible
            if max_height_in_range < mesh_min_z or (hf.reverse and (min_height_in_range > mesh_max_z)):
                skip_mesh = True

            if not skip_mesh:
                # Map rigid index -> per-mesh boundary arrays
                mesh_local_idx = self.rigid2MeshIndices[meshid]
                if mesh_local_idx >= 0:
                    node_offset = self.meshBoundaryNodeOffset[mesh_local_idx]
                    num_nodes = self.meshBoundaryNodeCount[mesh_local_idx]

                    for nidx in range(num_nodes):
                        pos = self.meshBoundaryCoords[node_offset + nidx]
                        x = pos[0]
                        z = pos[1]
                        foot, n, signed = hf.nearest_on_curve_2d(x, z)
                        if signed < 0.0:
                            cpoint = foot
                            self.cacheGroundContact(meshid, cpoint, n, ti.Vector.zero(ti.f32, self.d), signed)

    @ti.kernel
    def detectVoxel2Rigids_(self, rigidIdx: ti.i32, vox: ti.template()):
        """Detect collisions between a voxel grid (2D surface edges) and primitive rigids.

        Uses voxel boundary edges with outward normals for contact response.

        """
        j = rigidIdx
        rtype = self.rigidDomainIds[j][1]
        prim = self.rigidParams[j, 1]

        # 2D: edges distance
        if rtype == RigidType.BOX:
            num_verts = 4
            minExtent = prim.norm()
            for vi in range(num_verts):
                pos = self.get_box_vertex(j, vi)
                d, n, c = vox.signed_distance_to_edges_2d(pos, 0.0)
                if d < 0.0:
                    self.cacheGroundContact(j, c, n, ti.Vector.zero(ti.f32, self.d), d)
        elif rtype == RigidType.BALL:
            center = self.rigidParams[j, 0]
            radius = self.radius[j]
            d, n, c = vox.signed_distance_to_edges_2d(center, radius)
            if d < radius:
                self.cacheGroundContact(j, c, n, ti.Vector.zero(ti.f32, self.d), d - radius)
        elif rtype == RigidType.CAPSULE:
            center = self.rigidParams[j, 0]
            lcdir = self.rigidParams[j, 1]
            lc = self.cached_rotation_matrix[j] @ lcdir + center
            uc = center * 2.0 - lc
            radius = self.radius[j]
            for ep in ti.static(range(2)):
                test_p = lc if ep == 0 else uc
                d, n, c = vox.signed_distance_to_edges_2d(test_p, radius)
                if d < radius:
                    self.cacheGroundContact(j, c, n, ti.Vector.zero(ti.f32, self.d), d - radius)

    @ti.kernel
    def detectVoxel2MeshContacts_(self, rigidIdx: ti.i32, vox: ti.template()):
        """Detect collisions between voxel grid (2D) and mesh rigids via boundary nodes.
        Iterates over ground-mesh pairs buffer.

        TODO: Similar to analytical-plane vs mesh,
        we can apply contact manifold reduction here to avoid over-correction from too many boundary nodes.
        """

        if rigidIdx != -1:
            meshid = rigidIdx
            mesh_local = self.rigid2MeshIndices[meshid]

            if mesh_local >= 0:
                # Get global domain index from mesh rigid
                domain_idx = self.rigidDomainIds[meshid][0]
                # print("mesh id:", meshid, " local idx:", mesh_local)
                node_offset = self.meshBoundaryNodeOffset[mesh_local]
                num_nodes = self.meshBoundaryNodeCount[mesh_local]
                minExtent = (self.aabb[domain_idx, 1] - self.aabb[domain_idx, 0]).norm()
                for nidx in range(num_nodes):
                        pos = self.meshBoundaryCoords[node_offset + nidx]
                        d, n, c = vox.signed_distance_to_edges_2d(pos, 0.2 * minExtent)
                        if d < 0.0:
                            self.cacheGroundContact(meshid, c, n, ti.Vector.zero(ti.f32, self.d), d)

    # -------  End of Rigid-Ground contact related functions -----------------------------
    # ----------------------------------------------------------------------------------

    @ti.func
    def cacheContact(self, aid, bid, cpoint, normal, depth: ti.f32):
        """Cache a rigid-rigid contact (will use applyImpulsePair during iteration).

        Also registers the contact in per-env index lists for O(1) PGS env-lookup.

        Args:
            aid: First rigid index
            bid: Second rigid index
            cpoint: Contact point in world space
            normal: Contact normal (from aid to bid)
            depth: Signed penetration depth (negative = penetrating)
        """
        idx = ti.atomic_add(self.num_contacts[None], 1)
        if idx < self.MAX_CONTACTS:
            self.contact_rigid_a[idx] = aid
            self.contact_rigid_b[idx] = bid
            self.contact_point[idx] = cpoint
            self.contact_normal[idx] = normal
            self.contact_depth[idx] = depth
            ti.atomic_add(self.contact_count_per_rigid[aid], 1)
            ti.atomic_add(self.contact_count_per_rigid[bid], 1)
            # Compute restitution bounce target velocity (once, using pre-PGS velocity)
            ra = cpoint - self.rigidParams[aid, 0]
            rb = cpoint - self.rigidParams[bid, 0]
            e = 0.5 * (self.contactParams[aid][1] + self.contactParams[bid][1])
            v_threshold = self.restitution_velocity_threshold
            vn_pre = 0.0
            va = self.V[aid] + ti.Vector([-ra[1], ra[0]]) * self.RotV[aid][0]
            vb = self.V[bid] + ti.Vector([-rb[1], rb[0]]) * self.RotV[bid][0]
            vn_pre = (va - vb).dot(normal)
            if vn_pre < -v_threshold:
                self.contact_bounce_vel[idx] = -e * vn_pre
            else:
                self.contact_bounce_vel[idx] = 0.0
            # Build fixed tangent basis (stable across PGS iterations)
            self.contact_tangent1[idx] = ti.Vector([-normal[1], normal[0]])
            # Per-env index tracking (env_id < 0 maps to env 0 for single-env mode)
            env_id = ti.max(self.rigid_env_id[aid], 0)
            if env_id < self.MAX_ENVS_ALLOC:
                local_i = ti.atomic_add(self.contact_env_count[env_id], 1)
                if local_i < self.MAX_CC_PER_ENV:
                    self.contact_env_idx[env_id * self.MAX_CC_PER_ENV + local_i] = idx

    @ti.func
    def cacheGroundContact(self, rid, cpoint, normal, ground_vel, depth: ti.f32):
        """Cache a ground-rigid contact (will use applyImpulseAtPoint during iteration).

        Also registers the contact in per-env index lists for O(1) PGS env-lookup.

        Args:
            rid: Rigid index
            cpoint: Contact point in world space
            normal: Contact normal (pointing away from ground)
            ground_vel: Velocity of ground at contact point
            depth: Signed penetration depth (negative = penetrating, positive = margin)
        """
        idx = ti.atomic_add(self.num_ground_contacts[None], 1)
        if idx < self.MAX_GROUND_CONTACTS:
            self.ground_contact_rigid[idx] = rid
            self.ground_contact_point[idx] = cpoint
            self.ground_contact_normal[idx] = normal
            self.ground_contact_vel[idx] = ground_vel
            self.ground_contact_depth[idx] = depth
            # Compute restitution bounce target velocity (once, using pre-PGS velocity)
            lr = cpoint - self.rigidParams[rid, 0]
            e = self.contactParams[rid][1]
            v_threshold = self.restitution_velocity_threshold
            vn_pre = 0.0
            tlr = ti.Vector([-lr[1], lr[0]])
            v_point = self.V[rid] + tlr * self.RotV[rid][0]
            vn_pre = (v_point - ground_vel).dot(normal)
            if vn_pre < -v_threshold:
                self.ground_contact_bounce_vel[idx] = -e * vn_pre
            else:
                self.ground_contact_bounce_vel[idx] = 0.0
            # Build fixed tangent basis (stable across PGS iterations)
            # 2D: single tangent = 90-degree rotation of normal
            self.ground_contact_tangent1[idx] = ti.Vector([-normal[1], normal[0]])

            # Per-env index tracking (env_id < 0 maps to env 0 for single-env mode)
            env_id = ti.max(self.rigid_env_id[rid], 0)
            if env_id < self.MAX_ENVS_ALLOC:
                local_i = ti.atomic_add(self.ground_contact_env_count[env_id], 1)
                if local_i < self.MAX_GC_PER_ENV:
                    self.ground_contact_env_idx[env_id * self.MAX_GC_PER_ENV + local_i] = idx

    @ti.func
    def _add_pgs_row(
        self,
        aid,
        bid,
        jac_a,
        jac_b,
        rhs,
        lower,
        upper,
        parent_row,
    ):
        flag = -1
        ci = ti.atomic_add(self.numConstraints[None], 1)
        if ci < self.MAX_CONSTRAINTS:
            self.pgs_bodypair[ci] = ti.Vector([aid, bid])
            self.pgs_Jac_a[ci] = jac_a
            self.pgs_Jac_b[ci] = jac_b
            self.pgs_rhs[ci] = rhs
            self.pgs_limits[ci] = ti.Vector([lower, upper])
            self.pgs_lambda[ci] = 0.0
            self.pgs_parent_row[ci] = parent_row
            flag = ci
        return flag

    @ti.func
    def _assemble_ground_contact_rows(self, idx: ti.i32, dt: ti.f32):
        rid = self.ground_contact_rigid[idx]
        cpoint = self.ground_contact_point[idx]
        normal = self.ground_contact_normal[idx]
        ground_vel = self.ground_contact_vel[idx]
        depth = self.ground_contact_depth[idx]
        lr = cpoint - self.rigidParams[rid, 0]
        mu = self.contactParams[rid][0]

        bounce_vel = self.ground_contact_bounce_vel[idx]

        bias_vel = 0.0
        if depth < 0.0 and self.contact_erp > 0.0:
            bias_vel = ti.max(self.contact_erp * depth / dt, -5.0)

        # Properly decouple restitution and baumgarte separation velocities
        target_vel = bounce_vel
        if -bias_vel > bounce_vel:
            target_vel = -bias_vel

        jac_n = ti.Vector.zero(ti.f32, 6)
        rcn_s = vectorCrossProduct(lr, normal)[0]
        jac_n = ti.Vector([normal[0], normal[1], 0.0, 0.0, 0.0, rcn_s])

        normal_row = self._add_pgs_row(
            rid, -1, jac_n, ti.Vector.zero(ti.f32, 6), target_vel + normal.dot(ground_vel), 0.0, 1e10, -1
        )
        self.ground_contact_pgs_indices[idx] = ti.Vector([normal_row, -1, -1])
        if mu > 1e-12 and normal_row >= 0:
            t1 = self.ground_contact_tangent1[idx]
            jac_t1 = ti.Vector.zero(ti.f32, 6)
            rct1_s = vectorCrossProduct(lr, t1)[0]
            jac_t1 = ti.Vector([t1[0], t1[1], 0.0, 0.0, 0.0, rct1_s])
            rhs_t1 = t1.dot(ground_vel)
            tangent1_row = self._add_pgs_row(rid, -1, jac_t1, ti.Vector.zero(ti.f32, 6), rhs_t1, -mu, mu, normal_row)
            self.ground_contact_pgs_indices[idx][1] = tangent1_row


    @ti.func
    def _assemble_pair_contact_rows(self, idx: ti.i32, dt: ti.f32):
        aid = self.contact_rigid_a[idx]
        bid = self.contact_rigid_b[idx]
        cpoint = self.contact_point[idx]
        normal = self.contact_normal[idx]
        depth = self.contact_depth[idx]
        ra = cpoint - self.rigidParams[aid, 0]
        rb = cpoint - self.rigidParams[bid, 0]
        mu = 0.5 * (self.contactParams[aid][0] + self.contactParams[bid][0])

        bounce_vel = self.contact_bounce_vel[idx]

        bias_vel = 0.0
        if depth < 0.0 and self.contact_erp > 0.0:
            bias_vel = ti.max(self.contact_erp * depth / dt, -5.0)

        # Properly decouple restitution and baumgarte separation velocities
        # Both are positive requested separation velocities
        target_vel = bounce_vel
        if -bias_vel > bounce_vel:
            target_vel = -bias_vel

        jac_na = ti.Vector.zero(ti.f32, 6)
        jac_nb = ti.Vector.zero(ti.f32, 6)
        raxn_s = vectorCrossProduct(ra, normal)[0]
        rbxn_s = vectorCrossProduct(rb, normal)[0]
        jac_na = ti.Vector([normal[0], normal[1], 0.0, 0.0, 0.0, raxn_s])
        jac_nb = ti.Vector([normal[0], normal[1], 0.0, 0.0, 0.0, rbxn_s])

        normal_row = self._add_pgs_row(aid, bid, jac_na, jac_nb, target_vel, 0.0, 1e10, -1)
        self.contact_pgs_indices[idx] = ti.Vector([normal_row, -1, -1])

        if mu > 1e-12 and normal_row >= 0:
            t1 = self.contact_tangent1[idx]
            jac_t1a = ti.Vector.zero(ti.f32, 6)
            jac_t1b = ti.Vector.zero(ti.f32, 6)

            raxt1_s = vectorCrossProduct(ra, t1)[0]
            rbxt1_s = vectorCrossProduct(rb, t1)[0]
            jac_t1a = ti.Vector([t1[0], t1[1], 0.0, 0.0, 0.0, raxt1_s])
            jac_t1b = ti.Vector([t1[0], t1[1], 0.0, 0.0, 0.0, rbxt1_s])
            tangent1_row = self._add_pgs_row(aid, bid, jac_t1a, jac_t1b, 0.0, -mu, mu, normal_row)
            self.contact_pgs_indices[idx][1] = tangent1_row


    @ti.kernel
    def _assemble_contact_constraints_kernel(self, dt: ti.f32):
        for idx in range(self.num_ground_contacts[None]):
            self._assemble_ground_contact_rows(idx, dt)
        for idx in range(self.num_contacts[None]):
            self._assemble_pair_contact_rows(idx, dt)

    @ti.kernel
    def _compute_contact_forces_kernel(self, dt: ti.f32):
        """
        Compute contact forces from accumulated impulses after PGS solve.

        Contact force = impulse / dt

        For ground contacts:
            force = (lambda_n * normal + lambda_t1 * tangent1 + lambda_t2 * tangent2) / dt

        For rigid-rigid contacts:
            force = (lambda_n * normal + lambda_t1 * tangent1 + lambda_t2 * tangent2) / dt
        """
        eps = 1e-12
        dt_inv = 1.0 / (dt + eps)

        # Compute ground contact forces
        for idx in range(self.num_ground_contacts[None]):
            pgs_indices = self.ground_contact_pgs_indices[idx]
            normal_row = pgs_indices[0]
            tangent1_row = pgs_indices[1]
            tangent2_row = pgs_indices[2]

            lambda_n = 0.0
            if normal_row >= 0:
                lambda_n = self.pgs_lambda[normal_row]

            lambda_t1 = 0.0
            if tangent1_row >= 0:
                lambda_t1 = self.pgs_lambda[tangent1_row]

            lambda_t2 = 0.0
            if tangent2_row >= 0:
                lambda_t2 = self.pgs_lambda[tangent2_row]

            normal = self.ground_contact_normal[idx]
            tangent1 = self.ground_contact_tangent1[idx]

            force = lambda_n * normal + lambda_t1 * tangent1

            self.ground_contact_force[idx] = force * dt_inv

        # Compute rigid-rigid contact forces
        for idx in range(self.num_contacts[None]):
            pgs_indices = self.contact_pgs_indices[idx]
            normal_row = pgs_indices[0]
            tangent1_row = pgs_indices[1]
            tangent2_row = pgs_indices[2]

            lambda_n = 0.0
            if normal_row >= 0:
                lambda_n = self.pgs_lambda[normal_row]

            lambda_t1 = 0.0
            if tangent1_row >= 0:
                lambda_t1 = self.pgs_lambda[tangent1_row]

            lambda_t2 = 0.0
            if tangent2_row >= 0:
                lambda_t2 = self.pgs_lambda[tangent2_row]

            normal = self.contact_normal[idx]
            tangent1 = self.contact_tangent1[idx]

            force = lambda_n * normal + lambda_t1 * tangent1

            self.contact_force[idx] = force * dt_inv

    @ti.kernel
    def _assemble_joint_constraints_kernel(self, dt: ti.f32):
        for j_idx in range(self.numAnchors):
            assemble_single_joint_rows(self, dt, j_idx)

    @ti.kernel
    def reset_contact_caches_kernel(self):
        """Reset contact counters and force arrays in a single kernel launch.
        Replaces 2x fill() + 2x [None]=0 = 4 Python-Taichi round trips with 1.
        Uses bounded reset: only clears entries that were actually written.

        Also saves per-rigid total lambda from previous frame for warm-starting.
        """
        prev_nc = self.prev_num_contacts[None]
        prev_ngc = self.prev_num_ground_contacts[None]
        self.prev_num_contacts[None] = self.num_contacts[None]
        self.prev_num_ground_contacts[None] = self.num_ground_contacts[None]

        # Save per-rigid total lambda for warm-starting before clearing
        total_nodes = self.numRigids + self.numAnalytical

        self.num_contacts[None] = 0
        self.num_ground_contacts[None] = 0
        self.numConstraints[None] = 0
        # Reset per-env contact counters
        n_envs = ti.max(self.num_envs, 1)
        i = 0
        while i < n_envs:
            self.ground_contact_env_count[i] = 0
            self.contact_env_count[i] = 0
            i += 1
        for rid in range(total_nodes):
            self.contact_count_per_rigid[rid] = 0
        for j in range(prev_nc):
            self.contact_force[j] = ti.Vector.zero(ti.f32, self.d)
            self.contact_pgs_indices[j] = ti.Vector([-1, -1, -1])
            self.contact_bounce_vel[j] = 0.0
        for k in range(prev_ngc):
            self.ground_contact_force[k] = ti.Vector.zero(ti.f32, self.d)
            self.ground_contact_pgs_indices[k] = ti.Vector([-1, -1, -1])
            self.ground_contact_bounce_vel[k] = 0.0

    @ti.kernel
    def solve_pgs(self, pgs_iters: ti.i32):
        """
        Given that the PGS has been set up, solve the PGS.
        This function is called after the PGS has been set up, and before the next step.
        Now we have all:
            - V: current velocity
            - RotV: current angular velocity
            - pgs_Jac_a: Jacobian matrix
            - pgs_rhs: right-hand side vector
            - pgs_bodypair: body pairs, body a and body b
            - mass: mass matrix
            - inertia: inertia matrix
        """
        ti.loop_config(serialize=True)
        n_constraints = ti.min(self.numConstraints[None], ti.static(self.MAX_CONSTRAINTS))
        for _iter in range(pgs_iters):
            # print("Number of constraints: ", self.numConstraints[None])
            for i in range(n_constraints):
                self.solve_pgs_single_func(i)

            for i in range(n_constraints):
                k = n_constraints - 1 - i
                self.solve_pgs_single_func(k)

    @ti.func
    def solve_pgs_single_func(self, i: ti.i32):
        bodypair = self.pgs_bodypair[i]
        aid = bodypair[0]
        bid = bodypair[1]
        if aid >= 0:
            jac_a = self.pgs_Jac_a[i]
            jac_b = self.pgs_Jac_b[i]

            va6 = ti.Vector.zero(ti.f32, 6)
            vb6 = ti.Vector.zero(ti.f32, 6)

            massInvA = ti.Matrix.zero(ti.f32, 6, 6)
            massInvB = ti.Matrix.zero(ti.f32, 6, 6)

            inv_mass_a = 1.0 / (self.mass[aid] + 1e-12)
            massInvA[0, 0] = inv_mass_a
            massInvA[1, 1] = inv_mass_a
            massInvA[2, 2] = inv_mass_a

            va6 = ti.Vector([self.V[aid][0], self.V[aid][1], 0.0, 0.0, 0.0, self.RotV[aid][0]])
            inv_Ia = 1.0 / (self.inertia[aid] + 1e-12)
            massInvA[5, 5] = inv_Ia

            vel = jac_a.dot(va6)
            massInvJacA = massInvA @ jac_a
            W = jac_a.dot(massInvJacA)

            has_b = bid >= 0
            massInvJacB = ti.Vector.zero(ti.f32, 6)
            if has_b:
                inv_mass_b = 1.0 / (self.mass[bid] + 1e-12)
                massInvB[0, 0] = inv_mass_b
                massInvB[1, 1] = inv_mass_b
                massInvB[2, 2] = inv_mass_b

                vb6 = ti.Vector([self.V[bid][0], self.V[bid][1], 0.0, 0.0, 0.0, self.RotV[bid][0]])
                inv_Ib = 1.0 / (self.inertia[bid] + 1e-12)
                massInvB[5, 5] = inv_Ib

                vel -= jac_b.dot(vb6)
                massInvJacB = massInvB @ jac_b
                W += jac_b.dot(massInvJacB)

            rhs = self.pgs_rhs[i] - vel
            delta_lamb = rhs / (W + 1e-12)

            old_lamb = self.pgs_lambda[i]
            new_lamb = old_lamb + delta_lamb

            parent = self.pgs_parent_row[i]
            if parent >= 0:
                fric_lim_low = self.pgs_limits[i][0] * self.pgs_lambda[parent]
                fric_lim_upper = self.pgs_limits[i][1] * self.pgs_lambda[parent]
                new_lamb = ti.max(fric_lim_low, ti.min(fric_lim_upper, new_lamb))
            else:
                lower = self.pgs_limits[i][0]
                upper = self.pgs_limits[i][1]
                new_lamb = ti.max(lower, ti.min(upper, new_lamb))

            apply_lamb = new_lamb - old_lamb
            self.pgs_lambda[i] = new_lamb

            if apply_lamb != 0.0:
                deltaA = massInvJacA * apply_lamb

                self.V[aid] += ti.Vector([deltaA[0], deltaA[1]])
                self.RotV[aid][0] += deltaA[5]

                if has_b:
                    deltaB = massInvJacB * apply_lamb

                    self.V[bid] -= ti.Vector([deltaB[0], deltaB[1]])
                    self.RotV[bid][0] -= deltaB[5]

    @ti.kernel
    def precompute_rigid_transforms(self):
        """Precompute rotation matrices and world-space inertia tensors for all rigids.

        OPTIMIZATION: Called once before joint solving to cache expensive computations.
        Avoids recomputing the same rotation matrix and inertia transformation for
        each joint involving the same rigid body. For robots with 16-20 joints,
        this reduces computation by 50-70%.
        """
        for i in range(self.numRigids + self.numAnalytical):

            # Cache 2D rotation angle and inverse inertia
            self.cached_rotation_matrix[i] = cal2DRotationMat(self.quat[i][0] + self.visual_angle[i])
            I = self.inertia[i]
            if I > 0.0:
                self.cached_inertia_inv_2d[i] = 1.0 / I
            else:
                self.cached_inertia_inv_2d[i] = 1.0 / 1e-6

    @ti.func
    def _mask_allows_pair(self, idx_a: ti.i32, idx_b: ti.i32) -> ti.i32:
        allow_ab = (self.collide_bits[idx_a] & self.category_bits[idx_b]) != ti.u32(0)
        allow_ba = (self.collide_bits[idx_b] & self.category_bits[idx_a]) != ti.u32(0)
        return ti.cast(allow_ab and allow_ba, ti.i32)

    @ti.kernel
    def classify_collision_pairs_kernel(self, pairs: ti.template(), num_pairs: ti.i32):
        """Kernel to classify collision pairs into specific buffers."""
        # Clear pair buffers
        self.num_primitive_pairs[None] = 0
        self.num_ball_ball_pairs[None] = 0
        self.num_box_box_pairs[None] = 0
        self.num_box_ball_pairs[None] = 0
        self.num_seg_point_pairs[None] = 0
        self.num_seg_seg_pairs[None] = 0

        self.num_mesh_pairs[None] = 0
        self.num_mixed_pairs[None] = 0
        self.num_groundprim_pairs[None] = 0
        self.num_groundmesh_pairs[None] = 0

        for i in range(num_pairs):
            domain_a = pairs[i][0]
            domain_b = pairs[i][1]

            # Look up rigid indices from domain indices
            rigid_a = self.domainToRigid[domain_a]
            rigid_b = self.domainToRigid[domain_b]

            # Skip if either is not a rigid (-1 means not found)
            if rigid_a >= 0 and rigid_b >= 0:
                if self._mask_allows_pair(rigid_a, rigid_b) == 0:
                    continue

                is_anal_a = rigid_a >= self.numRigids
                is_anal_b = rigid_b >= self.numRigids

                # Classify pair type
                type_a = self.rigidDomainIds[rigid_a][1]
                type_b = self.rigidDomainIds[rigid_b][1]
                is_mesh_a = (type_a == RigidType.MESH) and (self.compound_count[rigid_a] == 0)
                is_mesh_b = (type_b == RigidType.MESH) and (self.compound_count[rigid_b] == 0)

                if is_anal_a or is_anal_b:
                    # Ground collision
                    # One must be analytical, one must be rigid/mesh
                    # Analytical domains effectively have infinite mass/fixed
                    # If both are analytical, we skip (logic below handles it)

                    if is_anal_a and not is_anal_b:
                        if is_mesh_b:
                            idx = ti.atomic_add(self.num_groundmesh_pairs[None], 1)
                            if idx < self.MAX_GROUND_PAIRS:
                                self.groundmesh_pairs_buffer[idx] = ti.Vector([rigid_a, rigid_b])
                        else:
                            idx = ti.atomic_add(self.num_groundprim_pairs[None], 1)
                            if idx < self.MAX_GROUND_PAIRS:
                                self.groundprim_pairs_buffer[idx] = ti.Vector([rigid_a, rigid_b])

                    elif is_anal_b and not is_anal_a:
                        if is_mesh_a:
                            idx = ti.atomic_add(self.num_groundmesh_pairs[None], 1)
                            if idx < self.MAX_GROUND_PAIRS:
                                self.groundmesh_pairs_buffer[idx] = ti.Vector([rigid_b, rigid_a])
                        else:
                            idx = ti.atomic_add(self.num_groundprim_pairs[None], 1)
                            if idx < self.MAX_GROUND_PAIRS:
                                self.groundprim_pairs_buffer[idx] = ti.Vector([rigid_b, rigid_a])
                else:
                    # Rigid-rigid collision
                    considerRigidPair = True

                    # **COLLISION FILTERING: Skip cross-environment collisions**
                    # For batched training, no rigid-rigid contact is considered, as for the same env, they are connected with joints.
                    # For different envs, they should not collide. This is a simple filter based on environment IDs.
                    # Ground (-1) is env-independent and should collide with anything.
                    env_a = self.rigid_env_id[rigid_a]
                    env_b = self.rigid_env_id[rigid_b]
                    if env_a >= 0 and env_b >= 0 and env_a != env_b:
                        considerRigidPair = False  # Different environments, skip

                    if considerRigidPair:
                        if is_mesh_a and is_mesh_b:
                            # Mesh-mesh
                            idx = ti.atomic_add(self.num_mesh_pairs[None], 1)
                            if idx < self.MAX_COLLISION_PAIRS:
                                self.mesh_pairs_buffer[idx] = ti.Vector([rigid_a, rigid_b])

                        elif is_mesh_a or is_mesh_b:
                            # Mixed (mesh-primitive)
                            idx = ti.atomic_add(self.num_mixed_pairs[None], 1)
                            if idx < self.MAX_COLLISION_PAIRS:
                                self.mixed_pairs_buffer[idx] = ti.Vector([rigid_a, rigid_b])

                        else:
                            # Primitive-primitive
                            contact_type = type_a | type_b

                            if contact_type == RigidContactType.BALLBALL:
                                idx = ti.atomic_add(self.num_ball_ball_pairs[None], 1)
                                if idx < self.MAX_COLLISION_PAIRS:
                                    self.ball_ball_pairs_buffer[idx] = ti.Vector([rigid_a, rigid_b])

                            elif contact_type == RigidContactType.BOXBOX:
                                idx = ti.atomic_add(self.num_box_box_pairs[None], 1)
                                if idx < self.MAX_COLLISION_PAIRS:
                                    self.box_box_pairs_buffer[idx] = ti.Vector([rigid_a, rigid_b])

                            elif contact_type == RigidContactType.BOXBALL:
                                idx = ti.atomic_add(self.num_box_ball_pairs[None], 1)
                                if idx < self.MAX_COLLISION_PAIRS:
                                    # Ensure rigid_a is box, rigid_b is ball
                                    r_a, r_b = rigid_a, rigid_b
                                    if type_a == RigidType.BALL:
                                        r_a, r_b = rigid_b, rigid_a
                                    self.box_ball_pairs_buffer[idx] = ti.Vector([r_a, r_b])

                            elif contact_type == RigidContactType.CAPSULEBOX:
                                idx = ti.atomic_add(self.num_seg_point_pairs[None], 1)
                                if idx < self.MAX_COLLISION_PAIRS:
                                    # Ensure rigid_a is segment, rigid_b is box
                                    r_a, r_b = rigid_a, rigid_b
                                    if type_a == RigidType.BOX:
                                        r_a, r_b = rigid_b, rigid_a
                                    self.seg_point_pairs_buffer[idx] = ti.Vector([r_a, r_b])

                            elif contact_type == RigidContactType.CAPSULEBALL:
                                idx = ti.atomic_add(self.num_seg_ball_pairs[None], 1)
                                if idx < self.MAX_COLLISION_PAIRS:
                                    # Ensure rigid_a is segment, rigid_b is box/ball
                                    r_a, r_b = rigid_a, rigid_b
                                    if type_a == RigidType.BALL:
                                        r_a, r_b = rigid_b, rigid_a
                                    self.seg_ball_pairs_buffer[idx] = ti.Vector([r_a, r_b])

                            elif contact_type == RigidContactType.CAPSULECAPSULE:
                                idx = ti.atomic_add(self.num_seg_seg_pairs[None], 1)
                                if idx < self.MAX_COLLISION_PAIRS:
                                    # Ensure rigid_a is capsule if mixed (optional)
                                    r_a, r_b = rigid_a, rigid_b
                                    self.seg_seg_pairs_buffer[idx] = ti.Vector([r_a, r_b])

                            # Also keep in general primitive buffer
                            idx = ti.atomic_add(self.num_primitive_pairs[None], 1)
                            if idx < self.MAX_COLLISION_PAIRS:
                                self.primitive_pairs_buffer[idx] = ti.Vector([rigid_a, rigid_b])

    def maybe_rebuild_spatial_hash(self, fem_lb=None, fem_ub=None, fem_margin=0.0):
        """Rebuild the mesh spatial hash if enough time has elapsed or
        max displacement since last rebuild exceeds 0.5 * cell_size.
        Called before detect_mesh_mesh_contacts_kernel each substep."""

        if self.spatialHash is None or not self._sh_mesh_needs_rebuild:
            return

        if self.considerRigidRigidContact:
            self.populate_spatial_hash_filtered()
        else:
            self.populate_spatial_hash_filtered(fem_lb=fem_lb, fem_ub=fem_ub)

        cell_size = float(self.spatialHash.gridSize[None].max())
        self._sh_mesh_max_v = self.get_max_linear_velocity()
        self._sh_contact_margin[None] = cell_size
        self._sh_mesh_cell_size = cell_size

        # print(
        #     f"\033[93m[Contact] Rebuilding mesh spatial hash... "
        #     f"(max displacement: {self._sh_mesh_max_v * self._sh_mesh_elapsed:.4f} m, "
        #     f"elapsed time: {self._sh_mesh_elapsed:.4f} s, "
        #     f"cell size: {self._sh_mesh_cell_size:.4f} m)\033[0m"
        # )

        self._sh_mesh_elapsed = 0.0
        self._sh_mesh_needs_rebuild = False

    def populate_spatial_hash_filtered(self, fem_lb=None, fem_ub=None):
        """Populate spatial hash with mesh elements filtered by FEM AABB intersection.

        For each mesh rigid, computes the intersection of the rigid AABB with
        the FEM contact node AABB.  Only elements overlapping this intersection
        (expanded by margin) are inserted, dramatically reducing the number of
        hashed elements when FEM nodes only cover part of the rigid surface.
        If fem_lb or fem_ub is None, inserts all mesh elements.
        """
        if self.spatialHash is None or self.numMesh == 0:
            return

        if fem_lb is None or fem_ub is None:
            fem_lb_field = self._sh_unbounded_lb
            fem_ub_field = self._sh_unbounded_ub
        else:
            fem_lb_field = fem_lb
            fem_ub_field = fem_ub

        self.spatialHash.reset()
        contact_margin = self._sh_contact_margin[None]
        velocity_buffer = 2.0 * self._sh_mesh_max_v * self._sh_mesh_rebuild_interval
        self._populate_spatial_hash_filtered_kernel(
            float(velocity_buffer),
            float(contact_margin),
            fem_lb_field,
            fem_ub_field,
        )
        self.spatialHash.build()
        self._sh_contact_margin[None] = float(contact_margin) + float(velocity_buffer)

    @ti.kernel
    def _populate_spatial_hash_filtered_kernel(
        self, velocity_buffer: ti.f32, contact_margin: ti.f32, fem_lb_field: ti.template(), fem_ub_field: ti.template()
    ):
        """Insert only mesh elements that overlap the FEM-rigid AABB intersection."""
        fem_lb = fem_lb_field[None]
        fem_ub = fem_ub_field[None]

        # Global bounding box for the grid — still covers all mesh rigids
        # so that SH cells are valid for any FEM query point.
        global_lb = ti.Vector([ti.f32(1e9) for _ in range(self.d)])
        global_ub = ti.Vector([ti.f32(-1e9) for _ in range(self.d)])

        for k in range(self.numMesh):
            rigid_id = self.mesh2RigidIndices[k]
            domain_id = self.rigidDomainIds[rigid_id][0]
            rigid_lb_k = self.aabb[domain_id, 0]
            rigid_ub_k = self.aabb[domain_id, 1]
            # Use FEM-rigid intersection for tighter grid bounds
            isect_lb_k = ti.max(rigid_lb_k, fem_lb) - contact_margin
            isect_ub_k = ti.min(rigid_ub_k, fem_ub) + contact_margin
            has_isect_k = True
            for d_idx in ti.static(range(self.d)):
                if isect_lb_k[d_idx] > isect_ub_k[d_idx]:
                    has_isect_k = False
            if has_isect_k:
                global_lb = ti.min(global_lb, isect_lb_k)
                global_ub = ti.max(global_ub, isect_ub_k)

        if global_lb[0] <= global_ub[0]:
            expand = (global_ub - global_lb).max() * 0.05 + contact_margin
            global_lb -= expand
            global_ub += expand

            for k in range(self.numMesh):
                rigid_id = self.mesh2RigidIndices[k]
                domain_id = self.rigidDomainIds[rigid_id][0]
                rigid_lb = self.aabb[domain_id, 0]
                rigid_ub = self.aabb[domain_id, 1]

                # Intersection of rigid AABB and FEM contact AABB (+ margin)
                isect_lb = ti.max(rigid_lb, fem_lb) - contact_margin
                isect_ub = ti.min(rigid_ub, fem_ub) + contact_margin

                # If no intersection, skip all elements of this rigid
                has_isect = True
                for d_idx in ti.static(range(self.d)):
                    if isect_lb[d_idx] > isect_ub[d_idx]:
                        has_isect = False

                elem_offset = self.meshBoundaryElementOffset[k]
                num_elems = self.meshBoundaryElementCount[k]
                for eidx in range(num_elems):
                    global_eidx = elem_offset + eidx
                    conn = self.meshBoundaryElements[global_eidx]

                    lb = ti.Vector([ti.f32(1e30) for _ in range(self.d)])
                    ub = ti.Vector([ti.f32(-1e30) for _ in range(self.d)])
                    for j in ti.static(range(3)):
                        if conn[j] >= 0:
                            coord = self.meshBoundaryCoords[conn[j]]
                            lb = ti.min(lb, coord)
                            ub = ti.max(ub, coord)

                    self.meshElemLB[global_eidx] = lb
                    self.meshElemUB[global_eidx] = ub
                    # consider element margin and velocity buffer
                    self.meshElemMarginBase[global_eidx] = (
                        self.meshRigidContactMarginRatio * (ub - lb).norm() + velocity_buffer
                    )

                    # Only insert if element AABB overlaps the FEM-rigid intersection
                    if has_isect:
                        overlaps = True
                        for d_idx in ti.static(range(self.d)):
                            if ub[d_idx] < isect_lb[d_idx] or lb[d_idx] > isect_ub[d_idx]:
                                overlaps = False
                        if overlaps:
                            self.spatialHash.addElement(lb, ub, rigid_id, global_eidx, velocity_buffer)
        else:
            global_lb = ti.Vector([ti.f32(0.0) for _ in range(self.d)])
            global_ub = ti.Vector([ti.f32(1.0) for _ in range(self.d)])

        self.spatialHash.setBounds(global_lb, global_ub)

    @ti.kernel
    def update_mesh_element_aabbs(self):
        """Refresh cached mesh element AABBs from current transformed coords."""
        for k in range(self.numMesh):
            elem_offset = self.meshBoundaryElementOffset[k]
            num_elems = self.meshBoundaryElementCount[k]
            for eidx in range(num_elems):
                global_eidx = elem_offset + eidx
                conn = self.meshBoundaryElements[global_eidx]
                lb = ti.Vector([1e30 for _ in range(self.d)])
                ub = ti.Vector([-1e30 for _ in range(self.d)])
                for j in ti.static(range(3)):
                    if conn[j] >= 0:
                        coord = self.meshBoundaryCoords[conn[j]]
                        lb = ti.min(lb, coord)
                        ub = ti.max(ub, coord)
                self.meshElemLB[global_eidx] = lb
                self.meshElemUB[global_eidx] = ub
                self.meshElemMarginBase[global_eidx] = self.meshRigidContactMarginRatio * (ub - lb).norm()

    # ════════════════════════════════════════════════════════════════════
    # FUSED KERNELS — minimize kernel launch overhead (~0.12ms per launch)
    # ════════════════════════════════════════════════════════════════════

    @ti.kernel
    def _rigidStep_and_precompute_kernel(self, dt: float, damping: float):
        """Fused: precompute_rigid_transforms + rigidStep in 1 kernel launch.
        Phase 1 precomputes world-frame inertia from current quaternion.
        Phase 2 uses it for velocity integration (world-frame).
        """
        # ── Phase 1: Precompute rotation matrices + world inertia ──
        for i in range(self.numRigids + self.numAnalytical):

            self.cached_rotation_matrix[i] = cal2DRotationMat(self.quat[i][0] + self.visual_angle[i])
            I = self.inertia[i]
            if I > 0.0:
                self.cached_inertia_inv_2d[i] = 1.0 / I
            else:
                self.cached_inertia_inv_2d[i] = 1.0 / 1e-6

        # ── Phase 2: Velocity integration (using world-frame inertia) ──
        for i in range(self.numRigids):
            self._calculate_bc_for_index(i, dt)
            V_i = self.V[i]
            mass_i = self.mass[i]
            I_i = self.accumulated_impulse[i]
            RI_i = self.accumulated_rotational_impulse[i]

            self.V[i] = V_i + I_i / mass_i
            inertia_i = self.inertia[i]
            self.RotV[i] += RI_i / (inertia_i + 1e-6)

            # Rigid body velocity damping
            damp_factor = ti.max(0.0, 1.0 - damping * dt)
            self.V[i] *= damp_factor
            self.RotV[i] *= damp_factor

        # ── Phase 3: Apply enforced velocity/acceleration BCs ──
        for i in range(self.numRigids + self.numAnalytical):
            bc_type = self.bcNodes[i]
            if (bc_type & ATYPE) != 0:
                self.V[i] += self.bcTValues[i] * dt
            if (bc_type & ROTATYPE) != 0:
                self.RotV[i] += self.bcRValues[i] * dt

    @ti.kernel
    def _updateU_and_BBox_kernel(self, dt: float, update_bbox: ti.i32):
        """Fused: position integration (updateU) + AABB update (updateBBox) in 1 kernel.
        Saves ~0.12ms by eliminating one Python↔Taichi round trip.
        """
        # ── Phase 1: Integrate positions and quaternions ──
        for i in range(self.numRigids + self.numAnalytical):
            self._update_bc_for_index(i)
            V_i = self.V[i]
            du = V_i * dt
            self.U[i] += du
            self.rigidParams[i, 0] += du
            omega = self.RotV[i]
            drot = omega * dt
            self.quat[i] += drot

            if self.quat[i][0] > ti.math.pi:
                self.quat[i][0] -= 2 * ti.math.pi
            elif self.quat[i][0] < -ti.math.pi:
                self.quat[i][0] += 2 * ti.math.pi

        # ── Phase 2: Clear accumulated impulses for next substep ──
        n_active = self.numRigids + self.numAnalytical
        for i in range(n_active):
            self.accumulated_impulse[i] = ti.Vector.zero(ti.f32, self.d)
            self.accumulated_rotational_impulse[i] = ti.Vector.zero(ti.f32, 1)

        # ── Phase 3: Update bounding boxes (conditional) ──
        if update_bbox == 1:
            # Primitive rigid AABBs
            for i in range(self.numRigids):
                self.getPrimitiveRigidBBox(i)

            # Mesh rigid AABBs + coordinate transform
            # Always update mesh coords and AABB — needed by FEM-rigid contact
            # even when rigid-rigid contact is disabled.
            for i in range(self.numMesh):
                rigidIndice = self.mesh2RigidIndices[i]
                pool_id = self.instance_pool_id[rigidIndice]
                pool_node_offset = 0
                num_nodes = 0
                if pool_id >= 0:
                    pool_node_offset = self.pool_node_offset[pool_id]
                    num_nodes = self.pool_node_count[pool_id]
                else:
                    pool_node_offset = self.meshBoundaryNodeOffset[i]
                    num_nodes = self.meshBoundaryNodeCount[i]
                legacy_node_offset = self.meshBoundaryNodeOffset[i]
                center = self.rigidParams[rigidIndice, 0]
                scale = self.meshRigidScale[rigidIndice]
                offset = self.meshRigidOffset[rigidIndice]
                lb = ti.Vector([1e30 for _ in range(self.d)])
                ub = ti.Vector([-1e30 for _ in range(self.d)])
                for nid in range(num_nodes):
                    lr = self.pool_boundary_lrs[pool_node_offset + nid]
                    scaled = ti.Vector([lr[k] * scale[k] for k in ti.static(range(self.d))])
                    coord = center + offset + self.cached_rotation_matrix[rigidIndice] @ scaled
                    self.meshBoundaryCoords[legacy_node_offset + nid] = coord
                    lb = ti.min(coord, lb)
                    ub = ti.max(coord, ub)
                domain_idx = self.rigidDomainIds[rigidIndice][0]
                self.aabb[domain_idx, 0] = lb
                self.aabb[domain_idx, 1] = ub

            # Analytical plane AABBs
            num_analytical = self.numAnalytical if self.movingAnalytical else 0
            buffer = 0.1
            large_span = 100.0
            for i in range(num_analytical):
                idx = i + self.numRigids
                normal_local = self.rigidParams[idx, 1]
                normal_world = self.cached_rotation_matrix[idx] @ normal_local
                p = self.rigidParams[idx, 0]
                lb = ti.Vector.zero(ti.f32, self.d)
                ub = ti.Vector.zero(ti.f32, self.d)
                tangent = ti.Vector([-normal_world[1], normal_world[0]])
                lo_raw = p - tangent * large_span - normal_world * buffer
                hi_raw = p + tangent * large_span + normal_world * buffer
                lb = ti.Vector([min(float(lo_raw[k]), float(hi_raw[k])) for k in range(self.d)])
                ub = ti.Vector([max(float(lo_raw[k]), float(hi_raw[k])) for k in range(self.d)])
                self.aabb[idx, 0] = lb
                self.aabb[idx, 1] = ub

    # ════════════════════════════════════════════════════════════════════

    @ti.kernel
    def _generate_ground_pairs_direct_kernel(self):
        """Generate ground-rigid contact pairs directly without BVH broadphase.

        For parallel RL mode: analytical planes are infinite, so every rigid
        is potentially in contact. Replaces BVH + classify for ground contacts.
        Rigid-rigid pairs are NOT generated (envs are spatially isolated).
        """
        # Reset all pair counters
        self.num_primitive_pairs[None] = 0
        self.num_ball_ball_pairs[None] = 0
        self.num_box_box_pairs[None] = 0
        self.num_box_ball_pairs[None] = 0
        self.num_seg_point_pairs[None] = 0
        self.num_seg_ball_pairs[None] = 0
        self.num_seg_seg_pairs[None] = 0
        self.num_mesh_pairs[None] = 0
        self.num_mixed_pairs[None] = 0
        self.num_groundprim_pairs[None] = 0
        self.num_groundmesh_pairs[None] = 0

        # Pair every rigid with every analytical plane
        for i in range(self.numRigids):
            type_i = self.rigidDomainIds[i][1]
            is_mesh_i = (type_i == RigidType.MESH) and (self.compound_count[i] == 0)
            for j in range(self.numAnalytical):
                anal_idx = self.numRigids + j
                if self._mask_allows_pair(anal_idx, i) == 0:
                    continue
                if is_mesh_i:
                    idx = ti.atomic_add(self.num_groundmesh_pairs[None], 1)
                    if idx < self.MAX_GROUND_PAIRS:
                        self.groundmesh_pairs_buffer[idx] = ti.Vector([anal_idx, i])
                else:
                    idx = ti.atomic_add(self.num_groundprim_pairs[None], 1)
                    if idx < self.MAX_GROUND_PAIRS:
                        self.groundprim_pairs_buffer[idx] = ti.Vector([anal_idx, i])

    @ti.kernel
    def update_joint_motor_velocity_targets_kernel(self, dt: ti.f32):
        """Convert command targets to per-substep velocity targets for motor rows.

        joint_control_target is treated as a command input and is never overwritten here.
        For acceleration command mode, this performs: v_target = v_rel + a_cmd * dt.
        """
        for j in range(self.numAnchors):
            if self.joint_has_motor[j] == 0:
                continue
            if self.joint_motor_target_mode[j] == 2:
                continue

            cmd = self.joint_control_target[j]
            vel_limit = self.joint_params[j][4]
            mode = self.joint_motor_target_mode[j]

            vel_target = self.joint_motor_target_vel[j]
            if mode == 1:
                vel_target += cmd * dt

                if vel_limit > 0.0:
                    vel_target = ti.min(ti.max(vel_target, -vel_limit), vel_limit)
                self.joint_motor_target_vel[j] = vel_target
            else:
                self.joint_motor_target_vel[j] = cmd  # direct velocity target mode

    @ti.kernel
    def apply_joint_pd_velocity_kernel(self, dt: ti.f32):
        """PD control with per-joint stiffness/damping — VELOCITY-MOTOR output.

        Computes a velocity target from the PD position error and writes it
        to the anchor's RotV so the constraint-solver velocity motor enforces
        it implicitly.  This avoids the instability caused by coupling
        explicit accumulated_rotational_impulse torques with the iterative PGS position constraint.

        Velocity target:  ω_target = kp · (θ_target − θ)
        The kd gain is used as optional pre-damping: the effective velocity
        written is  ω_target − (kd/kp) · ω_rel  when kp > 0.
        """
        for j in range(self.numAnchors):
            rigid_a = self.joint_id_a[j]
            rigid_b = self.joint_id_b[j]
            axis_local = self.joint_axis[j]
            jointType = self.joint_type[j]

            kp = self.kpd_field[j][0]
            kd = self.kpd_field[j][1]
            lower_limit = self.joint_params[j][2]  # joint lower limit
            upper_limit = self.joint_params[j][3]  # joint upper limit
            vel_limit = self.joint_params[j][4]  # velocity limit

            if jointType == JointType.Revolute:  # JointType.Revolute
                # Keep motor_mode = 0 so the velocity motor stays active
                # and enforces the target we compute here.
                self.joint_motor_target_mode[j] = 0  # set to velocity motor mode
                # Ensure anchor BC is ROTVTYPE so _calculate_bc_for_index
                # applies the PD output as a velocity (not acceleration).
                qa = self.quat[rigid_a]
                q0a = self.quat_initial[rigid_a]
                qb = self.quat[rigid_b]
                q0b = self.quat_initial[rigid_b]
                wa = self.RotV[rigid_a]
                wb = self.RotV[rigid_b]

                angle = ((qb - q0b) - (qa - q0a))[0]
                target_pos = self.joint_control_target[j]  # j = local anchor index

                # Clamp target to joint limits to prevent motor from fighting with limit constraint
                if lower_limit < upper_limit:
                    target_pos = ti.min(ti.max(target_pos, lower_limit), upper_limit)

                # Wrap error to [-pi, pi] to avoid discontinuity at ±pi
                pos_err = ti.atan2(ti.sin(target_pos - angle), ti.cos(target_pos - angle))
                # Optional damping pre-correction
                w_rel = (wb - wa)[0]
                ctrl_dt = self.control_dt
                vel_target = kp / ctrl_dt * pos_err - kd * w_rel
                # Control-period safety cap: avoid overshooting the target in one control step.
                # Uses control_dt (frame_dt or policy_dt), NOT substep_dt.
                # max_step_gain < 1 keeps some margin for solver/contact coupling.
                max_step_gain = 0.5
                ctrl_dt = self.control_dt
                if ctrl_dt > 0.0:
                    max_vel_from_error = max_step_gain * ti.abs(pos_err) / ctrl_dt
                    vel_target = ti.min(ti.max(vel_target, -max_vel_from_error), max_vel_from_error)
                # Clamp by velocity limit
                if vel_limit > 0.0:
                    vel_target = ti.min(ti.max(vel_target, -vel_limit), vel_limit)
                # Feed velocity motor target without mutating command input
                self.joint_motor_target_vel[j] = vel_target
            elif jointType == JointType.Prismatic:
                target_pos = self.joint_control_target[j]
                posA = self.rigidParams[rigid_a, 0]
                posB = self.rigidParams[rigid_b, 0]
                axis_world = ti.Vector.zero(ti.f32, self.d)
                axis_world = axis_local
                
                rel_pos = (posB - posA).dot(axis_world)
                pos_err = target_pos - rel_pos

                vel_err = (self.V[rigid_b] - self.V[rigid_a]).dot(axis_world)
                vel_target = kp * pos_err - kd * vel_err

                max_step_gain = 0.5
                ctrl_dt = self.control_dt
                if ctrl_dt > 0.0:
                    max_vel_from_error = max_step_gain * ti.abs(pos_err) / ctrl_dt
                    vel_target = ti.min(ti.max(vel_target, -max_vel_from_error), max_vel_from_error)

                if vel_limit > 0.0:
                    vel_target = ti.min(ti.max(vel_target, -vel_limit), vel_limit)

                self.joint_motor_target_vel[j] = vel_target

    @ti.kernel
    def apply_joint_pd_torque_kernel(self, dt: ti.f32):
        """PD control with per-joint stiffness/damping — TORQUE output.

        Computes τ = kp * (target − angle) − kd * ω_rel

        The physics engine's ``apply_motor_torques_kernel`` will convert
        to world frame and apply ±τ on the connected bodies as accumulated_rotational_impulse
        (external torque), identical to how MuJoCo / Isaac Lab operate.
        """

        for j in range(self.numAnchors):
            rigid_a = self.joint_id_a[j]
            rigid_b = self.joint_id_b[j]
            axis_local = self.joint_axis[j]
            jointType = self.joint_type[j]

            kp = self.kpd_field[j][0]
            kd = self.kpd_field[j][1]
            vel_limit = self.joint_params[j][4]  # velocity limit
            effort_lim = self.joint_params[j][5]  # effort limit stored in joint params row 5
            target_pos = self.joint_control_target[j]

            if jointType == 1:  # JointType.Revolute
                self.joint_motor_target_mode[j] = 2  # set to torque mode
                qa = self.quat[rigid_a]
                q0a = self.quat_initial[rigid_a]
                qb = self.quat[rigid_b]
                q0b = self.quat_initial[rigid_b]
                wa = self.RotV[rigid_a]
                wb = self.RotV[rigid_b]

                angle = ((qb - q0b) - (qa - q0a))[0]
                w_rel = (wb - wa)[0]
                # Wrap error to [-pi, pi] to avoid discontinuity at ±pi
                pos_err = ti.atan2(ti.sin(target_pos - angle), ti.cos(target_pos - angle))
                print(
                    "Check angle and w_rel for joint", j, " angle:", angle, " w_rel:", w_rel, " pos_err:", pos_err
                )
                torque_mag = kp * pos_err - kd * w_rel

                # Small deadband near target to suppress chatter.
                if ti.abs(pos_err) < 1e-3 and ti.abs(w_rel) < 1e-3:
                    torque_mag = 0.0

                # Soft effort saturation reduces bang-bang behavior at the limit.
                if effort_lim > 0.0:
                    torque_mag = effort_lim * ti.tanh(torque_mag / (effort_lim + 1e-6))

                # First apply effort saturation.
                if effort_lim > 0.0:
                    torque_mag = ti.min(ti.max(torque_mag, -effort_lim), effort_lim)

                # Predictive velocity limit: clamp torque so next-step relative
                # angular velocity cannot exceed +/- vel_limit.
                inertia_a = self.inertia[rigid_a]
                inertia_b = self.inertia[rigid_b]
                alpha_per_tau = 1.0 / (inertia_a + 1e-6) + 1.0 / (inertia_b + 1e-6)
                if vel_limit > 0.0 and dt > 0.0 and alpha_per_tau > 0.0:
                    tau_min = (-vel_limit - w_rel) / (dt * alpha_per_tau)
                    tau_max = (vel_limit - w_rel) / (dt * alpha_per_tau)
                    torque_mag = ti.min(ti.max(torque_mag, tau_min), tau_max)

                # Near target, if current speed cannot stop in time under effort
                # limit, force braking torque to avoid limit-cycle oscillation.
                if effort_lim > 0.0 and dt > 0.0 and alpha_per_tau > 0.0:
                    max_acc = effort_lim * alpha_per_tau
                    stopping_err = 0.5 * w_rel * w_rel / (max_acc + 1e-6)
                    if ti.abs(pos_err) < stopping_err and pos_err * w_rel > 0.0:
                        brake_tau = -w_rel / (dt * (alpha_per_tau + 1e-9))
                        torque_mag = ti.min(ti.max(brake_tau, -effort_lim), effort_lim)

                # 2D: torque is scalar, stored in [0]
                self.accumulated_rotational_impulse[rigid_a][0] -= torque_mag * dt
                self.accumulated_rotational_impulse[rigid_b][0] += torque_mag * dt
           


    @ti.func
    def _calculate_bc_for_index(self, idx, dt):
        """Accumulate boundary-condition forces/torques into accumulated_impulse
        and accumulated_rotational_impulse.

        NOTE: Also handles ROTVTYPE here so the motor target is available
        during PGS solve (before _updateU_and_BBox_kernel).
        """
        bc_type = self.bcNodes[idx]

        # Linear: accumulate into accumulated_impulse
        if (bc_type & ATYPE) != 0:
            self.accumulated_impulse[idx] = 0.0
        else:
            if (bc_type & GRAVITY) != 0:
                # Prescribed acceleration -> equivalent force (overwrite, ignoring prior accumulations)
                self.accumulated_impulse[idx] += self.mass[idx] * self.bcGValues[idx] * dt

            if (bc_type & FORCETYPE) != 0:
                self.accumulated_impulse[idx] += self.bcTValues[idx] * dt

        # Angular: accumulate into accumulated_rotational_impulse
        if (bc_type & ROTATYPE) != 0:
            self.accumulated_rotational_impulse[idx] = 0.0
        elif (bc_type & TORQUETYPE) != 0:
            self.accumulated_rotational_impulse[idx] += self.bcRValues[idx] * dt

    @ti.func
    def _update_bc_for_index(self, i):
        """Apply boundary conditions for rigid index i. Separated into ti.func for kernel reuse."""
        bc_type = self.bcNodes[i]

        if (bc_type & VTYPE) != 0:
            self.V[i] = self.bcTValues[i]
        elif (bc_type & UTYPE) != 0:
            self.V[i] = ti.Vector.zero(ti.f32, self.d)
        elif (bc_type & RTYPE) != 0:
            self.V[i] = ti.Vector.zero(ti.f32, self.d)
            self.RotV[i] = 0.0
        if (bc_type & ROTVTYPE) != 0:
            self.RotV[i] = self.bcRValues[i]

    @ti.func
    def get_box_vertex(self, rigidIdx: ti.i32, v_idx: ti.i32):
        center = self.rigidParams[rigidIdx, 0]
        extent = self.rigidParams[rigidIdx, 1]

        offset = ti.Vector.zero(ti.f32, self.d)

        # 0: - -, 1: + -, 2: + +, 3: - +
        sx = -1.0 if (v_idx == 0 or v_idx == 3) else 1.0
        sy = -1.0 if (v_idx == 0 or v_idx == 1) else 1.0

        local_pos = 0.5 * ti.Vector([sx * extent[0], sy * extent[1]])
        rotMat = self.cached_rotation_matrix[rigidIdx]
        offset = rotMat @ local_pos
       

        return center + offset

    @ti.func
    def getPrimitiveRigidBBox(self, rigidId: ti.i32):
        """Compute and store AABB for a single primitive rigid based on packed params."""
        rigid_type = self.rigidDomainIds[rigidId][1]
        # Mesh rigids have their AABB computed in the dedicated mesh loop;
        # skip here to avoid overwriting with zeros (no primitive geometry).
        if rigid_type != RigidType.MESH:
            center = self.rigidParams[rigidId, 0]
            rotMat = self.cached_rotation_matrix[rigidId]

            lb = ti.Vector.zero(ti.f32, self.d)
            ub = ti.Vector.zero(ti.f32, self.d)

            # ── Compound sub-colliders ──────────────────────────────────
            n_sub = self.compound_count[rigidId]
            if n_sub > 0:
                base = self.compound_offset[rigidId]
                # Initialize lb/ub from first sub-collider
                wp0 = rotMat @ self.compound_local_pos[base] + center
                r0 = self.compound_radius[base]
                for dim in ti.static(range(self.d)):
                    lb[dim] = wp0[dim] - r0
                    ub[dim] = wp0[dim] + r0
                for k in range(1, n_sub):
                    wp = rotMat @ self.compound_local_pos[base + k] + center
                    rk = self.compound_radius[base + k]
                    for dim in ti.static(range(self.d)):
                        lb[dim] = ti.min(lb[dim], wp[dim] - rk)
                        ub[dim] = ti.max(ub[dim], wp[dim] + rk)
            else:
                # ── Single shape ────────────────────────────────────────
                type = self.rigidDomainIds[rigidId][1]
                primary = self.rigidParams[rigidId, 1]
                radius = self.radius[rigidId]
                if type == RigidType.BALL:
                    lb, ub = getBallBBox(center, radius, rotMat)
                elif type == RigidType.BOX:
                    info = getBoxBBox(center, primary, rotMat)
                    lb = info[0]
                    ub = info[1]
                elif type == RigidType.CAPSULE:
                    lcdir = primary
                    lb, ub = getCapsuleBBox(center, lcdir, radius, rotMat)

            # Get global domain index and write to global AABB
            domain_idx = self.rigidDomainIds[rigidId][0]
            self.aabb[domain_idx, 0] = lb  # Lower bound
            self.aabb[domain_idx, 1] = ub  # Upper bound

    @ti.kernel
    def updateMeshCoords(self):
        """Transform mesh boundary vertices to world space (no AABB).

        Reads rest-pose vertices from the instancing pool, applies
        scale + cached_rotation_matrix + translation, and writes the
        result to ``meshBoundaryCoords``.  This is the lightweight
        alternative to ``updateBBox`` when only the transformed vertex
        positions are needed.
        """
        for i in range(self.numMesh):
            rigidIndice = self.mesh2RigidIndices[i]
            pool_id = self.instance_pool_id[rigidIndice]
            pool_node_offset = 0
            num_nodes = 0
            if pool_id >= 0:
                pool_node_offset = self.pool_node_offset[pool_id]
                num_nodes = self.pool_node_count[pool_id]
            else:
                pool_node_offset = self.meshBoundaryNodeOffset[i]
                num_nodes = self.meshBoundaryNodeCount[i]

            legacy_node_offset = self.meshBoundaryNodeOffset[i]
            center = self.rigidParams[rigidIndice, 0]
            scale = self.meshRigidScale[rigidIndice]
            offset = self.meshRigidOffset[rigidIndice]

            for nid in range(num_nodes):
                lr = self.pool_boundary_lrs[pool_node_offset + nid]
                scaled = ti.Vector([lr[k] * scale[k] for k in ti.static(range(self.d))])
                coord = center + offset + self.cached_rotation_matrix[rigidIndice] @ scaled
                self.meshBoundaryCoords[legacy_node_offset + nid] = coord

    @ti.kernel
    def updateBBox(self):
        """Recompute all primitive and mesh rigid bounding boxes (kernel)."""
        for i in range(self.numRigids):
            considercontact = self.rigidDomainIds[i][2]
            self.getPrimitiveRigidBBox(i)

        for i in range(self.numMesh):
            rigidIndice = self.mesh2RigidIndices[i]

            # Get pool geometry ID to read lrs directly from pool (memory efficient!)
            pool_id = self.instance_pool_id[rigidIndice]
            pool_node_offset = 0
            num_nodes = 0
            if pool_id >= 0:
                # Read from shared pool geometry
                pool_node_offset = self.pool_node_offset[pool_id]
                num_nodes = self.pool_node_count[pool_id]
            else:
                # Fallback to legacy (shouldn't happen with current implementation)
                pool_node_offset = self.meshBoundaryNodeOffset[i]
                num_nodes = self.meshBoundaryNodeCount[i]

            # Get legacy offset for writing transformed coords (per-instance)
            legacy_node_offset = self.meshBoundaryNodeOffset[i]

            # Get current transform (center, rotation, scale, offset)
            center = self.rigidParams[rigidIndice, 0]
            scale = self.meshRigidScale[rigidIndice]
            offset = self.meshRigidOffset[rigidIndice]

            # Initialize bbox with first transformed node
            lb = ti.Vector([1e30 for _ in range(self.d)])
            ub = ti.Vector([-1e30 for _ in range(self.d)])

            # Transform each boundary node and update bbox
            # Also update meshBoundaryCoords for contact detection
            for nid in range(num_nodes):
                # Read local position directly from pool (memory efficient - shared across all instances!)
                lr = self.pool_boundary_lrs[pool_node_offset + nid]

                # Apply scale, rotation, and translation (in that order)
                scaled = ti.Vector([lr[i] * scale[i] for i in ti.static(range(self.d))])
                coord = center + offset + self.cached_rotation_matrix[rigidIndice] @ scaled

                # CRITICAL: Update meshBoundaryCoords for contact detection (per-instance)
                self.meshBoundaryCoords[legacy_node_offset + nid] = coord

                lb = ti.min(coord, lb)
                ub = ti.max(coord, ub)

            # Get global domain index and write to global AABB
            domain_idx = self.rigidDomainIds[rigidIndice][0]
            self.aabb[domain_idx, 0] = lb  # Lower bound
            self.aabb[domain_idx, 1] = ub  # Upper bound

        num_analytical = self.numAnalytical if self.movingAnalytical else 0

        buffer = 0.1
        large_span = 100.0
        for i in range(num_analytical):
            idx = i + self.numRigids
            normal_local = self.rigidParams[idx, 1]
            normal_world = self.cached_rotation_matrix[idx] @ normal_local

            # update bbox
            p = self.rigidParams[idx, 0]  # point on plane
            lb = ti.Vector.zero(ti.f32, self.d)
            ub = ti.Vector.zero(ti.f32, self.d)
            tangent = ti.Vector([-normal_world[1], normal_world[0]])
            # single tangent in 2D (perpendicular to normal)
            lo_raw = p - tangent * large_span - normal_world * buffer
            hi_raw = p + tangent * large_span + normal_world * buffer
            # Ensure component-wise min/max so lo <= hi even if tangent points negative
            lb = ti.Vector([min(float(lo_raw[i]), float(hi_raw[i])) for i in range(self.d)])
            ub = ti.Vector([max(float(lo_raw[i]), float(hi_raw[i])) for i in range(self.d)])
         
            self.aabb[idx, 0] = lb
            self.aabb[idx, 1] = ub

    # ===============================================================================
    # Some util functions
    # ===============================================================================

    def get_max_linear_velocity(self):
        """Return max boundary-point linear speed across all rigids.

        Uses an upper bound per rigid:
            |v_boundary|max <= |v_center| + |omega| * r_max
        where r_max is the farthest boundary distance from rigid center.
        """
        n = self.numRigids
        if n == 0:
            return 0.0

        V_np = self.V.to_numpy()[:n]
        W_np = self.RotV.to_numpy()[:n]

        rigid_ids = self.rigidDomainIds.to_numpy()[:n]
        rigid_type = rigid_ids[:, 1]
        rigid_params_np = self.rigidParams.to_numpy()[:n]
        centers = rigid_params_np[:, 0, :]
        primary = rigid_params_np[:, 1, :]
        radius_np = self.radius.to_numpy()[:n]

        boundary_radius = np.zeros(n, dtype=np.float32)
        center_speed = np.linalg.norm(V_np, axis=1)
        omega_speed = np.linalg.norm(W_np, axis=1)

        # Primitive shapes
        ball_mask = rigid_type == int(RigidType.BALL)
        box_mask = rigid_type == int(RigidType.BOX)
        cyl_cap_mask = rigid_type == int(RigidType.CAPSULE)
        boundary_radius[ball_mask] = radius_np[ball_mask]
        boundary_radius[box_mask] = 0.5 * np.linalg.norm(primary[box_mask], axis=1)
        boundary_radius[cyl_cap_mask] = np.linalg.norm(primary[cyl_cap_mask], axis=1) + radius_np[cyl_cap_mask]

        # Compound sub-colliders: override with sub-collider envelope.
        compound_count = self.compound_count.to_numpy()[:n]
        compound_offset = self.compound_offset.to_numpy()[:n]
        if np.any(compound_count > 0):
            sub_pos = self.compound_local_pos.to_numpy()
            sub_radius = self.compound_radius.to_numpy()
            for rid in np.where(compound_count > 0)[0]:
                off = int(compound_offset[rid])
                cnt = int(compound_count[rid])
                if cnt <= 0:
                    continue
                p = sub_pos[off : off + cnt]
                r = sub_radius[off : off + cnt]
                boundary_radius[rid] = float(np.max(np.linalg.norm(p, axis=1) + r))

        # Mesh rigids: exact farthest transformed boundary node from center.
        mesh_mask = rigid_type == int(RigidType.MESH)
        if np.any(mesh_mask):
            rigid2mesh = self.rigid2MeshIndices.to_numpy()[:n]
            mesh_node_off = self.meshBoundaryNodeOffset.to_numpy()
            mesh_node_cnt = self.meshBoundaryNodeCount.to_numpy()
            mesh_coords = self.meshBoundaryCoords.to_numpy()
            for rid in np.where(mesh_mask)[0]:
                mid = int(rigid2mesh[rid])
                if mid < 0:
                    continue
                off = int(mesh_node_off[mid])
                cnt = int(mesh_node_cnt[mid])
                if cnt <= 0:
                    continue
                pts = mesh_coords[off : off + cnt]
                c = centers[rid]
                boundary_radius[rid] = float(np.max(np.linalg.norm(pts - c, axis=1)))

        boundary_speed_upper = center_speed + omega_speed * boundary_radius
        return float(boundary_speed_upper.max())

    def get_sh_update_interval(self):
        return self._sh_mesh_rebuild_interval

    def drawAll(self, gui, domains, colors=None, resolution=10):
        """Batch draw all rigids efficiently by caching to_numpy() calls.

        Args:
            gui: Taichi GUI object
            domains: List of domain objects
            colors: List of colors (one per domain), or None for default colors
            resolution: Resolution parameter for drawing
        """
        # Cache all numpy arrays once
        if self.numMesh > 0:
            all_boundary_coords = self.meshBoundaryCoords.to_numpy()
            all_boundary_elements = self.meshBoundaryElements.to_numpy()

        all_rigid_params = self.rigidParams.to_numpy()
        # all_shape_coords removed

        # Draw each domain using cached data
        for i, domain in enumerate(domains):
            if domain.type != DomainType.RIGID:
                continue

            color = colors[i] if colors is not None and i < len(colors) else 0xFFFFFF
            ndOffset = domain.ndOffset
            rtype = int(self.rigidDomainIds[ndOffset][1])
            rotMat = self.cached_rotation_matrix[ndOffset].to_numpy()

            if rtype == RigidType.MESH:
                # Draw mesh rigid
                mesh_local_id = self.rigid2MeshIndices[ndOffset]
                node_offset = self.meshBoundaryNodeOffset[mesh_local_id]
                num_nodes = self.meshBoundaryNodeCount[mesh_local_id]
                elem_offset = self.meshBoundaryElementOffset[mesh_local_id]
                num_elems = self.meshBoundaryElementCount[mesh_local_id]

                # Slice from cached arrays
                pos = all_boundary_coords[node_offset : node_offset + num_nodes, :2]
                elements = all_boundary_elements[elem_offset : elem_offset + num_elems]

                # 2D: draw edges
                a, b = elements[:, 0], elements[:, 1]
                gui.lines(pos[a], pos[b], radius=2, color=color)
                
            elif rtype == RigidType.BALL:
                # Draw ball
                center = all_rigid_params[ndOffset, 0, :2]
                radius = self.radius[ndOffset]
                gui.circle(center, radius=radius * resolution, color=color)

            elif rtype == RigidType.BOX:
                # Draw box
                center = all_rigid_params[ndOffset, 0]
                extent = all_rigid_params[ndOffset, 1]

                # 2D Box
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
                    gui.line(vertices[j], vertices[(j + 1) % 4], radius=2, color=color)
           

            elif rtype == RigidType.CAPSULE:
                # Draw capsule
                center = all_rigid_params[ndOffset, 0, :2]
                lcdir = all_rigid_params[ndOffset, 1, :]
                lc = (rotMat @ lcdir)[:2] + center
                uc = center * 2 - lc
                radius = self.radius[ndOffset]

                # Draw center line
                gui.line(lc, uc, radius=radius * resolution, color=color)

                if rtype == RigidType.CAPSULE:
                    # Draw end caps
                    gui.circle(lc, radius=radius * resolution, color=color)
                    gui.circle(uc, radius=radius * resolution, color=color)
