from bvh import CollisionDetector
from definitions import *
from femcontact import ContactFlexRigid, contactHelper, ContactSpringRigid
from femspringmanager import FemSpringManager
from mixedcontact import MixedContact
from rigidmanager import RigidManager
import numpy as np
import os
import taichi as ti
import time


@ti.data_oriented
class ExplicitLoop(MixedContact):
    def __init__(
        self,
        dt,
        domains,
        joints=[],
        ties=None,
        damping=0.0,
        useAdapativeDT=True,
        skip_spatial_hash=False,
        considerRigidRigidContact=True,
        use_pd=0,
        mass_scaling_dt=0.0,
        max_contact_velocity=50.0,
    ):
        """
        Args:
            dt: Initial time step (seconds) - may be overridden by adaptive stepping
            domains: List of DomainBase instances (FEMDomain, RigidBodyDomain, etc.)
            joints: List of joint objects (for RigidManager)
            ties: Optional tie specifications. Each item can be:
                - (domain_a, domain_b) as objects
                - (idx_a, idx_b) as domain indices
                - {'domain1': ..., 'domain2': ...} / {'slave': ..., 'master': ...}
            damping: Global damping factor (0.0 = no damping, 1.0 = critical damping)
            useAdapativeDT: If True, use adaptive time stepping based on stability limits
            skip_spatial_hash: If True, skip SpatialHashManager creation in RigidManager (saves VRAM, no broadphase)
            use_pd: 0 = no PD, 1 = velocity PD, 2 = torque PD for joints (only applies to RigidManager)
            mass_scaling_dt: If > 0, apply LS-DYNA-style selective mass scaling to ensure
                all element critical time steps >= this value (seconds).
            max_contact_velocity: Deprecated — ignored.  Spatial hashes are
                rebuilt lazily each substep / every 1/600 s with no velocity buffer.
        """

        self.domains = domains
        self._consider_rigid_rigid_contact = considerRigidRigidContact
        self.nnodes = 0
        self.d = domains[0].d
        self.joints = joints
        self.ties = list(ties) if ties is not None else []
        self._tie_pairs = set()
        self._refresh_tie_pairs()
        self.use_pd = use_pd
        self.skip_spatial_hash = skip_spatial_hash
        self.max_speed = 10.0

        # Create shared BVH for entire simulation (covers all domains)
        # Per-env sub-trees: each env of K objects uses up to 2K-1 nodes.
        tree_size = max(len(domains) * 4 + 1024, 2048)
        self.bvh = None

        # Initialize RigidManager for rigid/analytical domains, pass shared BVH
        self.rigidManager = None
        self.femSpringManager = None

        self.numDomains = len(domains)
        self._max_domains = self.numDomains + 10
        # Consolidated AABB field: aabb[i][0] = lower bound, aabb[i][1] = upper bound
        # Better memory locality when accessing domain bounding boxes
        # Indexed by global domain index
        self.aabb = ti.Vector.field(self.d, ti.f32, (self._max_domains, 2))
        self._bvh_pairs_work = ti.Vector.field(2, ti.i32, shape=tree_size)
        self._bvh_pairs_work_count = ti.field(ti.i32, shape=())
        self._bvh_pairs_work_count[None] = 0
        self._bvh_domain_stamp = ti.field(ti.i32, self._max_domains)
        self._bvh_active_stamp = ti.field(ti.i32, shape=())
        self._bvh_active_stamp[None] = 1
        self._use_bvh_domain_mask = ti.field(ti.i32, shape=())
        self._use_bvh_domain_mask[None] = 0
        self._has_fem_spring_bvh_pair = ti.field(ti.i32, shape=())
        self._has_fem_spring_bvh_pair[None] = 1
        self._domain_type = ti.field(ti.i32, self._max_domains)
        for i in range(self.numDomains):
            self._domain_type[i] = int(self.domains[i].type)

        self.stableTime = 1.0
        rigid_domains = [d for d in domains if d.type == DomainType.RIGID or d.type == DomainType.ANALYTICAL]
        self.fem_domains = [d for d in domains if d.type == DomainType.FEM]
        self.spring_domains = [d for d in domains if d.type == DomainType.SPRINGMASS]

        if len(rigid_domains) > 0:
            self.rigidManager = RigidManager(
                self.d,
                domains,
                joints,
                bvh=self.bvh,
                skip_spatial_hash=skip_spatial_hash,
                considerRigidRigidContact=considerRigidRigidContact,
                use_pd=self.use_pd,
            )
            self.stableTime = min(self.rigidManager.stableTime, self.stableTime)
            self.rigidManager.setGlobalAABB(self.aabb)
            self.rigid_update_times = 0

        if len(self.fem_domains) + len(self.spring_domains) > 0:
            self.femSpringManager = FemSpringManager(self.d, self.domains)
            self.stableTime = min(self.femSpringManager.stableTime, self.stableTime)
            self.femSpringManager.setGlobalAABB(self.aabb)
            if mass_scaling_dt > 0.0:
                self.femSpringManager.applyMassScaling(mass_scaling_dt)
                self.stableTime = max(self.stableTime, mass_scaling_dt)

        self.useAdapative = useAdapativeDT
        self.damping = damping
        self.contacts = []
        self._counter_py = 0  # Python-side counter (avoids [None] sync per substep)
     
        # Contact subcycling: auto-compute from stable-time ratio after contacts are built.
        self._rigid_contact_subcycle = 1
        self._domain_min_stable_time = self.stableTime
        self._rigid_contact_min_stable_time = self.stableTime
        self._rigid_contact_subcycle_max = 8

        self.numContacts = 0
        MAX_CONTACTS = max(128, len(domains) * 2)  # Heuristic max contacts based on domain count
        self.contactPairIds = ti.Vector.field(2, ti.i32, MAX_CONTACTS)  # Here the max number of contact is MAX_CONTACTS

        # Count objects that will be added to BVH
        num_contact_domains = sum(1 for d in domains if d.considerContact or d.considerGroundContact)
        max_bvh_nodes = tree_size
        if num_contact_domains > max_bvh_nodes:
            print(
                f"\033[91mError: Too many contact domains ({num_contact_domains}) for BVH! Maximum is {max_bvh_nodes}.\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum BVH nodes ({max_bvh_nodes})")

        print("Preparing FEM/Rigid contact detection...")

        # Pre-compute AABBs for rigid/analytical domains in one batch kernel
        # (avoids 60,000+ individual Python→Taichi reads via getBBox)
        if self.rigidManager is not None:
            self.rigidManager.precompute_rigid_transforms()
            self.rigidManager.updateBBox()

        # Now collect BVH candidates — getBBox just reads from the already-filled aabb field
        bvh_candidates = []  # (domain_index, env_id)
        for i in range(len(domains)):
            if self.domains[i].considerContact:
                # For rigid domains, aabb is already populated by updateBBox above
                # For FEM/spring domains, we still need to call getBBox individually
                if self.domains[i].type != DomainType.RIGID:
                    lb, ub = self.domains[i].getBBox()
                    self.aabb[i, 0] = lb
                    self.aabb[i, 1] = ub
                domain_env_id = getattr(self.domains[i], "env_id", -1)
                bvh_candidates.append((i, domain_env_id))

        ids_np = np.array([c[0] for c in bvh_candidates], dtype=np.int32)
        env_ids_np = np.array([c[1] for c in bvh_candidates], dtype=np.int32)
        self._single_contact_object = len(bvh_candidates) <= 1

        # Batch-add BVH objects only when broadphase can produce pair candidates.
        if len(bvh_candidates) > 1:
            self.bvh = CollisionDetector(self.d, tree_size)
            self.bvh.addObjects_batch(self.aabb, ids_np, env_ids_np)
        elif self._single_contact_object:
            print("Single contact object scene -- skipping BVH object setup/build")

        # Determine if we need the O(N²) FEM/Spring contact creation loop.
        # For pure rigid/analytical scenes (the common RL training case),
        # all pairs are RIGIDRIGID or RIGIDANALYTICAL and are handled by
        # RigidManager — so we can skip the expensive inner loop entirely.
        has_fem_or_spring = any(
            d.type == DomainType.FEM or d.type == DomainType.SPRINGMASS for d in domains if d.considerContact
        )

        if has_fem_or_spring:
            # Only run the O(N²) loop when there are FEM/Spring domains
            # that genuinely need contactHelper pairs
            contact_domains = [(i, domains[i]) for i in range(len(domains)) if domains[i].considerContact]
            print(f"Building FEM/Spring contacts among {len(contact_domains)} domains...")
            for idx_a, (i, dom_i) in enumerate(contact_domains):
                for idx_b in range(idx_a):
                    j, dom_j = contact_domains[idx_b]
                    ctype = dom_i.type | dom_j.type

                    if dom_i.category_bits & dom_j.collide_bits == 0:
                        continue

                    if dom_i.collide_bits & dom_j.category_bits == 0:
                        continue

                    # Skip pure rigid/analytical pairs (handled by RigidManager)
                    if (
                        ctype == ContactType.RIGIDRIGID
                        or ctype == ContactType.RIGIDANLAYTICAL
                        or ctype == ContactType.ANALYTICALANALYTICAL
                    ):
                        continue

                    if hasattr(dom_i, "env_id") and hasattr(dom_j, "env_id") and dom_i.env_id != dom_j.env_id:
                        print(
                            f"Warning: Skipping contact between domain {i} (env {dom_i.env_id}) and domain {j} (env {dom_j.env_id}) due to different env_ids"
                        )
                        continue
                    print(
                        "Creating contact between domain", i, "and domain", j, "of types", dom_i.type, "and", dom_j.type
                    )
                    contact = contactHelper(dom_i, dom_j, ctype)
                    if contact is not None:
                        if self._is_tie_pair(i, j):
                            print("  This pair is tied -- marking contact as tied")
                            print("Domain i:", dom_i, "name:", getattr(dom_i, "name", "N/A"))
                            print("Domain j:", dom_j, "name:", getattr(dom_j, "name", "N/A"))
                            contact.tied = True
                        self.contacts.append(contact)
                        self.contactPairIds[self.numContacts] = ti.Vector([i, j])
                        self.numContacts += 1
        else:
            print(f"Pure rigid/analytical scene -- skipping O(N^2) contact pair search ({len(bvh_candidates)} domains)")

        if self.bvh is not None:
            print("Starting BVH build with", self.bvh.object_count[None], "objects for contact detection...")
        else:
            print("BVH disabled (<=1 contact candidate) -- skipping BVH build")
        # Detect multi-env: if any domain has env_id >= 0, use per-env BVH
        self._bvh_env_groups = None  # kept for rebuild
        has_multi_env = any(eid >= 0 for eid in env_ids_np) if len(bvh_candidates) > 1 else False
        if self.bvh is not None and has_multi_env and self.bvh.object_count[None] > 1:
            # Group BVH object indices by env_id
            from collections import defaultdict

            env_groups = defaultdict(list)
            for bvh_obj_idx, (_, eid) in enumerate(bvh_candidates):
                if eid >= 0:
                    env_groups[eid].append(bvh_obj_idx)
                # env_id=-1 (ground/analytical) excluded from per-env BVH;
                # their contacts are handled by the batched contact kernels.
            self._bvh_env_groups = dict(env_groups)
            print(
                f"Per-env BVH: {len(env_groups)} envs, "
                f"skipping {sum(1 for _,e in bvh_candidates if e<0)} ground domains"
            )
            self.bvh.build_per_env(self._bvh_env_groups)
        elif self.bvh is not None and self.bvh.object_count[None] > 1:
            print("Building BVH for contact detection with", self.bvh.object_count[None], "objects...")
            self.bvh.build()

        # Stable time before contact stiffness constraints (domain-only baseline)
        self._domain_min_stable_time = self.stableTime

        # Warm up of these contacts
        print("Number of Mixed (fem-fem / fem-rigid) contacts here:", self.numContacts)
        rigid_contact_min_stable = 1e30
        for i in range(self.numContacts):
            self.contacts[i].calculate(0.0)
            self.stableTime = min(self.stableTime, self.contacts[i].stableTime)
            c = self.contacts[i]
            if isinstance(c, (ContactFlexRigid, ContactSpringRigid)):
                rigid_contact_min_stable = min(rigid_contact_min_stable, float(c.stableTime))

        if rigid_contact_min_stable < 1e29:
            self._rigid_contact_min_stable_time = rigid_contact_min_stable
        else:
            self._rigid_contact_min_stable_time = self._domain_min_stable_time

        # Auto subcycle count for Phase 2 (full SH query for new contacts).
        # Phase 1 (cached element narrow phase) still runs every substep,
        # maintaining correct forces for existing contacts.  Phase 2 only
        # finds NEW contacts, so deferring it by a few substeps is safe:
        # max delay = subcycle × dt which is tiny compared to contact margin.
        ratio = self._rigid_contact_min_stable_time / max(self._domain_min_stable_time, 1e-12)
        self._rigid_contact_subcycle = int(max(2, min(self._rigid_contact_subcycle_max, ratio)))

        if useAdapativeDT:
            self.dt = self.stableTime
            print("Using adaptive time stepping with initial dt =", self.dt)
        else:
            self.dt = dt
        self.default_dt = dt

        # ── Batched contact optimization ─────────────────────────────────
        # Replace N separate contact kernel launches with O(1) batched kernels.
        # Build flat work-lists for each contact type once at initialization.
        self._build_batched_contacts()
        # Active work-item compaction for rigid mesh contacts helps only when
        # BVH domain-mask can reject many items. Keep it off by default.
        self._use_rigid_mesh_compaction = self._batched_rigid_mesh_count > 0
        self._needs_bvh_domain_mask = self._use_rigid_mesh_compaction or self._batched_flex_count > 0
        # Flag: mesh rigid element AABBs need refresh after rigid body movement.
        # Between rigid substeps the mesh coordinates don't change, so
        # recomputing triangle AABBs every substep is wasted GPU work.
        self._rigid_mesh_aabb_dirty = True

        # Refine subcycle: when contact margin is known, allow higher subcycle
        # for scenarios with many substeps (small dt).  Phase 2 delay of
        # (subcycle-1) × dt must remain well below the contact margin.
        # Max penetration before correction = max_v × (subcycle-1) × dt.
        # Keep penetration < margin / safety.
        if hasattr(self, "_rigid_sh_contact_margin") and self._rigid_sh_contact_margin > 0:
            max_v_estimate = 2.0  # conservative peak relative velocity [m/s]
            safety = 2.0
            margin_steps = int(self._rigid_sh_contact_margin / (max_v_estimate * self._domain_min_stable_time * safety))
            self._rigid_contact_subcycle = max(2, min(self._rigid_contact_subcycle_max, margin_steps))
        print(
            f"[RigidSubcycle] domain_dt={self._domain_min_stable_time:.3e}, "
            f"contact_margin={getattr(self, '_rigid_sh_contact_margin', 0):.3e}, "
            f"subcycle={self._rigid_contact_subcycle}"
        )

        # Flag: contact preprocessing (activate_aabb, prefilter, build_workset)
        # needs refresh.  Between rigid substeps the rigid AABBs / SH are
        # unchanged, so the preprocessing results are nearly identical.
        self._rigid_contact_prefilter_dirty = True

        # Cache BVH activity check to avoid reading [None] every substep
        # Skip BVH broadphase only when NO subsystem needs it:
        #  - Unbatched contacts use BVH for pair activation
        #  - RigidManager uses BVH pairs for rigid-rigid/rigid-analytical detection
        #    (unless skip_bvh is set, in which case it generates ground pairs directly)
        print(
            f"RigidManager: {self.rigidManager.numRigidInContact if self.rigidManager is not None else 0} rigid-in-contact domains"
        )
        print(
            f"Batched contacts: {self._batched_anal_count} analytical, {self._batched_flex_count} FEM, "
            f"{self._batched_rigid_prim_count} rigid-primitive, {self._batched_rigid_mesh_count} rigid-mesh, "
            f"{self._batched_hf_count} heightfield, {self._batched_voxel_count} voxel"
        )
        rigid_needs_bvh = (
            self.rigidManager is not None
            and self.rigidManager.numRigidInContact > 1
            and self._consider_rigid_rigid_contact
        )
        self.skip_bvh = (self.bvh is None) or (
            self._batched_anal_count == 0
            and self._batched_hf_count == 0
            and self._batched_voxel_count == 0
            and self._unbatched_contacts_count == 0
            and not rigid_needs_bvh
        )

        if self.skip_bvh:
            if self.rigidManager is not None:
                self.rigidManager.skip_bvh = True
            self._use_bvh_domain_mask[None] = 0
            self._has_fem_spring_bvh_pair[None] = 0
            print("No BVH broadphase needed for contacts -- skipping BVH in RigidManager")

        # ── Lazy spatial hash rebuild ────────────────────────────────────
        # Spatial hashes are rebuilt lazily during substeps:
        #  - Rigid-rigid mesh: per-pair per-substep (handled by rigidmanager)
        #  - FEM-rigid / FEM-FEM: every interval seconds, or when
        #    max displacement exceeds 0.5 × cell_size.
        # Queries search 3×3×3 cells (±1 cell around the query point).
        # Interval must be much larger than dt so negative-cache skip epochs
        # persist across many substeps.  Displacement-based rebuild catches
        # fast-moving scenarios; time-based is a safety fallback.

        # Initial FEM spatial hash build (so first frame has valid data)
        if self._use_fem_spatial_hash and self.femSpringManager is not None:
            self.femSpringManager.populate_fem_spatial_hash(velocity_buffer=0.0)

        # Initial rigid spatial hash build for FEM-rigid mesh contacts
        if self._use_rigid_spatial_hash and self.rigidManager is not None and self.rigidManager.spatialHash is not None:
            margin = getattr(self, "_rigid_sh_contact_margin", 0.0)
            if hasattr(self, "_fem_contact_aabb_lb") and hasattr(self, "_bc_rigid_mesh_node_count"):
                self._compute_fem_contact_aabb(self._bc_rigid_mesh_node_count[None])
                print("contact margin for spatial hash:", margin)
                self.rigidManager.populate_spatial_hash_filtered(
                    fem_lb=self._fem_contact_aabb_lb,
                    fem_ub=self._fem_contact_aabb_ub,
                )
            else:
                self.rigidManager.populate_spatial_hash_all_meshes()

    def _parse_tie_entry(self, tie):
        if not isinstance(tie, (tuple, list)) or len(tie) != 2:
            return None

        a, b = tie
        if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)):
            return None

        ia = int(a)
        ib = int(b)

        # Accept Python-style negative indexing for tie specifications.
        # Example: (-1, -2) targets the last two domains.
        n_domains = len(self.domains)
        if ia < 0:
            ia += n_domains
        if ib < 0:
            ib += n_domains

        if ia < 0 or ib < 0 or ia >= n_domains or ib >= n_domains or ia == ib:
            return None
        return (min(ia, ib), max(ia, ib))

    def _refresh_tie_pairs(self):
        self._tie_pairs.clear()
        if len(self.ties) == 0:
            return

        unresolved = 0
        for tie in self.ties:
            pair = self._parse_tie_entry(tie)
            if pair is None:
                unresolved += 1
                continue
            self._tie_pairs.add(pair)

        if unresolved > 0:
            print(f"Warning: {unresolved} tie entries could not be resolved and were ignored")

    def _is_tie_pair(self, idx_a, idx_b):
        return (min(idx_a, idx_b), max(idx_a, idx_b)) in self._tie_pairs

    @ti.kernel
    def _scan_bvh_pairs_stamp(self, pairs: ti.template(), num_pairs: ti.i32, stamp: ti.i32):
        self._has_fem_spring_bvh_pair[None] = 0
        for p in range(num_pairs):
            a = pairs[p][0]
            b = pairs[p][1]
            if 0 <= a < self._max_domains:
                self._bvh_domain_stamp[a] = stamp
                ta = self._domain_type[a]
                if ta == int(DomainType.FEM) or ta == int(DomainType.SPRINGMASS):
                    self._has_fem_spring_bvh_pair[None] = 1
            if 0 <= b < self._max_domains:
                self._bvh_domain_stamp[b] = stamp
                tb = self._domain_type[b]
                if tb == int(DomainType.FEM) or tb == int(DomainType.SPRINGMASS):
                    self._has_fem_spring_bvh_pair[None] = 1

    @ti.kernel
    def _filter_out_rigid_rigid_pairs(self, num_pairs: ti.i32):
        """Copy BVH pairs excluding rigid-rigid pairs when disabled by config."""
        self._bvh_pairs_work_count[None] = 0
        rigid_type = ti.static(int(DomainType.RIGID))
        for p in range(num_pairs):
            a = self.bvh.collision_pairs[p][0]
            b = self.bvh.collision_pairs[p][1]
            keep = True
            if 0 <= a < self._max_domains and 0 <= b < self._max_domains:
                ta = self._domain_type[a]
                tb = self._domain_type[b]
                if ta == rigid_type and tb == rigid_type:
                    keep = False
            if keep:
                out_idx = ti.atomic_add(self._bvh_pairs_work_count[None], 1)
                self._bvh_pairs_work[out_idx] = ti.Vector([a, b])

    def advance(self):
        t_advance_start = time.perf_counter()
        num_bvh_pairs = 0
        pairs_field = self._bvh_pairs_work
        num_pairs_work = 0
        if not self.skip_bvh and self.bvh is not None:
            t0 = time.perf_counter()
            # Update BVH: full rebuild every 100 steps, refit otherwise
            if self._counter_py % 100 == 0:
                self.bvh.reset()
                self.bvh.update_objects(self.aabb, 1)
                if self._bvh_env_groups:
                    self.bvh.build_per_env(self._bvh_env_groups)
                else:
                    self.bvh.build()
            else:
                self.bvh.refit(self.aabb)

            if self.bvh._per_env_mode:
                self.bvh.detectInnerCollision_per_env()
            else:
                self.bvh.detectInnerCollision()
            num_bvh_pairs = self.bvh.pair_count[None]
            num_pairs_work = num_bvh_pairs
            pairs_field = self.bvh.collision_pairs
            if (not self._consider_rigid_rigid_contact) and num_bvh_pairs > 0:
                tf = time.perf_counter()
                self._filter_out_rigid_rigid_pairs(num_bvh_pairs)
                num_pairs_work = int(self._bvh_pairs_work_count[None])
                pairs_field = self._bvh_pairs_work

            if self._needs_bvh_domain_mask:
                if num_pairs_work > 0:
                    next_stamp = self._bvh_active_stamp[None] + 1
                    if next_stamp > 2000000000:
                        self._bvh_domain_stamp.fill(0)
                        next_stamp = 1
                    self._bvh_active_stamp[None] = next_stamp
                    self._use_bvh_domain_mask[None] = 1
                    self._scan_bvh_pairs_stamp(pairs_field, num_pairs_work, next_stamp)
                else:
                    self._use_bvh_domain_mask[None] = 1
                    self._has_fem_spring_bvh_pair[None] = 0
            else:
                self._use_bvh_domain_mask[None] = 0
                self._has_fem_spring_bvh_pair[None] = 1 if num_pairs_work > 0 else 0
        else:
            # BVH inactive: GPU fields already set once per frame in
            # advanceWithTime(); skip per-substep writes to avoid
            # cudaMemcpy(H2D) stream synchronization.
            pass

        # Only activate batched contact kernels when BVH reports potential
        # pairs involving at least one FEM/Spring domain.
        # has_fem_spring_bvh_pair = self._has_fem_spring_bvh_pair[None] == 1
        # run_batched_contacts = (not self._bvh_active) or has_fem_spring_bvh_pair

        # ── Batched contact kernels (see mixedcontact.py) ──
        self.run_batched_contacts(pairs_field, num_pairs_work)

        # Calculate FEM domains
        if self.femSpringManager is not None:
            t0 = time.perf_counter()
            self.femSpringManager.substep(self.dt, self.damping)

        # Rigid body substep — pass Taichi field directly (no numpy conversion)
        if self.rigidManager is not None:
            t0 = time.perf_counter()
            N = max(1, self.rigidManager.stableTime // self.dt)
            self.rigid_update_times += 1
            if self.rigid_update_times % N == 0:
                if hasattr(self, "_fem_contact_aabb_lb") and hasattr(self, "_bc_rigid_mesh_node_count"):
                    self._compute_fem_contact_aabb(self._bc_rigid_mesh_node_count[None])
                    self.rigidManager.substep(
                        pairs_field,
                        num_pairs_work,
                        N * self.dt,
                        self.damping,
                        fem_lb=self._fem_contact_aabb_lb,
                        fem_ub=self._fem_contact_aabb_ub,
                        fem_margin=getattr(self, "_rigid_sh_contact_margin", 0.0),
                    )
                else:
                    self.rigidManager.substep(pairs_field, num_pairs_work, N * self.dt, self.damping)

            N2 = max(1, self.rigidManager.get_sh_update_interval() // self.dt)  # sh update interval
            if self.rigid_update_times % N2 == 0:
                self._rigid_contact_prefilter_dirty = True
                self._rigid_mesh_aabb_dirty = True  # mesh coords changed

                # several state variables need to be reset
                if hasattr(self, "_bc_rigid_skip_epoch"):
                    self._bc_rigid_skip_epoch.fill(-1)

        self._counter_py += 1


    # ===================== Fixed-frame stepping helpers =====================
    def advanceWithTime(self, frame_dt: float = 1.0 / 60.0, verbose: bool = False):
        """Advance the simulation by an exact amount of simulated time using
        adaptive substeps based on current stableTime. This ensures a fixed
        frame time (e.g., 1/60s) regardless of stability constraints.

        Args:
            frame_dt: Target simulated time to advance (seconds), default 1/60.
            verbose: If True, print debug information about substeps.
        """

        # Set BVH-related GPU state ONCE per frame (not per substep) to avoid
        # cudaMemcpy(H2D) stream synchronization on every substep.
        if self.skip_bvh or self.bvh is None:
            self._use_bvh_domain_mask[None] = 0
            self._has_fem_spring_bvh_pair[None] = 0

        remaining = float(frame_dt)
        eps = 1e-12
        # Loop with substeps not exceeding current stability estimate
        if verbose:
            print(f"Advancing fixed time step of {frame_dt} seconds")
            print(f"Current stable time step: {self.stableTime:.3e} seconds")

        substep_count = 0
        while remaining > eps:
            # Use the most recent stability limit if adaptive, else fixed dt
            current_stable = self.stableTime if self.useAdapative else self.default_dt
            # Clamp substep to remaining time
            sub_dt = max(min(current_stable, remaining), 1e-8)
            self.dt = sub_dt
            self.advance()
            remaining -= sub_dt
            substep_count += 1

        if verbose:
            print(f"Completed {substep_count} substeps for {frame_dt} seconds of simulation time.")

    def getAllJointAngles(self, include_non_revolute: bool = False):
        """Return current joint angles in radians.

        Uses joint ordering from ``self.rigidManager.joints`` and returns a
        dictionary keyed by joint name (or ``joint_<index>`` when unnamed).

        Args:
            include_non_revolute: If True, include non-revolute joints with
                value ``None``. If False, only revolute joints are returned.

        Returns:
            dict[str, float | None]: Joint angle map in radians.
        """
        manager = self.rigidManager
        if manager is None or not hasattr(manager, "joints"):
            return {}

        angle_map = {}
        for j_idx, joint in enumerate(manager.joints):
            joint_name = getattr(joint, "name", "") or f"joint_{j_idx}"
            is_revolute = getattr(joint, "jointType", None) == JointType.Revolute

            if not is_revolute:
                if include_non_revolute:
                    angle_map[joint_name] = None
                continue

            idx_a = int(manager.joint_id_a[j_idx])
            idx_b = int(manager.joint_id_b[j_idx])

            angle_a = float(manager.quat[idx_a][0] - manager.quat_initial[idx_a][0])
            angle_b = float(manager.quat[idx_b][0] - manager.quat_initial[idx_b][0])
            angle = angle_b - angle_a
            angle = float(np.arctan2(np.sin(angle), np.cos(angle)))
         
            angle_map[joint_name] = float(angle)

        return angle_map

    def reset(self):
        """
        Reset the simulation to initial state.

        This method resets:
        - Time counter
        - All domain states (positions, velocities, accelerations)
        - Rigid body states via RigidManager

        Note: This does NOT reset boundary conditions - those should be
        managed by the caller (e.g., RL environment).
        """
        # Reset time counter
        self._counter_py = 0

        # Reset rigid bodies through RigidManager
        # RigidManager.reset() will re-pack rigid parameters from initial origins
        if self.rigidManager is not None and (
            self.rigidManager.numRigids > 0 or self.rigidManager.numAnalytical > 0
        ):
            self.rigidManager.reset()

        # Reset other domain types
        for domain in self.domains:
            if domain.type == DomainType.FEM or domain.type == DomainType.SPRINGMASS:
                pass

        # Reset BVH and contact states
        if self.bvh is not None and self.bvh.object_count[None] > 1:
            self.bvh.reset()
            self.bvh.update_objects(self.aabb, 1)
            if self._bvh_env_groups:
                self.bvh.build_per_env(self._bvh_env_groups)
            else:
                self.bvh.build()

        # Reset fixed FEM-rigid-mesh tie state.
        if hasattr(self, "_fixed_rigid_mesh_tie_initialized"):
            self._fixed_rigid_mesh_tie_initialized[None] = 0

        if hasattr(self, "_fixed_rigid_mesh_tie_found_count"):
            self._fixed_rigid_mesh_tie_found_count[None] = 0

        if hasattr(self, "_bc_rigid_tie_resolved"):
            self._bc_rigid_tie_resolved.fill(0)

        if hasattr(self, "_bc_rigid_cache_elem"):
            self._bc_rigid_cache_elem.fill(-1)

        if hasattr(self, "_bc_rigid_tie_weights"):
            self._bc_rigid_tie_weights.fill(0.0)
