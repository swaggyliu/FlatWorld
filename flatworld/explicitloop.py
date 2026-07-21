from bvh import CollisionDetector
from definitions import *
from femcontact import ContactFlexRigid, contactHelper, ContactSpringRigid
from femspringmanager import FemSpringManager
from mixedcontact import MixedContact, _assign_scalar, _host_np
from rigidmanager import RigidManager
from wp_init import ensure_warp
import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _scan_bvh_pairs_stamp_kernel(
    pairs: wp.array(dtype=wp.vec2i),
    num_pairs: int,
    stamp: int,
    max_domains: int,
    domain_type: wp.array(dtype=int),
    bvh_domain_stamp: wp.array(dtype=int),
    has_fem_spring_bvh_pair: wp.array(dtype=int),
    fem_type: int,
    spring_type: int,
):
    has_fem_spring_bvh_pair[0] = 0
    for p in range(num_pairs):
        a = pairs[p][0]
        b = pairs[p][1]
        if 0 <= a < max_domains:
            bvh_domain_stamp[a] = stamp
            ta = domain_type[a]
            if ta == fem_type or ta == spring_type:
                has_fem_spring_bvh_pair[0] = 1
        if 0 <= b < max_domains:
            bvh_domain_stamp[b] = stamp
            tb = domain_type[b]
            if tb == fem_type or tb == spring_type:
                has_fem_spring_bvh_pair[0] = 1


@wp.kernel
def _filter_out_rigid_rigid_pairs_kernel(
    collision_pairs: wp.array(dtype=wp.vec2i),
    num_pairs: int,
    max_domains: int,
    domain_type: wp.array(dtype=int),
    rigid_type: int,
    out_pairs: wp.array(dtype=wp.vec2i),
    out_count: wp.array(dtype=int),
):
    """Copy BVH pairs excluding rigid-rigid pairs when disabled by config."""
    out_count[0] = 0
    for p in range(num_pairs):
        a = collision_pairs[p][0]
        b = collision_pairs[p][1]
        keep = True
        if 0 <= a < max_domains and 0 <= b < max_domains:
            ta = domain_type[a]
            tb = domain_type[b]
            if ta == rigid_type and tb == rigid_type:
                keep = False
        if keep:
            out_idx = int(wp.atomic_add(out_count, 0, 1))
            out_pairs[out_idx] = wp.vec2i(a, b)


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
        ensure_warp()

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
        # Consolidated AABB field: aabb[i, 0] = lower bound, aabb[i, 1] = upper bound
        self.aabb = wp.zeros((self._max_domains, 2), dtype=wp.vec2)
        self._bvh_pairs_work = wp.zeros(tree_size, dtype=wp.vec2i)
        self._bvh_pairs_work_count = wp.zeros(1, dtype=int)
        self._bvh_domain_stamp = wp.zeros(self._max_domains, dtype=int)
        self._bvh_active_stamp = wp.zeros(1, dtype=int)
        _assign_scalar(self._bvh_active_stamp, 1)
        self._use_bvh_domain_mask = wp.zeros(1, dtype=int)
        self._has_fem_spring_bvh_pair = wp.zeros(1, dtype=int)
        _assign_scalar(self._has_fem_spring_bvh_pair, 1)

        domain_types = np.zeros(self._max_domains, dtype=np.int32)
        for i in range(self.numDomains):
            domain_types[i] = int(self.domains[i].type)
        self._domain_type = wp.array(domain_types, dtype=int)

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
        self._counter_py = 0  # Python-side counter (avoids scalar sync per substep)

        # Contact subcycling: auto-compute from stable-time ratio after contacts are built.
        self._rigid_contact_subcycle = 1
        self._domain_min_stable_time = self.stableTime
        self._rigid_contact_min_stable_time = self.stableTime
        self._rigid_contact_subcycle_max = 8

        self.numContacts = 0
        MAX_CONTACTS = max(128, len(domains) * 2)
        contact_pair_buf = np.zeros((MAX_CONTACTS, 2), dtype=np.int32)

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
        if self.rigidManager is not None:
            self.rigidManager.precompute_rigid_transforms()
            self.rigidManager.updateBBox()

        # Now collect BVH candidates — getBBox just reads from the already-filled aabb field
        bvh_candidates = []  # (domain_index, env_id)
        aabb_np = self.aabb.numpy()
        for i in range(len(domains)):
            if self.domains[i].considerContact:
                # For rigid domains, aabb is already populated by updateBBox above
                # For FEM/spring domains, we still need to call getBBox individually
                if self.domains[i].type != DomainType.RIGID:
                    lb, ub = self.domains[i].getBBox()
                    aabb_np[i, 0] = np.asarray(lb, dtype=np.float32).reshape(2)
                    aabb_np[i, 1] = np.asarray(ub, dtype=np.float32).reshape(2)
                domain_env_id = getattr(self.domains[i], "env_id", -1)
                bvh_candidates.append((i, domain_env_id))
        self.aabb.assign(aabb_np)

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
                        contact_pair_buf[self.numContacts, 0] = i
                        contact_pair_buf[self.numContacts, 1] = j
                        self.numContacts += 1
        else:
            print(f"Pure rigid/analytical scene -- skipping O(N^2) contact pair search ({len(bvh_candidates)} domains)")

        self.contactPairIds = wp.array(contact_pair_buf, dtype=wp.vec2i)

        if self.bvh is not None:
            print("Starting BVH build with", int(self.bvh.object_count.numpy()[0]), "objects for contact detection...")
        else:
            print("BVH disabled (<=1 contact candidate) -- skipping BVH build")
        # Detect multi-env: if any domain has env_id >= 0, use per-env BVH
        self._bvh_env_groups = None  # kept for rebuild
        has_multi_env = any(eid >= 0 for eid in env_ids_np) if len(bvh_candidates) > 1 else False
        if self.bvh is not None and has_multi_env and int(self.bvh.object_count.numpy()[0]) > 1:
            from collections import defaultdict

            env_groups = defaultdict(list)
            for bvh_obj_idx, (_, eid) in enumerate(bvh_candidates):
                if eid >= 0:
                    env_groups[eid].append(bvh_obj_idx)
            self._bvh_env_groups = dict(env_groups)
            print(
                f"Per-env BVH: {len(env_groups)} envs, "
                f"skipping {sum(1 for _,e in bvh_candidates if e<0)} ground domains"
            )
            self.bvh.build_per_env(self._bvh_env_groups)
        elif self.bvh is not None and int(self.bvh.object_count.numpy()[0]) > 1:
            print("Building BVH for contact detection with", int(self.bvh.object_count.numpy()[0]), "objects...")
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
        ratio = self._rigid_contact_min_stable_time / max(self._domain_min_stable_time, 1e-12)
        self._rigid_contact_subcycle = int(max(2, min(self._rigid_contact_subcycle_max, ratio)))

        if useAdapativeDT:
            self.dt = self.stableTime
            print("Using adaptive time stepping with initial dt =", self.dt)
        else:
            self.dt = dt
        self.default_dt = dt

        # ── Batched contact optimization ─────────────────────────────────
        self._build_batched_contacts()
        self._use_rigid_mesh_compaction = self._batched_rigid_mesh_count > 0
        self._needs_bvh_domain_mask = self._use_rigid_mesh_compaction or self._batched_flex_count > 0
        self._rigid_mesh_aabb_dirty = True

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

        self._rigid_contact_prefilter_dirty = True

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
            _assign_scalar(self._use_bvh_domain_mask, 0)
            _assign_scalar(self._has_fem_spring_bvh_pair, 0)
            print("No BVH broadphase needed for contacts -- skipping BVH in RigidManager")

        # Initial FEM spatial hash build (so first frame has valid data)
        if self._use_fem_spatial_hash and self.femSpringManager is not None:
            self.femSpringManager.populate_fem_spatial_hash(velocity_buffer=0.0)

        # Initial rigid spatial hash build for FEM-rigid mesh contacts
        if self._use_rigid_spatial_hash and self.rigidManager is not None and self.rigidManager.spatialHash is not None:
            margin = getattr(self, "_rigid_sh_contact_margin", 0.0)
            if hasattr(self, "_fem_contact_aabb_lb") and hasattr(self, "_bc_rigid_mesh_node_count"):
                self._compute_fem_contact_aabb(int(_host_np(self._bc_rigid_mesh_node_count)[0]))
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

    def _scan_bvh_pairs_stamp(self, pairs, num_pairs, stamp):
        wp.launch(
            _scan_bvh_pairs_stamp_kernel,
            dim=1,
            inputs=[
                pairs,
                int(num_pairs),
                int(stamp),
                int(self._max_domains),
                self._domain_type,
                self._bvh_domain_stamp,
                self._has_fem_spring_bvh_pair,
                int(DomainType.FEM),
                int(DomainType.SPRINGMASS),
            ],
        )

    def _filter_out_rigid_rigid_pairs(self, num_pairs):
        wp.launch(
            _filter_out_rigid_rigid_pairs_kernel,
            dim=1,
            inputs=[
                self.bvh.collision_pairs,
                int(num_pairs),
                int(self._max_domains),
                self._domain_type,
                int(DomainType.RIGID),
                self._bvh_pairs_work,
                self._bvh_pairs_work_count,
            ],
        )

    def advance(self):
        num_bvh_pairs = 0
        pairs_field = self._bvh_pairs_work
        num_pairs_work = 0
        if not self.skip_bvh and self.bvh is not None:
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
            num_bvh_pairs = int(self.bvh.pair_count.numpy()[0])
            num_pairs_work = num_bvh_pairs
            pairs_field = self.bvh.collision_pairs
            if (not self._consider_rigid_rigid_contact) and num_bvh_pairs > 0:
                self._filter_out_rigid_rigid_pairs(num_bvh_pairs)
                num_pairs_work = int(self._bvh_pairs_work_count.numpy()[0])
                pairs_field = self._bvh_pairs_work

            if self._needs_bvh_domain_mask:
                if num_pairs_work > 0:
                    next_stamp = int(self._bvh_active_stamp.numpy()[0]) + 1
                    if next_stamp > 2000000000:
                        self._bvh_domain_stamp.zero_()
                        next_stamp = 1
                    _assign_scalar(self._bvh_active_stamp, next_stamp)
                    _assign_scalar(self._use_bvh_domain_mask, 1)
                    self._scan_bvh_pairs_stamp(pairs_field, num_pairs_work, next_stamp)
                else:
                    _assign_scalar(self._use_bvh_domain_mask, 1)
                    _assign_scalar(self._has_fem_spring_bvh_pair, 0)
            else:
                _assign_scalar(self._use_bvh_domain_mask, 0)
                _assign_scalar(self._has_fem_spring_bvh_pair, 1 if num_pairs_work > 0 else 0)
        else:
            # BVH inactive: GPU fields already set once per frame in
            # advanceWithTime(); skip per-substep writes.
            pass

        # ── Batched contact kernels (see mixedcontact.py) ──
        self.run_batched_contacts(pairs_field, num_pairs_work)

        # Calculate FEM domains
        if self.femSpringManager is not None:
            self.femSpringManager.substep(self.dt, self.damping)

        # Rigid body substep
        if self.rigidManager is not None:
            N = max(1, self.rigidManager.stableTime // self.dt)
            self.rigid_update_times += 1
            if self.rigid_update_times % N == 0:
                if hasattr(self, "_fem_contact_aabb_lb") and hasattr(self, "_bc_rigid_mesh_node_count"):
                    self._compute_fem_contact_aabb(int(_host_np(self._bc_rigid_mesh_node_count)[0]))
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

            N2 = max(1, self.rigidManager.get_sh_update_interval() // self.dt)
            if self.rigid_update_times % N2 == 0:
                self._rigid_contact_prefilter_dirty = True
                self._rigid_mesh_aabb_dirty = True

                if hasattr(self, "_bc_rigid_skip_epoch"):
                    self._bc_rigid_skip_epoch.fill_(-1)

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

        # Set BVH-related GPU state ONCE per frame (not per substep).
        if self.skip_bvh or self.bvh is None:
            _assign_scalar(self._use_bvh_domain_mask, 0)
            _assign_scalar(self._has_fem_spring_bvh_pair, 0)

        remaining = float(frame_dt)
        eps = 1e-12
        if verbose:
            print(f"Advancing fixed time step of {frame_dt} seconds")
            print(f"Current stable time step: {self.stableTime:.3e} seconds")

        substep_count = 0
        while remaining > eps:
            current_stable = self.stableTime if self.useAdapative else self.default_dt
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

        joint_id_a = manager.joint_id_a.numpy()
        joint_id_b = manager.joint_id_b.numpy()
        quat = manager.quat.numpy()
        quat_initial = manager.quat_initial.numpy()

        angle_map = {}
        for j_idx, joint in enumerate(manager.joints):
            joint_name = getattr(joint, "name", "") or f"joint_{j_idx}"
            is_revolute = getattr(joint, "jointType", None) == JointType.Revolute

            if not is_revolute:
                if include_non_revolute:
                    angle_map[joint_name] = None
                continue

            idx_a = int(joint_id_a[j_idx])
            idx_b = int(joint_id_b[j_idx])

            angle_a = float(quat[idx_a]) - float(quat_initial[idx_a])
            angle_b = float(quat[idx_b]) - float(quat_initial[idx_b])
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
        self._counter_py = 0

        if self.rigidManager is not None and (
            self.rigidManager.numRigids > 0 or self.rigidManager.numAnalytical > 0
        ):
            self.rigidManager.reset()

        for domain in self.domains:
            if domain.type == DomainType.FEM or domain.type == DomainType.SPRINGMASS:
                pass

        if self.bvh is not None and int(self.bvh.object_count.numpy()[0]) > 1:
            self.bvh.reset()
            self.bvh.update_objects(self.aabb, 1)
            if self._bvh_env_groups:
                self.bvh.build_per_env(self._bvh_env_groups)
            else:
                self.bvh.build()

        if hasattr(self, "_fixed_rigid_mesh_tie_initialized"):
            _assign_scalar(self._fixed_rigid_mesh_tie_initialized, 0)

        if hasattr(self, "_fixed_rigid_mesh_tie_found_count"):
            _assign_scalar(self._fixed_rigid_mesh_tie_found_count, 0)

        if hasattr(self, "_bc_rigid_tie_resolved"):
            self._bc_rigid_tie_resolved.fill_(0)

        if hasattr(self, "_bc_rigid_cache_elem"):
            self._bc_rigid_cache_elem.fill_(-1)

        if hasattr(self, "_bc_rigid_tie_weights"):
            self._bc_rigid_tie_weights.fill_(0.0)
