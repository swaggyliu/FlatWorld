"""Batched mixed-domain contact kernels for ExplicitLoop.

Handles penalty contact between FEM/Spring domains and analytical planes,
other FEM domains, rigid bodies, height fields, and voxel maps.  Work lists
are built once at init; each substep launches O(1) kernels per contact type.
"""

from contact_detection import detectPointToMeshBoundary, detectPointToPrimitive, pointToEdgeContact
from definitions import *
import numpy as np
import taichi as ti
import time


@ti.data_oriented
class MixedContact:
    """Mixin providing batched contact build/run/kernels for ExplicitLoop."""

    def run_batched_contacts(self, pairs_field, num_pairs_work):
        """Execute all batched contact kernels for one substep."""
        if self._batched_anal_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_anal_pair_a, self._bc_anal_pair_b, self._bc_anal_active
            )
            self._batched_analytical_contact_kernel(self._batched_anal_count)


        if self._batched_flex_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_flex_pair_a, self._bc_flex_pair_b, self._bc_flex_active
            )

            flex_tie_initialized = (
                hasattr(self, "_fixed_flexflex_tie_initialized") and self._fixed_flexflex_tie_initialized[None] == 1
            )

            need_flex_hash_refresh = self._nontied_flex_item_count > 0 or (
                self._tied_flex_item_count > 0 and not flex_tie_initialized
            )
            if self._use_fem_spatial_hash and need_flex_hash_refresh:
                if self.femSpringManager is not None:
                    self.femSpringManager.maybe_rebuild_fem_spatial_hash(self.dt)


            if self._tied_flex_item_count > 0:
                if self._fixed_flexflex_tie_initialized[None] == 0:
                    self._initialize_fem_fem_ties_once_kernel(self._batched_flex_count)
                    self._fixed_flexflex_tie_initialized[None] = 1
                    print(
                        f"Initialized FEM-FEM ties: resolved {int(self._fixed_flexflex_tie_found_count[None])} / "
                        f"{self._tied_flex_item_count} work items"
                    )

                self._apply_fem_fem_ties_fixed_kernel(self._batched_flex_count)

            if self._nontied_flex_item_count > 0:
                if self._use_fem_spatial_hash:
                    self._batched_flexflex_contact_kernel_sh(self._batched_flex_count)
                else:
                    self._batched_flexflex_contact_kernel(self._batched_flex_count)

        if self._batched_rigid_count > 0:
            if self._rigid_contact_prefilter_dirty:
                if self.skip_bvh:
                    self._activate_pairs_by_aabb_kernel(
                        self._bc_rigid_pair_a,
                        self._bc_rigid_pair_b,
                        self._bc_rigid_active,
                        self._bc_rigid_active.shape[0],
                    )
                else:
                    self._activate_or_fill_pairs(
                        pairs_field, num_pairs_work, self._bc_rigid_pair_a, self._bc_rigid_pair_b, self._bc_rigid_active
                    )
            if self._batched_rigid_prim_count > 0:
                self._batched_rigid_prim_kernel(self._batched_rigid_prim_count)
            if self._batched_rigid_mesh_count > 0:
                tie_initialized = (
                    hasattr(self, "_fixed_rigid_mesh_tie_initialized")
                    and self._fixed_rigid_mesh_tie_initialized[None] == 1
                )

                needs_mesh_element_aabb = self._nontied_rigid_mesh_item_count > 0 or (
                    self._tied_rigid_mesh_item_count > 0 and not tie_initialized
                )

                if self._rigid_mesh_aabb_dirty and needs_mesh_element_aabb:
                    self.rigidManager.update_mesh_element_aabbs()
                    self._rigid_mesh_aabb_dirty = False

                if self._tied_rigid_mesh_item_count > 0:
                    if self._fixed_rigid_mesh_tie_initialized[None] == 0:
                        self._initialize_fem_rigid_mesh_ties_once_kernel(
                            self._batched_rigid_mesh_count,
                        )
                        self._fixed_rigid_mesh_tie_initialized[None] = 1

                    self._apply_fem_rigid_mesh_ties_fixed_kernel(self._batched_rigid_mesh_count)

                if self._nontied_rigid_mesh_item_count > 0:
                    run_full_mesh_contact = 1
                    if self._rigid_contact_subcycle > 1 and (self._counter_py % self._rigid_contact_subcycle) != 0:
                        run_full_mesh_contact = 0


                    self._bc_rigid_node_near_any.fill(0)
                    self._prefilter_mesh_node_activity()

                    self._build_active_rigid_mesh_workset(
                        self._batched_rigid_mesh_count,
                    )
                    self._batched_rigid_mesh_kernel_sh(run_full_mesh_contact)

        if self._batched_hf_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_hf_pair_a, self._bc_hf_pair_b, self._bc_hf_active
            )
            self._batched_heightfield_contact_kernel(self._batched_hf_count, self.dt)

        if self._batched_voxel_count > 0:
            self._activate_or_fill_pairs(
                pairs_field, num_pairs_work, self._bc_voxel_pair_a, self._bc_voxel_pair_b, self._bc_voxel_active
            )
            self._batched_voxelmap_contact_kernel(self._batched_voxel_count, self.dt)

    def _build_batched_contacts(self):
        """Build flat Taichi work-lists for all contact pairs so that each
        contact *type* is evaluated by a single kernel launch rather than
        one kernel per pair.

        Supported batched types:
          - FlexAnalytical + SpringAnalytical → _batched_analytical_contact_kernel
          - FlexFlex                         → _batched_flexflex_contact_kernel
          - FlexRigid + SpringRigid          → _batched_rigid_prim_kernel / _batched_rigid_mesh_kernel
          - FlexHeightField + SpringHeightField → _batched_heightfield_contact_kernel
          - FlexVoxelMap                     → _batched_voxelmap_contact_kernel
        """
        from femcontact import (
            ContactFlexAnalytical,
            ContactFlexFlex,
            ContactFlexHeightField,
            ContactFlexRigid,
            ContactFlexVoxelMap,
            ContactSpringAnalytical,
            ContactSpringHeightField,
            ContactSpringRigid,
        )

        anal_contacts = []  # (contact, index) - FlexAnalytical + SpringAnalytical
        flex_contacts = []  # (contact, index) - FlexFlex
        rigid_contacts = []  # (contact, index) - FlexRigid + SpringRigid
        hf_contacts = []  # (contact, index) - FlexHeightField + SpringHeightField
        voxel_contacts = []  # (contact, index) - FlexVoxelMap
        unbatched = []
        unbatched_pair_ids = []

        for idx, c in enumerate(self.contacts):
            if isinstance(c, (ContactFlexAnalytical, ContactSpringAnalytical)):
                anal_contacts.append((c, idx))
            elif isinstance(c, ContactFlexFlex):
                flex_contacts.append((c, idx))
            elif isinstance(c, (ContactFlexRigid, ContactSpringRigid)):
                rigid_contacts.append((c, idx))
            elif isinstance(c, (ContactFlexHeightField, ContactSpringHeightField)):
                hf_contacts.append((c, idx))
            elif isinstance(c, ContactFlexVoxelMap):
                voxel_contacts.append((c, idx))
            else:
                unbatched.append(c)
                unbatched_pair_ids.append((int(self.contactPairIds[idx][0]), int(self.contactPairIds[idx][1])))

        # ── FlexAnalytical + SpringAnalytical batching ──
        self._batched_anal_count = 0
        if anal_contacts and self.femSpringManager is not None:
            work_bn = []
            work_penalty = []
            work_plane_pt = []
            work_plane_nm = []
            work_dom_did = []
            work_pair = []

            self._anal_pair_domain_ids = [
                (int(self.contactPairIds[orig_idx][0]), int(self.contactPairIds[orig_idx][1]))
                for _, (_, orig_idx) in enumerate(anal_contacts)
            ]

            for pair_idx, (c, _) in enumerate(anal_contacts):
                is_spring = isinstance(c, ContactSpringAnalytical)
                d_idx = c.domain1.domainIdx
                anal_idx = c.domain2.ndOffset
                pp = [float(c.domain2.rigidManager.rigidParams[anal_idx, 0][k]) for k in range(self.d)]
                pn = [float(c.domain2.rigidManager.rigidParams[anal_idx, 1][k]) for k in range(self.d)]


                if is_spring:
                    # SpringAnalytical: check all nodes (not just boundary)
                    off = int(self.femSpringManager.domainNodeOffset[d_idx])
                    cnt = c.domain1.nnodes
                    for i in range(cnt):
                        work_bn.append(off + i)
                        work_penalty.append(float(c.penalty))
                        work_plane_pt.append(pp)
                        work_plane_nm.append(pn)
                        work_dom_did.append(d_idx)
                        work_pair.append(pair_idx)
                else:
                    # FlexAnalytical: boundary nodes only
                    off = int(self.femSpringManager.domainBoundaryNodeOffset[d_idx])
                    cnt = int(self.femSpringManager.domainBoundaryNodeCount[d_idx])
                    for i in range(cnt):
                        work_bn.append(int(self.femSpringManager.boundaryNodes[off + i]))
                        work_penalty.append(float(c.penalty))
                        work_plane_pt.append(pp)
                        work_plane_nm.append(pn)
                        work_dom_did.append(d_idx)
                        work_pair.append(pair_idx)

            total = len(work_bn)
            if total > 0:
                self._bc_anal_node = ti.field(ti.i32, total)
                self._bc_anal_penalty = ti.field(ti.f32, total)
                self._bc_anal_pp = ti.Vector.field(self.d, ti.f32, total)
                self._bc_anal_pn = ti.Vector.field(self.d, ti.f32, total)
                self._bc_anal_dom_did = ti.field(ti.i32, total)
                self._bc_anal_pair = ti.field(ti.i32, total)
                self._bc_anal_pair_a, self._bc_anal_pair_b, self._bc_anal_active = self._create_pair_activation_fields(
                    self._anal_pair_domain_ids
                )

                self._bc_anal_node.from_numpy(np.array(work_bn, dtype=np.int32))
                self._bc_anal_penalty.from_numpy(np.array(work_penalty, dtype=np.float32))
                self._bc_anal_pp.from_numpy(np.array(work_plane_pt, dtype=np.float32))
                self._bc_anal_pn.from_numpy(np.array(work_plane_nm, dtype=np.float32))
                self._bc_anal_dom_did.from_numpy(np.array(work_dom_did, dtype=np.int32))
                self._bc_anal_pair.from_numpy(np.array(work_pair, dtype=np.int32))
                self._batched_anal_count = total
                print(f"  Batched Analytical: {len(anal_contacts)} pairs → 1 kernel " f"({total} work items)")

        # ── FlexFlex batching ──
        self._batched_flex_count = 0
        self._use_fem_spatial_hash = False
        self._tied_flex_item_count = 0
        self._nontied_flex_item_count = 0
        if flex_contacts and self.femSpringManager is not None:
            work_node = []
            work_penalty = []
            work_be_off = []
            work_be_cnt = []
            work_pair = []
            work_friction = []
            work_target_did = []  # FEM-local domain index of target for spatial hash query
            visited_fem_domain = set()
            work_tied = []  # whether this contact pair is a tied contact

            for pair_idx, (c, _) in enumerate(flex_contacts):
                fric = max(c.domain1.friction, c.domain2.friction)
                pen = float(c.penalty)

                tied = 1 if (hasattr(c, "tied") and c.tied) else 0

                # Pass 1: dom2 boundary nodes vs dom1 boundary elements
                d2_idx = c.domain2.domainIdx
                off2 = int(self.femSpringManager.domainBoundaryNodeOffset[d2_idx])
                cnt2 = int(self.femSpringManager.domainBoundaryNodeCount[d2_idx])
                d1_be_off = int(self.femSpringManager.domainBoundaryElemOffset[c.domain1.domainIdx])
                d1_be_cnt = int(c.domain1.mesh.numBoundElements)
                d1_did = int(c.domain1.domainIdx)  # FEM-local index for spatial hash
                for i in range(cnt2):
                    nid = int(self.femSpringManager.boundaryNodes[off2 + i])
                    work_node.append(nid)
                    work_penalty.append(pen)
                    work_be_off.append(d1_be_off)
                    work_be_cnt.append(d1_be_cnt)
                    work_pair.append(pair_idx)
                    work_friction.append(fric)
                    work_target_did.append(d1_did)
                    work_tied.append(tied)

                # Pass 2: dom1 boundary nodes vs dom2 boundary elements
                d1_idx = c.domain1.domainIdx
                off1 = int(self.femSpringManager.domainBoundaryNodeOffset[d1_idx])
                cnt1 = int(self.femSpringManager.domainBoundaryNodeCount[d1_idx])
                d2_be_off = int(self.femSpringManager.domainBoundaryElemOffset[c.domain2.domainIdx])
                d2_be_cnt = int(c.domain2.mesh.numBoundElements)
                d2_did = int(c.domain2.domainIdx)  # FEM-local index for spatial hash
                for i in range(cnt1):
                    nid = int(self.femSpringManager.boundaryNodes[off1 + i])
                    work_node.append(nid)
                    work_penalty.append(pen)
                    work_be_off.append(d2_be_off)
                    work_be_cnt.append(d2_be_cnt)
                    work_pair.append(pair_idx)
                    work_friction.append(fric)
                    work_target_did.append(d2_did)
                    work_tied.append(tied)

            total = len(work_node)
            if total > 0:
                self._bc_flex_node = ti.field(ti.i32, total)
                self._bc_flex_penalty = ti.field(ti.f32, total)
                self._bc_flex_be_off = ti.field(ti.i32, total)
                self._bc_flex_be_cnt = ti.field(ti.i32, total)
                self._bc_flex_pair = ti.field(ti.i32, total)
                self._bc_flex_friction = ti.field(ti.f32, total)
                self._bc_flex_target_did = ti.field(ti.i32, total)
                self._bc_flex_cache_elem = ti.field(ti.i32, total)
                self._bc_flex_kind = ti.field(ti.i32, total)
                self._bc_flex_tied = ti.field(ti.i32, total)
                num_pairs = len(flex_contacts)
                self._bc_flex_active = ti.field(ti.i32, num_pairs)
                work_tied_np = np.array(work_tied, dtype=np.int32)

                self._bc_flex_node.from_numpy(np.array(work_node, dtype=np.int32))
                self._bc_flex_penalty.from_numpy(np.array(work_penalty, dtype=np.float32))
                self._bc_flex_be_off.from_numpy(np.array(work_be_off, dtype=np.int32))
                self._bc_flex_be_cnt.from_numpy(np.array(work_be_cnt, dtype=np.int32))
                self._bc_flex_pair.from_numpy(np.array(work_pair, dtype=np.int32))
                self._bc_flex_friction.from_numpy(np.array(work_friction, dtype=np.float32))
                self._bc_flex_target_did.from_numpy(np.array(work_target_did, dtype=np.int32))
                self._bc_flex_tied.from_numpy(work_tied_np)

                self._bc_flex_cache_elem.fill(-1)

                self._tied_flex_item_count = int(work_tied_np.sum())
                self._nontied_flex_item_count = total - self._tied_flex_item_count
                if self._tied_flex_item_count > 0:
                    self._bc_flex_tie_resolved = ti.field(ti.i32, total)
                    self._bc_flex_tie_elem = ti.field(ti.i32, total)
                    self._bc_flex_tie_weights = ti.Vector.field(self.d, ti.f32, total)
                    self._bc_flex_tie_gap = ti.field(ti.f32, total)
                    self._bc_flex_tie_resolved.fill(0)
                    self._bc_flex_tie_elem.fill(-1)
                    self._bc_flex_tie_weights.fill(0.0)
                    self._bc_flex_tie_gap.fill(0.0)
                    self._fixed_flexflex_tie_initialized = ti.field(ti.i32, shape=())
                    self._fixed_flexflex_tie_initialized[None] = 0
                    self._fixed_flexflex_tie_found_count = ti.field(ti.i32, shape=())
                    self._fixed_flexflex_tie_found_count[None] = 0

                self._flex_pair_domain_ids = [
                    (int(self.contactPairIds[orig_idx][0]), int(self.contactPairIds[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(flex_contacts)
                ]
                self._bc_flex_pair_a, self._bc_flex_pair_b, self._bc_flex_active = self._create_pair_activation_fields(
                    self._flex_pair_domain_ids
                )

                self._batched_flex_count = total
                self._use_fem_spatial_hash = self.femSpringManager.spatialHash is not None
                print(
                    f"  Batched FlexFlex: {len(flex_contacts)} pairs → 1 kernel "
                    f"({total} work items, tied={self._tied_flex_item_count}, "
                    f"spatial_hash={'ON' if self._use_fem_spatial_hash else 'OFF'})"
                )

        # ── FlexRigid + SpringRigid batching ──
        self._batched_rigid_count = 0
        self._batched_rigid_mesh_count = 0
        self._batched_rigid_prim_count = 0
        self._use_rigid_spatial_hash = False
        self._tied_rigid_mesh_item_count = 0
        self._nontied_rigid_mesh_item_count = 0
        if rigid_contacts and self.femSpringManager is not None and self.rigidManager is not None:
            work_node = []
            work_normals = []
            work_rigid_idx = []
            work_penalty = []
            work_friction = []
            work_is_mesh = []
            work_fem_did = []
            work_rigid_did = []
            work_pair = []
            work_tied = []

            for pair_idx, (c, orig_idx) in enumerate(rigid_contacts):
                is_spring = isinstance(c, ContactSpringRigid)
                d_idx = c.domain1.domainIdx
                rigid_idx = c.domain2.ndOffset
                pen = float(c.penalty)
                fric = float(c.domain2.friction)

                tied = 1 if (hasattr(c, "tied") and c.tied) else 0

                rigid_type_val = int(c.domain2.rigidManager.rigidDomainIds[rigid_idx][1])
                is_mesh = 1 if rigid_type_val == int(RigidType.MESH) else 0

                fem_gid = int(c.domain1.femManager.femDomainIds[c.domain1.domainIdx]) if is_mesh else 0
                # SpatialHash stores mesh owner as rigid index (not global domain index).
                rigid_gid = int(rigid_idx) if is_mesh else 0

                if is_spring:
                    off = int(self.femSpringManager.domainNodeOffset[d_idx])
                    cnt = c.domain1.nnodes
                    for i in range(cnt):
                        work_node.append(off + i)
                        work_rigid_idx.append(rigid_idx)
                        work_penalty.append(pen)
                        work_friction.append(fric)
                        work_is_mesh.append(is_mesh)
                        work_fem_did.append(fem_gid)
                        work_rigid_did.append(rigid_gid)
                        work_pair.append(pair_idx)
                        work_tied.append(tied)
                else:
                    off = int(self.femSpringManager.domainBoundaryNodeOffset[d_idx])
                    cnt = int(self.femSpringManager.domainBoundaryNodeCount[d_idx])
                    for i in range(cnt):
                        nd = int(self.femSpringManager.boundaryNodes[off + i])
                        work_node.append(nd)
                        work_normals.append(self.femSpringManager.boundaryNodeNormals[nd])
                        work_rigid_idx.append(rigid_idx)
                        work_penalty.append(pen)
                        work_friction.append(fric)
                        work_is_mesh.append(is_mesh)
                        work_fem_did.append(fem_gid)
                        work_rigid_did.append(rigid_gid)
                        work_pair.append(pair_idx)
                        work_tied.append(tied)
            total = len(work_node)
            if total > 0:
                self._bc_rigid_node = ti.field(ti.i32, total)
                self._bc_rigid_idx = ti.field(ti.i32, total)
                self._bc_rigid_penalty = ti.field(ti.f32, total)
                self._bc_rigid_friction = ti.field(ti.f32, total)
                self._bc_rigid_is_mesh = ti.field(ti.i32, total)
                self._bc_rigid_fem_did = ti.field(ti.i32, total)
                self._bc_rigid_did = ti.field(ti.i32, total)
                self._bc_rigid_pair = ti.field(ti.i32, total)
                self._bc_rigid_tied = ti.field(ti.i32, total)

                self._bc_rigid_node.from_numpy(np.array(work_node, dtype=np.int32))
                self._bc_rigid_idx.from_numpy(np.array(work_rigid_idx, dtype=np.int32))
                self._bc_rigid_penalty.from_numpy(np.array(work_penalty, dtype=np.float32))
                self._bc_rigid_friction.from_numpy(np.array(work_friction, dtype=np.float32))
                self._bc_rigid_is_mesh.from_numpy(np.array(work_is_mesh, dtype=np.int32))
                self._bc_rigid_fem_did.from_numpy(np.array(work_fem_did, dtype=np.int32))
                self._bc_rigid_did.from_numpy(np.array(work_rigid_did, dtype=np.int32))
                self._bc_rigid_pair.from_numpy(np.array(work_pair, dtype=np.int32))
                self._bc_rigid_tied.from_numpy(np.array(work_tied, dtype=np.int32))

                self._rigid_pair_domain_ids = [
                    (int(self.contactPairIds[orig_idx][0]), int(self.contactPairIds[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(rigid_contacts)
                ]
                self._bc_rigid_pair_a, self._bc_rigid_pair_b, self._bc_rigid_active = (
                    self._create_pair_activation_fields(self._rigid_pair_domain_ids)
                )

                self._batched_rigid_count = total

                # Build separate index arrays for mesh vs primitive items.
                # This lets us launch separate smaller kernels (faster JIT).
                mesh_indices = [i for i, m in enumerate(work_is_mesh) if m == 1]
                prim_indices = [i for i, m in enumerate(work_is_mesh) if m == 0]
                self._batched_rigid_mesh_count = len(mesh_indices)
                self._batched_rigid_prim_count = len(prim_indices)
                self._tied_rigid_mesh_item_count = sum(1 for i in mesh_indices if work_tied[i] == 1)
                self._nontied_rigid_mesh_item_count = len(mesh_indices) - self._tied_rigid_mesh_item_count
                if mesh_indices:
                    self._bc_rigid_mesh_idx = ti.field(ti.i32, len(mesh_indices))
                    self._bc_rigid_mesh_idx.from_numpy(np.array(mesh_indices, dtype=np.int32))
                    self._bc_rigid_normals = ti.Vector.field(self.d, ti.f32, len(mesh_indices))
                    self._bc_rigid_normals.from_numpy(np.array(work_normals, dtype=np.float32))
                    self._bc_rigid_mesh_active_idx = ti.field(ti.i32, len(mesh_indices))
                    self._bc_rigid_mesh_active_count = ti.field(ti.i32, shape=())
                    self._bc_rigid_mesh_active_count[None] = len(mesh_indices)
                    # Node resolution flag to skip redundant rigid targets once contact is found
                    self._bc_rigid_node_resolved = ti.field(ti.i32, self.femSpringManager.MAX_NODES)
                    # Per-mesh-work-item element cache.
                    # For normal contact, stores the last-hit rigid mesh element to avoid
                    # full spatial-hash searches while the same local contact remains valid.
                    # For tied FEM-rigid mesh contact, stores the fixed rigid mesh element
                    # resolved during one-time tie initialization; the stored element is then
                    # reused every substep together with _bc_rigid_tie_weights.
                    self._bc_rigid_cache_elem = ti.Vector.field(2, ti.i32, len(mesh_indices))
                    self._bc_rigid_cache_elem.fill(-1)
                    # LS-DYNA friction history for mesh contact, indexed by mesh-local work-item
                    self._bc_rigid_fric_prev_elem = ti.field(ti.i32, len(mesh_indices))
                    self._bc_rigid_fric_prev_valid = ti.field(ti.i32, len(mesh_indices))
                    self._bc_rigid_fric_prev_force = ti.Vector.field(self.d, ti.f32, len(mesh_indices))
                    self._bc_rigid_fric_prev_weights = ti.Vector.field(self.d, ti.f32, len(mesh_indices))
                    self._bc_rigid_fric_prev_penetration = ti.field(ti.f32, len(mesh_indices))
                    self._bc_rigid_fric_prev_elem.fill(-1)
                    self._bc_rigid_fric_prev_valid.fill(0)
                    self._bc_rigid_fric_prev_force.fill(0.0)
                    self._bc_rigid_fric_prev_weights.fill(0.0)
                    self._bc_rigid_fric_prev_penetration.fill(0.0)
                    # Tie lock: once resolved on first detection, keep using
                    # the same cached element pair and skip SH lookups.
                    self._bc_rigid_tie_resolved = ti.field(ti.i32, len(mesh_indices))
                    self._bc_rigid_tie_resolved.fill(0)

                    # Fixed FEM-rigid-mesh tie data.
                    self._bc_rigid_tie_weights = ti.Vector.field(self.d, ti.f32, len(mesh_indices))
                    self._bc_rigid_tie_weights.fill(0.0)
                    # Enable fixed one-time tie mode for FEM-rigid mesh.
                    self._fixed_rigid_mesh_tie_enabled = ti.field(ti.i32, shape=())
                    self._fixed_rigid_mesh_tie_enabled[None] = 1
                    # Has the one-time initialization been performed
                    self._fixed_rigid_mesh_tie_initialized = ti.field(ti.i32, shape=())
                    self._fixed_rigid_mesh_tie_initialized[None] = 0
                    # Optional counters for diagnostics.
                    self._fixed_rigid_mesh_tie_found_count = ti.field(ti.i32, shape=())
                    self._fixed_rigid_mesh_tie_found_count[None] = 0

                    # Epoch-based negative cache: when SH query returns 0 candidates,
                    # record the current SH epoch.  Skip SH query on subsequent
                    # substeps until the next SH rebuild bumps the epoch.
                    self._bc_rigid_skip_epoch = ti.field(ti.i32, len(mesh_indices))
                    self._bc_rigid_skip_epoch.fill(-1)

                    mesh_nodes_unique = sorted(set(work_node[i] for i in mesh_indices))
                    mesh_rigid_ids = sorted(set(work_rigid_did[i] for i in mesh_indices))
                    rigid_margin = []
                    for rid in mesh_rigid_ids:
                        rigid_margin.append(1e-4)

                    self._bc_rigid_mesh_nodes = ti.field(ti.i32, len(mesh_nodes_unique))
                    self._bc_rigid_mesh_nodes.from_numpy(np.array(mesh_nodes_unique, dtype=np.int32))
                    self._bc_rigid_mesh_node_count = ti.field(ti.i32, shape=())
                    self._bc_rigid_mesh_node_count[None] = len(mesh_nodes_unique)

                    self._bc_rigid_mesh_rigid_ids = ti.field(ti.i32, len(mesh_rigid_ids))
                    self._bc_rigid_mesh_rigid_ids.from_numpy(np.array(mesh_rigid_ids, dtype=np.int32))
                    self._bc_rigid_mesh_rigid_margin = ti.field(ti.f32, len(mesh_rigid_ids))
                    self._bc_rigid_mesh_rigid_margin.from_numpy(np.array(rigid_margin, dtype=np.float32))
                    self._bc_rigid_mesh_rigid_count = ti.field(ti.i32, shape=())
                    self._bc_rigid_mesh_rigid_count[None] = len(mesh_rigid_ids)

                    use_node_mask = 0  # TODO: disable this mask function, recover it later, it can help with large mesh + many contacts cases
                    self._bc_rigid_node_mask = ti.field(ti.i32, self.femSpringManager.MAX_NODES)
                    self._bc_rigid_node_near_any = ti.field(ti.i32, self.femSpringManager.MAX_NODES)
                    self._bc_rigid_use_node_mask = ti.field(ti.i32, shape=())
                    self._bc_rigid_use_node_mask[None] = use_node_mask
                    if use_node_mask == 1:
                        for rid in mesh_rigid_ids:
                            if rid >= 31:
                                use_node_mask = 0
                                self._bc_rigid_use_node_mask[None] = 0
                                break
                if prim_indices:
                    self._bc_rigid_prim_idx = ti.field(ti.i32, len(prim_indices))
                    self._bc_rigid_prim_idx.from_numpy(np.array(prim_indices, dtype=np.int32))
                    self._bc_rigid_node_resolved = ti.field(ti.i32, self.femSpringManager.MAX_NODES)

                has_mesh = self._batched_rigid_mesh_count > 0
                self._use_rigid_spatial_hash = (
                    has_mesh and self.rigidManager is not None and self.rigidManager.spatialHash is not None
                )
                # Max contact half-thickness for mesh rigid contacts — used as
                # spatial hash grid margin so approaching FEM nodes are inside.
                self._rigid_sh_contact_margin = 0.0

                # FEM contact AABB — computed each SH rebuild to filter
                # which mesh elements enter the spatial hash.
                self._fem_contact_aabb_lb = ti.Vector.field(self.d, ti.f32, shape=())
                self._fem_contact_aabb_ub = ti.Vector.field(self.d, ti.f32, shape=())

                print(
                    f"  Batched Rigid: {len(rigid_contacts)} pairs → split kernels "
                    f"({self._batched_rigid_mesh_count} mesh + {self._batched_rigid_prim_count} prim items, "
                    f"spatial_hash={'ON' if self._use_rigid_spatial_hash else 'OFF'})"
                )
        # ── FlexHeightField + SpringHeightField batching ──
        # Groups all contacts by their heightfield domain; if there is exactly
        # one unique heightfield (the common RL case), a single batched kernel
        # handles everything.  Multiple heightfields fall back to unbatched.
        self._batched_hf_count = 0
        if hf_contacts and self.femSpringManager is not None:
            hf_domains = {}
            for c, idx in hf_contacts:
                key = id(c.domain2)
                if key not in hf_domains:
                    hf_domains[key] = c.domain2

            if len(hf_domains) <= 1:
                self._hf_domain = list(hf_domains.values())[0]
                work_node = []
                work_penalty = []
                work_dom_did = []
                work_pair = []

                self._hf_pair_domain_ids = [
                    (int(self.contactPairIds[orig_idx][0]), int(self.contactPairIds[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(hf_contacts)
                ]

                for pair_idx, (c, _) in enumerate(hf_contacts):
                    is_spring = isinstance(c, ContactSpringHeightField)
                    d_idx = c.domain1.domainIdx
                    pen = float(c.penalty)


                    if is_spring:
                        off = int(self.femSpringManager.domainNodeOffset[d_idx])
                        cnt = c.domain1.nnodes
                        for i in range(cnt):
                            work_node.append(off + i)
                            work_penalty.append(pen)
                            work_dom_did.append(d_idx)
                            work_pair.append(pair_idx)
                    else:
                        off = int(self.femSpringManager.domainBoundaryNodeOffset[d_idx])
                        cnt = int(self.femSpringManager.domainBoundaryNodeCount[d_idx])
                        for i in range(cnt):
                            work_node.append(int(self.femSpringManager.boundaryNodes[off + i]))
                            work_penalty.append(pen)
                            work_dom_did.append(d_idx)
                            work_pair.append(pair_idx)

                total = len(work_node)
                if total > 0:
                    self._bc_hf_node = ti.field(ti.i32, total)
                    self._bc_hf_penalty = ti.field(ti.f32, total)
                    self._bc_hf_dom_did = ti.field(ti.i32, total)
                    self._bc_hf_pair = ti.field(ti.i32, total)
                    self._bc_hf_pair_a, self._bc_hf_pair_b, self._bc_hf_active = self._create_pair_activation_fields(
                        self._hf_pair_domain_ids
                    )
                    self._bc_hf_node.from_numpy(np.array(work_node, dtype=np.int32))
                    self._bc_hf_penalty.from_numpy(np.array(work_penalty, dtype=np.float32))
                    self._bc_hf_dom_did.from_numpy(np.array(work_dom_did, dtype=np.int32))
                    self._bc_hf_pair.from_numpy(np.array(work_pair, dtype=np.int32))
                    self._batched_hf_count = total
                    print(f"  Batched HeightField: {len(hf_contacts)} pairs → 1 kernel " f"({total} work items)")
            else:
                for c, idx in hf_contacts:
                    unbatched.append(c)
                    unbatched_pair_ids.append((int(self.contactPairIds[idx][0]), int(self.contactPairIds[idx][1])))
                print(f"  HeightField: {len(hf_domains)} unique domains, keeping {len(hf_contacts)} unbatched")

        # ── FlexVoxelMap batching ──
        # Same grouping strategy as HeightField.
        self._batched_voxel_count = 0
        if voxel_contacts and self.femSpringManager is not None:
            voxel_domains = {}
            for c, idx in voxel_contacts:
                key = id(c.domain2)
                if key not in voxel_domains:
                    voxel_domains[key] = c.domain2

            if len(voxel_domains) <= 1:
                self._voxel_domain = list(voxel_domains.values())[0]
                work_node = []
                work_penalty = []
                work_dom_did = []
                work_pair = []

                self._voxel_pair_domain_ids = [
                    (int(self.contactPairIds[orig_idx][0]), int(self.contactPairIds[orig_idx][1]))
                    for _, (_, orig_idx) in enumerate(voxel_contacts)
                ]

                for pair_idx, (c, _) in enumerate(voxel_contacts):
                    d_idx = c.domain1.domainIdx
                    pen = float(c.penalty)

                    off = int(self.femSpringManager.domainBoundaryNodeOffset[d_idx])
                    cnt = int(self.femSpringManager.domainBoundaryNodeCount[d_idx])
                    for i in range(cnt):
                        work_node.append(int(self.femSpringManager.boundaryNodes[off + i]))
                        work_penalty.append(pen)
                        work_dom_did.append(d_idx)
                        work_pair.append(pair_idx)

                total = len(work_node)
                if total > 0:
                    self._bc_voxel_node = ti.field(ti.i32, total)
                    self._bc_voxel_penalty = ti.field(ti.f32, total)
                    self._bc_voxel_dom_did = ti.field(ti.i32, total)
                    self._bc_voxel_pair = ti.field(ti.i32, total)
                    self._bc_voxel_pair_a, self._bc_voxel_pair_b, self._bc_voxel_active = (
                        self._create_pair_activation_fields(self._voxel_pair_domain_ids)
                    )
                    self._bc_voxel_node.from_numpy(np.array(work_node, dtype=np.int32))
                    self._bc_voxel_penalty.from_numpy(np.array(work_penalty, dtype=np.float32))
                    self._bc_voxel_dom_did.from_numpy(np.array(work_dom_did, dtype=np.int32))
                    self._bc_voxel_pair.from_numpy(np.array(work_pair, dtype=np.int32))
                    self._batched_voxel_count = total
                    print(f"  Batched VoxelMap: {len(voxel_contacts)} pairs → 1 kernel " f"({total} work items)")
            else:
                for c, idx in voxel_contacts:
                    unbatched.append(c)
                    unbatched_pair_ids.append((int(self.contactPairIds[idx][0]), int(self.contactPairIds[idx][1])))
                print(f"  VoxelMap: {len(voxel_domains)} unique domains, keeping {len(voxel_contacts)} unbatched")

        self._unbatched_contacts = unbatched
        self._unbatched_contacts_count = len(unbatched)
        self._unbatched_pair_ids = unbatched_pair_ids
        if self._unbatched_contacts_count > 0:
            self._bc_unbatched_pair_a, self._bc_unbatched_pair_b, self._bc_unbatched_active = (
                self._create_pair_activation_fields(self._unbatched_pair_ids)
            )

    # ── Pair activation helpers ────────────────────────────────────

    def _create_pair_activation_fields(self, pair_domain_ids):
        """Create Taichi fields (pair_a, pair_b, active) for BVH pair activation."""
        n = len(pair_domain_ids)
        pair_a = ti.field(ti.i32, shape=n)
        pair_b = ti.field(ti.i32, shape=n)
        active = ti.field(ti.i32, shape=n)
        pair_a.from_numpy(np.array([p[0] for p in pair_domain_ids], dtype=np.int32))
        pair_b.from_numpy(np.array([p[1] for p in pair_domain_ids], dtype=np.int32))
        active.fill(0)
        return pair_a, pair_b, active

    def _activate_or_fill_pairs(self, pairs_field, num_pairs_work, pair_a, pair_b, active):
        """Activate/deactivate contact pairs based on BVH collision results."""
        n = active.shape[0]
        if not self.skip_bvh:
            if num_pairs_work > 0:
                self._activate_pairs_kernel(pairs_field, num_pairs_work, pair_a, pair_b, active, n)
            else:
                active.fill(0)
        else:
            active.fill(1)

    @ti.kernel
    def _activate_pairs_by_aabb_kernel(
        self, pair_a: ti.template(), pair_b: ti.template(), active: ti.template(), num_contact_pairs: ti.i32
    ):
        """Activate pair when current domain AABBs overlap (for skip_bvh mode)."""
        for i in range(num_contact_pairs):
            a = pair_a[i]
            b = pair_b[i]
            is_active = 1
            if 0 <= a < self._max_domains and 0 <= b < self._max_domains:
                for k in ti.static(range(self.d)):
                    if self.aabb[a, 1][k] < self.aabb[b, 0][k] or self.aabb[b, 1][k] < self.aabb[a, 0][k]:
                        is_active = 0
            active[i] = is_active

    @ti.kernel
    def _activate_pairs_kernel(
        self,
        pairs: ti.template(),
        num_pairs: ti.i32,
        pair_a: ti.template(),
        pair_b: ti.template(),
        active: ti.template(),
        num_contact_pairs: ti.i32,
    ):
        """Unified BVH pair activation: marks which contact pairs overlap in BVH."""
        for i in range(num_contact_pairs):
            active[i] = 0
        for p in range(num_pairs):
            a = pairs[p][0]
            b = pairs[p][1]
            pa = ti.max(a, b)
            pb = ti.min(a, b)
            for i in range(num_contact_pairs):
                if pair_a[i] == pa and pair_b[i] == pb:
                    active[i] = 1

    @ti.kernel
    def _build_active_rigid_mesh_workset(self, total_mesh_items: ti.i32):
        """Compact non-tied rigid-mesh work items to an active set.

        Check order (cheapest first):
        0. Per-node prefilter (single int compare, skips all pairs for far nodes)
        1. Pair active check
        2. Tie resolved override
        3. Skip epoch check (SH-based, persists until next SH rebuild)
        4. Per-rigid AABB check (only for nodes that passed prefilter)
        """
        self._bc_rigid_mesh_active_count[None] = 0
        for i in range(total_mesh_items):
            wm = self._bc_rigid_mesh_idx[i]

            # Cheapest check first: per-node prefilter rejects nodes
            # not near any rigid AABB (1 int compare vs 9 AABB tests).
            nid = self._bc_rigid_node[wm]
            tied = self._bc_rigid_tied[wm]

            # This workset is only for ordinary non-tied contact.
            if tied == 1:
                continue

            if self._bc_rigid_node_near_any[nid] == 0:
                self._clear_rigid_mesh_friction_state(i)
                continue

            pair_idx = self._bc_rigid_pair[wm]
            keep = self._bc_rigid_active[pair_idx] != 0

            # Skip-epoch check: SH returned 0 candidates and SH has not
            # been rebuilt since. If it passes, do the per-rigid AABB check.
            if keep:
                if self._bc_rigid_skip_epoch[i] == 1:
                    keep = False
                else:
                    rid = self._bc_rigid_did[wm]
                    if self._bc_rigid_use_node_mask[None] == 1 and rid < 31:
                        if (self._bc_rigid_node_mask[nid] & (1 << rid)) == 0:
                            keep = False
                    else:
                        rigid_domain_idx = self.rigidManager.rigidDomainIds[rid][0]
                        node_coord = self.femSpringManager.coords[nid]
                        margin = 1e-4
                        rigid_lb = self.aabb[rigid_domain_idx, 0] - margin
                        rigid_ub = self.aabb[rigid_domain_idx, 1] + margin

                        if (node_coord - rigid_lb).min() < 0.0 or (rigid_ub - node_coord).min() < 0.0:
                            keep = False

            if keep:
                dst = ti.atomic_add(self._bc_rigid_mesh_active_count[None], 1)
                self._bc_rigid_mesh_active_idx[dst] = i
            else:
                self._clear_rigid_mesh_friction_state(i)

    @ti.kernel
    def _compute_fem_contact_aabb(self, num_nodes: ti.i32):
        """Compute the bounding box of all FEM nodes in rigid mesh contact."""
        for k in ti.static(range(self.d)):
            self._fem_contact_aabb_lb[None][k] = ti.f32(1e9)
            self._fem_contact_aabb_ub[None][k] = ti.f32(-1e9)
        for i in range(num_nodes):
            nid = self._bc_rigid_mesh_nodes[i]
            coord = self.femSpringManager.coords[nid]
            for k in ti.static(range(self.d)):
                ti.atomic_min(self._fem_contact_aabb_lb[None][k], coord[k])
                ti.atomic_max(self._fem_contact_aabb_ub[None][k], coord[k])

    @ti.kernel
    def _prefilter_mesh_node_activity(self):
        """Mark FEM nodes inside at least one rigid's expanded AABB.

        Runs over unique mesh nodes (37K) instead of all work items (336K).
        Each thread checks the node against all rigids (9) and writes 1
        if the node is near any.  The workset builder then skips
        all pairs for unmarked nodes with a single integer comparison.
        """
        for i in range(self._bc_rigid_mesh_node_count[None]):
            nid = self._bc_rigid_mesh_nodes[i]
            coord = self.femSpringManager.coords[nid]
            near_any = False
            for j in range(self._bc_rigid_mesh_rigid_count[None]):
                if not near_any:
                    rid = self._bc_rigid_mesh_rigid_ids[j]
                    domain_idx = self.rigidManager.rigidDomainIds[rid][0]
                    m = self._bc_rigid_mesh_rigid_margin[j]
                    r_lb = self.aabb[domain_idx, 0] - m
                    r_ub = self.aabb[domain_idx, 1] + m
                    if (coord - r_lb).min() >= 0.0 and (r_ub - coord).min() >= 0.0:
                        near_any = True
            if near_any:
                self._bc_rigid_node_near_any[nid] = 1

    @ti.kernel
    def _batched_analytical_contact_kernel(self, total: ti.i32):
        """Single kernel for ALL FlexAnalytical contact pairs.
        Processes every boundary-node–plane work item in parallel."""
        for w in range(total):
            pair_idx = self._bc_anal_pair[w]
            if self._bc_anal_active[pair_idx] == 0:
                continue
            nid = self._bc_anal_node[w]
            nodeCoord = self.femSpringManager.coords[nid]
            pp = self._bc_anal_pp[w]
            pn = self._bc_anal_pn[w]
            pen = (nodeCoord - pp).dot(pn)
            if pen < 0.0:
                self.femSpringManager.Fext[nid] -= pn * self._bc_anal_penalty[w] * pen

    # @ti.func
    # def _eval_fixed_fem_fem_tie_point(self, elem_idx: ti.i32, weights):
    #     """Evaluate the fixed material point on a FEM boundary element."""
    #     conn = self.femSpringManager.boundaryElements[elem_idx]
    #     cp = ti.Vector.zero(ti.f32, ti.static(self.d))
    #     for a in ti.static(range(self.d)):
    #         cp += self.femSpringManager.coords[conn[a]] * weights[a]
    #     return cp

    @ti.func
    def _eval_fixed_tie_point(
        self,
        elem_idx: ti.i32,
        weights,
        boundary_elements: ti.template(),
        boundary_coords: ti.template(),
    ):
        conn = boundary_elements[elem_idx]
        cp = ti.Vector.zero(ti.f32, ti.static(self.d))

        for a in ti.static(range(self.d)):
            cp += boundary_coords[conn[a]] * weights[a]

        return cp

    @ti.func
    def _fem_fem_tie_elem_normal(self, elem_idx: ti.i32):
        conn = self.femSpringManager.boundaryElements[elem_idx]
        p0 = self.femSpringManager.coords[conn[0]]
        p1 = self.femSpringManager.coords[conn[1]]
        n = ti.Vector.zero(ti.f32, ti.static(self.d))

        t = (p1 - p0).normalized(1e-10)
        n = ti.Vector([-t[1], t[0]])

        return n

    @ti.kernel
    def _initialize_fem_fem_ties_once_kernel(self, total: ti.i32):
        """
        The one-time tied-contact initialization uses the same geometric
        detection criterion as ordinary FEM-FEM contact: a slave boundary node
        must project inside a target boundary element and satisfy
        """
        for w in range(total):
            if self._bc_flex_tied[w] == 0:
                continue
            if self._bc_flex_tie_resolved[w] == 1:
                continue

            nid = self._bc_flex_node[w]
            node_coord = self.femSpringManager.coords[nid]
            be_off = self._bc_flex_be_off[w]
            be_cnt = self._bc_flex_be_cnt[w]
            target_did = self._bc_flex_target_did[w]
            contact_kind = self._bc_flex_kind[w]

            best_pen = ti.f32(1e9)
            best_elem = -1
            best_weights = ti.Vector.zero(ti.f32, ti.static(self.d))
            found = False

            # One-time narrow phase.  No BVH active-pair mask is used here:
            # tied pairs must be tested exactly once in the initial state.
            min_buf = self.femSpringManager.spatialHash.gridSize[None].max()
            query_buf = ti.max(ti.max(1e-4, min_buf), 0.001)  # TODO: why 0.016?
            print("Check query_buf: ", query_buf)
            potentialEls, dids, numPotentials = self.femSpringManager.spatialHash.queryPointWithBuffer(
                node_coord, query_buf, target_did
            )
            for j in range(numPotentials):
                elem_idx = potentialEls[j]
                elem_conn = self.femSpringManager.boundaryElements[elem_idx]
                pen, normal, cp, curr_weights = detectPointToMeshBoundary(
                    node_coord, self.femSpringManager.coords, elem_conn, limit_penetration=query_buf
                )

                if ti.abs(pen) < ti.abs(best_pen) and ti.abs(pen) < query_buf:
                    best_weights = curr_weights
                    found = True
                    best_elem = elem_idx
                    best_pen = pen
                # elif pen < best_pen:
                #     print("Check this pen:", pen)

            if found:
                cpoint = self._eval_fixed_tie_point(
                    best_elem,
                    best_weights,
                    self.femSpringManager.boundaryElements,
                    self.femSpringManager.coords,
                )
                gap = node_coord - cpoint
                normal = self._fem_fem_tie_elem_normal(best_elem)

                self._bc_flex_tie_resolved[w] = 1
                self._bc_flex_tie_elem[w] = best_elem
                self._bc_flex_tie_weights[w] = best_weights
                self._bc_flex_tie_gap[w] = gap.dot(normal)
                ti.atomic_add(self._fixed_flexflex_tie_found_count[None], 1)
            else:
                print("check node : ", nid, "coords: ", node_coord)

    @ti.kernel
    def _apply_fem_fem_ties_fixed_kernel(self, total: ti.i32):
        """Apply fixed FEM-FEM tied-contact forces using cached initial ties."""
        for w in range(total):
            if self._bc_flex_tied[w] == 0:
                continue
            if self._bc_flex_tie_resolved[w] == 0:
                continue

            nid = self._bc_flex_node[w]
            elem_idx = self._bc_flex_tie_elem[w]
            weights = self._bc_flex_tie_weights[w]
            gap = self._bc_flex_tie_gap[w]
            penalty = self._bc_flex_penalty[w]

            # Apply one fixed FEM-FEM tied-contact penalty force.
            cpoint = self._eval_fixed_tie_point(
                elem_idx,
                weights,
                self.femSpringManager.boundaryElements,
                self.femSpringManager.coords,
            )
            normal = self._fem_fem_tie_elem_normal(elem_idx)
            target = cpoint + gap * normal
            move = target - self.femSpringManager.coords[nid]
            total_force = penalty * move
            self.femSpringManager.Fext[nid] += total_force
            conn = self.femSpringManager.boundaryElements[elem_idx]
            for a in ti.static(range(self.d)):
                self.femSpringManager.Fext[conn[a]] -= total_force * weights[a]

    @ti.kernel
    def _batched_flexflex_contact_kernel(self, total: ti.i32):
        """Single kernel for ALL FlexFlex contact pairs (brute-force path).
        Each work item checks one boundary node against ALL target boundary elements."""
        for w in range(total):
            pair_idx = self._bc_flex_pair[w]
            if self._bc_flex_active[pair_idx] == 0:
                continue
            target_did = self._bc_flex_target_did[w]
            if (
                self._use_bvh_domain_mask[None] == 1
                and self._bvh_domain_stamp[target_did] != self._bvh_active_stamp[None]
            ):
                continue

            nid = self._bc_flex_node[w]
            penalty = self._bc_flex_penalty[w]
            be_off = self._bc_flex_be_off[w]
            be_cnt = self._bc_flex_be_cnt[w]
            friction_coeff = self._bc_flex_friction[w]

            nodeCoord = self.femSpringManager.coords[nid]
            node_vel = self.femSpringManager.V[nid]

            best_pen = ti.f32(1e9)
            best_normal = ti.Vector.zero(ti.f32, ti.static(self.d))
            best_weights = ti.Vector.zero(ti.f32, ti.static(self.d))
            best_elem_conn = ti.Vector.zero(ti.i32, ti.static(self.d))
            found = False

            for j in range(be_cnt):
                elem_conn = self.femSpringManager.boundaryElements[be_off + j]
                pen = ti.f32(1e9)
                normal = ti.Vector.zero(ti.f32, ti.static(self.d))
                is_inside = False

                n0 = self.femSpringManager.coords[elem_conn[0]]
                n1 = self.femSpringManager.coords[elem_conn[1]]
                pen, normal, cp, is_inside, weights = pointToEdgeContact(nodeCoord, n0, n1, self.d)
               
                if pen < best_pen and ti.abs(pen) < 1.0 and is_inside:
                    best_pen = pen
                    best_normal = normal
                    best_weights = weights
                    best_elem_conn = elem_conn
                    found = True

            if found and best_pen < 0.0:
                normal_force = -best_normal * penalty * best_pen
                total_force = normal_force
                if friction_coeff > 1e-9:
                    surf_vel = ti.Vector.zero(ti.f32, ti.static(self.d))
                    for k in ti.static(range(self.d)):
                        surf_vel += self.femSpringManager.V[best_elem_conn[k]] * best_weights[k]
                    relative_vel = node_vel - surf_vel
                    tangential_vel = relative_vel - relative_vel.dot(best_normal) * best_normal
                    if tangential_vel.norm() > 1e-9:
                        friction_dir = -tangential_vel.normalized()
                        total_force += friction_dir * friction_coeff * normal_force.norm()
                self.femSpringManager.Fext[nid] += total_force

    @ti.kernel
    def _batched_flexflex_contact_kernel_sh(self, total: ti.i32):
        """Spatial-hash-accelerated FlexFlex contact kernel with element cache.

        Phase 1:
        Retest the last-hit target boundary element stored in _bc_flex_cache_elem[w].

        Phase 2:
        If cache miss, query FEM spatial hash for nearby target elements.

        Notes:
        - Cache stores GLOBAL boundary-element indices.
        - Keep the current force-transfer logic:
                Fext[nid] += total_force
                Fext[target element nodes] -= total_force * weights
        - No rigid-style lever arm / rotational impulse calculation.
        """
        for w in range(total):
            if self._bc_flex_tied[w] == 1:
                continue
            pair_idx = self._bc_flex_pair[w]
            if self._bc_flex_active[pair_idx] == 0:
                continue

            if (
                self._use_bvh_domain_mask[None] == 1
                and self._bvh_domain_stamp[self._bc_flex_target_did[w]] != self._bvh_active_stamp[None]
            ):
                continue

            nid = self._bc_flex_node[w]
            target_did = self._bc_flex_target_did[w]
            friction_coeff = self._bc_flex_friction[w]
            be_off = self._bc_flex_be_off[w]
            be_cnt = self._bc_flex_be_cnt[w]
            contact_kind = self._bc_flex_kind[w]
            nodeCoord = self.femSpringManager.coords[nid]
            node_vel = self.femSpringManager.V[nid]
            penalty = self._bc_flex_penalty[w]

            min_buf = self.femSpringManager.spatialHash.gridSize[None].max()
            query_buf = ti.max(ti.max(1e-4, min_buf), 0.016)  # TODO: why this 0.016

            best_pen = ti.f32(1e9)
            best_normal = ti.Vector.zero(ti.f32, ti.static(self.d))
            weights = ti.Vector.zero(ti.f32, self.d)  # barycentric weights for force distribution
            elem_conn = ti.Vector.zero(ti.i32, 3)  # max 3 nodes per element (triangles)
            found = False

            # -------------------------------------------------------------
            # Phase 1: Check cached element
            # -------------------------------------------------------------
            cached_elem = self._bc_flex_cache_elem[w]

            # Only trust cache if it still belongs to this work item's target range.
            if cached_elem >= be_off and cached_elem < be_off + be_cnt:
                curr_conn = self.femSpringManager.boundaryElements[cached_elem]
                # Rebuild AABB of cached element from current node positions
                lb = ti.Vector.zero(ti.f32, ti.static(self.d)) + 1e30
                ub = ti.Vector.zero(ti.f32, ti.static(self.d)) - 1e30
                for k in ti.static(range(self.d)):
                    coord = self.femSpringManager.coords[curr_conn[k]]
                    lb = ti.min(lb, coord)
                    ub = ti.max(ub, coord)
                margin = ti.max(1e-4, 1e-2 * (ub - lb).norm())
                lb -= margin
                ub += margin

                # Cache bbox miss -> DO NOT continue outer loop.
                # Just treat it as cache miss and fall through to SH fallback.
                if (nodeCoord - lb).min() >= 0.0 and (ub - nodeCoord).min() >= 0.0:
                    pen, normal, cp, curr_weights = detectPointToMeshBoundary(
                        nodeCoord, self.femSpringManager.coords, curr_conn, limit_penetration=margin
                    )

                    if pen < 1e-4 and ti.abs(pen) < margin:
                        # solid-solid
                        best_pen = pen
                        best_normal = normal
                        weights = curr_weights
                        elem_conn = curr_conn
                        found = True

            # -------------------------------------------------------------
            # Phase 2: cache miss -> SH fallback
            # -------------------------------------------------------------
            if not found:
                potentialEls, dids, numPotentials = self.femSpringManager.spatialHash.queryPointWithBuffer(
                    nodeCoord, query_buf, target_did
                )

                for j in range(numPotentials):
                    global_elem_idx = potentialEls[j]
                    curr_conn = self.femSpringManager.boundaryElements[global_elem_idx]
                    # lb = ti.Vector.zero(ti.f32, ti.static(self.d)) + 1e30
                    # ub = ti.Vector.zero(ti.f32, ti.static(self.d)) - 1e30
                    # for k in ti.static(range(self.d)):
                    #     coord = self.femSpringManager.coords[curr_conn[k]]
                    #     lb = ti.min(lb, coord)
                    #     ub = ti.max(ub, coord)
                    pen, normal, cp, curr_weights = detectPointToMeshBoundary(
                        nodeCoord, self.femSpringManager.coords, curr_conn, limit_penetration=query_buf
                    )  # If query_buf is changed to margin, the test_3Dfem_contact case fails.

                    if pen < best_pen and pen < 1e-4 and ti.abs(pen) < query_buf:
                        best_pen = pen
                        best_normal = normal
                        weights = curr_weights
                        elem_conn = curr_conn
                        found = True

            # If both cache and SH miss, invalidate cache
            if not found:
                self._bc_flex_cache_elem[w] = -1

            if found and best_pen < 0.0:
                normal_force = -best_normal * penalty * best_pen
                total_force = normal_force
                if friction_coeff > 1e-9:
                    surf_vel = ti.Vector.zero(ti.f32, ti.static(self.d))
                    for k in ti.static(range(self.d)):
                        surf_vel += self.femSpringManager.V[elem_conn[k]] * weights[k]
                    relative_vel = node_vel - surf_vel
                    tangential_vel = relative_vel - relative_vel.dot(best_normal) * best_normal
                    if tangential_vel.norm() > 1e-9:
                        friction_dir = -tangential_vel.normalized()
                        total_force += friction_dir * friction_coeff * normal_force.norm()
                self.femSpringManager.Fext[nid] += total_force
                for i in range(self.d):
                    self.femSpringManager.Fext[elem_conn[i]] -= total_force * weights[i]

    # ── FEM-rigid split kernels: mesh and primitive compiled separately ──
    # Each kernel only contains one code path, reducing JIT compilation time.

    @ti.kernel
    def _batched_rigid_prim_kernel(self, num_prim: ti.i32):
        """FEM-rigid contact for primitive rigids only (no mesh code compiled)."""
        for wp in range(num_prim):
            w = self._bc_rigid_prim_idx[wp]
            pair_idx = self._bc_rigid_pair[w]
            if self._bc_rigid_active[pair_idx] == 0:
                continue
            nid = self._bc_rigid_node[w]
            rigid_idx = self._bc_rigid_idx[w]
            tied = self._bc_rigid_tied[w]

            lb = self.aabb[self.rigidManager.rigidDomainIds[rigid_idx][0], 0]
            ub = self.aabb[self.rigidManager.rigidDomainIds[rigid_idx][0], 1]
            nodeCoord = self.femSpringManager.coords[nid]
            node_vel = self.femSpringManager.V[nid]
            center = self.rigidManager.rigidParams[rigid_idx, 0]
            not_in_bbox = ((nodeCoord - lb).min() < 0.0) or ((nodeCoord - ub).max() > 0.0)
            if not_in_bbox:
                continue

            rigid_type = self.rigidManager.rigidDomainIds[rigid_idx][1]
            prim = self.rigidManager.rigidParams[rigid_idx, 1]
            rot = self.rigidManager.cached_rotation_matrix[rigid_idx]
            radius = self.rigidManager.radius[rigid_idx]

            pen, norm, cp = detectPointToPrimitive(nodeCoord, rigid_type, center, prim, rot, radius)
            margin = (ub - lb).norm() * 0.5

            if tied == 1 and pen < margin:
                penalty = self._bc_rigid_penalty[w]
                move = cp - nodeCoord
                self.femSpringManager.Fext[nid] += penalty * move
                self.rigidManager.accumulated_impulse[rigid_idx] -= penalty * move * self.dt
                lever = cp - center
                self.rigidManager.accumulated_rotational_impulse[rigid_idx][0] += (
                    lever[0] * (-penalty * move)[1] - lever[1] * (-penalty * move)[0]
                ) * self.dt
            
            elif pen < margin:
                penetration = pen - 1e-4
                if penetration < 0.0:
                    penalty = self._bc_rigid_penalty[w]
                    friction_coeff = self._bc_rigid_friction[w]

                    normal_force = -norm * penalty * penetration
                    total_force = normal_force

                    if friction_coeff > 1e-9:
                        rigid_vel = self.rigidManager.V[rigid_idx]
                        rigid_omega = self.rigidManager.RotV[rigid_idx]
                        surface_vel = ti.Vector.zero(ti.f32, ti.static(self.d))
                        r_contact = cp - center
                        surface_vel = rigid_vel + ti.Vector(
                            [-rigid_omega[0] * r_contact[1], rigid_omega[0] * r_contact[0]]
                        )
                      
                        relative_vel = node_vel - surface_vel
                        tangential_vel = relative_vel - relative_vel.dot(norm) * norm
                        if tangential_vel.norm() > 1e-9:
                            friction_dir = -tangential_vel.normalized()
                            total_force += friction_dir * friction_coeff * normal_force.norm()

                    self.femSpringManager.Fext[nid] += total_force
                    self.rigidManager.accumulated_impulse[rigid_idx] -= total_force * self.dt

                    lever = cp - center
                    self.rigidManager.accumulated_rotational_impulse[rigid_idx][0] += (
                            lever[0] * (-total_force)[1] - lever[1] * (-total_force)[0]
                        ) * self.dt

    @ti.func
    def _point_aabb_distance_sq(self, p, lb, ub):
        """Lower bound squared distance from point to AABB."""
        dist2 = ti.f32(0.0)
        for d in ti.static(range(self.d)):
            v = ti.f32(0.0)
            if p[d] < lb[d]:
                v = lb[d] - p[d]
            elif p[d] > ub[d]:
                v = p[d] - ub[d]
            dist2 += v * v
        return dist2

    @ti.func
    def _cell_id_to_bounds(self, cid):
        """Decode cell id to world-space cell AABB."""
        grid_lb = self.rigidManager.spatialHash.globalbbox[0]
        grid_sz = self.rigidManager.spatialHash.gridSize[None]
        cn = self.rigidManager.spatialHash.cellNumbers[None]
        cell_lb = ti.Vector.zero(ti.f32, ti.static(self.d))
        cell_ub = ti.Vector.zero(ti.f32, ti.static(self.d))
        nx = cn[0]
        ix = cid % nx
        iy = cid // nx
        cell_lb = grid_lb + ti.Vector([ti.cast(ix, ti.f32) * grid_sz[0], ti.cast(iy, ti.f32) * grid_sz[1]])
        cell_ub = cell_lb + grid_sz
       
        return cell_lb, cell_ub

    @ti.func
    def _clear_rigid_mesh_friction_state(self, i: ti.i32):
        self._bc_rigid_fric_prev_elem[i] = -1
        self._bc_rigid_fric_prev_valid[i] = 0
        self._bc_rigid_fric_prev_weights[i] = ti.Vector.zero(ti.f32, ti.static(self.d))
        self._bc_rigid_fric_prev_force[i] = ti.Vector.zero(ti.f32, ti.static(self.d))
        self._bc_rigid_fric_prev_penetration[i] = 0.0

    # @ti.func
    # def _eval_fixed_rigid_mesh_tie_point(self, elem_idx: ti.i32, weights):
    #     conn = self.rigidManager.meshBoundaryElements[elem_idx]
    #     cp = ti.Vector.zero(ti.f32, ti.static(self.d))

    #     for a in ti.static(range(self.d)):
    #         cp += self.rigidManager.meshBoundaryCoords[conn[a]] * weights[a]

    #     return cp

    @ti.func
    def _apply_one_fixed_rigid_mesh_tie(self, nid: ti.i32, rigid_idx: ti.i32, cpoint, penalty: ti.f32):
        node_coord = self.femSpringManager.coords[nid]
        move = cpoint - node_coord

        # FEM side
        self.femSpringManager.Fext[nid] += penalty * move

        # Rigid linear impulse
        self.rigidManager.accumulated_impulse[rigid_idx] -= penalty * move * self.dt

        # Rigid angular impulse
        lever = cpoint - self.rigidManager.rigidParams[rigid_idx, 0]
        self.rigidManager.accumulated_rotational_impulse[rigid_idx][0] += (
            lever[0] * (-penalty * move)[1] - lever[1] * (-penalty * move)[0]
        ) * self.dt

    @ti.func
    def _point_to_cell_coord(self, pos):
        """Map world point to clamped grid cell coordinate."""
        rel = ti.floor(
            (pos - self.rigidManager.spatialHash.globalbbox[0]) / self.rigidManager.spatialHash.gridSize[None]
        ).cast(ti.i32)
        rel = ti.max(rel, ti.Vector.zero(ti.i32, ti.static(self.d)))
        rel = ti.min(rel, self.rigidManager.spatialHash.cellNumbers[None] - 1)
        return rel

    @ti.func
    def _process_rigid_sh_cell(
        self,
        cid,
        nodeCoord,
        target_rigid_did,
        penetration,
        normal,
        cpoint,
        weights,
        elem_idx,
        rigid_did,
        found,
        i_a,
        has_cand,
    ):
        """Test all elements in one SH cell against a query node (inlined)."""
        if 0 <= cid < self.rigidManager.spatialHash.total_cells[None]:
            skip_cell = False
            # Cell-level pruning: if this cell's AABB is already farther than
            # current best |penetration| lower-bound, skip entire cell.
            best_abs = ti.abs(penetration)
            if (not skip_cell) and found and best_abs < 1e8:
                c_lb, c_ub = self._cell_id_to_bounds(cid)
                if self._point_aabb_distance_sq(nodeCoord, c_lb, c_ub) > best_abs * best_abs:
                    skip_cell = True

            if not skip_cell:
                for p in range(
                    self.rigidManager.spatialHash.cellStart[cid], self.rigidManager.spatialHash.cellEnd[cid]
                ):
                    ei = self.rigidManager.spatialHash._sortedElemIdx[p]
                    if ei >= 0:
                        global_eidx = self.rigidManager.spatialHash.domainIds[ei][1]
                        did = self.rigidManager.spatialHash.domainIds[ei][0]
                        if did != target_rigid_did:
                            continue
                        has_cand = True
                        conn = self.rigidManager.meshBoundaryElements[global_eidx]
                        el_lb = ti.Vector.zero(ti.f32, ti.static(self.d)) + 1e30
                        el_ub = ti.Vector.zero(ti.f32, ti.static(self.d)) - 1e30
                        for vj in ti.static(range(self.d)):
                            c = self.rigidManager.meshBoundaryCoords[conn[vj]]
                            el_lb = ti.min(el_lb, c)
                            el_ub = ti.max(el_ub, c)
                        margin = ti.max(1e-4, self.rigidManager.meshElemMarginBase[global_eidx])
                        el_lb -= margin
                        el_ub += margin

                        # Sweep-and-prune style pruning using the current best
                        # |penetration| as a dynamic search radius lower bound.
                        best_abs = ti.abs(penetration)
                        if found and best_abs < 1e8:
                            if self._point_aabb_distance_sq(nodeCoord, el_lb, el_ub) > best_abs * best_abs:
                                continue

                        if not ((nodeCoord - el_lb).min() < 0.0 or (el_ub - nodeCoord).min() < 0.0):
                            conn = self.rigidManager.meshBoundaryElements[global_eidx]
                            pen, norm, cp, curr_weights = detectPointToMeshBoundary(
                                nodeCoord, self.rigidManager.meshBoundaryCoords, conn, limit_penetration=margin
                            )
                            if ti.abs(pen) < ti.abs(penetration):
                                penetration = pen
                                normal = norm
                                cpoint = cp
                                weights = curr_weights
                                elem_idx = global_eidx
                                rigid_did = did
                                found = True
                                self._bc_rigid_cache_elem[i_a] = [global_eidx, did]
        return penetration, normal, cpoint, weights, elem_idx, rigid_did, found, has_cand

    @ti.kernel
    def _batched_rigid_mesh_kernel_sh(self, run_full: ti.i32):
        """FEM-rigid contact for mesh rigids with spatial hash (no primitive code compiled).

        active_count_field is read on GPU (no CPU sync).  Threads beyond the
        active count exit immediately.
        """
        for k in range(self._batched_rigid_mesh_count):
            if k >= self._bc_rigid_mesh_active_count[None]:
                continue
            i_a = self._bc_rigid_mesh_active_idx[k]
            wm = self._bc_rigid_mesh_idx[i_a]
            nid = self._bc_rigid_node[wm]

            penalty = self._bc_rigid_penalty[wm]
            friction_coeff = self._bc_rigid_friction[wm]
            s_normal = self._bc_rigid_normals[wm]
            target_rigid_did = self._bc_rigid_did[wm]
            tied = self._bc_rigid_tied[wm]

            # This kernel only handles ordinary non-tied contact.
            if tied == 1:
                continue

            nodeCoord = self.femSpringManager.coords[nid]
            node_vel = self.femSpringManager.V[nid]

            # AABB reject is handled by active-workset builder for all
            # non-tied items.  No need to duplicate here.

            rigid_did = 0
            min_buf = self.rigidManager.spatialHash.gridSize[None].max()
            query_buf_small = ti.max(1e-4, min_buf)
            penetration = ti.f32(1e9)
            normal = ti.Vector.zero(ti.f32, ti.static(self.d))
            cpoint = ti.Vector.zero(ti.f32, ti.static(self.d))
            curr_weights = ti.Vector.zero(ti.f32, ti.static(self.d))
            curr_elem = -1
            found = False
            # print("Query buffer size:", query_buf_small)

            # Phase 1: Check cached element
            cached_el = self._bc_rigid_cache_elem[i_a][0]
            cached_did = self._bc_rigid_cache_elem[i_a][1]
            if cached_el >= 0 and cached_did == target_rigid_did:
                lb = self.rigidManager.meshElemLB[cached_el]
                ub = self.rigidManager.meshElemUB[cached_el]
                margin = ti.max(
                    1e-4, self.rigidManager.meshElemMarginBase[cached_el]
                )  # TODO: why 5.0 and why 0.02 of element size as margin!
                # print("margin for cached element:", margin)
                lb -= margin
                ub += margin
                if not ((nodeCoord - lb).min() < 0.0 or (ub - nodeCoord).min() < 0.0):
                    conn = self.rigidManager.meshBoundaryElements[cached_el]
                    pen, norm, cp, weights = detectPointToMeshBoundary(
                        nodeCoord, self.rigidManager.meshBoundaryCoords, conn, limit_penetration=margin
                    )
                    if ti.abs(pen) < ti.abs(penetration):  # additional normal check to reduce false positives
                        penetration = pen
                        normal = norm
                        cpoint = cp
                        curr_weights = weights
                        curr_elem = cached_el
                        rigid_did = cached_did
                        found = True
            # phase1 end

            # Phase 2: Inline SH cell iteration (skip if negative-cache epoch match)
            if not found and self._bc_rigid_skip_epoch[i_a] == 1:
                # print("Skipping SH query for work item", i_a, "due to negative cache hit")
                self._clear_rigid_mesh_friction_state(i_a)
                continue
            if not found and run_full == 0:
                # print("Skipping SH query for work item", i_a, "due to early-out optimization")
                self._clear_rigid_mesh_friction_state(i_a)
                continue
            if not found:
                sh_has_cand = False
                if self.rigidManager.spatialHash.total_cells[None] > 0:
                    qlb = nodeCoord - query_buf_small
                    qub = nodeCoord + query_buf_small
                    sh_lpos = ti.floor(
                        (qlb - self.rigidManager.spatialHash.globalbbox[0])
                        / self.rigidManager.spatialHash.gridSize[None]
                    ).cast(ti.i32)
                    sh_upos = ti.floor(
                        (qub - self.rigidManager.spatialHash.globalbbox[0])
                        / self.rigidManager.spatialHash.gridSize[None]
                    ).cast(ti.i32)
                    sh_lpos = ti.max(sh_lpos, ti.Vector.zero(ti.i32, ti.static(self.d)))
                    sh_lpos = ti.min(sh_lpos, self.rigidManager.spatialHash.cellNumbers[None] - 1)
                    sh_upos = ti.max(sh_upos, ti.Vector.zero(ti.i32, ti.static(self.d)))
                    sh_upos = ti.min(sh_upos, self.rigidManager.spatialHash.cellNumbers[None] - 1)

                    # Prefer scanning a local neighborhood around the last-hit
                    # cached element cell before sweeping the entire query box.
                    use_local_first = 0
                    hint_lpos = ti.Vector.zero(ti.i32, ti.static(self.d))
                    hint_upos = ti.Vector.zero(ti.i32, ti.static(self.d))
                    if cached_el >= 0 and cached_did == target_rigid_did:
                        use_local_first = 1
                        hint_center = 0.5 * (
                            self.rigidManager.meshElemLB[cached_el] + self.rigidManager.meshElemUB[cached_el]
                        )
                        hint_cell = self._point_to_cell_coord(hint_center)
                        hint_lpos = ti.max(hint_cell, sh_lpos)
                        hint_upos = ti.min(hint_cell, sh_upos)

                    if use_local_first == 1:
                        for I in ti.grouped(
                            ti.ndrange((hint_lpos[0], hint_upos[0] + 1), (hint_lpos[1], hint_upos[1] + 1))
                        ):
                            cid = I[0] + I[1] * self.rigidManager.spatialHash.cellNumbers[None][0]
                            penetration, normal, cpoint, curr_weights, curr_elem, rigid_did, found, sh_has_cand = (
                                self._process_rigid_sh_cell(
                                    cid,
                                    nodeCoord,
                                    target_rigid_did,
                                    penetration,
                                    normal,
                                    cpoint,
                                    curr_weights,
                                    curr_elem,
                                    rigid_did,
                                    found,
                                    i_a,
                                    sh_has_cand,
                                )
                            )

                    for I in ti.grouped(ti.ndrange((sh_lpos[0], sh_upos[0] + 1), (sh_lpos[1], sh_upos[1] + 1))):
                        if use_local_first == 1:
                            if (
                                I[0] >= hint_lpos[0]
                                and I[0] <= hint_upos[0]
                                and I[1] >= hint_lpos[1]
                                and I[1] <= hint_upos[1]
                            ):
                                continue
                        cid = I[0] + I[1] * self.rigidManager.spatialHash.cellNumbers[None][0]
                        penetration, normal, cpoint, curr_weights, curr_elem, rigid_did, found, sh_has_cand = (
                            self._process_rigid_sh_cell(
                                cid,
                                nodeCoord,
                                target_rigid_did,
                                penetration,
                                normal,
                                cpoint,
                                curr_weights,
                                curr_elem,
                                rigid_did,
                                found,
                                i_a,
                                sh_has_cand,
                            )
                        )

                if not sh_has_cand:
                    self._bc_rigid_skip_epoch[i_a] = 1
            # phase2 end

            if found and penetration < 0.0:
                # rigid_did comes from SpatialHash.domainIds[][0], which stores rigid index.
                rigid_idx = rigid_did
                center = self.rigidManager.rigidParams[rigid_idx, 0]
                normal_force = -normal * penalty * penetration
                total_force = normal_force
                friction_force = ti.Vector.zero(ti.f32, ti.static(self.d))

                if friction_coeff > 1e-9:
                    # Use the normal force at this time step as the basis for calculating the yield friction force.LSDYNAdifferent
                    Fy = friction_coeff * normal_force.norm()
                    # It can be further considered that the penalty parameters in the tangential direction and the normal direction are different.
                    k_t = penalty * 1

                    prev_force = ti.Vector.zero(ti.f32, ti.static(self.d))
                    slip = ti.Vector.zero(ti.f32, ti.static(self.d))
                    if self._bc_rigid_fric_prev_valid[i_a] == 1:
                        prev_elem = self._bc_rigid_fric_prev_elem[i_a]
                        prev_conn = self.rigidManager.meshBoundaryElements[prev_elem]
                        prev_weights = self._bc_rigid_fric_prev_weights[i_a]
                        curr_conn = self.rigidManager.meshBoundaryElements[curr_elem]

                        prev_surface_point = ti.Vector.zero(ti.f32, ti.static(self.d))
                        curr_surface_point = ti.Vector.zero(ti.f32, ti.static(self.d))
                        curr_surface_point = (
                            self.rigidManager.meshBoundaryCoords[curr_conn[0]] * curr_weights[0]
                            + self.rigidManager.meshBoundaryCoords[curr_conn[1]] * curr_weights[1]
                        )
                        prev_surface_point = (
                            self.rigidManager.meshBoundaryCoords[prev_conn[0]] * prev_weights[0]
                            + self.rigidManager.meshBoundaryCoords[prev_conn[1]] * prev_weights[1]
                        )
                    
                        slip = curr_surface_point - prev_surface_point
                        slip = slip - slip.dot(normal) * normal
                        prev_force = self._bc_rigid_fric_prev_force[i_a]
                        prev_force = prev_force - prev_force.dot(normal) * normal

                    f_trial = prev_force + k_t * slip
                    f_trial_norm = f_trial.norm()

                    if f_trial_norm <= Fy:
                        friction_force = f_trial
                    else:
                        friction_force = f_trial / f_trial_norm * Fy

                    self._bc_rigid_fric_prev_elem[i_a] = curr_elem
                    self._bc_rigid_fric_prev_valid[i_a] = 1
                    self._bc_rigid_fric_prev_weights[i_a] = curr_weights
                    self._bc_rigid_fric_prev_force[i_a] = friction_force
                    self._bc_rigid_fric_prev_penetration[i_a] = penetration
                    total_force += friction_force
                else:
                    self._clear_rigid_mesh_friction_state(i_a)

                self.femSpringManager.Fext[nid] += total_force
                self.rigidManager.accumulated_impulse[rigid_idx] -= total_force * self.dt

                lever = cpoint - center
                self.rigidManager.accumulated_rotational_impulse[rigid_idx][0] += (
                    lever[0] * (-total_force)[1] - lever[1] * (-total_force)[0]
                ) * self.dt

            else:
                self._clear_rigid_mesh_friction_state(i_a)

                # if not found:
                #     self._bc_rigid_cache_elem[nid] = -1

    @ti.kernel
    def _initialize_fem_rigid_mesh_ties_once_kernel(self, total_mesh_items: ti.i32):
        for i_a in range(total_mesh_items):
            wm = self._bc_rigid_mesh_idx[i_a]

            # Only handle tied FEM-rigid-mesh items.
            if self._bc_rigid_tied[wm] == 0:
                continue

            # Already resolved.
            if self._bc_rigid_tie_resolved[i_a] == 1:
                continue

            nid = self._bc_rigid_node[wm]
            target_rigid_did = self._bc_rigid_did[wm]
            node_coord = self.femSpringManager.coords[nid]

            min_buf = self.rigidManager.spatialHash.gridSize[None].max()
            query_buf = ti.max(1e-4, min_buf)

            penetration = ti.f32(1e9)
            normal = ti.Vector.zero(ti.f32, ti.static(self.d))
            cpoint = ti.Vector.zero(ti.f32, ti.static(self.d))
            weights = ti.Vector.zero(ti.f32, ti.static(self.d))
            elem_idx = -1
            rigid_did = -1
            found = False
            has_cand = False

            if self.rigidManager.spatialHash.total_cells[None] > 0:
                qlb = node_coord - query_buf
                qub = node_coord + query_buf

                sh_lpos = ti.floor(
                    (qlb - self.rigidManager.spatialHash.globalbbox[0]) / self.rigidManager.spatialHash.gridSize[None]
                ).cast(ti.i32)

                sh_upos = ti.floor(
                    (qub - self.rigidManager.spatialHash.globalbbox[0]) / self.rigidManager.spatialHash.gridSize[None]
                ).cast(ti.i32)

                sh_lpos = ti.max(sh_lpos, ti.Vector.zero(ti.i32, ti.static(self.d)))
                sh_lpos = ti.min(sh_lpos, self.rigidManager.spatialHash.cellNumbers[None] - 1)
                sh_upos = ti.max(sh_upos, ti.Vector.zero(ti.i32, ti.static(self.d)))
                sh_upos = ti.min(sh_upos, self.rigidManager.spatialHash.cellNumbers[None] - 1)

                for I in ti.grouped(
                    ti.ndrange(
                        (sh_lpos[0], sh_upos[0] + 1),
                        (sh_lpos[1], sh_upos[1] + 1),
                    )
                ):
                    cid = I[0] + I[1] * self.rigidManager.spatialHash.cellNumbers[None][0]
                    penetration, normal, cpoint, weights, elem_idx, rigid_did, found, has_cand = (
                        self._process_rigid_sh_cell(
                            cid,
                            node_coord,
                            target_rigid_did,
                            penetration,
                            normal,
                            cpoint,
                            weights,
                            elem_idx,
                            rigid_did,
                            found,
                            i_a,
                            has_cand,
                        )
                    )

            if found:
                self._bc_rigid_tie_resolved[i_a] = 1
                self._bc_rigid_cache_elem[i_a] = ti.Vector([elem_idx, rigid_did])
                self._bc_rigid_tie_weights[i_a] = weights
                ti.atomic_add(self._fixed_rigid_mesh_tie_found_count[None], 1)

    @ti.kernel
    def _apply_fem_rigid_mesh_ties_fixed_kernel(self, total_mesh_items: ti.i32):
        for i_a in range(total_mesh_items):
            wm = self._bc_rigid_mesh_idx[i_a]

            if self._bc_rigid_tied[wm] == 0:
                continue

            if self._bc_rigid_tie_resolved[i_a] == 0:
                continue

            nid = self._bc_rigid_node[wm]
            penalty = self._bc_rigid_penalty[wm]
            elem_idx = self._bc_rigid_cache_elem[i_a][0]
            rigid_idx = self._bc_rigid_cache_elem[i_a][1]
            weights = self._bc_rigid_tie_weights[i_a]
            cpoint = self._eval_fixed_tie_point(
                elem_idx,
                weights,
                self.rigidManager.meshBoundaryElements,
                self.rigidManager.meshBoundaryCoords,
            )

            self._apply_one_fixed_rigid_mesh_tie(nid, rigid_idx, cpoint, penalty)

    @ti.kernel
    def _batched_heightfield_contact_kernel(self, total: ti.i32, dt: ti.f32):
        """Single kernel for ALL FlexHeightField + SpringHeightField pairs.
        All contacts must reference the same heightfield domain (_hf_domain)."""
        for w in range(total):
            pair_idx = self._bc_hf_pair[w]
            if self._bc_hf_active[pair_idx] == 0:
                continue
            nid = self._bc_hf_node[w]
            penalty = self._bc_hf_penalty[w]
            node_pos = self.femSpringManager.coords[nid]
            node_vel = self.femSpringManager.V[nid]

            damping_coeff = ti.f32(0.5)
            foot = ti.Vector.zero(ti.f32, ti.static(self.d))
            n = ti.Vector.zero(ti.f32, ti.static(self.d))
            signed = ti.f32(1e9)

            x = node_pos[0]
            y = node_pos[1]
            foot, n, signed = self._hf_domain.nearest_on_curve_2d(x, y)

            if signed < 0.0:
                contact_force = -n * penalty * signed
                vn = node_vel.dot(n)
                if vn < 0.0:
                    contact_force += -n * damping_coeff * penalty * vn * dt
                self.femSpringManager.Fext[nid] += contact_force

    @ti.kernel
    def _batched_voxelmap_contact_kernel(self, total: ti.i32, dt: ti.f32):
        """Single kernel for ALL FlexVoxelMap contact pairs.
        All contacts must reference the same voxelmap domain (_voxel_domain)."""
        for w in range(total):
            pair_idx = self._bc_voxel_pair[w]
            if self._bc_voxel_active[pair_idx] == 0:
                continue
            nid = self._bc_voxel_node[w]
            penalty = self._bc_voxel_penalty[w]
            node_pos = self.femSpringManager.coords[nid]
            node_vel = self.femSpringManager.V[nid]

            damping_coeff = ti.f32(0.5)
            n = ti.Vector.zero(ti.f32, ti.static(self.d))
            signed_dist = ti.f32(1e9)
            cpoint = ti.Vector.zero(ti.f32, ti.static(self.d))

            signed_dist, n, cpoint = self._voxel_domain.signed_distance_to_edges_2d(node_pos, ti.f32(0.0))
           
            if signed_dist < 0.0:
                penetration = -signed_dist
                contact_force = n * penalty * penetration
                vn = node_vel.dot(n)
                if vn < 0.0:
                    contact_force += -n * damping_coeff * penalty * vn * dt
                self.femSpringManager.Fext[nid] += contact_force
