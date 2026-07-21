"""2D BVH collision detector (NVIDIA Warp).

Public class: ``CollisionDetector`` — used by ExplicitLoop / RigidManager.
Device kernels take explicit ``wp.array`` AABB arguments (no Taichi templates).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from wp_init import ensure_warp

# BVH configuration (module-level for use inside @wp.func / @wp.kernel)
MAX_STACK_DEPTH = 64
MAX_TREE_DEPTH = 10
MAX_OBJECTS_PER_LEAF = 1


@wp.struct
class BVHNode2D:
    aabb_min: wp.vec2
    aabb_max: wp.vec2
    left_child: int
    right_child: int
    is_leaf: int
    primitive_id: int


@wp.struct
class Object2D:
    aabb_min: wp.vec2
    aabb_max: wp.vec2
    center: wp.vec2
    primitive_id: int
    env_id: int


@wp.func
def aabb_intersect(min_a: wp.vec2, max_a: wp.vec2, min_b: wp.vec2, max_b: wp.vec2):
    return (
        min_a[0] <= max_b[0]
        and max_a[0] >= min_b[0]
        and min_a[1] <= max_b[1]
        and max_a[1] >= min_b[1]
    )


@wp.func
def _partition_objects(
    start: int,
    end: int,
    mid: int,
    axis: int,
    sorted_indices: wp.array(dtype=int),
    objects: wp.array(dtype=Object2D),
):
    """Insertion-sort partition of sorted_indices[start:end] by center[axis]."""
    for i in range(start + 1, end):
        key_idx = sorted_indices[i]
        key_value = objects[key_idx].center[axis]
        j = int(i - 1)
        done = int(0)
        for _ in range(end - start):
            if done == 0:
                do_shift = int(0)
                if j >= start:
                    if objects[sorted_indices[j]].center[axis] > key_value:
                        do_shift = 1
                if do_shift == 1:
                    sorted_indices[j + 1] = sorted_indices[j]
                    j = j - 1
                else:
                    done = 1
        sorted_indices[j + 1] = key_idx


@wp.func
def _build_node(
    objects: wp.array(dtype=Object2D),
    nodes: wp.array(dtype=BVHNode2D),
    sorted_indices: wp.array(dtype=int),
    starts: wp.array(dtype=int),
    ends: wp.array(dtype=int),
    object_count: wp.array(dtype=int),
    node_count: wp.array(dtype=int),
):
    """Build a single BVH over sorted_indices[0:object_count]."""
    ends[0] = object_count[0]
    starts[0] = 0
    node_index = int(0)

    for itn in range(MAX_TREE_DEPTH):
        find_children = int(0)
        level_width = 1 << itn
        for lp in range(level_width):
            offset = level_width - 1
            start = starts[offset + lp]
            end = ends[offset + lp]
            node = nodes[node_index]

            count = end - start
            if count > 0:
                aabb_min = wp.vec2(1e9, 1e9)
                aabb_max = wp.vec2(-1e9, -1e9)

                for i in range(start, end):
                    idx = sorted_indices[i]
                    obj = objects[idx]
                    aabb_min = wp.min(aabb_min, obj.aabb_min)
                    aabb_max = wp.max(aabb_max, obj.aabb_max)

                node.aabb_min = aabb_min
                node.aabb_max = aabb_max

                if count <= MAX_OBJECTS_PER_LEAF:
                    node.primitive_id = objects[sorted_indices[start]].primitive_id
                    node.left_child = -1
                    node.right_child = -1
                    node.is_leaf = 1
                else:
                    extent = aabb_max - aabb_min
                    split_axis = int(0)
                    if extent[1] > extent[0]:
                        split_axis = 1

                    mid = start + count // 2
                    _partition_objects(start, end, mid, split_axis, sorted_indices, objects)

                    left_index = int(1)
                    right_index = int(1)
                    if mid - start < MAX_OBJECTS_PER_LEAF:
                        left_index = -1
                    else:
                        node_count[0] = node_count[0] + 1
                        left_index = node_count[0]

                    if end - mid < MAX_OBJECTS_PER_LEAF:
                        right_index = -1
                    else:
                        node_count[0] = node_count[0] + 1
                        right_index = node_count[0]

                    child_offset = (1 << (itn + 1)) - 1
                    starts[child_offset + 2 * lp] = start
                    ends[child_offset + 2 * lp] = mid
                    starts[child_offset + 2 * lp + 1] = mid
                    ends[child_offset + 2 * lp + 1] = end

                    node.left_child = left_index
                    node.right_child = right_index
                    node.primitive_id = -1
                    if left_index == -1 and right_index == -1:
                        node.is_leaf = 1
                    else:
                        find_children = 1

                nodes[node_index] = node
                node_index = node_index + 1

        if find_children == 0:
            break

    node_count[0] = node_count[0] + 1


@wp.func
def _build_subtree_func(
    obj_count: int,
    objects: wp.array(dtype=Object2D),
    nodes: wp.array(dtype=BVHNode2D),
    sorted_indices: wp.array(dtype=int),
    starts: wp.array(dtype=int),
    ends: wp.array(dtype=int),
    node_count: wp.array(dtype=int),
):
    """Build a sub-tree for sorted_indices[0:obj_count] starting at node_count."""
    starts[0] = 0
    ends[0] = obj_count
    node_index = int(node_count[0])

    for itn in range(MAX_TREE_DEPTH):
        find_children = int(0)
        level_width = 1 << itn
        for lp in range(level_width):
            offset = level_width - 1
            start = starts[offset + lp]
            end = ends[offset + lp]
            node = nodes[node_index]

            count = end - start
            if count > 0:
                aabb_min = wp.vec2(1e9, 1e9)
                aabb_max = wp.vec2(-1e9, -1e9)

                for i in range(start, end):
                    idx = sorted_indices[i]
                    obj = objects[idx]
                    aabb_min = wp.min(aabb_min, obj.aabb_min)
                    aabb_max = wp.max(aabb_max, obj.aabb_max)

                node.aabb_min = aabb_min
                node.aabb_max = aabb_max

                if count <= MAX_OBJECTS_PER_LEAF:
                    node.primitive_id = objects[sorted_indices[start]].primitive_id
                    node.left_child = -1
                    node.right_child = -1
                    node.is_leaf = 1
                else:
                    extent = aabb_max - aabb_min
                    split_axis = int(0)
                    if extent[1] > extent[0]:
                        split_axis = 1

                    mid = start + count // 2
                    _partition_objects(start, end, mid, split_axis, sorted_indices, objects)

                    left_index = int(1)
                    right_index = int(1)
                    if mid - start < MAX_OBJECTS_PER_LEAF:
                        left_index = -1
                    else:
                        node_count[0] = node_count[0] + 1
                        left_index = node_count[0]

                    if end - mid < MAX_OBJECTS_PER_LEAF:
                        right_index = -1
                    else:
                        node_count[0] = node_count[0] + 1
                        right_index = node_count[0]

                    child_offset = (1 << (itn + 1)) - 1
                    starts[child_offset + 2 * lp] = start
                    ends[child_offset + 2 * lp] = mid
                    starts[child_offset + 2 * lp + 1] = mid
                    ends[child_offset + 2 * lp + 1] = end

                    node.left_child = left_index
                    node.right_child = right_index
                    node.primitive_id = -1
                    if left_index == -1 and right_index == -1:
                        node.is_leaf = 1
                    else:
                        find_children = 1

                nodes[node_index] = node
                node_index = node_index + 1

        if find_children == 0:
            break

    node_count[0] = node_count[0] + 1


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _add_one_object_kernel(
    aabb_arrays: wp.array(dtype=wp.vec2, ndim=2),
    pid: int,
    set_buffer_zone: int,
    env_id: int,
    max_nodes: int,
    objects: wp.array(dtype=Object2D),
    object_count: wp.array(dtype=int),
):
    lb = aabb_arrays[pid, 0]
    ub = aabb_arrays[pid, 1]
    idx = object_count[0]
    if idx < max_nodes:
        buffer = wp.vec2(0.0, 0.0)
        if set_buffer_zone == 1:
            buffer = 0.1 * (ub - lb)
        obj = Object2D()
        obj.aabb_min = lb - buffer
        obj.aabb_max = ub + buffer
        obj.center = (lb + ub) * 0.5
        obj.primitive_id = pid
        obj.env_id = env_id
        objects[idx] = obj
        object_count[0] = idx + 1


@wp.kernel
def _add_objects_batch_kernel(
    aabb_arrays: wp.array(dtype=wp.vec2, ndim=2),
    ids: wp.array(dtype=int),
    env_ids: wp.array(dtype=int),
    count: int,
    max_nodes: int,
    objects: wp.array(dtype=Object2D),
    object_count: wp.array(dtype=int),
):
    for k in range(count):
        pid = ids[k]
        eid = env_ids[k]
        lb = aabb_arrays[pid, 0]
        ub = aabb_arrays[pid, 1]
        idx = object_count[0] + k
        if idx < max_nodes:
            buffer = 0.1 * (ub - lb)
            obj = Object2D()
            obj.aabb_min = lb - buffer
            obj.aabb_max = ub + buffer
            obj.center = (lb + ub) * 0.5
            obj.primitive_id = pid
            obj.env_id = eid
            objects[idx] = obj
    object_count[0] = object_count[0] + count


@wp.kernel
def _update_objects_kernel(
    aabb_arrays: wp.array(dtype=wp.vec2, ndim=2),
    set_buffer_zone: int,
    objects: wp.array(dtype=Object2D),
    object_count: wp.array(dtype=int),
):
    i = wp.tid()
    if i >= object_count[0]:
        return
    obj = objects[i]
    pid = obj.primitive_id
    lb = aabb_arrays[pid, 0]
    ub = aabb_arrays[pid, 1]
    if set_buffer_zone == 1:
        buffer = 0.1 * (ub - lb)
        lb = lb - buffer
        ub = ub + buffer
    obj.aabb_min = lb
    obj.aabb_max = ub
    obj.center = 0.5 * (lb + ub)
    objects[i] = obj


@wp.kernel
def _build_kernel(
    objects: wp.array(dtype=Object2D),
    nodes: wp.array(dtype=BVHNode2D),
    sorted_indices: wp.array(dtype=int),
    starts: wp.array(dtype=int),
    ends: wp.array(dtype=int),
    object_count: wp.array(dtype=int),
    node_count: wp.array(dtype=int),
):
    if object_count[0] != 0:
        for i in range(object_count[0]):
            sorted_indices[i] = i
        node_count[0] = 0
        _build_node(objects, nodes, sorted_indices, starts, ends, object_count, node_count)


@wp.kernel
def _build_subtree_kernel(
    obj_indices: wp.array(dtype=int),
    obj_count: int,
    objects: wp.array(dtype=Object2D),
    nodes: wp.array(dtype=BVHNode2D),
    sorted_indices: wp.array(dtype=int),
    starts: wp.array(dtype=int),
    ends: wp.array(dtype=int),
    node_count: wp.array(dtype=int),
):
    if obj_count > 0:
        for i in range(obj_count):
            sorted_indices[i] = obj_indices[i]
        _build_subtree_func(
            obj_count, objects, nodes, sorted_indices, starts, ends, node_count
        )


@wp.kernel
def _set_node_root_kernel(
    start: int,
    count: int,
    root: int,
    node_subtree_root: wp.array(dtype=int),
):
    i = wp.tid()
    if i < count:
        node_subtree_root[start + i] = root


@wp.kernel
def _refit_kernel(
    aabb_arrays: wp.array(dtype=wp.vec2, ndim=2),
    nodes: wp.array(dtype=BVHNode2D),
    node_count: wp.array(dtype=int),
):
    # Serial bottom-up refit (launch dim=1) — children must be ready first.
    nc = node_count[0]
    for istep in range(nc):
        i = nc - istep - 1
        node_i = nodes[i]
        if node_i.is_leaf == 1:
            pid = node_i.primitive_id
            node_i.aabb_min = aabb_arrays[pid, 0]
            node_i.aabb_max = aabb_arrays[pid, 1]
            nodes[i] = node_i
        else:
            left_node = nodes[node_i.left_child]
            right_node = nodes[node_i.right_child]
            node_i.aabb_min = wp.min(left_node.aabb_min, right_node.aabb_min)
            node_i.aabb_max = wp.max(left_node.aabb_max, right_node.aabb_max)
            nodes[i] = node_i


@wp.kernel
def _detect_inner_collision_kernel(
    nodes: wp.array(dtype=BVHNode2D),
    objects: wp.array(dtype=Object2D),
    stack: wp.array(dtype=int, ndim=2),
    collision_pairs: wp.array(dtype=wp.vec2i),
    pair_count: wp.array(dtype=int),
    node_count: wp.array(dtype=int),
    max_pairs: int,
):
    ia = wp.tid()
    nc = node_count[0]
    if ia >= nc:
        return

    node_a_idx = nc - 1 - ia
    stack[ia, 0] = 0
    stack_size = int(1)

    node_a = nodes[node_a_idx]
    node_a_is_leaf = node_a.is_leaf
    node_a_aabb_min = node_a.aabb_min
    node_a_aabb_max = node_a.aabb_max
    node_a_prim_id = node_a.primitive_id

    while stack_size > 0:
        stack_size = stack_size - 1
        node_b_idx = stack[ia, stack_size]

        node_b = nodes[node_b_idx]
        node_b_is_leaf = node_b.is_leaf
        node_b_aabb_min = node_b.aabb_min
        node_b_aabb_max = node_b.aabb_max
        node_b_prim_id = node_b.primitive_id
        node_b_left = node_b.left_child
        node_b_right = node_b.right_child

        if (node_a_is_leaf == 1) and (node_a_idx > node_b_idx):
            if aabb_intersect(node_a_aabb_min, node_a_aabb_max, node_b_aabb_min, node_b_aabb_max):
                if node_b_is_leaf == 1:
                    if node_a_prim_id != node_b_prim_id:
                        # Index objects by primitive_id (same as original Taichi).
                        obj_a = objects[node_a_prim_id]
                        obj_b = objects[node_b_prim_id]
                        skip_pair = int(0)
                        if obj_a.env_id >= 0 and obj_b.env_id >= 0:
                            skip_pair = 1
                        if skip_pair == 0:
                            pair_index = int(wp.atomic_add(pair_count, 0, 1))
                            if pair_index < max_pairs:
                                collision_pairs[pair_index] = wp.vec2i(
                                    node_a_prim_id, node_b_prim_id
                                )
                elif node_b_is_leaf != 1:
                    if stack_size + 2 < MAX_STACK_DEPTH:
                        stack[ia, stack_size] = node_b_left
                        stack_size = stack_size + 1
                        stack[ia, stack_size] = node_b_right
                        stack_size = stack_size + 1


@wp.kernel
def _detect_inner_collision_per_env_kernel(
    nodes: wp.array(dtype=BVHNode2D),
    stack: wp.array(dtype=int, ndim=2),
    node_subtree_root: wp.array(dtype=int),
    collision_pairs: wp.array(dtype=wp.vec2i),
    pair_count: wp.array(dtype=int),
    node_count: wp.array(dtype=int),
    max_pairs: int,
):
    ia = wp.tid()
    nc = node_count[0]
    if ia >= nc:
        return

    node_a_idx = nc - 1 - ia
    node_a = nodes[node_a_idx]
    node_a_is_leaf = node_a.is_leaf
    node_a_aabb_min = node_a.aabb_min
    node_a_aabb_max = node_a.aabb_max
    node_a_prim_id = node_a.primitive_id

    root = node_subtree_root[node_a_idx]
    stack[ia, 0] = root
    stack_size = int(1)

    while stack_size > 0:
        stack_size = stack_size - 1
        node_b_idx = stack[ia, stack_size]

        node_b = nodes[node_b_idx]
        node_b_is_leaf = node_b.is_leaf
        node_b_aabb_min = node_b.aabb_min
        node_b_aabb_max = node_b.aabb_max
        node_b_prim_id = node_b.primitive_id
        node_b_left = node_b.left_child
        node_b_right = node_b.right_child

        if (node_a_is_leaf == 1) and (node_a_idx > node_b_idx):
            if aabb_intersect(node_a_aabb_min, node_a_aabb_max, node_b_aabb_min, node_b_aabb_max):
                if node_b_is_leaf == 1:
                    if node_a_prim_id != node_b_prim_id:
                        pair_index = int(wp.atomic_add(pair_count, 0, 1))
                        if pair_index < max_pairs:
                            collision_pairs[pair_index] = wp.vec2i(
                                node_a_prim_id, node_b_prim_id
                            )
                elif node_b_is_leaf != 1:
                    if stack_size + 2 < MAX_STACK_DEPTH:
                        stack[ia, stack_size] = node_b_left
                        stack_size = stack_size + 1
                        stack[ia, stack_size] = node_b_right
                        stack_size = stack_size + 1


class CollisionDetector:
    """2D BVH broad-phase collision detector."""

    MAX_STACK_DEPTH = MAX_STACK_DEPTH
    MAX_TREE_DEPTH = MAX_TREE_DEPTH
    MAX_OBJECTS_PER_LEAF = MAX_OBJECTS_PER_LEAF

    def __init__(self, d, max_nodes=1024):
        ensure_warp()
        if d != 2:
            raise ValueError(f"CollisionDetector Warp migration supports d=2 only (got {d})")

        self.d = d
        self.max_nodes = max_nodes
        self.nodes = wp.zeros(max_nodes, dtype=BVHNode2D)
        self.objects = wp.zeros(max_nodes, dtype=Object2D)

        self.node_count = wp.zeros(1, dtype=int)
        self.object_count = wp.zeros(1, dtype=int)

        self.max_pairs = max_nodes * 10
        self.collision_pairs = wp.zeros(self.max_pairs, dtype=wp.vec2i)
        self.pair_count = wp.zeros(1, dtype=int)

        self.sorted_indices = wp.zeros(max_nodes, dtype=int)
        self.starts = wp.zeros(max_nodes, dtype=int)
        self.ends = wp.zeros(max_nodes, dtype=int)

        self.stack = wp.zeros((max_nodes, MAX_STACK_DEPTH), dtype=int)

        self._max_envs = max(max_nodes // 2, 16)
        self.env_root = wp.zeros(self._max_envs, dtype=int)
        self.env_node_count = wp.zeros(self._max_envs, dtype=int)
        self.num_envs = wp.zeros(1, dtype=int)
        self._node_subtree_root = wp.zeros(max_nodes, dtype=int)
        self._per_env_mode = False

        self._temp_ids = wp.zeros(max_nodes, dtype=int)
        self._temp_env_ids = wp.zeros(max_nodes, dtype=int)

    def reset(self):
        self.pair_count.zero_()
        self.node_count.zero_()
        self._per_env_mode = False

    def addOneObject(self, aabbArrays, id: int, setbufferZone: int, env_id: int):
        """Add one object from an AABB array of shape (N, 2) dtype=vec2."""
        wp.launch(
            _add_one_object_kernel,
            dim=1,
            inputs=[
                aabbArrays,
                int(id),
                int(setbufferZone),
                int(env_id),
                self.max_nodes,
                self.objects,
                self.object_count,
            ],
        )

    def addObjects_batch(self, aabbArrays, ids_np, env_ids_np):
        """Batch-add objects from numpy id arrays into BVH."""
        count = len(ids_np)
        if count == 0:
            return
        buf_ids = np.zeros(self.max_nodes, dtype=np.int32)
        buf_eids = np.zeros(self.max_nodes, dtype=np.int32)
        buf_ids[:count] = ids_np
        buf_eids[:count] = env_ids_np
        self._temp_ids.assign(buf_ids)
        self._temp_env_ids.assign(buf_eids)
        wp.launch(
            _add_objects_batch_kernel,
            dim=1,
            inputs=[
                aabbArrays,
                self._temp_ids,
                self._temp_env_ids,
                int(count),
                self.max_nodes,
                self.objects,
                self.object_count,
            ],
        )

    def update_objects(self, aabbArrays, setbufferZone: int):
        oc = int(self.object_count.numpy()[0])
        if oc <= 0:
            return
        wp.launch(
            _update_objects_kernel,
            dim=oc,
            inputs=[aabbArrays, int(setbufferZone), self.objects, self.object_count],
        )

    def build(self):
        wp.launch(
            _build_kernel,
            dim=1,
            inputs=[
                self.objects,
                self.nodes,
                self.sorted_indices,
                self.starts,
                self.ends,
                self.object_count,
                self.node_count,
            ],
        )

    def build_per_env(self, env_groups):
        """Build independent sub-trees for each environment group.

        Args:
            env_groups: dict mapping env_id -> list of BVH object indices
                        (indices into self.objects, NOT domain ids).
        """
        self.node_count.zero_()
        self._per_env_mode = True
        envs = sorted(env_groups.keys())
        num_envs = len(envs)
        if num_envs > self._max_envs:
            raise RuntimeError(
                f"Too many env groups ({num_envs}) for BVH per-env "
                f"(max {self._max_envs})"
            )
        self.num_envs.fill_(num_envs)
        buf = np.zeros(self.max_nodes, dtype=np.int32)

        for env_idx, eid in enumerate(envs):
            obj_indices = np.array(env_groups[eid], dtype=np.int32)
            obj_count = len(obj_indices)

            buf[:obj_count] = obj_indices
            self._temp_ids.assign(buf)

            node_base = int(self.node_count.numpy()[0])
            wp.launch(
                _build_subtree_kernel,
                dim=1,
                inputs=[
                    self._temp_ids,
                    int(obj_count),
                    self.objects,
                    self.nodes,
                    self.sorted_indices,
                    self.starts,
                    self.ends,
                    self.node_count,
                ],
            )
            node_count_env = int(self.node_count.numpy()[0]) - node_base

            # Write env metadata on host (small)
            er = self.env_root.numpy()
            enc = self.env_node_count.numpy()
            er[env_idx] = node_base
            enc[env_idx] = node_count_env
            self.env_root.assign(er)
            self.env_node_count.assign(enc)

            if node_count_env > 0:
                wp.launch(
                    _set_node_root_kernel,
                    dim=node_count_env,
                    inputs=[
                        node_base,
                        node_count_env,
                        node_base,
                        self._node_subtree_root,
                    ],
                )

        print(
            f"BVH: Built {num_envs} per-env sub-trees, "
            f"total {int(self.node_count.numpy()[0])} nodes"
        )

    def refit(self, aabbArrays):
        nc = int(self.node_count.numpy()[0])
        if nc <= 0:
            return
        wp.launch(
            _refit_kernel,
            dim=1,
            inputs=[aabbArrays, self.nodes, self.node_count],
        )

    def detectInnerCollision(self):
        self.pair_count.zero_()
        nc = int(self.node_count.numpy()[0])
        if nc <= 0:
            return
        wp.launch(
            _detect_inner_collision_kernel,
            dim=nc,
            inputs=[
                self.nodes,
                self.objects,
                self.stack,
                self.collision_pairs,
                self.pair_count,
                self.node_count,
                self.max_pairs,
            ],
        )

    def detectInnerCollision_per_env(self):
        self.pair_count.zero_()
        nc = int(self.node_count.numpy()[0])
        if nc <= 0:
            return
        wp.launch(
            _detect_inner_collision_per_env_kernel,
            dim=nc,
            inputs=[
                self.nodes,
                self.stack,
                self._node_subtree_root,
                self.collision_pairs,
                self.pair_count,
                self.node_count,
                self.max_pairs,
            ],
        )

    def get_collision_pairs(self):
        count = int(self.pair_count.numpy()[0])
        pairs = self.collision_pairs.numpy()[:count]
        return pairs

    def __str__(self):
        nc = int(self.node_count.numpy()[0])
        oc = int(self.object_count.numpy()[0])
        result = ["BVH Tree Structure:"]
        result.append(f"Total nodes: {nc}, Objects: {oc}")
        result.append("-" * 60)

        nodes_np = self.nodes.numpy()

        def format_node(node_idx, prefix="", is_last=True):
            if node_idx < 0 or node_idx >= nc:
                return
            node = nodes_np[node_idx]
            connector = "└── " if is_last else "├── "
            if int(node["is_leaf"]) == 1:
                node_info = f"Node[{node_idx}] LEAF → Primitive {int(node['primitive_id'])}"
            else:
                node_info = f"Node[{node_idx}] INTERNAL"
            amin = node["aabb_min"]
            amax = node["aabb_max"]
            aabb_info = (
                f" AABB[{amin[0]:.2f},{amin[1]:.2f}"
                f" → {amax[0]:.2f},{amax[1]:.2f}]"
            )
            result.append(prefix + connector + node_info + aabb_info)
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension
            if int(node["is_leaf"]) != 1:
                left = int(node["left_child"])
                right = int(node["right_child"])
                if left >= 0:
                    format_node(left, new_prefix, right < 0)
                if right >= 0:
                    format_node(right, new_prefix, True)

        if nc > 0:
            format_node(0)
        else:
            result.append("(Empty tree)")
        return "\n".join(result)
