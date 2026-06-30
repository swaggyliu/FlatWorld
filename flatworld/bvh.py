from definitions import *
import numpy as np
import taichi as ti

@ti.dataclass
class BVHNode2D:
    aabb_min: ti.types.vector(2, ti.f32)  # Bounding box minimum
    aabb_max: ti.types.vector(2, ti.f32)  # bounding box maximum value
    left_child: ti.i32  # Left child node index，-1Indicates no child node
    right_child: ti.i32  # Right child node index，-1Indicates no child node
    is_leaf: ti.i32
    primitive_id: ti.i32  # associated primitives（like triangle）index，-1Represents an internal node


@ti.dataclass
class Object2D:
    aabb_min: ti.types.vector(2, ti.f32)  # Bounding box minimum
    aabb_max: ti.types.vector(2, ti.f32)  # bounding box maximum value
    center: ti.types.vector(2, ti.f32)
    primitive_id: ti.i32  # associated primitives（like triangle）index，-1Represents an internal node
    env_id: ti.i32  # environmentID，Used to filter collisions within the same environment during batch training

@ti.data_oriented
class CollisionDetector:
    # Class constants for BVH configuration
    MAX_STACK_DEPTH = 64  # Maximum depth for collision detection stack
    MAX_TREE_DEPTH = 10  # Maximum depth of BVH tree
    MAX_OBJECTS_PER_LEAF = 1  # Maximum objects allowed in a leaf node

    def __init__(self, d, max_nodes=1024):
        self.d = d
        self.max_nodes = max_nodes
        self.nodes = BVHNode2D.field(shape=(max_nodes,))
        self.objects = Object2D.field(shape=(max_nodes,))

        self.node_count = ti.field(ti.i32, shape=())
        self.object_count = ti.field(ti.i32, shape=())

        # Used to store potential collision pair results。Each collision pair contains twoprimitive_id。
        self.collision_pairs = ti.Vector.field(2, ti.i32, shape=(max_nodes * 10,))
        self.pair_count = ti.field(ti.i32, shape=())

        # temporary sorted array
        self.sorted_indices = ti.field(ti.i32, shape=max_nodes)
        self.starts = ti.field(ti.i32, max_nodes)
        self.ends = ti.field(ti.i32, max_nodes)

        # For contact detection - make stack per-thread to avoid GPU race conditions
        self.stack = ti.field(ti.i32, shape=(max_nodes, self.MAX_STACK_DEPTH))

        # ── Per-env sub-tree metadata ──
        self._max_envs = max(max_nodes // 2, 16)
        self.env_root = ti.field(ti.i32, self._max_envs)
        self.env_node_count = ti.field(ti.i32, self._max_envs)
        self.num_envs = ti.field(ti.i32, ())
        self._node_subtree_root = ti.field(ti.i32, max_nodes)
        self._per_env_mode = False  # Python flag

        # ── Reusable temp buffers (avoid per-call ti.field allocation) ──
        # Used by addObjects_batch and build_per_env to avoid creating new
        # ti.field objects on each call, which triggers kernel recompilation
        # via ti.template() and crashes GPU at high env counts.
        self._temp_ids = ti.field(ti.i32, max_nodes)
        self._temp_env_ids = ti.field(ti.i32, max_nodes)

    def reset(self):
        self.pair_count[None] = 0
        self.node_count[None] = 0
        self._per_env_mode = False
        # self.object_count[None] = 0

    @ti.kernel
    def addOneObject(self, aabbArrays: ti.template(), id: ti.i32, setbufferZone: ti.i32, env_id: ti.i32):
        """Add an object toBVH"""
        lb = aabbArrays[id, 0]
        ub = aabbArrays[id, 1]

        idx = self.object_count[None]
        if idx < self.max_nodes:
            buffer = ti.Vector.zero(ti.f32, self.d)
            if setbufferZone == 1:
                # Increased buffer zone to avoid penetration during high-speed movement
                buffer = 0.1 * (ub - lb)

            self.objects[idx].aabb_min = lb - buffer
            self.objects[idx].aabb_max = ub + buffer

            self.objects[idx].center = (lb + ub) * 0.5
            self.objects[idx].primitive_id = id
            self.objects[idx].env_id = env_id
            self.object_count[None] += 1

    def addObjects_batch(self, aabbArrays, ids_np, env_ids_np):
        """Batch-add multiple objects to BVH from numpy arrays.
        Avoids N separate kernel launches by copying via temp fields.

        Args:
            aabbArrays: Taichi AABB field (shared).
            ids_np: (M,) int32 numpy array of domain indices.
            env_ids_np: (M,) int32 numpy array of env ids.
        """
        import numpy as np

        count = len(ids_np)
        if count == 0:
            return
        # Reuse pre-allocated temp buffers (avoid creating new ti.field per call)
        buf_ids = np.zeros(self.max_nodes, dtype=np.int32)
        buf_eids = np.zeros(self.max_nodes, dtype=np.int32)
        buf_ids[:count] = ids_np
        buf_eids[:count] = env_ids_np
        self._temp_ids.from_numpy(buf_ids)
        self._temp_env_ids.from_numpy(buf_eids)
        self._addObjects_batch_kernel(aabbArrays, self._temp_ids, self._temp_env_ids, count)

    @ti.kernel
    def _addObjects_batch_kernel(
        self, aabbArrays: ti.template(), ids: ti.template(), env_ids: ti.template(), count: ti.i32
    ):
        """Kernel: batch-add objects. Runs serially to maintain ordering."""
        for k in range(count):
            pid = ids[k]
            eid = env_ids[k]
            lb = aabbArrays[pid, 0]
            ub = aabbArrays[pid, 1]
            # idx = ti.atomic_add(self.object_count[None], 1)  # thread-safe
            idx = self.object_count[None] + k  # sequential layout
            if idx < self.max_nodes:
                buffer = 0.1 * (ub - lb)
                self.objects[idx].aabb_min = lb - buffer
                self.objects[idx].aabb_max = ub + buffer
                self.objects[idx].center = (lb + ub) * 0.5
                self.objects[idx].primitive_id = pid
                self.objects[idx].env_id = eid
        self.object_count[None] += count

    @ti.kernel
    def update_objects(self, aabbArrays: ti.template(), setbufferZone: ti.i32):
        """renewBVHobjects in"""
        for i in range(self.object_count[None]):
            pid = self.objects[i].primitive_id
            lb = aabbArrays[pid, 0]
            ub = aabbArrays[pid, 1]
            if setbufferZone == 1:
                # Increased buffer zone to avoid penetration during high-speed movement
                buffer = 0.1 * (ub - lb)
                lb -= buffer
                ub += buffer
            self.objects[i].aabb_min = lb
            self.objects[i].aabb_max = ub
            self.objects[i].center = 0.5 * (lb + ub)

    @ti.kernel
    def build(self):
        """buildBVHTree"""
        if self.object_count[None] != 0:
            # Initialize sorted array
            for i in range(self.object_count[None]):
                self.sorted_indices[i] = i

            # buildBVHTree
            self.node_count[None] = 0
            self._build_node(0)

    # ── Per-env sub-tree build ─────────────────────────────────────

    def build_per_env(self, env_groups):
        """Build independent sub-trees for each environment group.

        Each env's sub-tree occupies a contiguous section of the nodes array.
        This avoids the global tree depth limit (MAX_TREE_DEPTH=10 → 1024 leaves)
        because each sub-tree only contains objects from one env (~10 objects).

        Args:
            env_groups: dict mapping env_id -> list of BVH *object* indices
                        (indices into self.objects array, NOT domain ids).
        """
        self.node_count[None] = 0
        self._per_env_mode = True
        envs = sorted(env_groups.keys())
        num_envs = len(envs)
        if num_envs > self._max_envs:
            raise RuntimeError(f"Too many env groups ({num_envs}) for BVH per-env " f"(max {self._max_envs})")
        self.num_envs = num_envs
        buf = np.zeros(self.max_nodes, dtype=np.int32)

        for env_idx, eid in enumerate(envs):
            obj_indices = np.array(env_groups[eid], dtype=np.int32)
            obj_count = len(obj_indices)

            # Reuse pre-allocated buffer (avoid creating new ti.field per env,
            # which triggers kernel recompilation via ti.template() and crashes GPU)
            buf[:obj_count] = obj_indices
            self._temp_ids.from_numpy(buf)

            node_base = self.node_count[None]
            self._build_subtree_kernel(self._temp_ids, obj_count)
            node_count_env = self.node_count[None] - node_base

            self.env_root[env_idx] = node_base
            self.env_node_count[env_idx] = node_count_env
            self._set_node_root_kernel(node_base, node_count_env, node_base)

        print(f"BVH: Built {num_envs} per-env sub-trees, " f"total {self.node_count[None]} nodes")

    @ti.kernel
    def _build_subtree_kernel(self, obj_indices: ti.template(), obj_count: ti.i32):
        """Build one sub-tree for the given object indices.
        Writes nodes starting at self.node_count[None]."""
        if obj_count > 0:
            for i in range(obj_count):
                self.sorted_indices[i] = obj_indices[i]
            self._build_subtree_func(obj_count)

    @ti.func
    def _build_subtree_func(self, obj_count: ti.i32):
        """Build a sub-tree for sorted_indices[0:obj_count].
        Identical to _build_node but parameterised on obj_count and
        starting node offset from self.node_count[None]."""
        self.starts[0] = 0
        self.ends[0] = obj_count
        node_index = self.node_count[None]

        for itn in range(ti.static(self.MAX_TREE_DEPTH)):
            findChildren = False
            for lp in range(2 ** (itn)):
                offset = 2 ** (itn) - 1
                start = self.starts[offset + lp]
                end = self.ends[offset + lp]
                node = self.nodes[node_index]

                object_count = end - start
                if object_count > 0:
                    aabb_min = ti.Vector([ti.f32(1e9) for i in ti.static(range(self.d))])
                    aabb_max = ti.Vector([ti.f32(-1e9) for i in ti.static(range(self.d))])

                    for i in range(start, end):
                        idx = self.sorted_indices[i]
                        obj_min = self.objects[idx].aabb_min
                        obj_max = self.objects[idx].aabb_max
                        aabb_min = ti.min(aabb_min, obj_min)
                        aabb_max = ti.max(aabb_max, obj_max)

                    node.aabb_min = aabb_min
                    node.aabb_max = aabb_max
                    if object_count <= ti.static(self.MAX_OBJECTS_PER_LEAF):
                        node.primitive_id = self.objects[self.sorted_indices[start]].primitive_id
                        node.left_child = -1
                        node.right_child = -1
                        node.is_leaf = 1
                    else:
                        extent = self.nodes[node_index].aabb_max - self.nodes[node_index].aabb_min
                        split_axis = 0
                        if extent.y > extent.x:
                            split_axis = 1

                        mid = start + object_count // 2
                        self._partition_objects(start, end, mid, split_axis)

                        left_index, right_index = 1, 1
                        if mid - start < ti.static(self.MAX_OBJECTS_PER_LEAF):
                            left_index = -1
                        else:
                            self.node_count[None] += 1
                            left_index = self.node_count[None]

                        if end - mid < ti.static(self.MAX_OBJECTS_PER_LEAF):
                            right_index = -1
                        else:
                            self.node_count[None] += 1
                            right_index = self.node_count[None]

                        offset = 2 ** (itn + 1) - 1
                        self.starts[offset + 2 * lp] = start
                        self.ends[offset + 2 * lp] = mid
                        self.starts[offset + 2 * lp + 1] = mid
                        self.ends[offset + 2 * lp + 1] = end

                        node.left_child = left_index
                        node.right_child = right_index
                        node.primitive_id = -1
                        if left_index == -1 and right_index == -1:
                            node.is_leaf = 1
                        else:
                            findChildren = True
                    self.nodes[node_index] = node
                    node_index += 1

            if not findChildren:
                break
        self.node_count[None] += 1

    @ti.kernel
    def _set_node_root_kernel(self, start: ti.i32, count: ti.i32, root: ti.i32):
        for i in range(count):
            self._node_subtree_root[start + i] = root

    @ti.kernel
    def detectInnerCollision_per_env(self):
        """Per-env collision detection.
        Each leaf traverses only its own env's sub-tree (via _node_subtree_root).
        No env_id filtering needed — sub-trees are env-pure."""
        self.pair_count[None] = 0

        for ia in range(self.node_count[None]):
            node_a_idx = self.node_count[None] - 1 - ia

            node_a = self.nodes[node_a_idx]
            node_a_is_leaf = node_a.is_leaf
            node_a_aabb_min = node_a.aabb_min
            node_a_aabb_max = node_a.aabb_max
            node_a_prim_id = node_a.primitive_id

            root = self._node_subtree_root[node_a_idx]
            self.stack[ia, 0] = root
            stack_size = 1

            while stack_size > 0:
                stack_size -= 1
                node_b_idx = self.stack[ia, stack_size]

                node_b = self.nodes[node_b_idx]
                node_b_is_leaf = node_b.is_leaf
                node_b_aabb_min = node_b.aabb_min
                node_b_aabb_max = node_b.aabb_max
                node_b_prim_id = node_b.primitive_id
                node_b_left = node_b.left_child
                node_b_right = node_b.right_child

                if (node_a_is_leaf == 1) and (node_a_idx > node_b_idx):
                    if self.aabb_intersect(node_a_aabb_min, node_a_aabb_max, node_b_aabb_min, node_b_aabb_max):
                        if node_b_is_leaf == 1:
                            if node_a_prim_id != node_b_prim_id:
                                pair_index = ti.atomic_add(self.pair_count[None], 1)
                                if pair_index < self.collision_pairs.shape[0]:
                                    self.collision_pairs[pair_index] = ti.Vector([node_a_prim_id, node_b_prim_id])
                        elif node_b_is_leaf != 1:
                            if stack_size + 2 < ti.static(self.MAX_STACK_DEPTH):
                                self.stack[ia, stack_size] = node_b_left
                                stack_size += 1
                                self.stack[ia, stack_size] = node_b_right
                                stack_size += 1

    @ti.func
    def _partition_objects(self, start: ti.i32, end: ti.i32, mid: ti.i32, axis: ti.i32):
        """Partial sorting using insertion sort，GPUfriendly
        Although it isO(n²)，But each partition is usually small，and avoidedwhileloop inGPUquestion on
        """
        # Simple insertion sort，Very efficient for small data sets andGPUSafety
        for i in range(start + 1, end):
            key_idx = self.sorted_indices[i]
            key_value = self.objects[key_idx].center[axis]
            j = i - 1

            # Use a fixed number of loops insteadwhile，avoidGPUbranching problem
            for _ in range(end - start):
                if j >= start and self.objects[self.sorted_indices[j]].center[axis] > key_value:
                    self.sorted_indices[j + 1] = self.sorted_indices[j]
                    j -= 1
                else:
                    break

            self.sorted_indices[j + 1] = key_idx

    @ti.func
    def _build_node(self, depth: ti.i32):
        self.ends[0] = self.object_count[None]
        node_index = 0

        for itn in range(ti.static(self.MAX_TREE_DEPTH)):
            findChildren = False
            for lp in range(2 ** (itn)):
                offset = 2 ** (itn) - 1
                start = self.starts[offset + lp]
                end = self.ends[offset + lp]
                node = self.nodes[node_index]

                # If the number of objects is less than the threshold，Create leaf nodes
                object_count = end - start
                if object_count > 0:
                    # Calculate the bounding box of the current node
                    aabb_min = ti.Vector([ti.f32(1e9) for i in ti.static(range(self.d))])
                    aabb_max = ti.Vector([ti.f32(-1e9) for i in ti.static(range(self.d))])

                    for i in range(start, end):
                        idx = self.sorted_indices[i]
                        obj_min = self.objects[idx].aabb_min
                        obj_max = self.objects[idx].aabb_max
                        aabb_min = ti.min(aabb_min, obj_min)
                        aabb_max = ti.max(aabb_max, obj_max)

                    node.aabb_min = aabb_min
                    node.aabb_max = aabb_max
                    if object_count <= ti.static(self.MAX_OBJECTS_PER_LEAF):
                        node.primitive_id = self.objects[self.sorted_indices[start]].primitive_id

                        node.left_child = -1
                        node.right_child = -1
                        node.is_leaf = 1
                    else:
                        # Select split axis（Based on maximum extension axis）
                        extent = self.nodes[node_index].aabb_max - self.nodes[node_index].aabb_min
                        split_axis = 0
                        if extent.y > extent.x:
                            split_axis = 1

                        # Use fast partitioning algorithm (O(n) instead of O(n²))
                        mid = start + object_count // 2
                        self._partition_objects(start, end, mid, split_axis)

                        left_index, right_index = 1, 1
                        if mid - start < ti.static(self.MAX_OBJECTS_PER_LEAF):
                            left_index = -1
                        else:
                            self.node_count[None] += 1
                            left_index = self.node_count[None]

                        if end - mid < ti.static(self.MAX_OBJECTS_PER_LEAF):
                            right_index = -1
                        else:
                            self.node_count[None] += 1
                            right_index = self.node_count[None]

                        offset = 2 ** (itn + 1) - 1
                        self.starts[offset + 2 * lp] = start
                        self.ends[offset + 2 * lp] = mid
                        self.starts[offset + 2 * lp + 1] = mid
                        self.ends[offset + 2 * lp + 1] = end

                        node.left_child = left_index
                        node.right_child = right_index
                        node.primitive_id = -1
                        if left_index == -1 and right_index == -1:
                            node.is_leaf = 1
                        else:
                            findChildren = True
                    self.nodes[node_index] = node
                    node_index += 1

            if not findChildren:
                break
        self.node_count[None] += 1

    def __str__(self):
        """Debug method: Print BVH tree structure in a readable hierarchical format"""
        result = ["BVH Tree Structure:"]
        result.append(f"Total nodes: {self.node_count[None]}, Objects: {self.object_count[None]}")
        result.append("-" * 60)

        def format_node(node_idx, prefix="", is_last=True):
            """Recursively format a node and its children"""
            if node_idx < 0 or node_idx >= self.node_count[None]:
                return

            node = self.nodes[node_idx]

            # Draw branch connector
            connector = "└── " if is_last else "├── "

            # Format node info
            if node.is_leaf == 1:
                node_info = f"Node[{node_idx}] LEAF → Primitive {node.primitive_id}"
            else:
                node_info = f"Node[{node_idx}] INTERNAL"

            # Add AABB info
            aabb_info = f" AABB[{node.aabb_min[0]:.2f},{node.aabb_min[1]:.2f}"
            aabb_info += f" → {node.aabb_max[0]:.2f},{node.aabb_max[1]:.2f}"
            aabb_info += "]"

            result.append(prefix + connector + node_info + aabb_info)

            # Prepare prefix for children
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension

            # Recursively process children (if internal node)
            if node.is_leaf != 1:
                if node.left_child >= 0:
                    format_node(node.left_child, new_prefix, node.right_child < 0)
                if node.right_child >= 0:
                    format_node(node.right_child, new_prefix, True)

        # Start from root (node 0)
        if self.node_count[None] > 0:
            format_node(0)
        else:
            result.append("(Empty tree)")

        return "\n".join(result)

    @ti.kernel
    def refit(self, aabbArrays: ti.template()):
        """Refit BVH bounds bottom-up. OPTIMIZED: Cache field reads."""
        for istep in range(self.node_count[None]):
            i = self.node_count[None] - istep - 1
            node_i = self.nodes[i]
            is_leaf = node_i.is_leaf

            if is_leaf == 1:
                pid = node_i.primitive_id
                self.nodes[i].aabb_min = aabbArrays[pid, 0]
                self.nodes[i].aabb_max = aabbArrays[pid, 1]
            else:
                # Cache child indices
                left_idx = node_i.left_child
                right_idx = node_i.right_child

                # Read children
                left_node = self.nodes[left_idx]
                right_node = self.nodes[right_idx]

                # Cache bounds
                left_min = left_node.aabb_min
                left_max = left_node.aabb_max
                right_min = right_node.aabb_min
                right_max = right_node.aabb_max

                aabb_min = ti.min(left_min, right_min)
                aabb_max = ti.max(left_max, right_max)
                self.nodes[i].aabb_min = aabb_min
                self.nodes[i].aabb_max = aabb_max

    @ti.kernel
    def detectInnerCollision(self):
        """
        existBVHThe core kernel that performs self-collision detection internally。
        """
        # Clear the results of the previous frame
        self.pair_count[None] = 0

        # Use two stacks to store node pairs to be detected.。
        # This kind of“Dual stack”Structure is the key to traversing different branches within the same tree。

        for ia in range(self.node_count[None]):
            node_a_idx = self.node_count[None] - 1 - ia
            self.stack[ia, 0] = 0  # Use per-thread stack
            stack_size = 1

            # OPTIMIZATION: Cache node_a data (read once per outer loop)
            node_a = self.nodes[node_a_idx]
            node_a_is_leaf = node_a.is_leaf
            node_a_aabb_min = node_a.aabb_min
            node_a_aabb_max = node_a.aabb_max
            node_a_prim_id = node_a.primitive_id

            while stack_size > 0:
                stack_size -= 1
                node_b_idx = self.stack[ia, stack_size]  # Use per-thread stack

                # OPTIMIZATION: Cache node_b data (read once per inner iteration)
                node_b = self.nodes[node_b_idx]
                node_b_is_leaf = node_b.is_leaf
                node_b_aabb_min = node_b.aabb_min
                node_b_aabb_max = node_b.aabb_max
                node_b_prim_id = node_b.primitive_id
                node_b_left = node_b.left_child
                node_b_right = node_b.right_child

                # first step：Check if the bounding boxes of two nodes intersect。
                if (node_a_is_leaf == 1) and (node_a_idx > node_b_idx):

                    if self.aabb_intersect(node_a_aabb_min, node_a_aabb_max, node_b_aabb_min, node_b_aabb_max):
                        # Step 2：Determine node pair type，Decide what to do next。
                        # Condition1：Both nodes are leaf nodes -> Report potential collision pairs。
                        if node_b_is_leaf == 1:
                            # make sureprimitive_idefficient，And avoid collision with yourself（node_a_idx != node_b_idx）
                            # FIX: Use atomic operation to avoid race condition on GPU
                            if node_a_prim_id != node_b_prim_id:
                                # Filter out same-environment collisions (batched training)
                                obj_a = self.objects[node_a_prim_id]
                                obj_b = self.objects[node_b_prim_id]

                                # Skip if both are in different/same environments (both >= 0) for batched training, but allow if either is ground (env_id = -1)
                                # Allow: ground (-1) vs anything
                                skip_pair = False
                                if obj_a.env_id >= 0 and obj_b.env_id >= 0:
                                    skip_pair = True

                                if not skip_pair:
                                    pair_index = ti.atomic_add(self.pair_count[None], 1)
                                    if pair_index < self.collision_pairs.shape[0]:
                                        # Store primitivesIDright，Instead of tree node index
                                        self.collision_pairs[pair_index] = ti.Vector([node_a_prim_id, node_b_prim_id])

                        # Condition2：nodeAis a leaf node，nodeBis an internal tree node -> WillAandBThe child nodes of。
                        elif node_b_is_leaf != 1:
                            if stack_size + 2 < ti.static(self.MAX_STACK_DEPTH):  # Check against stack depth limit
                                self.stack[ia, stack_size] = node_b_left  # Use per-thread stack
                                stack_size += 1

                                self.stack[ia, stack_size] = node_b_right  # Use per-thread stack
                                stack_size += 1

    @ti.func
    def aabb_intersect(self, min_a, max_a, min_b, max_b):
        """judge twoAABBWhether it intersects"""
        return min_a.x <= max_b.x and max_a.x >= min_b.x and min_a.y <= max_b.y and max_a.y >= min_b.y
  
    def get_collision_pairs(self):
        """existPythonGet the detected collision pair results"""
        count = self.pair_count[None]
        pairs = self.collision_pairs.to_numpy()[:count]
        return pairs
