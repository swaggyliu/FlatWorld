import taichi as ti


@ti.data_oriented
class SpatialHashManager:
    """Sort-based spatial hash for fast point-to-element queries.

    Algorithm (counting-sort, **parallelized**):
        1. ``addElement`` — register each element's AABB + domain tag
           (called from a parallel ``@ti.kernel``).
        2. ``setBounds``  — store grid bounding box (call from same kernel).
        3. ``build()``    — **Python method** that launches a sequence of
           parallel ``@ti.kernel`` s:

           a. Compute grid dimensions, clamp cell size to avoid overflow.
           b. Expand elements to (cellID, elemIdx) pairs — parallel.
           c. Count pairs per cell — parallel.
           d. Exclusive prefix-sum → cell offsets — serial (cells only).
           e. Scatter pairs + write ``cellStart``/``cellEnd`` — parallel.

        4. ``queryPoint`` / ``queryPointWithBuffer`` — map query position to
           cell(s), iterate the contiguous slice, filter by ``domainID``.

    Memory is O(N + C).  No per-cell capacity limit.
    Build is O(N + C) work, spread across GPU threads.

    Args:
        d:            Spatial dimension (2 or 3).
        max_elements: Upper bound on boundary elements.
        max_cells:    Upper bound on grid cells.
        max_elements_per_cell: *Ignored* (API compatibility).
        max_query_results: Max results returned per query call.
    """

    def __init__(
        self,
        d: int,
        max_elements: int = 100000,
        max_cells: int = 100000,
        max_elements_per_cell: int = 200,
        max_query_results: int = 512,
    ):
        self.d = d
        self.MAX_ELEMENTS = max_elements
        self.MAX_CELLS = max_cells
        self.max_cells_per_element = 3**d  # worst-case cells/elem
        self.MAX_PAIRS = max_elements * self.max_cells_per_element
        self.MAX_QUERY = max_query_results
        self.MAX_ELEMENTS_PER_CELL = max_elements_per_cell  # legacy alias

        mem_mb = (
            self.MAX_ELEMENTS * (d * 4 * 2 + 8)
            + self.MAX_PAIRS * 8  # pairCellId + pairElemIdx
            + self.MAX_PAIRS * 4  # _sortedElemIdx
            + self.MAX_CELLS * 16  # count + offset + start + end
        ) / 1e6
        print(
            f"[SortedSpatialHash] d={d}, elems={max_elements}, "
            f"cells={max_cells}, pairs={self.MAX_PAIRS} (~{mem_mb:.0f} MB)"
        )

        # ── Grid metadata ──
        self.globalbbox = ti.Vector.field(d, ti.f32, shape=2)
        self.cellNumbers = ti.Vector.field(d, ti.i32, shape=())
        self.total_cells = ti.field(ti.i32, shape=())
        self.gridSize = ti.Vector.field(d, ti.f32, shape=())
        self.estimateSize = ti.field(ti.f32, shape=())
        self._sumSize = ti.field(ti.f32, shape=())

        # ── Per-element (filled by addElement) ──
        self.numElements = ti.field(ti.i32, shape=())
        self.domainIds = ti.Vector.field(2, ti.i32, shape=self.MAX_ELEMENTS)
        self.elementbbox = ti.Vector.field(d, ti.f32, shape=(self.MAX_ELEMENTS, 2))

        # ── Pair arrays: unsorted input ──
        self.numPairs = ti.field(ti.i32, shape=())
        self.pairCellId = ti.field(ti.i32, shape=self.MAX_PAIRS)
        self.pairElemIdx = ti.field(ti.i32, shape=self.MAX_PAIRS)

        # ── Sorted output (cellId implicit via cellStart/End) ──
        self._sortedElemIdx = ti.field(ti.i32, shape=self.MAX_PAIRS)

        # ── Cell scratch + result ──
        self._cellCount = ti.field(ti.i32, shape=self.MAX_CELLS)
        self._cellOffset = ti.field(ti.i32, shape=self.MAX_CELLS)
        self.cellStart = ti.field(ti.i32, shape=self.MAX_CELLS)
        self.cellEnd = ti.field(ti.i32, shape=self.MAX_CELLS)

    # ================================================================
    #  Reset  (call from Python before each populate cycle)
    # ================================================================
    @ti.kernel
    def reset(self):
        self.numElements[None] = 0
        self.numPairs[None] = 0
        self.estimateSize[None] = 1e30
        self._sumSize[None] = 0.0
        self.total_cells[None] = 0

    # ================================================================
    #  addElement  (@ti.func — call from a parallel kernel)
    # ================================================================
    @ti.func
    def addElement(self, lb, ub, domainID: ti.i32, elementid: ti.i32, buffer):
        idx = ti.atomic_add(self.numElements[None], 1)
        if idx < self.MAX_ELEMENTS:
            self.domainIds[idx][0] = domainID
            self.domainIds[idx][1] = elementid

            bufferlb = lb - buffer
            bufferub = ub + buffer
            self.elementbbox[idx, 0] = bufferlb
            self.elementbbox[idx, 1] = bufferub

            diag = (bufferub - bufferlb).norm()
            ti.atomic_min(self.estimateSize[None], diag)
            ti.atomic_add(self._sumSize[None], diag)
        else:
            print(f"\033[91m[SortedSH] MAX_ELEMENTS ({self.MAX_ELEMENTS}) " f"exceeded!\033[0m")

    # ================================================================
    #  setBounds  (@ti.func — call at end of addElement kernel)
    # ================================================================
    @ti.func
    def setBounds(self, lb, ub):
        """Store grid bounding box from inside a Taichi kernel."""
        self.globalbbox[0] = lb
        self.globalbbox[1] = ub

    # ================================================================
    #  build  (Python method — launches parallel kernels)
    # ================================================================
    def build(self, lb=None, ub=None):
        """Build the spatial hash after elements have been added.

        Call from **Python** (not from inside a kernel).

        Args:
            lb, ub: Optional grid bounds (list / ndarray).  If omitted the
                    bounds stored by ``setBounds()`` inside the addElement
                    kernel are used.
        """
        if lb is not None:
            self.globalbbox[0] = lb
            self.globalbbox[1] = ub
        self._build_grid_kernel()
        if self.numElements[None] == 0:
            return
        self._expand_pairs_kernel()
        self._count_pairs_kernel()
        self._prefix_sum_kernel()
        self._scatter_and_index_kernel()

    # ── K1: cell sizing (single-thread, clamp to MAX_CELLS) ─────────
    @ti.kernel
    def _build_grid_kernel(self):
        cellSize = self.estimateSize[None] * 1.5

        # print(f"Estimated cell size: {cellSize:.4f} (sumSize={self._sumSize[None]:.4f}, estimateSize={self.estimateSize[None]:.4f})")

        extent = self.globalbbox[1] - self.globalbbox[0]

        # Clamp cell size so total_cells cannot exceed MAX_CELLS
        max_cells_f = ti.cast(self.MAX_CELLS, ti.f32)
        max_per_dim = ti.pow(max_cells_f, 1.0 / self.d)
        max_per_dim = ti.max(max_per_dim, 1.0)
        for k in ti.static(range(self.d)):
            min_cs = extent[k] / max_per_dim
            cellSize = ti.max(cellSize, min_cs)

        gridNums = extent / (cellSize + 1e-6)
        self.cellNumbers[None] = ti.max(ti.ceil(gridNums).cast(ti.i32), ti.Vector([1 for _ in range(self.d)]))
        self.gridSize[None] = extent / self.cellNumbers[None].cast(ti.f32)
        # print(f"SpatialHash gridSize: {self.gridSize[None]}, cellNumbers: {self.cellNumbers[None]}")

        total = 1
        for k in ti.static(range(self.d)):
            total *= self.cellNumbers[None][k]
        self.total_cells[None] = ti.min(total, self.MAX_CELLS)

    # ── K2: expand elements → (cell, elem) pairs  (parallel) ────────
    @ti.kernel
    def _expand_pairs_kernel(self):
        self.numPairs[None] = 0
        for i in range(self.numElements[None]):
            elb = self.elementbbox[i, 0]
            eub = self.elementbbox[i, 1]

            lpos = ti.floor((elb - self.globalbbox[0]) / self.gridSize[None]).cast(ti.i32)
            upos = ti.floor((eub - self.globalbbox[0]) / self.gridSize[None]).cast(ti.i32)

            lpos = ti.max(lpos, ti.Vector.zero(ti.i32, self.d))
            lpos = ti.min(lpos, self.cellNumbers[None] - 1)
            upos = ti.max(upos, ti.Vector.zero(ti.i32, self.d))
            upos = ti.min(upos, self.cellNumbers[None] - 1)

            for ix in range(lpos[0], upos[0] + 1):
                for iy in range(lpos[1], upos[1] + 1):
                    cid = ix + iy * self.cellNumbers[None][0]
                    if cid < self.total_cells[None]:
                        pidx = ti.atomic_add(self.numPairs[None], 1)
                        if pidx < self.MAX_PAIRS:
                            self.pairCellId[pidx] = cid
                            self.pairElemIdx[pidx] = i
           
    # ── K3: count pairs per cell  (parallel) ────────────────────────
    @ti.kernel
    def _count_pairs_kernel(self):
        tc = self.total_cells[None]
        np_ = ti.min(self.numPairs[None], self.MAX_PAIRS)
        # Clear counts
        for c in range(tc):
            self._cellCount[c] = 0
        # Accumulate (atomic — safe under parallelism)
        for p in range(np_):
            cid = self.pairCellId[p]
            if 0 <= cid < tc:
                ti.atomic_add(self._cellCount[cid], 1)

    # ── K4: exclusive prefix-sum  (serial over cells) ───────────────
    @ti.kernel
    def _prefix_sum_kernel(self):
        tc = self.total_cells[None]
        self._cellOffset[0] = 0
        ti.loop_config(serialize=True)
        for c in range(1, tc):
            self._cellOffset[c] = self._cellOffset[c - 1] + self._cellCount[c - 1]

    # ── K5: scatter pairs + set cell ranges  (parallel) ─────────────
    @ti.kernel
    def _scatter_and_index_kernel(self):
        tc = self.total_cells[None]
        np_ = ti.min(self.numPairs[None], self.MAX_PAIRS)
        # Reset cursors to offsets
        for c in range(tc):
            self._cellCount[c] = self._cellOffset[c]
        # Scatter (atomic cursor increment)
        for p in range(np_):
            cid = self.pairCellId[p]
            if 0 <= cid < tc:
                dest = ti.atomic_add(self._cellCount[cid], 1)
                if dest < self.MAX_PAIRS:
                    self._sortedElemIdx[dest] = self.pairElemIdx[p]
        # Final cell ranges
        for c in range(tc):
            self.cellStart[c] = self._cellOffset[c]
            self.cellEnd[c] = self._cellCount[c]

    # ================================================================
    #  Point → linearised cell ID
    # ================================================================
    @ti.func
    def _point_to_cell_id(self, pos):
        cid = -1
        if self.total_cells[None] > 0:
            rel = ti.floor((pos - self.globalbbox[0]) / self.gridSize[None]).cast(ti.i32)
            valid = True
            for k in ti.static(range(self.d)):
                if rel[k] < 0 or rel[k] >= self.cellNumbers[None][k]:
                    valid = False
            if valid:
                cid = rel[0] + rel[1] * self.cellNumbers[None][0]
             
        return cid

    # ================================================================
    #  queryPoint  (single cell, filter by domainID)
    # ================================================================
    @ti.func
    def queryPoint(self, pos: ti.template(), domainID: ti.i32):
        elids = ti.Vector([-1] * self.MAX_QUERY, ti.i32)
        dids = ti.Vector([-1] * self.MAX_QUERY, ti.i32)
        numPotentials = 0

        cid = self._point_to_cell_id(pos)
        if 0 <= cid < self.total_cells[None]:
            start = self.cellStart[cid]
            end = self.cellEnd[cid]
            for p in range(start, end):
                ei = self._sortedElemIdx[p]
                if (ei >= 0 and self.domainIds[ei][0] == domainID) or (ei >= 0 and domainID == -1):
                    if numPotentials < self.MAX_QUERY:
                        elids[numPotentials] = self.domainIds[ei][1]
                        dids[numPotentials] = self.domainIds[ei][0]
                        numPotentials += 1
        return elids, dids, numPotentials

    # ================================================================
    #  queryPointWithBuffer  (multi-cell, filter, deduplicate)
    # ================================================================
    @ti.func
    def queryPointWithBuffer(self, pos: ti.template(), buffer: ti.f32, domainID: ti.i32):
        elids = ti.Vector([-1] * self.MAX_QUERY, ti.i32)
        dids = ti.Vector([-1] * self.MAX_QUERY, ti.i32)
        visited = ti.Vector([-1] * self.MAX_QUERY, ti.i32)
        numPotentials = 0

        if self.total_cells[None] > 0:
            qlb = pos - buffer
            qub = pos + buffer

            lpos = ti.floor((qlb - self.globalbbox[0]) / self.gridSize[None]).cast(ti.i32)
            upos = ti.floor((qub - self.globalbbox[0]) / self.gridSize[None]).cast(ti.i32)

            lpos = ti.max(lpos, ti.Vector.zero(ti.i32, self.d))
            lpos = ti.min(lpos, self.cellNumbers[None] - 1)
            upos = ti.max(upos, ti.Vector.zero(ti.i32, self.d))
            upos = ti.min(upos, self.cellNumbers[None] - 1)

            # print(f"Querying SH with buffer={buffer:.4f}, lpos={lpos}, upos={upos}")

            for I in ti.grouped(ti.ndrange((lpos[0], upos[0] + 1), (lpos[1], upos[1] + 1))):
                cid = I[0] + I[1] * self.cellNumbers[None][0]
                if 0 <= cid < self.total_cells[None]:
                    start = self.cellStart[cid]
                    end = self.cellEnd[cid]
                    for p in range(start, end):
                        ei = self._sortedElemIdx[p]
                        if (ei >= 0 and self.domainIds[ei][0] == domainID) or (ei >= 0 and domainID == -1):
                            already = False
                            for k in range(numPotentials):
                                if visited[k] == ei:
                                    already = True
                                    break
                            if not already and numPotentials < self.MAX_QUERY:
                                elids[numPotentials] = self.domainIds[ei][1]
                                dids[numPotentials] = self.domainIds[ei][0]
                                visited[numPotentials] = ei
                                numPotentials += 1
           
        return elids, dids, numPotentials

    # ================================================================
    #  Pretty-print
    # ================================================================
    def __str__(self):
        lines = [
            f"SortedSpatialHash ({self.d}D)",
            f"  Elements: {self.numElements[None]}/{self.MAX_ELEMENTS}",
            f"  Pairs (elem x cell): {self.numPairs[None]}/{self.MAX_PAIRS}",
            f"  Grid: {self.cellNumbers[None].to_numpy()}",
            f"  Cell size: {self.gridSize[None].to_numpy()}",
            f"  Total cells: {self.total_cells[None]}/{self.MAX_CELLS}",
            f"  BBox: {self.globalbbox[0].to_numpy()} -> {self.globalbbox[1].to_numpy()}",
            "",
        ]

        non_empty = 0
        max_occ = 0
        for i in range(self.total_cells[None]):
            cnt = self.cellEnd[i] - self.cellStart[i]
            if cnt > 0:
                non_empty += 1
                max_occ = max(max_occ, cnt)

        lines.append(
            f"  Non-empty cells: {non_empty}/{self.total_cells[None]} "
            f"({100*non_empty/max(self.total_cells[None], 1):.1f}%)"
        )
        lines.append(f"  Max elements in a cell: {max_occ}")

        lines.append("  Cell contents (first 10 non-empty):")
        shown = 0
        for i in range(min(self.total_cells[None], 10000)):
            s, e = self.cellStart[i], self.cellEnd[i]
            cnt = e - s
            if cnt > 0:
                elems = [int(self._sortedElemIdx[j]) for j in range(s, min(e, s + 10))]
                tag = f"... ({cnt} total)" if cnt > 10 else ""
                lines.append(f"    Cell {i}: {elems}{tag}")
                shown += 1
                if shown >= 10:
                    break

        return "\n".join(lines)
