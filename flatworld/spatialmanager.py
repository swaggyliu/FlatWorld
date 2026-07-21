"""Sort-based spatial hash for fast point-to-element queries (NVIDIA Warp).

Algorithm (counting-sort, parallelized):
    1. ``add_element`` — register each element's AABB + domain tag
       (call from a parallel ``@wp.kernel``).
    2. ``set_bounds``  — store grid bounding box (call from same kernel).
    3. ``build()``     — Python method that launches a sequence of kernels.
    4. ``query_point`` / ``query_point_with_buffer`` — map query position to
       cell(s), iterate the contiguous slice, filter by ``domainID``.

Device code must call the module-level ``@wp.func`` helpers with explicit
array arguments (Warp kernels cannot access ``self`` fields).  Class methods
``addElement`` / ``setBounds`` / ``queryPoint`` / ``queryPointWithBuffer`` are
host helpers that wrap those funcs for Python / smoke tests.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from wp_init import ensure_warp


# ---------------------------------------------------------------------------
# Device helpers (call from @wp.kernel / @wp.func with manager arrays)
# ---------------------------------------------------------------------------


@wp.func
def add_element(
    lb: wp.vec2,
    ub: wp.vec2,
    domain_id: int,
    element_id: int,
    buffer: float,
    max_elements: int,
    num_elements: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    element_bbox: wp.array(dtype=wp.vec2, ndim=2),
    estimate_size: wp.array(dtype=float),
    sum_size: wp.array(dtype=float),
):
    """Register one element AABB into the spatial hash (device)."""
    idx = int(wp.atomic_add(num_elements, 0, 1))
    if idx < max_elements:
        domain_ids[idx] = wp.vec2i(domain_id, element_id)
        buf = wp.vec2(buffer, buffer)
        buffer_lb = lb - buf
        buffer_ub = ub + buf
        element_bbox[idx, 0] = buffer_lb
        element_bbox[idx, 1] = buffer_ub
        diag = wp.length(buffer_ub - buffer_lb)
        wp.atomic_min(estimate_size, 0, diag)
        wp.atomic_add(sum_size, 0, diag)
    else:
        wp.printf("[SortedSH] MAX_ELEMENTS exceeded!\n")


@wp.func
def set_bounds(
    lb: wp.vec2,
    ub: wp.vec2,
    global_bbox: wp.array(dtype=wp.vec2),
):
    """Store grid bounding box (device)."""
    global_bbox[0] = lb
    global_bbox[1] = ub


@wp.func
def _point_to_cell_id(
    pos: wp.vec2,
    total_cells: wp.array(dtype=int),
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
):
    cid = int(-1)
    if total_cells[0] > 0:
        gs = grid_size[0]
        cn = cell_numbers[0]
        delta = pos - global_bbox[0]
        rel = wp.vec2i(
            int(wp.floor(delta[0] / gs[0])),
            int(wp.floor(delta[1] / gs[1])),
        )
        valid = True
        if rel[0] < 0 or rel[0] >= cn[0]:
            valid = False
        if rel[1] < 0 or rel[1] >= cn[1]:
            valid = False
        if valid:
            cid = rel[0] + rel[1] * cn[0]
    return cid


@wp.func
def query_point(
    pos: wp.vec2,
    domain_id: int,
    max_query: int,
    total_cells: wp.array(dtype=int),
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
    cell_start: wp.array(dtype=int),
    cell_end: wp.array(dtype=int),
    sorted_elem_idx: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    query_elids: wp.array(dtype=int),
):
    """Single-cell query; writes element ids into ``query_elids``, returns count."""
    num_potentials = int(0)
    cid = _point_to_cell_id(pos, total_cells, global_bbox, grid_size, cell_numbers)
    tc = total_cells[0]
    if 0 <= cid < tc:
        start = cell_start[cid]
        end = cell_end[cid]
        for p in range(start, end):
            ei = sorted_elem_idx[p]
            if ei >= 0:
                did = domain_ids[ei][0]
                if did == domain_id or domain_id == -1:
                    if num_potentials < max_query:
                        query_elids[num_potentials] = domain_ids[ei][1]
                        num_potentials += 1
    return num_potentials


@wp.func
def query_point_with_buffer(
    pos: wp.vec2,
    buffer: float,
    domain_id: int,
    max_query: int,
    total_cells: wp.array(dtype=int),
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
    cell_start: wp.array(dtype=int),
    cell_end: wp.array(dtype=int),
    sorted_elem_idx: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    query_elids: wp.array(dtype=int),
):
    """Multi-cell query with AABB buffer; deduplicates by element id."""
    num_potentials = int(0)
    tc = total_cells[0]
    if tc > 0:
        buf = wp.vec2(buffer, buffer)
        qlb = pos - buf
        qub = pos + buf
        gs = grid_size[0]
        cn = cell_numbers[0]
        origin = global_bbox[0]

        dlb = qlb - origin
        dub = qub - origin
        lx = int(wp.floor(dlb[0] / gs[0]))
        ly = int(wp.floor(dlb[1] / gs[1]))
        ux = int(wp.floor(dub[0] / gs[0]))
        uy = int(wp.floor(dub[1] / gs[1]))

        lx = int(wp.max(lx, 0))
        ly = int(wp.max(ly, 0))
        lx = int(wp.min(lx, cn[0] - 1))
        ly = int(wp.min(ly, cn[1] - 1))
        ux = int(wp.max(ux, 0))
        uy = int(wp.max(uy, 0))
        ux = int(wp.min(ux, cn[0] - 1))
        uy = int(wp.min(uy, cn[1] - 1))

        for ix in range(lx, ux + 1):
            for iy in range(ly, uy + 1):
                cid = ix + iy * cn[0]
                if 0 <= cid < tc:
                    start = cell_start[cid]
                    end = cell_end[cid]
                    for p in range(start, end):
                        ei = sorted_elem_idx[p]
                        if ei >= 0:
                            did = domain_ids[ei][0]
                            if did == domain_id or domain_id == -1:
                                elem_id = domain_ids[ei][1]
                                already = int(0)
                                for k in range(num_potentials):
                                    # Dedup by element id (not internal index ei)
                                    if query_elids[k] == elem_id:
                                        already = 1
                                if already == 0 and num_potentials < max_query:
                                    query_elids[num_potentials] = elem_id
                                    num_potentials += 1
    return num_potentials


# ---------------------------------------------------------------------------
# Build kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _reset_kernel(
    num_elements: wp.array(dtype=int),
    num_pairs: wp.array(dtype=int),
    estimate_size: wp.array(dtype=float),
    sum_size: wp.array(dtype=float),
    total_cells: wp.array(dtype=int),
):
    num_elements[0] = 0
    num_pairs[0] = 0
    estimate_size[0] = 1e30
    sum_size[0] = 0.0
    total_cells[0] = 0


@wp.kernel
def _build_grid_kernel(
    max_cells: int,
    estimate_size: wp.array(dtype=float),
    global_bbox: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
    grid_size: wp.array(dtype=wp.vec2),
    total_cells: wp.array(dtype=int),
):
    cell_size = estimate_size[0] * 1.5
    extent = global_bbox[1] - global_bbox[0]

    max_cells_f = float(max_cells)
    max_per_dim = wp.pow(max_cells_f, 0.5)
    max_per_dim = wp.max(max_per_dim, 1.0)

    min_cs_x = extent[0] / max_per_dim
    min_cs_y = extent[1] / max_per_dim
    cell_size = wp.max(cell_size, min_cs_x)
    cell_size = wp.max(cell_size, min_cs_y)

    cs = cell_size + 1e-6
    nx = int(wp.max(wp.ceil(extent[0] / cs), 1.0))
    ny = int(wp.max(wp.ceil(extent[1] / cs), 1.0))
    cell_numbers[0] = wp.vec2i(nx, ny)
    grid_size[0] = wp.vec2(extent[0] / float(nx), extent[1] / float(ny))

    total = nx * ny
    total_cells[0] = int(wp.min(total, max_cells))


@wp.kernel
def _expand_pairs_kernel(
    num_elements: wp.array(dtype=int),
    num_pairs: wp.array(dtype=int),
    max_pairs: int,
    total_cells: wp.array(dtype=int),
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
    element_bbox: wp.array(dtype=wp.vec2, ndim=2),
    pair_cell_id: wp.array(dtype=int),
    pair_elem_idx: wp.array(dtype=int),
):
    i = wp.tid()
    ne = num_elements[0]
    if i >= ne:
        return

    elb = element_bbox[i, 0]
    eub = element_bbox[i, 1]
    gs = grid_size[0]
    cn = cell_numbers[0]
    origin = global_bbox[0]
    tc = total_cells[0]

    dlb = elb - origin
    dub = eub - origin
    lx = int(wp.floor(dlb[0] / gs[0]))
    ly = int(wp.floor(dlb[1] / gs[1]))
    ux = int(wp.floor(dub[0] / gs[0]))
    uy = int(wp.floor(dub[1] / gs[1]))

    lx = int(wp.max(lx, 0))
    ly = int(wp.max(ly, 0))
    lx = int(wp.min(lx, cn[0] - 1))
    ly = int(wp.min(ly, cn[1] - 1))
    ux = int(wp.max(ux, 0))
    uy = int(wp.max(uy, 0))
    ux = int(wp.min(ux, cn[0] - 1))
    uy = int(wp.min(uy, cn[1] - 1))

    for ix in range(lx, ux + 1):
        for iy in range(ly, uy + 1):
            cid = ix + iy * cn[0]
            if cid < tc:
                pidx = int(wp.atomic_add(num_pairs, 0, 1))
                if pidx < max_pairs:
                    pair_cell_id[pidx] = cid
                    pair_elem_idx[pidx] = i


@wp.kernel
def _clear_cell_count_kernel(
    cell_count: wp.array(dtype=int),
    total_cells: wp.array(dtype=int),
):
    c = wp.tid()
    if c < total_cells[0]:
        cell_count[c] = 0


@wp.kernel
def _count_pairs_kernel(
    num_pairs: wp.array(dtype=int),
    max_pairs: int,
    total_cells: wp.array(dtype=int),
    pair_cell_id: wp.array(dtype=int),
    cell_count: wp.array(dtype=int),
):
    p = wp.tid()
    np_ = int(wp.min(num_pairs[0], max_pairs))
    if p >= np_:
        return
    tc = total_cells[0]
    cid = pair_cell_id[p]
    if 0 <= cid < tc:
        wp.atomic_add(cell_count, cid, 1)


@wp.kernel
def _prefix_sum_kernel(
    cell_count: wp.array(dtype=int),
    cell_offset: wp.array(dtype=int),
    total_cells: wp.array(dtype=int),
):
    # Serial exclusive prefix-sum over cells (launch dim=1).
    tc = total_cells[0]
    if tc > 0:
        cell_offset[0] = 0
        for c in range(1, tc):
            cell_offset[c] = cell_offset[c - 1] + cell_count[c - 1]


@wp.kernel
def _scatter_reset_cursors_kernel(
    cell_count: wp.array(dtype=int),
    cell_offset: wp.array(dtype=int),
    total_cells: wp.array(dtype=int),
):
    c = wp.tid()
    if c < total_cells[0]:
        cell_count[c] = cell_offset[c]


@wp.kernel
def _scatter_pairs_kernel(
    num_pairs: wp.array(dtype=int),
    max_pairs: int,
    total_cells: wp.array(dtype=int),
    pair_cell_id: wp.array(dtype=int),
    pair_elem_idx: wp.array(dtype=int),
    cell_count: wp.array(dtype=int),
    sorted_elem_idx: wp.array(dtype=int),
):
    p = wp.tid()
    np_ = int(wp.min(num_pairs[0], max_pairs))
    if p >= np_:
        return
    tc = total_cells[0]
    cid = pair_cell_id[p]
    if 0 <= cid < tc:
        dest = int(wp.atomic_add(cell_count, cid, 1))
        if dest < max_pairs:
            sorted_elem_idx[dest] = pair_elem_idx[p]


@wp.kernel
def _write_cell_ranges_kernel(
    cell_count: wp.array(dtype=int),
    cell_offset: wp.array(dtype=int),
    cell_start: wp.array(dtype=int),
    cell_end: wp.array(dtype=int),
    total_cells: wp.array(dtype=int),
):
    c = wp.tid()
    if c < total_cells[0]:
        cell_start[c] = cell_offset[c]
        cell_end[c] = cell_count[c]


@wp.kernel
def _host_add_element_kernel(
    lb: wp.vec2,
    ub: wp.vec2,
    domain_id: int,
    element_id: int,
    buffer: float,
    max_elements: int,
    num_elements: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    element_bbox: wp.array(dtype=wp.vec2, ndim=2),
    estimate_size: wp.array(dtype=float),
    sum_size: wp.array(dtype=float),
):
    add_element(
        lb,
        ub,
        domain_id,
        element_id,
        buffer,
        max_elements,
        num_elements,
        domain_ids,
        element_bbox,
        estimate_size,
        sum_size,
    )


@wp.kernel
def _host_set_bounds_kernel(
    lb: wp.vec2,
    ub: wp.vec2,
    global_bbox: wp.array(dtype=wp.vec2),
):
    set_bounds(lb, ub, global_bbox)


@wp.kernel
def _host_query_point_kernel(
    pos: wp.vec2,
    domain_id: int,
    max_query: int,
    total_cells: wp.array(dtype=int),
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
    cell_start: wp.array(dtype=int),
    cell_end: wp.array(dtype=int),
    sorted_elem_idx: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    query_elids: wp.array(dtype=int),
    out_count: wp.array(dtype=int),
):
    out_count[0] = query_point(
        pos,
        domain_id,
        max_query,
        total_cells,
        global_bbox,
        grid_size,
        cell_numbers,
        cell_start,
        cell_end,
        sorted_elem_idx,
        domain_ids,
        query_elids,
    )


@wp.kernel
def _host_query_point_with_buffer_kernel(
    pos: wp.vec2,
    buffer: float,
    domain_id: int,
    max_query: int,
    total_cells: wp.array(dtype=int),
    global_bbox: wp.array(dtype=wp.vec2),
    grid_size: wp.array(dtype=wp.vec2),
    cell_numbers: wp.array(dtype=wp.vec2i),
    cell_start: wp.array(dtype=int),
    cell_end: wp.array(dtype=int),
    sorted_elem_idx: wp.array(dtype=int),
    domain_ids: wp.array(dtype=wp.vec2i),
    query_elids: wp.array(dtype=int),
    out_count: wp.array(dtype=int),
):
    out_count[0] = query_point_with_buffer(
        pos,
        buffer,
        domain_id,
        max_query,
        total_cells,
        global_bbox,
        grid_size,
        cell_numbers,
        cell_start,
        cell_end,
        sorted_elem_idx,
        domain_ids,
        query_elids,
    )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SpatialHashManager:
    """Sort-based spatial hash for fast point-to-element queries.

    Args:
        d: Spatial dimension (2 only; 3D expand path was never complete).
        max_elements: Upper bound on boundary elements.
        max_cells: Upper bound on grid cells.
        max_elements_per_cell: Ignored (API compatibility).
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
        ensure_warp()
        if d != 2:
            raise ValueError(f"SpatialHashManager Warp migration supports d=2 only (got {d})")

        self.d = d
        self.MAX_ELEMENTS = max_elements
        self.MAX_CELLS = max_cells
        self.max_cells_per_element = 3**d
        self.MAX_PAIRS = max_elements * self.max_cells_per_element
        self.MAX_QUERY = max_query_results
        self.MAX_ELEMENTS_PER_CELL = max_elements_per_cell

        mem_mb = (
            self.MAX_ELEMENTS * (d * 4 * 2 + 8)
            + self.MAX_PAIRS * 8
            + self.MAX_PAIRS * 4
            + self.MAX_CELLS * 16
        ) / 1e6
        print(
            f"[SortedSpatialHash] d={d}, elems={max_elements}, "
            f"cells={max_cells}, pairs={self.MAX_PAIRS} (~{mem_mb:.0f} MB)"
        )

        # Grid metadata
        self.globalbbox = wp.zeros(2, dtype=wp.vec2)
        self.cellNumbers = wp.zeros(1, dtype=wp.vec2i)
        self.total_cells = wp.zeros(1, dtype=int)
        self.gridSize = wp.zeros(1, dtype=wp.vec2)
        self.estimateSize = wp.zeros(1, dtype=float)
        self._sumSize = wp.zeros(1, dtype=float)

        # Per-element
        self.numElements = wp.zeros(1, dtype=int)
        self.domainIds = wp.zeros(self.MAX_ELEMENTS, dtype=wp.vec2i)
        self.elementbbox = wp.zeros((self.MAX_ELEMENTS, 2), dtype=wp.vec2)

        # Pair arrays
        self.numPairs = wp.zeros(1, dtype=int)
        self.pairCellId = wp.zeros(self.MAX_PAIRS, dtype=int)
        self.pairElemIdx = wp.zeros(self.MAX_PAIRS, dtype=int)

        self._sortedElemIdx = wp.zeros(self.MAX_PAIRS, dtype=int)

        self._cellCount = wp.zeros(self.MAX_CELLS, dtype=int)
        self._cellOffset = wp.zeros(self.MAX_CELLS, dtype=int)
        self.cellStart = wp.zeros(self.MAX_CELLS, dtype=int)
        self.cellEnd = wp.zeros(self.MAX_CELLS, dtype=int)

        self.queryElids = wp.zeros(self.MAX_QUERY, dtype=int)
        self._query_count = wp.zeros(1, dtype=int)

    def reset(self):
        """Reset counters before a populate cycle (host)."""
        wp.launch(
            _reset_kernel,
            dim=1,
            inputs=[
                self.numElements,
                self.numPairs,
                self.estimateSize,
                self._sumSize,
                self.total_cells,
            ],
        )

    def addElement(self, lb, ub, domainID: int, elementid: int, buffer: float):
        """Host helper to register one element. Device code: ``add_element``."""
        lb_v = wp.vec2(float(lb[0]), float(lb[1]))
        ub_v = wp.vec2(float(ub[0]), float(ub[1]))
        wp.launch(
            _host_add_element_kernel,
            dim=1,
            inputs=[
                lb_v,
                ub_v,
                int(domainID),
                int(elementid),
                float(buffer),
                self.MAX_ELEMENTS,
                self.numElements,
                self.domainIds,
                self.elementbbox,
                self.estimateSize,
                self._sumSize,
            ],
        )

    def setBounds(self, lb, ub):
        """Host helper to store grid bounds. Device code: ``set_bounds``."""
        lb_v = wp.vec2(float(lb[0]), float(lb[1]))
        ub_v = wp.vec2(float(ub[0]), float(ub[1]))
        wp.launch(
            _host_set_bounds_kernel,
            dim=1,
            inputs=[lb_v, ub_v, self.globalbbox],
        )

    def build(self, lb=None, ub=None):
        """Build the spatial hash after elements have been added (host)."""
        if lb is not None:
            self.setBounds(lb, ub)

        wp.launch(
            _build_grid_kernel,
            dim=1,
            inputs=[
                self.MAX_CELLS,
                self.estimateSize,
                self.globalbbox,
                self.cellNumbers,
                self.gridSize,
                self.total_cells,
            ],
        )

        ne = int(self.numElements.numpy()[0])
        if ne == 0:
            return

        self.numPairs.zero_()
        wp.launch(
            _expand_pairs_kernel,
            dim=ne,
            inputs=[
                self.numElements,
                self.numPairs,
                self.MAX_PAIRS,
                self.total_cells,
                self.globalbbox,
                self.gridSize,
                self.cellNumbers,
                self.elementbbox,
                self.pairCellId,
                self.pairElemIdx,
            ],
        )

        tc = int(self.total_cells.numpy()[0])
        np_ = int(min(int(self.numPairs.numpy()[0]), self.MAX_PAIRS))

        if tc > 0:
            wp.launch(
                _clear_cell_count_kernel,
                dim=tc,
                inputs=[self._cellCount, self.total_cells],
            )
        if np_ > 0:
            wp.launch(
                _count_pairs_kernel,
                dim=np_,
                inputs=[
                    self.numPairs,
                    self.MAX_PAIRS,
                    self.total_cells,
                    self.pairCellId,
                    self._cellCount,
                ],
            )

        wp.launch(
            _prefix_sum_kernel,
            dim=1,
            inputs=[self._cellCount, self._cellOffset, self.total_cells],
        )

        if tc > 0:
            wp.launch(
                _scatter_reset_cursors_kernel,
                dim=tc,
                inputs=[self._cellCount, self._cellOffset, self.total_cells],
            )
        if np_ > 0:
            wp.launch(
                _scatter_pairs_kernel,
                dim=np_,
                inputs=[
                    self.numPairs,
                    self.MAX_PAIRS,
                    self.total_cells,
                    self.pairCellId,
                    self.pairElemIdx,
                    self._cellCount,
                    self._sortedElemIdx,
                ],
            )
        if tc > 0:
            wp.launch(
                _write_cell_ranges_kernel,
                dim=tc,
                inputs=[
                    self._cellCount,
                    self._cellOffset,
                    self.cellStart,
                    self.cellEnd,
                    self.total_cells,
                ],
            )

    def queryPoint(self, pos, domainID: int) -> int:
        """Host helper. Device code: ``query_point`` with explicit arrays."""
        pos_v = wp.vec2(float(pos[0]), float(pos[1]))
        wp.launch(
            _host_query_point_kernel,
            dim=1,
            inputs=[
                pos_v,
                int(domainID),
                self.MAX_QUERY,
                self.total_cells,
                self.globalbbox,
                self.gridSize,
                self.cellNumbers,
                self.cellStart,
                self.cellEnd,
                self._sortedElemIdx,
                self.domainIds,
                self.queryElids,
                self._query_count,
            ],
        )
        return int(self._query_count.numpy()[0])

    def queryPointWithBuffer(self, pos, buffer: float, domainID: int) -> int:
        """Host helper. Device code: ``query_point_with_buffer``."""
        pos_v = wp.vec2(float(pos[0]), float(pos[1]))
        wp.launch(
            _host_query_point_with_buffer_kernel,
            dim=1,
            inputs=[
                pos_v,
                float(buffer),
                int(domainID),
                self.MAX_QUERY,
                self.total_cells,
                self.globalbbox,
                self.gridSize,
                self.cellNumbers,
                self.cellStart,
                self.cellEnd,
                self._sortedElemIdx,
                self.domainIds,
                self.queryElids,
                self._query_count,
            ],
        )
        return int(self._query_count.numpy()[0])

    def __str__(self):
        ne = int(self.numElements.numpy()[0])
        np_ = int(self.numPairs.numpy()[0])
        tc = int(self.total_cells.numpy()[0])
        cn = self.cellNumbers.numpy()[0]
        gs = self.gridSize.numpy()[0]
        bb0 = self.globalbbox.numpy()[0]
        bb1 = self.globalbbox.numpy()[1]
        cell_start = self.cellStart.numpy()
        cell_end = self.cellEnd.numpy()
        sorted_idx = self._sortedElemIdx.numpy()

        lines = [
            f"SortedSpatialHash ({self.d}D)",
            f"  Elements: {ne}/{self.MAX_ELEMENTS}",
            f"  Pairs (elem x cell): {np_}/{self.MAX_PAIRS}",
            f"  Grid: {cn}",
            f"  Cell size: {gs}",
            f"  Total cells: {tc}/{self.MAX_CELLS}",
            f"  BBox: {bb0} -> {bb1}",
            "",
        ]

        non_empty = 0
        max_occ = 0
        for i in range(tc):
            cnt = int(cell_end[i] - cell_start[i])
            if cnt > 0:
                non_empty += 1
                max_occ = max(max_occ, cnt)

        lines.append(
            f"  Non-empty cells: {non_empty}/{tc} "
            f"({100 * non_empty / max(tc, 1):.1f}%)"
        )
        lines.append(f"  Max elements in a cell: {max_occ}")
        lines.append("  Cell contents (first 10 non-empty):")

        shown = 0
        for i in range(min(tc, 10000)):
            s, e = int(cell_start[i]), int(cell_end[i])
            cnt = e - s
            if cnt > 0:
                elems = [int(sorted_idx[j]) for j in range(s, min(e, s + 10))]
                tag = f"... ({cnt} total)" if cnt > 10 else ""
                lines.append(f"    Cell {i}: {elems}{tag}")
                shown += 1
                if shown >= 10:
                    break

        return "\n".join(lines)
