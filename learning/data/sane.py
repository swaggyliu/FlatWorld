"""Reject unphysical rollouts (PGS explosions, initial interpenetration)."""

import numpy as np


def _skip_pair_set(skip_pairs) -> set:
    """Normalize skip_pairs from a set, list, or empty npz array."""
    if skip_pairs is None:
        return set()
    if isinstance(skip_pairs, (set, frozenset)):
        pairs = list(skip_pairs)
    else:
        arr = np.asarray(skip_pairs)
        if arr.size == 0:
            return set()
        pairs = np.asarray(arr, dtype=np.int64).reshape(-1, 2)
    skip = set()
    for a, b in pairs:
        ia, ib = int(a), int(b)
        skip.add((ia, ib))
        skip.add((ib, ia))
    return skip


def frame0_min_gap(states: np.ndarray, geom: np.ndarray,
                   skip_pairs=None) -> float:
    """AABB separating-axis gap on frame 0. Negative = overlap."""
    s0 = np.asarray(states[0], dtype=np.float64)
    g = np.asarray(geom, dtype=np.float64)
    worst = 1.0e9
    n = s0.shape[0]
    skip = _skip_pair_set(skip_pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in skip:
                continue
            gx = abs(s0[i, 0] - s0[j, 0]) - (g[i, 0] + g[j, 0])
            gy = abs(s0[i, 1] - s0[j, 1]) - (g[i, 1] + g[j, 1])
            worst = min(worst, max(gx, gy))
    return float(worst)


def rollout_is_physical(states: np.ndarray, geom: np.ndarray = None,
                        xy_lim: float = 2.4, v_lim: float = 24.0,
                        pen_tol: float = 0.01, skip_pairs=None) -> bool:
    """True if the episode is safe to train on."""
    st = np.asarray(states)
    xy = st[..., :2]
    if not np.isfinite(st).all():
        return False
    if float(np.max(np.abs(xy))) > xy_lim:
        return False
    if float(np.max(np.abs(st[..., 3:5]))) > v_lim:
        return False
    if geom is not None and frame0_min_gap(st, geom, skip_pairs) < -pen_tol:
        return False
    return True
