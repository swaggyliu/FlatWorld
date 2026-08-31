"""Reject unphysical rollouts (PGS explosions, initial interpenetration)."""

import numpy as np


def frame0_min_gap(states: np.ndarray, geom: np.ndarray) -> float:
    """AABB separating-axis gap on frame 0. Negative = overlap."""
    s0 = np.asarray(states[0], dtype=np.float64)
    g = np.asarray(geom, dtype=np.float64)
    worst = 1.0e9
    n = s0.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            gx = abs(s0[i, 0] - s0[j, 0]) - (g[i, 0] + g[j, 0])
            gy = abs(s0[i, 1] - s0[j, 1]) - (g[i, 1] + g[j, 1])
            worst = min(worst, max(gx, gy))
    return float(worst)


def rollout_is_physical(states: np.ndarray, geom: np.ndarray = None,
                        xy_lim: float = 2.4, v_lim: float = 8.0,
                        pen_tol: float = 0.01) -> bool:
    """True if the episode is safe to train on."""
    st = np.asarray(states)
    xy = st[..., :2]
    if not np.isfinite(st).all():
        return False
    if float(np.max(np.abs(xy))) > xy_lim:
        return False
    if float(np.max(np.abs(st[..., 3:5]))) > v_lim:
        return False
    if geom is not None and frame0_min_gap(st, geom) < -pen_tol:
        return False
    return True
