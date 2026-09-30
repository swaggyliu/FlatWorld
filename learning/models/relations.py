"""Typed GNN edges: contact vs kinematic joints.

    0 contact     (solver / signed-gap, not a joint)
    1 revolute
    2 weld
    3 prismatic   (reserved; unused in current scenes)
    4 none        (k-NN spatial neighbour, no special physics)

Joints always occupy an edge. Contact overlays only on non-joint pairs.
"""

from __future__ import annotations

import numpy as np
import torch

REL_CONTACT = 0
REL_REVOLUTE = 1
REL_WELD = 2
REL_PRISMATIC = 3
REL_NONE = 4
NUM_REL = 5
REL_EMB_DIM = 16


def empty_rel(n: int, dtype=np.int64) -> np.ndarray:
    return np.full((n, n), REL_NONE, dtype=dtype)


def set_pair(R: np.ndarray, i: int, j: int, t: int) -> None:
    R[int(i), int(j)] = R[int(j), int(i)] = int(t)


def is_joint(rel_type: torch.Tensor) -> torch.Tensor:
    return (rel_type >= REL_REVOLUTE) & (rel_type <= REL_PRISMATIC)


def as_rel_tensor(rel_type, B: int, N: int, device) -> torch.Tensor:
    if rel_type is None:
        return torch.full((B, N, N), REL_NONE, device=device, dtype=torch.long)
    t = rel_type if torch.is_tensor(rel_type) else torch.as_tensor(
        rel_type, device=device, dtype=torch.long)
    t = t.to(device=device, dtype=torch.long)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.shape[0] == 1 and B > 1:
        t = t.expand(B, -1, -1)
    return t.contiguous()


def overlay_contact(rel_type: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    """Keep joint ids; mark contacting non-joint pairs as REL_CONTACT."""
    joint = is_joint(rel_type)
    hit = contact.to(dtype=torch.bool)
    if hit.dim() == rel_type.dim() + 1:
        hit = hit.squeeze(-1)
    return torch.where(hit & ~joint, torch.zeros_like(rel_type), rel_type)


def infer_rel_type(n: int, skip_pairs=None) -> np.ndarray:
    """Rebuild a static joint matrix from collected ``skip_pairs``.

    Push-T stores one weld pair. Reacher stores two revolutes plus a
    non-joint skip (base–link2); only the actual joints are typed.
    """
    R = empty_rel(n)
    if skip_pairs is None:
        return R
    arr = np.asarray(skip_pairs)
    if arr.size == 0:
        return R
    pairs = arr.reshape(-1, 2)
    if len(pairs) == 1:
        set_pair(R, int(pairs[0, 0]), int(pairs[0, 1]), REL_WELD)
        return R
    if n == 3:
        set_pair(R, 0, 1, REL_REVOLUTE)
        set_pair(R, 0, 2, REL_REVOLUTE)
        return R
    for p in pairs:
        set_pair(R, int(p[0]), int(p[1]), REL_WELD)
    return R
