"""Sequence-window dataset over collected rollout npz files.

Each __getitem__ samples a random time window of length window+1 from a
random rollout and returns normalized tensors:
    obj_types        (N,)            long
    obj_states       (T_w+1, N, 6)   float (normalized)
    actions          (T_w, 2)        float (normalized)
    contact_feat     (T_w+1, K, 7)   float (normalized, padding rows zeroed)
    contact_mask     (T_w+1, K)      float
    tactile_summary  (T_w+1, 4)      float (normalized)

Supervision: (s_t, a_t) -> s_{t+1} for t in [0, T_w).
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from learning.data.normalizer import Normalizer
from learning.data.sane import rollout_is_physical


def _catch_event_frames(states: np.ndarray, obj_types: np.ndarray,
                        geom: np.ndarray, v_hi: float = 0.30,
                        v_lo: float = 0.06, win: int = 6):
    """Frames where a rolling ball is stopped by a collision (raw units).

    A frame t is a catch-event frame when some ball i has speed > v_hi at
    t-win and < v_lo at t. Rolling friction is ~0, so only collisions do
    this; ordinary EE pushing never produces a >0.3 m/s -> <0.06 m/s drop.
    Returns a list of such t. Used for event-anchored window sampling: these
    transitions carry the "ball hits a (quasi-)static obstacle and stops"
    physics the baseline distribution almost never visits.
    """
    frames = []
    T = states.shape[0]
    if obj_types.ndim != 1:
        return frames
    for i in range(states.shape[1]):
        if int(obj_types[i]) != 2:
            continue
        if abs(float(geom[i, 0]) - float(geom[i, 1])) > 1e-6:
            continue  # not a ball
        v = np.linalg.norm(states[:, i, 3:5], axis=-1)
        for t in range(win, T):
            if v[t] < v_lo and v[t - win] > v_hi:
                # keep the first qualifying frame of each stop
                if not frames or t - frames[-1] > win:
                    frames.append(t)
    return frames


def _pair_labels(xy: np.ndarray, geom: np.ndarray, gap_eps: float = 0.015):
    """Fallback pair / ground flags from signed gap (old npz without solver).

    Prefer ``pair_contact`` / ``ground_contact`` written by the collector
    from the PGS contact cache. This circle-gap approx is only for files
    that have not been relabeled yet.
    """
    rel = xy[:, :, None, :] - xy[:, None, :, :]
    dist = np.linalg.norm(rel, axis=-1)
    rad = np.max(np.abs(geom), axis=-1)
    gap = dist - rad[None, :, None] - rad[None, None, :]
    n = xy.shape[1]
    touch = (gap < gap_eps) & (~np.eye(n, dtype=bool))
    ground = (xy[..., 1] - geom[None, :, 1]) < gap_eps
    return touch.astype(np.float32), ground.astype(np.float32)


class PushWindowDataset(Dataset):
    def __init__(self, rollouts_dir: str, window: int = 20,
                 normalizer: Normalizer = None, files: list = None,
                 stride: int = 1):
        """stride > 1 implements frame skipping / action repeat: one model
        step then covers `stride` simulation frames and the effective action
        is the mean force over the chunk. This magnifies the (otherwise
        tiny) per-step force effect so the dynamics cannot ignore it."""
        self.window = window
        self.stride = stride
        self.files = files if files is not None else sorted(
            glob.glob(os.path.join(rollouts_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no rollout npz files under {rollouts_dir}")
        if normalizer is None:
            normalizer = Normalizer.fit_from_dir(rollouts_dir)
        self.norm = normalizer

        # Load all rollouts into memory (small: ~100 x (201, 6, 6) floats)
        self.rollouts = []
        skipped = 0
        kept_files = []
        for f in self.files:
            d = np.load(f)
            states = d["obj_states"].astype(np.float32)
            actions = d["actions"].astype(np.float32)
            cfeat = d["contact_feat"].astype(np.float32)
            cmask = d["contact_mask"].astype(np.float32)
            tsum = d["tactile_summary"].astype(np.float32)
            if "obj_geom" in d.files:
                geom = d["obj_geom"].astype(np.float32)
            else:
                geom = Normalizer._default_geom(d).astype(np.float32)
            if not rollout_is_physical(states, geom):
                skipped += 1
                continue

            pair_c = d["pair_contact"].astype(np.float32) if "pair_contact" in d.files \
                else None
            ground_c = d["ground_contact"].astype(np.float32) if "ground_contact" in d.files \
                else None
            if pair_c is None or ground_c is None:
                g_pair, g_ground = _pair_labels(states[..., :2], geom)
                if pair_c is None:
                    pair_c = g_pair
                if ground_c is None:
                    ground_c = g_ground

            # Catch-event frames (raw physical units, before normalization):
            # a ball decelerates from >0.15 m/s to <0.06 m/s within a few
            # frames while EE is in contact. These are the rare "ball hits a
            # static EE and stops" transitions the baseline distribution
            # almost never visits.
            ev_frames = _catch_event_frames(states, d["obj_types"], geom)

            if stride > 1:
                T = actions.shape[0]
                n_chunk = T // stride
                states = states[: n_chunk * stride + 1 : stride]
                actions = actions[: n_chunk * stride].reshape(
                    n_chunk, stride, -1).mean(axis=1)
                cfeat = cfeat[: n_chunk * stride + 1 : stride]
                cmask = cmask[: n_chunk * stride + 1 : stride]
                tsum = tsum[: n_chunk * stride + 1 : stride]
                pair_c = pair_c[: n_chunk * stride + 1 : stride]
                ground_c = ground_c[: n_chunk * stride + 1 : stride]

            states = self.norm.normalize("obj_states", states)
            actions = self.norm.normalize("actions", actions)
            cfeat = self.norm.normalize("contact_feat", cfeat)
            cfeat = cfeat * cmask[..., None]  # zero padding rows after normalization
            tsum = self.norm.normalize("tactile_summary", tsum)
            if "obj_geom" in self.norm.stats:
                geom = self.norm.normalize("obj_geom", geom)
            self.rollouts.append({
                "obj_types": torch.from_numpy(d["obj_types"].astype(np.int64)),
                "obj_states": torch.from_numpy(states),
                "actions": torch.from_numpy(actions),
                "contact_feat": torch.from_numpy(cfeat),
                "contact_mask": torch.from_numpy(cmask),
                "tactile_summary": torch.from_numpy(tsum),
                "obj_geom": torch.from_numpy(geom),
                "pair_contact": torch.from_numpy(pair_c),
                "ground_contact": torch.from_numpy(ground_c),
                "ev_frames": ev_frames,
            })
            kept_files.append(f)
        self.files = kept_files
        if skipped:
            print(f"dataset: skipped {skipped} unphysical rollouts "
                  f"(kept {len(self.rollouts)})")

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        r = self.rollouts[idx]
        T = r["actions"].shape[0]                     # number of action frames
        ev = r.get("ev_frames")
        anchor = None
        if ev and len(ev) > 0:
            p_ev = float(getattr(self, "event_anchor_p", 0.2))
            if torch.rand(1).item() < p_ev:
                # uniform choice over event frames, window centered so the
                # deceleration lands inside [t0, t1]
                f = int(ev[int(torch.randint(0, len(ev), (1,)).item())])
                f = min(max(f - self.window // 2, 0), T - self.window)
                anchor = f
        if anchor is not None:
            t0 = anchor
        else:
            t0 = int(torch.randint(0, T - self.window + 1, (1,)).item())
        t1 = t0 + self.window
        return {
            "obj_types": r["obj_types"],                              # (N,)
            "obj_states": r["obj_states"][t0:t1 + 1],                 # (T_w+1, N, 6)
            "actions": r["actions"][t0:t1],                           # (T_w, 2)
            "contact_feat": r["contact_feat"][t0:t1 + 1],             # (T_w+1, K, 7)
            "contact_mask": r["contact_mask"][t0:t1 + 1],             # (T_w+1, K)
            "tactile_summary": r["tactile_summary"][t0:t1 + 1],       # (T_w+1, 4)
            "obj_geom": r["obj_geom"],                                # (N, 2)
            "pair_contact": r["pair_contact"][t0:t1 + 1],             # (T_w+1, N, N)
            "ground_contact": r["ground_contact"][t0:t1 + 1],         # (T_w+1, N)
        }


def train_val_split(dataset: PushWindowDataset, val_ratio: float = 0.1,
                    seed: int = 0):
    """Stratified per-file split (no window leakage).

    Splits old and new (rollout index >= 4220) batches separately so
    neither train nor val is dominated by one batch. Previously the last
    N files (all new) landed in val; a seeded shuffle mitigates but
    still does not guarantee proportional coverage.
    """
    rng = np.random.default_rng(seed)
    old_ix, new_ix = [], []
    for i, f in enumerate(dataset.files):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            num = int(stem.split("_")[-1])
        except ValueError:
            num = 0
        (new_ix if num >= 4220 else old_ix).append(i)

    def _stratified_split(ix):
        ix = list(ix)
        rng.shuffle(ix)
        n_val = max(1, int(len(ix) * val_ratio)) if ix else 0
        return ix[:n_val], ix[n_val:]

    val_old, train_old = _stratified_split(old_ix)
    val_new, train_new = _stratified_split(new_ix)
    val_ix = val_old + val_new
    train_ix = train_old + train_new

    def _subset(ix):
        ds = PushWindowDataset.__new__(PushWindowDataset)
        ds.window, ds.norm = dataset.window, dataset.norm
        ds.stride = getattr(dataset, "stride", 1)
        ds.event_anchor_p = getattr(dataset, "event_anchor_p", 0.2)
        ds.files = [dataset.files[int(i)] for i in ix]
        ds.rollouts = [dataset.rollouts[int(i)] for i in ix]
        return ds

    train_ds, val_ds = _subset(train_ix), _subset(val_ix)
    print(f"stratified split seed={seed}: "
          f"train {len(train_ds)} (old {len(train_old)}/new {len(train_new)}) / "
          f"val {len(val_ds)} (old {len(val_old)}/new {len(val_new)})")
    return train_ds, val_ds
