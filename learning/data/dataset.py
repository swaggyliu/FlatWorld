"""Sequence-window dataset over collected rollout npz files.

Each __getitem__ samples a random time window of length window+1 from a
random rollout and returns tensors. Continuous fields are z-scored;
flags and types are left as-is:
    obj_types        (N,)            long (0=EE, 1=box, 2=ball)
    obj_states       (T_w+1, N, 6)   float (normalized)
    actions          (T_w, 2)        float (normalized)
    contact_feat     (T_w+1, K, 7)   float (normalized, padding rows zeroed)
    contact_mask     (T_w+1, K)      float 0/1
    tactile_summary  (T_w+1, 4)      float (normalized)
    obj_geom         (N, 2)          float (normalized)
    pair_contact     (T_w+1, N, N)   float 0/1 (solver rigid–rigid)
    ground_contact   (T_w+1, N)      float 0/1 (solver body–plane)

Supervision: (s_t, a_t) -> s_{t+1} for t in [0, T_w).
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from learning.data.normalizer import Normalizer
from learning.data.sane import rollout_is_physical


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
            if "obj_geom" not in d.files:
                skipped += 1
                continue
            geom = d["obj_geom"].astype(np.float32)
            if "pair_contact" not in d.files or "ground_contact" not in d.files:
                skipped += 1
                continue
            pair_c = d["pair_contact"].astype(np.float32)
            ground_c = d["ground_contact"].astype(np.float32)
            if not rollout_is_physical(states, geom):
                skipped += 1
                continue

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
            })
            kept_files.append(f)
        self.files = kept_files
        if skipped:
            print(f"dataset: skipped {skipped} rollouts "
                  f"(missing fields or unphysical; kept {len(self.rollouts)})")

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        r = self.rollouts[idx]
        T = r["actions"].shape[0]                     # number of action frames
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
    """Per-file shuffle split (no window leakage across train/val)."""
    rng = np.random.default_rng(seed)
    ix = np.arange(len(dataset))
    rng.shuffle(ix)
    n_val = max(1, int(len(ix) * val_ratio))
    val_ix, train_ix = ix[:n_val], ix[n_val:]

    def _subset(idx):
        ds = PushWindowDataset.__new__(PushWindowDataset)
        ds.window, ds.norm = dataset.window, dataset.norm
        ds.stride = getattr(dataset, "stride", 1)
        ds.files = [dataset.files[int(i)] for i in idx]
        ds.rollouts = [dataset.rollouts[int(i)] for i in idx]
        return ds

    train_ds, val_ds = _subset(train_ix), _subset(val_ix)
    print(f"split seed={seed}: train {len(train_ds)} / val {len(val_ds)}")
    return train_ds, val_ds
