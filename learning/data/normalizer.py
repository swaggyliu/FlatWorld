"""Input normalization statistics for states, actions and tactile signals.

Scales differ strongly across quantities (positions ~1.0, forces ~10 N,
penetration ~1e-3), so per-dimension standardization is required before
feeding the world model. Statistics are fitted over all rollouts in a
directory and can be saved to / loaded from a JSON file.

Angles are already wrapped to [-pi, pi] at collection time; they are
standardized like any other dimension (std of wrapped angles is
well-defined for this scene since rotations stay small).
"""

import glob
import json
import os

import numpy as np


class Normalizer:
    """Per-dimension standardization for flat feature groups."""

    KEYS = ("obj_states", "actions", "contact_feat", "tactile_summary")

    def __init__(self):
        self.stats = {}  # key -> {"mean": (D,), "std": (D,)}

    # ---------------- fitting ----------------
    @staticmethod
    def _collect_flat(files: list, key: str, masked: bool = False):
        chunks = []
        masks = []
        for f in files:
            d = np.load(f)
            arr = d[key].astype(np.float64)
            chunks.append(arr.reshape(-1, arr.shape[-1]))
            if masked:
                m = d["contact_mask"].astype(np.float64).reshape(-1, 1)
                masks.append(m)
        flat = np.concatenate(chunks, axis=0)
        if masked:
            m = np.concatenate(masks, axis=0) > 0.5
            flat = flat * m  # zero out padding rows
            n = max(m.sum(), 1)
            mean = flat.sum(axis=0) / n
            var = (((flat - mean) * m) ** 2).sum(axis=0) / n
        else:
            mean = flat.mean(axis=0)
            var = flat.var(axis=0)
        return mean, np.sqrt(var)

    @classmethod
    def fit_from_dir(cls, rollouts_dir: str) -> "Normalizer":
        files = sorted(glob.glob(os.path.join(rollouts_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"no rollout npz files under {rollouts_dir}")
        norm = cls()
        for key in cls.KEYS:
            mean, std = cls._collect_flat(files, key, masked=(key == "contact_feat"))
            norm.stats[key] = {
                "mean": mean.astype(np.float32),
                "std": np.maximum(std, 1e-6).astype(np.float32),
            }
        return norm

    # ---------------- save / load ----------------
    def save(self, path: str):
        payload = {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
                   for k, v in self.stats.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        norm = cls()
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for k, v in payload.items():
            norm.stats[k] = {
                "mean": np.asarray(v["mean"], dtype=np.float32),
                "std": np.asarray(v["std"], dtype=np.float32),
            }
        return norm

    # ---------------- transform ----------------
    def normalize(self, key: str, x: np.ndarray) -> np.ndarray:
        s = self.stats[key]
        return (x - s["mean"]) / s["std"]

    def denormalize(self, key: str, x: np.ndarray) -> np.ndarray:
        s = self.stats[key]
        return x * s["std"] + s["mean"]

    def to_tensors(self, device="cpu"):
        import torch
        out = {}
        for k, v in self.stats.items():
            out[k] = (
                torch.as_tensor(v["mean"], device=device),
                torch.as_tensor(v["std"], device=device),
            )
        return out
