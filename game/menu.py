"""Start-screen model picker."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass


@dataclass
class ModelOption:
    label: str
    path: str
    full_ensemble: bool
    detail: str


def _is_last(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name.endswith("_last.pt") or name == "last.pt"


def discover_models(*extra_paths: str) -> list[ModelOption]:
    """Find .pt checkpoints the player can drive the spirit with."""
    roots = [
        "learning/checkpoints_pair19_ens",
        "learning/world_models",
    ]
    for p in extra_paths:
        if p:
            roots.append(p)

    seen = set()
    options: list[ModelOption] = []

    def add(opt: ModelOption):
        key = (os.path.normpath(opt.path), opt.full_ensemble)
        if key in seen or not os.path.exists(opt.path):
            return
        seen.add(key)
        options.append(opt)

    for root in roots:
        if not os.path.isdir(root):
            continue
        ens = sorted(
            f for f in glob.glob(os.path.join(root, "ens_*.pt"))
            if not _is_last(f)
        )
        folder = os.path.basename(os.path.normpath(root))
        if len(ens) >= 2:
            add(ModelOption(
                label=f"{folder}  -  full ensemble",
                path=root,
                full_ensemble=True,
                detail=f"{len(ens)} members  (heavier CEM)",
            ))
        for f in ens:
            add(ModelOption(
                label=os.path.basename(f),
                path=f,
                full_ensemble=False,
                detail=folder,
            ))
        for name in ("best.pt", "best_last.pt"):
            f = os.path.join(root, name)
            if os.path.isfile(f) and not _is_last(f):
                add(ModelOption(
                    label=os.path.basename(f),
                    path=f,
                    full_ensemble=False,
                    detail=folder,
                ))
        for f in sorted(glob.glob(os.path.join(root, "*.pt"))):
            if _is_last(f):
                continue
            add(ModelOption(
                label=os.path.basename(f),
                path=f,
                full_ensemble=False,
                detail=folder,
            ))

    for p in extra_paths:
        if p and os.path.isfile(p) and p.endswith(".pt") and not _is_last(p):
            add(ModelOption(
                label=os.path.basename(p),
                path=p,
                full_ensemble=False,
                detail="custom file",
            ))

    return options


def ensure_user_model_dir() -> str:
    path = os.path.join("learning", "world_models")
    os.makedirs(path, exist_ok=True)
    return path
