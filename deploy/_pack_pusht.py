"""Build a lean tarball for aliyun Push-T web deploy."""
from __future__ import annotations

import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("TEMP", "/tmp")) / "flatworld-pusht.tgz"

INCLUDE = [
    "flatworld",
    "game",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "learning/__init__.py",
    "learning/planner.py",
    "learning/solver_planner.py",
    "learning/train_rsl_push_t.py",
    "learning/train_rsl_wm_push_t.py",
    "learning/tasks",
    "learning/rl",
    "learning/models",
    "learning/env",
    "learning/world_models",
    "learning/data/__init__.py",
    "learning/data/normalizer.py",
    "learning/data/dataset.py",
    "learning/data/collect.py",
    "learning/data/sane.py",
    "learning/checkpoints/push_t",
    "learning/checkpoints/push_t_wm_rsl/best.pt",
    "learning/checkpoints/push_t_wm_rsl/train_wm_rsl_summary.json",
]

SKIP = {".git", "__pycache__", ".venv", "logs", "_game_logs", "_promo_logs", "_smoke"}


def ok(p: Path) -> bool:
    return not any(part in SKIP or part.endswith(".pyc") for part in p.parts)


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    n = 0
    with tarfile.open(OUT, "w:gz") as tar:
        for rel in INCLUDE:
            p = ROOT / rel
            if not p.exists():
                print("missing", rel)
                continue
            if p.is_file():
                if ok(p):
                    tar.add(p, arcname=rel.replace("\\", "/"))
                    n += 1
            else:
                for f in p.rglob("*"):
                    if f.is_file() and ok(f):
                        tar.add(f, arcname=f.relative_to(ROOT).as_posix())
                        n += 1
    print(f"wrote {OUT} files={n} MB={OUT.stat().st_size / 1e6:.1f}")


if __name__ == "__main__":
    main()
