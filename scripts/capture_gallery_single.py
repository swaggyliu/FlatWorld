#!/usr/bin/env python3
"""Capture a single gallery frame (spawned in a fresh Python process)."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test2D"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--module", required=True)
    parser.add_argument("--func", required=True)
    parser.add_argument("--kwargs", default="{}")
    args = parser.parse_args()

    from gallery_capture import install_gallery_capture

    install_gallery_capture(args.output, args.capture_frame, args.max_frames)

    module = importlib.import_module(args.module)
    if args.module == "test_2Drigid_bcs":
        module.BENCHMARK_MODE = False
    func = getattr(module, args.func)
    kwargs = json.loads(args.kwargs)

    try:
        func(**kwargs)
    except (AssertionError, UnicodeEncodeError):
        # Demos often assert terminal state; keep the screenshot if we got one.
        output = Path(args.output)
        if output.exists():
            return 0
        raise
    except Exception:
        output = Path(args.output)
        if output.exists():
            return 0
        raise

    output = Path(args.output)
    return 0 if output.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
