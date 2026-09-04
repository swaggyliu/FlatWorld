#!/usr/bin/env python3
"""Capture Spirit Push screenshots for the homepage README.

Usage (repo root):
    python scripts/capture_game.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HEADLESS", "1")
os.environ.setdefault("PYTHONPATH", str(ROOT))

OUT = ROOT / "docs" / "game"
W, H = 1280, 720


def _shot(path: Path):
    import pyray as rl
    rel = path.relative_to(ROOT).as_posix()
    img = rl.load_image_from_screen()
    ok = rl.export_image(img, rel)
    rl.unload_image(img)
    print(f"{'wrote' if ok else 'FAILED'} {rel}")


def main():
    import pyray as rl
    import torch

    from game.camera import Camera
    from game.render import (
        BG, draw_equipment_panel, draw_hud, draw_scene, draw_start_menu,
        load_ui_font, unload_ui_font,
    )
    from game.session import GameSession

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = str(ROOT / "learning" / "checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT | rl.FLAG_WINDOW_HIGHDPI)
    rl.init_window(W, H, "FlatWorld — capture")
    rl.set_target_fps(30)
    load_ui_font()
    cam = Camera(W, H)

    # Homepage shot should match the shipped checkpoint, not local extra .pt files.
    label, detail = "best.pt", "learning/checkpoints"

    rl.begin_drawing()
    rl.clear_background(BG)
    draw_start_menu(W, H, label, detail, 0, 1)
    rl.end_drawing()
    rl.begin_drawing()
    rl.clear_background(BG)
    draw_start_menu(W, H, label, detail, 0, 1)
    rl.end_drawing()
    _shot(OUT / "menu.png")

    session = GameSession(ckpt, device, full_ensemble=False)
    session.level_seed_base = 42
    help_lines = [
        "gold = goal     red = obstacle",
        "click gold, then a spot     faster + fewer pushes = more stars",
    ]
    shots = (("push.png", 1), ("bank.png", 3), ("gauntlet.png", 5))
    for name, level in shots:
        session.level = level
        session._boot_level(apply_loadout=True)
        spec = session.spec
        obs = session.obs
        for _ in range(2):
            rl.begin_drawing()
            rl.clear_background(BG)
            draw_scene(
                cam, obs["obj_states"], obs["obj_types"], obs["obj_geom"],
                session.level_target_idx, session.hazard_idx, session.level_goal,
                session.trail, 0.0, push_idx=session.level_target_idx,
                restitution=session.restitution, friction=session.friction)
            draw_hud(session.status, help_lines, W,
                     level=session.level,
                     elapsed=0.0, total_score=0, pace=0, par=spec.par,
                     pushes=0, push_cap=spec.push_cap, hint=spec.hint)
            draw_equipment_panel(W, session.restitution, session.friction,
                                 session.scale)
            rl.end_drawing()
        _shot(OUT / name)

    unload_ui_font()
    rl.close_window()
    print(f"captures in {OUT}")


if __name__ == "__main__":
    main()
