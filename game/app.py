"""Python + raylib prototype: Spirit Push (CEM) and Push-T You-vs-AI.

Usage (repo root):
    python -m game
    python -m game --checkpoint learning/checkpoints/push_game
    python -m game --mode pusht          # split-screen: WASD vs WM+PPO
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyray as rl
import torch

from .camera import Camera
from .menu import discover_models, ensure_user_model_dir
from .render import (
    BG, _hit, draw_equipment_panel, draw_hud, draw_scene, draw_start_menu,
    draw_pusht_versus,
    equipment_layout, load_ui_font, slider_value, t_to_scale, unload_ui_font,
)
from .session import GameSession

W, H = 1280, 720


def _run_push(args, device: str):
    ensure_user_model_dir()
    models = discover_models(args.checkpoint)
    if not models:
        models = discover_models("learning/checkpoints/push_game")
    pick = 0
    for i, m in enumerate(models):
        if os.path.normpath(m.path) == os.path.normpath(args.checkpoint):
            pick = i
            break

    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT | rl.FLAG_WINDOW_HIGHDPI)
    rl.init_window(args.width, args.height, "FlatWorld — Spirit Push")
    rl.set_target_fps(60)
    load_ui_font()

    cam = Camera(args.width, args.height)
    session = None
    page = "menu"
    error = ""
    loading = False
    drag = None

    while not rl.window_should_close():
        if page == "menu":
            models = discover_models(args.checkpoint) or models
            pick = min(pick, max(len(models) - 1, 0))
            if rl.is_key_pressed(rl.KEY_ESCAPE):
                break
            if models and rl.is_key_pressed(rl.KEY_LEFT):
                pick = (pick - 1) % len(models)
            if models and rl.is_key_pressed(rl.KEY_RIGHT):
                pick = (pick + 1) % len(models)
            if rl.is_key_pressed(rl.KEY_ENTER) and models:
                loading = True

            rl.begin_drawing()
            rl.clear_background(BG)
            if models:
                opt = models[pick]
                prev, nxt, start = draw_start_menu(
                    args.width, args.height, opt.label, opt.detail,
                    pick, len(models), error, loading)
            else:
                prev, nxt, start = draw_start_menu(
                    args.width, args.height, "(no checkpoints found)",
                    "train or drop a .pt into learning/world_models",
                    0, 0, error, False)
            rl.end_drawing()

            clicked = rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)
            mx, my = rl.get_mouse_x(), rl.get_mouse_y()
            if clicked and models:
                if _hit(prev, mx, my):
                    pick = (pick - 1) % len(models)
                    error = ""
                elif _hit(nxt, mx, my):
                    pick = (pick + 1) % len(models)
                    error = ""
                elif _hit(start, mx, my):
                    loading = True
            if loading and models:
                opt = models[pick]
                try:
                    session = GameSession(
                        opt.path, device, full_ensemble=opt.full_ensemble)
                    error = ""
                    page = "play"
                except Exception as exc:
                    error = f"failed to load: {exc}"
                    session = None
                loading = False
            continue

        if rl.is_key_pressed(rl.KEY_ESCAPE):
            break
        if rl.is_key_pressed(rl.KEY_R) or rl.is_key_pressed(rl.KEY_SPACE):
            session.reset()
        if rl.is_key_pressed(rl.KEY_N):
            session.next_level()

        lay = equipment_layout(args.width)
        mx, my = rl.get_mouse_x(), rl.get_mouse_y()
        on_ui = _hit(lay["panel"], mx, my) or _hit(lay["hud"], mx, my)
        if rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT):
            if drag is None:
                if _hit(lay["spring"], mx, my):
                    drag = "spring"
                elif _hit(lay["grip"], mx, my):
                    drag = "grip"
                elif _hit(lay["size"], mx, my):
                    drag = "size"
            if drag == "spring":
                session.restitution = slider_value(lay["spring"], mx)
                session.apply_equipment()
            elif drag == "grip":
                session.friction = slider_value(lay["grip"], mx)
                session.apply_equipment()
            elif drag == "size":
                session.scale = t_to_scale(slider_value(lay["size"], mx))
                session.apply_equipment()
        else:
            drag = None

        if (rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)
                and drag is None and not on_ui):
            wx, wy = cam.to_world(mx, my)
            if wy > -0.05:
                session.handle_click(wx, wy)
        session.tick()

        obs = session.obs
        rl.begin_drawing()
        rl.clear_background(BG)
        draw_scene(cam, obs["obj_states"], obs["obj_types"], obs["obj_geom"],
                   session.level_target_idx, session.hazard_idx, session.level_goal,
                   session.trail, session.hit_flash,
                   happy=session.status.startswith("WIN"),
                   push_idx=session.push_idx, steer_goal=session.steer_goal,
                   restitution=session.restitution, friction=session.friction,
                   force=session.action if session.running else None)
        spec = session.spec
        help_lines = [
            "gold = goal     red = obstacle",
            "click gold, then a spot     faster + fewer pushes = more stars",
            "R / Space = retry     N = skip (no score)     Esc = quit",
        ]
        draw_hud(session.status, help_lines, args.width,
                 level=session.level,
                 elapsed=session.elapsed, total_score=session.display_score(),
                 pace=session.pace_score(), par=spec.par,
                 pushes=session.pushes, push_cap=spec.push_cap,
                 best=session.best_for_level(), stars=session.last_stars,
                 hint=spec.hint)
        draw_equipment_panel(args.width, session.restitution, session.friction,
                             session.scale)
        rl.end_drawing()

    unload_ui_font()
    rl.close_window()


def _run_pusht(args, device: str):
    from .pusht_session import PushTVersusSession

    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT | rl.FLAG_WINDOW_HIGHDPI)
    rl.init_window(args.width, args.height, "FlatWorld — You vs AI")
    rl.set_target_fps(60)
    load_ui_font()

    error = ""
    session = None
    try:
        session = PushTVersusSession(
            wm_path=args.wm, policy_path=args.policy, device=device,
            hold=args.hold, seed=args.seed)
    except Exception as exc:
        error = str(exc)

    key_map = {
        rl.KEY_W: "w", rl.KEY_A: "a", rl.KEY_S: "s", rl.KEY_D: "d",
    }
    while not rl.window_should_close():
        if rl.is_key_pressed(rl.KEY_ESCAPE):
            break
        if session is not None:
            if rl.is_key_pressed(rl.KEY_R) or rl.is_key_pressed(rl.KEY_SPACE):
                session.next_episode()
            if rl.is_key_pressed(rl.KEY_P):
                session.paused = not session.paused
            for code, name in key_map.items():
                if rl.is_key_down(code):
                    session.set_key(name, True)
                else:
                    session.set_key(name, False)
            session.tick()

        rl.begin_drawing()
        rl.clear_background(BG)
        if session is None:
            draw_pusht_versus(
                args.width, args.height,
                {"obj_states": np.zeros((3, 6)), "obj_geom": np.ones((3, 2)) * 0.05},
                {"obj_states": np.zeros((3, 6)), "obj_geom": np.ones((3, 2)) * 0.05},
                [], [], None, None,
                title="You vs AI",
                subtitle=f"load failed: {error}",
                hint="train push_t_wm_rsl then retry")
        else:
            cd = None
            if session.phase == "countdown":
                n = max(1, int(np.ceil(session.countdown_left)))
                cd = str(n) if session.countdown_left > 0 else "GO!"
            # Large overlay only after the race ends (not during countdown/race).
            overlay = session.status if session.phase == "over" else ""
            left_a = session.left.action if session.phase == "race" else None
            right_a = session.right.action if session.phase == "race" else None
            draw_pusht_versus(
                args.width, args.height,
                session.left.task.observe(),
                session.right.task.observe(),
                session.left.trail, session.right.trail,
                session.sg, session.bg,
                left_action=left_a, right_action=right_a,
                left_happy=session.left.ok, right_happy=session.right.ok,
                title="You vs AI",
                subtitle=overlay,
                countdown=cd,
                score_line=(
                    f"YOU {session.player_wins}   vs   AI {session.ai_wins}"
                    f"   (draw {session.draws})"),
                hint=("WASD move   Space/R rematch   P pause   Esc quit"
                      + ("   · PAUSED" if session.paused else "")),
                goal_theta=float(session.gth))
        rl.end_drawing()

    if session is not None:
        session.close()
    unload_ui_font()
    rl.close_window()


def main():
    parser = argparse.ArgumentParser(description="FlatWorld spirit push")
    parser.add_argument("--mode", choices=("push", "pusht"), default="push",
                        help="push = WM+CEM Spirit Push; pusht = WM+PPO Push-T")
    parser.add_argument("--checkpoint", type=str,
                        default="learning/checkpoints/push_game")
    parser.add_argument("--wm", type=str,
                        default="learning/checkpoints/push_t")
    parser.add_argument("--policy", type=str,
                        default="learning/checkpoints/push_t_wm_rsl/best.pt")
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--full", action="store_true",
                        help="prefer the full ensemble when a folder is selected")
    parser.add_argument("--width", type=int, default=W)
    parser.add_argument("--height", type=int, default=H)
    args = parser.parse_args()

    os.environ.setdefault("HEADLESS", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "pusht":
        _run_pusht(args, device)
    else:
        _run_push(args, device)


if __name__ == "__main__":
    main()
