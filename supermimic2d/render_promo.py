"""Polished promo reel: FlatWorld reproduces SuperMimic (2D).

Title → multi-skill overlay → end card. Landscape, website-ready.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .character import CharacterConfig, CharacterEnv
from .motion import make_lafan2d_library
from .render_showcase import rollout_segment
from .render_video import (
    HEX,
    REF_HEX,
    _device,
    _draw_bodies,
    body_geoms,
    load_policy,
)


def _blank(ax, facecolor="#121820"):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(facecolor)


def draw_title_card(ax, t: float, duration: float = 2.4):
    """Brand-first title: FlatWorld dominates, SuperMimic as subtitle."""
    _blank(ax, "#0e141c")
    # soft vertical wash
    yy = np.linspace(0, 1, 64)
    for i, y0 in enumerate(yy[:-1]):
        a = 0.04 + 0.06 * (1.0 - abs(y0 - 0.45))
        ax.fill_between([0, 1], y0, yy[i + 1], color="#1c2a3a", alpha=a, zorder=0)
    ax.axhline(0.05, color="#c9a227", lw=1.4, alpha=0.85, zorder=1)

    fade = min(1.0, t / 0.35) * min(1.0, (duration - t) / 0.35)
    ax.text(
        0.50, 0.58, "FlatWorld",
        ha="center", va="center",
        fontsize=42, fontweight="bold", color="#f2f5f8",
        alpha=fade, zorder=2,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        0.50, 0.42, "SuperMimic  ·  2D",
        ha="center", va="center",
        fontsize=18, color="#9bb8d9",
        alpha=fade * 0.95, zorder=2,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        0.50, 0.28, "Motion imitation with a learned world model",
        ha="center", va="center",
        fontsize=11, color="#8a96a8",
        alpha=fade * 0.9, zorder=2,
        fontfamily="DejaVu Sans",
    )


def draw_end_card(ax, t: float, duration: float = 2.0):
    _blank(ax, "#0e141c")
    fade = min(1.0, t / 0.3) * min(1.0, (duration - t) / 0.35)
    ax.text(
        0.50, 0.55, "FlatWorld",
        ha="center", va="center",
        fontsize=36, fontweight="bold", color="#f2f5f8",
        alpha=fade, fontfamily="DejaVu Sans",
    )
    ax.text(
        0.50, 0.40, "SuperMimic reproduced in 2D",
        ha="center", va="center",
        fontsize=14, color="#9bb8d9",
        alpha=fade * 0.95, fontfamily="DejaVu Sans",
    )
    ax.axhline(0.05, color="#c9a227", lw=1.4, alpha=0.7 * fade)


def draw_promo_frame(ax, bodies, geoms, ground_y: float, skill: str,
                     ref_bodies=None, ref_alpha: float = 0.32):
    ax.clear()
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.0, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#121820")
    # atmosphere
    ax.fill_between([0, 1], 0.55, 1.0, color="#182230", alpha=0.55, zorder=0)
    ax.fill_between([0.0, 1.0], 0.0, ground_y, color="#1a222e", zorder=0)
    ax.plot([0.0, 1.0], [ground_y, ground_y], color="#c9a227", lw=1.3,
            zorder=0, solid_capstyle="butt")

    if ref_bodies is not None:
        _draw_bodies(
            ax, ref_bodies, geoms, REF_HEX,
            alpha=ref_alpha, edge="#c5ced8", zorder=1,
        )
    _draw_bodies(ax, bodies, geoms, HEX, alpha=0.96, edge="white", zorder=2)

    # Brand + skill — brand stays the stronger signal
    ax.text(
        0.04, 0.94, "FlatWorld",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=13, fontweight="bold", color="#e8eef4",
        fontfamily="DejaVu Sans", zorder=5,
    )
    ax.text(
        0.04, 0.88, f"SuperMimic  ·  {skill}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, color="#9bb8d9",
        fontfamily="DejaVu Sans", zorder=5,
    )


def write_promo(segments, geoms, cfg, path: str, fps: float,
                ref_alpha: float = 0.32, title_sec: float = 2.4, end_sec: float = 2.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=140)
    writer = imageio.get_writer(
        path, fps=int(fps), format="FFMPEG", mode="I",
        codec="libx264", pixelformat="yuv420p", quality=6, macro_block_size=1,
    )
    total = 0

    def push():
        nonlocal total
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = np.asarray(buf)[..., :3].copy()
        H, W = rgb.shape[:2]
        rgb = rgb[: H - (H % 2), : W - (W % 2)]
        writer.append_data(rgb)
        total += 1
        return rgb

    try:
        n_title = max(1, int(round(title_sec * fps)))
        for i in range(n_title):
            draw_title_card(ax, i / fps, title_sec)
            push()

        for label, frames, refs in segments:
            last = None
            for i in range(len(frames)):
                ref_b = refs[i] if refs is not None else None
                draw_promo_frame(
                    ax, frames[i], geoms, cfg.ground_y, label,
                    ref_bodies=ref_b, ref_alpha=ref_alpha,
                )
                last = push()
            # short hold between skills
            if last is not None:
                for _ in range(8):
                    writer.append_data(last)
                    total += 1

        n_end = max(1, int(round(end_sec * fps)))
        for i in range(n_end):
            draw_end_card(ax, i / fps, end_sec)
            push()
    finally:
        writer.close()
        plt.close(fig)
    return path, total


# Lean on skills that track best; keep variety for the story.
PROMO_REEL = (
    ("walk", 3.5, "Walk"),
    ("run", 3.0, "Run"),
    ("dance", 4.5, "Dance"),
    ("jump", 3.2, "Jump"),
    ("fight", 3.2, "Fight"),
    ("sprint", 2.8, "Sprint"),
    ("crouch", 2.8, "Crouch"),
    ("fall_getup", 3.8, "Fall & Get Up"),
)


def main():
    p = argparse.ArgumentParser(description="FlatWorld × SuperMimic promo reel")
    p.add_argument("--ckpt", type=str, default="supermimic2d/runs/lafan2d/best.pt")
    p.add_argument("--out", type=str, default="supermimic2d/runs/lafan2d/promo_supermimic.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--root-kp", type=float, default=160.0)
    p.add_argument("--ref-alpha", type=float, default=0.30)
    args = p.parse_args()

    fps = float(args.fps)
    cfg = CharacterConfig(
        control_hz=fps, max_episode_steps=10_000,
        root_kp=args.root_kp, action_noise_std=0.0, obs_noise_std=0.0,
        fall_height=0.05,
    )
    lib = make_lafan2d_library(fps=fps)
    device = _device()
    geoms = body_geoms(cfg)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    tmp = CharacterEnv(cfg, lib["walk"])
    policy, meta = load_policy(str(ckpt_path), tmp, device)
    print(f"loaded {ckpt_path}  eval={meta.get('eval')}", flush=True)

    segments = []
    for name, sec, label in PROMO_REEL:
        n_steps = max(1, int(round(sec * fps)))
        env = CharacterEnv(cfg, lib[name])
        env.cfg.max_episode_steps = n_steps + 5
        frames, refs, rewards = rollout_segment(env, policy, device, n_steps)
        segments.append((label, frames, refs))
        print(f"  {label:14s}  {len(frames):3d} frames  mean_r={float(np.mean(rewards)):.3f}",
              flush=True)

    path, n = write_promo(
        segments, geoms, cfg, args.out, fps, ref_alpha=args.ref_alpha,
    )
    print(f"\nwrote {path}")
    print(f"  frames={n}  duration≈{n / fps:.1f}s")


if __name__ == "__main__":
    main()
