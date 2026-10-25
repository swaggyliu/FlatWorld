"""Build a ~30s multi-skill showcase reel for the website.

Sequence (default ~30s @ 30fps):
  walk → run → sprint → dance → jump → crouch → fight → fall_getup

Each segment starts at standing rest (clip ease-in). Solid = policy,
ghost = full kinematic reference (clip joints + ground-clamped root).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .character import CharacterConfig, CharacterEnv
from .motion import make_lafan2d_library
from .render_video import (
    HEX,
    REF_HEX,
    _device,
    body_geoms,
    draw_frame,
    load_policy,
)


# (name, seconds, display label)
DEFAULT_REEL = (
    ("walk", 4.0, "Walk"),
    ("run", 3.5, "Run"),
    ("sprint", 3.0, "Sprint"),
    ("dance", 5.0, "Dance"),
    ("jump", 3.5, "Jump"),
    ("crouch", 3.0, "Crouch"),
    ("fight", 3.5, "Fight"),
    ("fall_getup", 4.5, "Fall & Get Up"),
)


@torch.no_grad()
def rollout_segment(env: CharacterEnv, policy, device, n_steps: int):
    obs = env.reset(phase=0.0)
    frames, refs, rewards = [], [], []
    for _ in range(n_steps):
        frames.append(env.body_states().copy())
        q_ref, root_ref = env.clip.sample(env.phase)
        # Full planted clip pose (shin-OBB contact). Keep ref height/joints so
        # the ghost matches jump_ref; only share live X to reduce lateral drift.
        root_draw = root_ref.copy()
        root_draw[0] = float(env.root_state()[0])
        refs.append(env.kinematic_bodies(q_ref, root_draw))
        if policy is None:
            action = q_ref
        else:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy(o).squeeze(0).cpu().numpy()
        obs, r, done, info = env.step(action, action_noise=0.0)
        rewards.append(float(r))
        if done:
            # Continue until budget — soft reset keeps reel length stable
            obs = env.reset(phase=0.0)
    return frames, refs, rewards


def write_showcase(
    segments,
    geoms,
    cfg: CharacterConfig,
    path: str,
    fps: float,
    ref_alpha: float = 0.40,
    figsize=(8.0, 4.5),
    dpi: int = 120,
):
    """Landscape website-friendly reel with per-segment titles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    writer = imageio.get_writer(
        path, fps=int(fps), format="FFMPEG", mode="I",
        codec="libx264", pixelformat="yuv420p", quality=7, macro_block_size=1,
    )
    total = 0
    try:
        for label, frames, refs in segments:
            n = len(frames)
            for i in range(n):
                title = f"SuperMimic-2D  ·  {label}   ({i / fps:.1f}s)"
                draw_frame(
                    ax, frames[i], geoms, HEX, cfg.ground_y, title,
                    ref_bodies=refs[i], ref_colors=REF_HEX, ref_alpha=ref_alpha,
                )
                # Widen camera a bit for landscape
                ax.set_xlim(0.05, 0.95)
                ax.set_ylim(0.0, 1.0)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                rgb = np.asarray(buf)[..., :3].copy()
                H, W = rgb.shape[:2]
                rgb = rgb[: H - (H % 2), : W - (W % 2)]
                writer.append_data(rgb)
                total += 1
            # 6-frame hold on last pose between skills
            if frames:
                for _ in range(6):
                    writer.append_data(rgb)
                    total += 1
    finally:
        writer.close()
        plt.close(fig)
    return path, total


def main():
    p = argparse.ArgumentParser(description="30s SuperMimic-2D website showcase")
    p.add_argument("--ckpt", type=str, default="supermimic2d/runs/lafan2d/best.pt")
    p.add_argument("--out", type=str, default="supermimic2d/runs/lafan2d/showcase_30s.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--root-kp", type=float, default=140.0)
    p.add_argument("--ref-alpha", type=float, default=0.40)
    p.add_argument("--reference-only", action="store_true",
                   help="Use reference PD instead of learned policy")
    args = p.parse_args()

    fps = float(args.fps)
    cfg = CharacterConfig(
        control_hz=fps,
        max_episode_steps=10_000,
        root_kp=args.root_kp,
        action_noise_std=0.0,
        obs_noise_std=0.0,
        # Don't truncate mid-jump: crouch COM dips near the default fall line.
        fall_height=0.05,
    )
    lib = make_lafan2d_library(fps=fps)
    device = _device()
    geoms = body_geoms(cfg)

    policy = None
    if not args.reference_only:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise SystemExit(f"checkpoint not found: {ckpt_path}")
        tmp = CharacterEnv(cfg, lib["walk"])
        policy, meta = load_policy(str(ckpt_path), tmp, device)
        print(f"loaded {ckpt_path}  eval={meta.get('eval')}", flush=True)
    else:
        print("reference-PD showcase (no policy)", flush=True)

    segments = []
    reel_sec = 0.0
    for name, sec, label in DEFAULT_REEL:
        n_steps = max(1, int(round(sec * fps)))
        env = CharacterEnv(cfg, lib[name])
        # Cap episode length so soft-reset doesn't fire mid-segment
        env.cfg.max_episode_steps = n_steps + 5
        frames, refs, rewards = rollout_segment(env, policy, device, n_steps)
        segments.append((label, frames, refs))
        reel_sec += len(frames) / fps
        print(f"  {label:14s}  {len(frames):3d} frames  "
              f"mean_r={float(np.mean(rewards)):.3f}", flush=True)

    path, n = write_showcase(
        segments, geoms, cfg, args.out, fps, ref_alpha=args.ref_alpha,
    )
    print(f"\nwrote {path}")
    print(f"  frames={n}  duration≈{n / fps:.1f}s  (@ {fps:.0f} fps)")


if __name__ == "__main__":
    main()
