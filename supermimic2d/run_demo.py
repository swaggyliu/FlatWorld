"""One-shot: train SuperMimic-2D on a dance clip, then write an MP4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .render_video import body_geoms, load_policy, rollout_frames, write_mp4
from .train import build_argparser, train
from .character import CharacterConfig, CharacterEnv
from .motion import make_dance_clip
from .render_video import _device


def main():
    # Reuse train CLI; add render path.
    p = build_argparser()
    p.add_argument("--video", type=str, default=None,
                   help="MP4 path (default: <out>/dance.mp4)")
    p.add_argument("--video-steps", type=int, default=120)
    p.add_argument("--skip-train", action="store_true",
                   help="Only render from existing checkpoint")
    args = p.parse_args()

    out = Path(args.out)
    video = Path(args.video) if args.video else out / "dance.mp4"

    if not args.skip_train:
        ckpt = train(args)
    else:
        ckpt = out / "best.pt"
        if not ckpt.exists():
            ckpt = out / "last.pt"
        if not ckpt.exists():
            print(f"no checkpoint under {out}", file=sys.stderr)
            sys.exit(1)

    cfg = CharacterConfig(
        control_hz=args.fps,
        max_episode_steps=args.video_steps + 5,
        root_kp=max(10.0, 0.3 * args.root_kp),
        action_noise_std=0.0,
        obs_noise_std=0.0,
    )
    env = CharacterEnv(cfg, make_dance_clip(fps=args.fps, duration=args.clip_duration))
    device = _device()
    policy, meta = load_policy(str(ckpt), env, device)
    frames, rewards = rollout_frames(env, policy, device, args.video_steps)
    import numpy as np
    ev = meta.get("eval", {})
    label = f"SuperMimic-2D dance  eval_ret={ev.get('return', float('nan')):.1f}"
    titles = [
        f"{label}   t={i / args.fps:.2f}s   r={rewards[min(i, len(rewards) - 1)]:.2f}"
        for i in range(len(frames))
    ]
    path = write_mp4(frames, body_geoms(cfg), cfg, str(video), args.fps, titles=titles)
    print(f"\n=== demo complete ===\nvideo: {path}\nmean reward: {float(np.mean(rewards)):.3f}")


if __name__ == "__main__":
    main()
