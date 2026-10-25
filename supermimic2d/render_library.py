"""Render reference (+ optional overlay) videos for the LAFAN-2D motion suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .character import CharacterConfig, CharacterEnv
from .motion import make_lafan2d_library
from .render_video import (
    _device,
    body_geoms,
    load_policy,
    rollout_frames,
    write_mp4,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="supermimic2d/runs/lafan2d")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--root-kp", type=float, default=100.0)
    p.add_argument("--ref-alpha", type=float, default=0.42)
    p.add_argument("--overlay", action="store_true",
                   help="Also write policy+kinematic-ref overlays (needs --ckpt)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CharacterConfig(
        control_hz=args.fps, max_episode_steps=args.steps + 5,
        root_kp=args.root_kp, action_noise_std=0.0, obs_noise_std=0.0,
        fall_height=0.05,
    )
    device = _device()
    geoms = body_geoms(cfg)
    lib = make_lafan2d_library(fps=args.fps)

    policy = None
    if args.overlay:
        if not args.ckpt:
            raise SystemExit("--overlay requires --ckpt")
        # load against a throwaway env for dims
        tmp = CharacterEnv(cfg, next(iter(lib.values())))
        policy, meta = load_policy(args.ckpt, tmp, device)
        print(f"loaded policy eval={meta.get('eval')}")

    for name, clip in lib.items():
        env = CharacterEnv(cfg, clip)
        frames_ref, rewards_ref = rollout_frames(env, None, device, args.steps)
        titles = [f"LAFAN2D/{name}  reference" for _ in frames_ref]
        path = write_mp4(frames_ref, geoms, cfg, str(out / f"{name}_ref.mp4"),
                         args.fps, titles=titles)
        print(f"[ref] {name:12s}  mean_r={np.mean(rewards_ref):.3f}  → {path}")

        if policy is not None:
            env_p = CharacterEnv(cfg, clip)
            frames, rewards, ref_frames = rollout_frames(
                env_p, policy, device, args.steps, with_kinematic_ref=True,
            )
            titles = [f"LAFAN2D/{name}  policy+ghost  r={rewards[i]:.2f}"
                      for i in range(len(frames))]
            path = write_mp4(
                frames, geoms, cfg, str(out / f"{name}_overlay.mp4"),
                args.fps, titles=titles, ref_frames=ref_frames, ref_alpha=args.ref_alpha,
            )
            print(f"[pol] {name:12s}  mean_r={np.mean(rewards):.3f}  → {path}")


if __name__ == "__main__":
    main()
