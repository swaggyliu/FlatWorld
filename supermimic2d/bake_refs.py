"""Physics-bake procedural refs by PD-tracking them in FlatWorld.

Why: LAFAN1 refs are mocap + retarget (already near-physical). Our procedural
clips are kinematic drafts — feet can hover / skate. Without a trained policy
we still have a physics engine: track the draft with PD + light root assist,
record the *simulated* (q, root) as the new reference. Ground contact and
friction then enforce plant / no-skate where the draft asked for contact.

Pipeline:  procedural draft → bake_refs → MotionClip npz → make_lafan2d_library
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .character import CharacterConfig, CharacterEnv
from .motion import (
    CONTACT_Y,
    GROUND_Y,
    MotionClip,
    N_JOINTS,
    _smooth_traj,
    clip_ground_stats,
    default_baked_dir,
    fit_root_to_contact,
    foot_sole_heights,
    procedural_lafan2d_library,
    save_clip,
)


def _foot_tip_xy(q: np.ndarray, root: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L/R foot tip (x, y) from shin distal end — for skate diagnostics."""
    from .motion import SHIN_SIZE, _IDX_SHIN_L, _IDX_SHIN_R, _rot, fk_body_poses

    poses = fk_body_poses(q, root)
    half = 0.5 * SHIN_SIZE[1]
    tips = []
    for idx in (_IDX_SHIN_L, _IDX_SHIN_R):
        x, y, th = float(poses[idx, 0]), float(poses[idx, 1]), float(poses[idx, 2])
        tip = np.array([x, y]) + _rot(th) @ np.array([0.0, -half])
        tips.append(tip)
    return tips[0], tips[1]


def skate_stats(clip: MotionClip, contact_tol: float = 0.025) -> dict[str, float]:
    """Mean |foot_vx| while that foot is near the ground (skating score)."""
    dt = clip.dt
    speeds = []
    for i in range(clip.n_frames - 1):
        fl0, fr0 = _foot_tip_xy(clip.joint_q[i], clip.root_xy_theta[i])
        fl1, fr1 = _foot_tip_xy(clip.joint_q[i + 1], clip.root_xy_theta[i + 1])
        yl, yr = foot_sole_heights(clip.joint_q[i], clip.root_xy_theta[i])
        if yl <= CONTACT_Y + contact_tol:
            speeds.append(abs(float(fl1[0] - fl0[0]) / dt))
        if yr <= CONTACT_Y + contact_tol:
            speeds.append(abs(float(fr1[0] - fr0[0]) / dt))
    arr = np.asarray(speeds, dtype=np.float64) if speeds else np.zeros(1)
    return {
        "skate_mean": float(arr.mean()),
        "skate_max": float(arr.max()),
        "skate_n": int(arr.size),
    }


def bake_clip_pd(
    draft: MotionClip,
    *,
    root_kp: float = 70.0,
    root_kd: float = 10.0,
    settle_steps: int = 2,
) -> MotionClip:
    """PD-track ``draft`` once through FlatWorld; return simulated trajectory."""
    n = draft.n_frames
    cfg = CharacterConfig(
        control_hz=draft.fps,
        max_episode_steps=n + 20,
        root_kp=root_kp,
        root_kd=root_kd,
        action_noise_std=0.0,
        obs_noise_std=0.0,
        fall_height=0.02,  # never early-stop a bake
    )
    env = CharacterEnv(cfg, draft)
    env.reset(phase=0.0)
    for _ in range(settle_steps):
        env.set_joint_targets(draft.joint_q[0])
        env._apply_root_assist(draft.root_xy_theta[0])
        env.looper.advanceWithTime(env.dt)

    qs = np.zeros((n, N_JOINTS), dtype=np.float32)
    roots = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        qs[i] = env.joint_angles()
        roots[i] = env.root_state()[:3]
        q_ref = draft.joint_q[i]
        root_ref = draft.root_xy_theta[i]
        env.phase = float(i) / max(n - 1, 1)
        env.set_joint_targets(q_ref)
        env._apply_root_assist(root_ref)
        env.looper.advanceWithTime(env.dt)

    # Light smooth + penetration cleanup (sim can still micro-penetrate).
    qs = _smooth_traj(qs, sigma=0.7)
    roots = _smooth_traj(roots, sigma=0.7)
    qs[0] = draft.joint_q[0]
    roots[0] = draft.root_xy_theta[0]
    roots = fit_root_to_contact(
        qs, roots, contact="feet" if draft.name != "fall_getup" else "support",
        contact_y=CONTACT_Y, smooth_sigma=1.5,
    )
    return MotionClip(
        fps=draft.fps,
        duration=draft.duration,
        joint_q=qs,
        root_xy_theta=roots,
        name=draft.name,
    )


def bake_library(
    out_dir: Path | None = None,
    fps: float = 30.0,
    root_kp: float = 70.0,
) -> dict[str, MotionClip]:
    out_dir = Path(out_dir) if out_dir is not None else default_baked_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    drafts = procedural_lafan2d_library(fps=fps)
    baked: dict[str, MotionClip] = {}
    print(f"baking → {out_dir}")
    print(f"{'clip':12s}  {'foot_min':>8s}  {'pen':>4s}  {'skate':>7s}  {'→skate':>7s}")
    for name, draft in drafts.items():
        before = {**clip_ground_stats(draft), **skate_stats(draft)}
        clip = bake_clip_pd(draft, root_kp=root_kp)
        after = {**clip_ground_stats(clip), **skate_stats(clip)}
        path = out_dir / f"{name}.npz"
        save_clip(clip, path)
        baked[name] = clip
        print(
            f"{name:12s}  {after['min_foot_y']:8.4f}  {after['n_penetrating']:4d}  "
            f"{before['skate_mean']:7.3f}  {after['skate_mean']:7.3f}  → {path.name}"
        )
    return baked


def main():
    p = argparse.ArgumentParser(description="Physics-bake LAFAN-2D reference motions")
    p.add_argument("--out", type=str, default=str(default_baked_dir()))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--root-kp", type=float, default=70.0,
                   help="Root assist while baking (lower → more contact-driven)")
    args = p.parse_args()
    bake_library(Path(args.out), fps=args.fps, root_kp=args.root_kp)
    print("done. make_lafan2d_library(use_baked=True) will load these.")


if __name__ == "__main__":
    main()
