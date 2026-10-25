"""Roll out a trained (or reference) policy and write an MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .character import CharacterConfig, CharacterEnv
from .models import Policy
from .motion import make_clip, make_dance_clip, make_lafan2d_library


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _draw_bodies(ax, bodies, geoms, colors, *, alpha: float = 0.95,
                 edge: str = "white", zorder: int = 2):
    from matplotlib.patches import Circle, Polygon
    for i, st in enumerate(bodies):
        x, y, th = float(st[0]), float(st[1]), float(st[2])
        kind, a, b = geoms[i]
        col = colors[i]
        if kind == "ball":
            ax.add_patch(Circle(
                (x, y), a, facecolor=col, edgecolor=edge,
                lw=0.6, alpha=alpha, zorder=zorder,
            ))
        else:
            hw, hh = a, b
            corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])
            world = corners @ R.T + np.array([x, y])
            ax.add_patch(Polygon(
                world, closed=True, facecolor=col, edgecolor=edge,
                lw=0.6, alpha=alpha, zorder=zorder,
            ))


def draw_frame(ax, bodies, geoms, colors, ground_y: float, title: str,
               ref_xy=None, ref_bodies=None, ref_colors=None, ref_alpha: float = 0.35):
    ax.clear()
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#1a2332")
    # Fill strictly below the contact plane; keep the edge thin so it does not
    # paint over shin pixels (a thick axhline looked like foot penetration).
    ax.fill_between([0.0, 1.0], 0.0, ground_y, color="#2a3344", zorder=0)
    ax.plot(
        [0.0, 1.0], [ground_y, ground_y],
        color="#c9a227", lw=1.2, zorder=0, solid_capstyle="butt",
    )

    # Ghost reference first (semi-transparent), then solid learned pose.
    if ref_bodies is not None:
        _draw_bodies(
            ax, ref_bodies, geoms, ref_colors or REF_HEX,
            alpha=ref_alpha, edge="#d0d8e0", zorder=1,
        )
    _draw_bodies(ax, bodies, geoms, colors, alpha=0.95, edge="white", zorder=2)

    if ref_xy is not None:
        ax.plot(ref_xy[0], ref_xy[1], "o", color="#7dffb3", ms=5, alpha=0.7)
    ax.set_title(title, color="white", fontsize=11, pad=8)


def plt_circle(x, y, r, color):
    from matplotlib.patches import Circle
    return Circle((x, y), r, facecolor=color, edgecolor="white", lw=0.6, alpha=0.95)


def plt_poly(world, color):
    from matplotlib.patches import Polygon
    return Polygon(world, closed=True, facecolor=color, edgecolor="white",
                   lw=0.6, alpha=0.95)


def body_geoms(cfg: CharacterConfig):
    tw, th = cfg.torso_size
    aw, ah = cfg.arm_size
    uw, uh = cfg.thigh_size
    sw, sh = cfg.shin_size
    # half-extents for boxes; radius for head
    return [
        ("box", 0.5 * tw, 0.5 * th),
        ("ball", cfg.head_radius, cfg.head_radius),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * uw, 0.5 * uh),
        ("box", 0.5 * sw, 0.5 * sh),
        ("box", 0.5 * uw, 0.5 * uh),
        ("box", 0.5 * sw, 0.5 * sh),
    ]


# torso, head, arms×4, legs×4 — arms share one color, legs share one color
HEX = [
    "#E8C4A0",  # torso
    "#FFDD88",  # head
    "#4A90D9", "#4A90D9", "#4A90D9", "#4A90D9",  # arms L/R
    "#6B8F71", "#6B8F71", "#6B8F71", "#6B8F71",  # legs L/R
]
REF_HEX = [
    "#E8C4A0",
    "#FFDD88",
    "#9BB8D9", "#9BB8D9", "#9BB8D9", "#9BB8D9",  # pale arms (ghost)
    "#A8C4AE", "#A8C4AE", "#A8C4AE", "#A8C4AE",  # pale legs (ghost)
]


def rollout_reference_kinematic(env: CharacterEnv, n_steps: int, phase0: float = 0.0):
    """Play the planted clip via FK — no soft-contact sim digging into the floor."""
    env.reset(phase=phase0)
    frames, rewards = [], []
    n = max(env.clip.n_frames - 1, 1)
    for _ in range(n_steps):
        q_ref, root_ref = env.clip.sample(env.phase)
        frames.append(env.kinematic_bodies(q_ref, root_ref))
        rewards.append(0.0)
        env.phase = (env.phase + 1.0 / n) % 1.0
    return frames, rewards


def rollout_frames(env: CharacterEnv, policy: Policy | None, device,
                   n_steps: int, phase0: float = 0.0, with_kinematic_ref: bool = False):
    # Pure reference videos: exact planted motion (matches training targets).
    if policy is None and not with_kinematic_ref:
        return rollout_reference_kinematic(env, n_steps, phase0=phase0)

    obs = env.reset(phase=phase0)
    frames_state = []
    ref_frames = []
    rewards = []
    for t in range(n_steps):
        frames_state.append(env.body_states().copy())
        if with_kinematic_ref:
            # Ghost = planted clip pose; share live X only (keep ref Y / joints).
            q_ref, root_ref = env.clip.sample(env.phase)
            root_draw = root_ref.copy()
            root_draw[0] = float(env.root_state()[0])
            ref_frames.append(env.kinematic_bodies(q_ref, root_draw))
        if policy is None:
            q_ref, _ = env.clip.sample(env.phase)
            action = q_ref
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(o).squeeze(0).cpu().numpy()
        obs, r, done, info = env.step(action, action_noise=0.0)
        rewards.append(float(r))
        if done:
            break
    if with_kinematic_ref:
        return frames_state, rewards, ref_frames
    return frames_state, rewards


def write_mp4(frames, geoms, cfg: CharacterConfig, path: str, fps: float,
              titles=None, ref_frames=None, ref_alpha: float = 0.35):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=120)
    writer = imageio.get_writer(
        path, fps=int(fps), format="FFMPEG", mode="I",
        codec="libx264", pixelformat="yuv420p", quality=8, macro_block_size=1,
    )
    n = len(frames)
    try:
        for i, bodies in enumerate(frames):
            title = titles[i] if titles else f"frame {i}"
            ref_b = None
            if ref_frames is not None:
                ref_b = ref_frames[min(i, len(ref_frames) - 1)]
            draw_frame(
                ax, bodies, geoms, HEX, cfg.ground_y, title,
                ref_bodies=ref_b, ref_colors=REF_HEX, ref_alpha=ref_alpha,
            )
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            rgb = np.asarray(buf)[..., :3].copy()
            H, W = rgb.shape[:2]
            rgb = rgb[: H - (H % 2), : W - (W % 2)]
            writer.append_data(rgb)
    finally:
        writer.close()
        plt.close(fig)
    return path


def load_policy(ckpt_path: str, env: CharacterEnv, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    proprio = int(ckpt.get("proprio_dim", env.proprio_dim))
    policy = Policy(
        ckpt["obs_dim"], ckpt["action_dim"], proprio_dim=proprio,
        hidden=int(cfg.get("hidden", 256)),
        delta_scale=float(cfg.get("delta_scale", 0.6)),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="supermimic2d/runs/dance/best.pt")
    p.add_argument("--out", type=str, default="supermimic2d/runs/dance/dance.mp4")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--reference", action="store_true",
                   help="Render pure reference PD (no learned policy)")
    p.add_argument("--overlay", action="store_true",
                   help="Overlay translucent reference on learned policy")
    p.add_argument("--ref-alpha", type=float, default=0.38)
    p.add_argument("--root-kp", type=float, default=35.0)
    p.add_argument("--clip", type=str, default="dance",
                   help="Motion clip name, or 'all' to dump the LAFAN-2D suite")
    args = p.parse_args()

    cfg = CharacterConfig(
        control_hz=args.fps, max_episode_steps=args.steps + 5,
        root_kp=args.root_kp, action_noise_std=0.0, obs_noise_std=0.0,
        fall_height=0.05,
    )
    device = _device()
    geoms = body_geoms(cfg)

    if args.clip.lower() in ("all", "lafan2d"):
        out_dir = Path(args.out)
        if out_dir.suffix.lower() == ".mp4":
            out_dir = out_dir.parent / "lafan2d"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, clip in make_lafan2d_library(fps=args.fps).items():
            env = CharacterEnv(cfg, clip)
            frames, rewards = rollout_frames(env, None, device, args.steps, phase0=0.0)
            titles = [f"ref {name}   t={i / args.fps:.2f}s" for i in range(len(frames))]
            path = write_mp4(frames, geoms, cfg, str(out_dir / f"{name}_ref.mp4"),
                             args.fps, titles=titles)
            print(f"wrote {path}  mean_r={np.mean(rewards):.3f}")
        return

    clip = make_clip(args.clip, fps=args.fps) if args.clip != "dance" \
        else make_dance_clip(fps=args.fps, duration=4.0)

    if args.overlay:
        env_l = CharacterEnv(cfg, clip)
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise SystemExit(f"checkpoint not found: {ckpt_path}")
        policy, ckpt = load_policy(str(ckpt_path), env_l, device)
        ev = ckpt.get("eval", {})
        frames, rewards, ref_frames = rollout_frames(
            env_l, policy, device, args.steps, phase0=0.0, with_kinematic_ref=True,
        )
        n = len(frames)
        label = f"policy + kinematic ref  ret={ev.get('return', float('nan')):.1f}"
        titles = [f"{label}   t={i / args.fps:.2f}s   r={rewards[min(i, len(rewards)-1)]:.2f}"
                  for i in range(n)]
        out = args.out if args.out != "supermimic2d/runs/dance/dance.mp4" \
            else "supermimic2d/runs/dance/overlay_compare.mp4"
        path = write_mp4(frames, geoms, cfg, out, args.fps, titles=titles,
                         ref_frames=ref_frames, ref_alpha=args.ref_alpha)
        print(f"wrote {path}  ({n} frames, mean_r={np.mean(rewards):.3f}, "
              f"ref_alpha={args.ref_alpha})")
        return

    env = CharacterEnv(cfg, clip)
    policy = None
    label = "reference PD"
    if not args.reference:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise SystemExit(f"checkpoint not found: {ckpt_path}")
        policy, ckpt = load_policy(str(ckpt_path), env, device)
        ev = ckpt.get("eval", {})
        label = f"SuperMimic-2D  ret={ev.get('return', float('nan')):.1f}"

    frames, rewards = rollout_frames(env, policy, device, args.steps, phase0=0.0)
    titles = [f"{label}   t={i / args.fps:.2f}s   r={rewards[min(i, len(rewards)-1)]:.2f}"
              for i in range(len(frames))]
    path = write_mp4(frames, geoms, cfg, args.out, args.fps, titles=titles)
    print(f"wrote {path}  ({len(frames)} frames, mean_r={np.mean(rewards):.3f})")


if __name__ == "__main__":
    main()
