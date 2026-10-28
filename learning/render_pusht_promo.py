"""Render WM+PPO Push-T clips in the live web / You-vs-AI style.

Draws a semi-transparent red ghost-T goal, dark panel background, and a
setup overlay (position random · unrestricted · simple reward · …).
Searches seeds until enough successes, writes per-episode mp4s plus an
optional combined promo.

  python -m learning.render_pusht_promo \\
    --policy learning/checkpoints/push_t_wm_rsl/best_tol012_90pct.pt \\
    --wm learning/checkpoints/push_t \\
    --tol 0.12 --need-ok 5 \\
    --out-dir docs/media --out docs/media/pusht_promo.mp4
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import imageio.v2 as imageio


# Match docs/pusht/app.js ghost + body colors.
_BG = "#182430"
_GHOST_FACE = (230 / 255, 90 / 255, 100 / 255, 0.30)
_GHOST_EDGE = (230 / 255, 70 / 255, 85 / 255, 0.55)
_STEM = "#e9c46a"
_BAR = "#c9b8a0"
_EE = "#62cfbe"
_TRAIL = (98 / 255, 207 / 255, 190 / 255, 0.55)
_TEXT = "#e8eef4"
_MUTED = "#9db0c0"


def _pick_font():
    """Prefer a CJK-capable face so Chinese overlay labels render."""
    candidates = (
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun",
        "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS",
        "DejaVu Sans",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def _obb(x, y, th, hw, hh):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    corners = (R @ np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]]).T).T
    return corners + np.array([x, y])


def _setup_lines(tol: float, bilingual: bool = True):
    if bilingual:
        return [
            "WM + PPO  ·  Push-T",
            f"position random / 位置随机   ·   unrestricted / 无限制",
            f"simple reward / 简单奖励   ·   tol={tol:.2f} m",
            "frozen WM obs  ·  real-physics PPO  ·  max_ee_t_dist=∞",
        ]
    return [
        "WM + PPO  ·  Push-T",
        "position random  ·  unrestricted (no EE range limit)",
        f"simple reward (potential + sparse success)  ·  tol={tol:.2f} m",
        "frozen WM encoder obs  ·  real FlatWorld dynamics",
    ]


def _frame(states, geom, sg, bg, gth, trail, lines, status,
           w=960, h=540):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.set_xlim(0.05, 2.05)
    ax.set_ylim(-0.05, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)

    # Ghost T (stem=1, bar=2) under live bodies.
    if len(states) > 1 and sg is not None:
        hw, hh = float(geom[1, 0]), float(geom[1, 1])
        ax.add_patch(Polygon(
            _obb(float(sg[0]), float(sg[1]), float(gth), hw, hh),
            closed=True, fc=_GHOST_FACE, ec=_GHOST_EDGE, lw=1.4, zorder=2))
    if len(states) > 2 and bg is not None:
        hw, hh = float(geom[2, 0]), float(geom[2, 1])
        ax.add_patch(Polygon(
            _obb(float(bg[0]), float(bg[1]), float(gth), hw, hh),
            closed=True, fc=_GHOST_FACE, ec=_GHOST_EDGE, lw=1.4, zorder=2))

    if trail:
        tr = np.asarray(trail)
        if len(tr) >= 2:
            ax.plot(tr[:, 0], tr[:, 1], color=_TRAIL, lw=2.0, zorder=3)

    for i in range(1, len(states)):
        x, y, th = states[i, 0], states[i, 1], states[i, 2]
        hw, hh = float(geom[i, 0]), float(geom[i, 1])
        face = _STEM if i == 1 else _BAR
        ax.add_patch(Polygon(
            _obb(x, y, th, hw, hh), closed=True,
            fc=face, ec="#2b2b2b", lw=0.8, zorder=4))
    if len(states) > 0:
        ax.add_patch(Circle(
            (states[0, 0], states[0, 1]), float(geom[0, 0]),
            fc=_EE, ec="#1a3a36", lw=1.0, zorder=5))

    # Setup overlay panel (top-left).
    ax.add_patch(Rectangle(
        (0.02, 0.72), 0.96, 0.26, transform=ax.transAxes,
        fc=(12 / 255, 18 / 255, 28 / 255, 0.72), ec="none",
        zorder=10, clip_on=False))
    y = 0.94
    for i, line in enumerate(lines):
        ax.text(
            0.04, y, line, transform=ax.transAxes,
            fontsize=12 if i == 0 else 9.5,
            fontweight="bold" if i == 0 else "normal",
            color=_TEXT if i == 0 else _MUTED, va="top", zorder=11)
        y -= 0.055 if i == 0 else 0.048
    if status:
        ax.text(
            0.98, 0.06, status, transform=ax.transAxes,
            fontsize=11, color=_EE, ha="right", va="bottom",
            fontweight="bold", zorder=11)

    fig.tight_layout(pad=0.15)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _card(lines, w=960, h=540, bg="#1d3557"):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.22), 0.84, 0.56, boxstyle="round,pad=0.02",
        facecolor="#2a4a6a", edgecolor="none"))
    y = 0.66
    for i, line in enumerate(lines):
        ax.text(0.5, y, line, ha="center", va="center",
                fontsize=20 if i == 0 else 13,
                fontweight="bold" if i == 0 else "normal",
                color="white")
        y -= 0.09
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _rollout_episode(task, wm, policy, hold, seed, lines, fps,
                     settle_s=0.8, max_t=None):
    """Run one episode; return (ok, frames, settle_t)."""
    from learning.rl.imagine import task_features
    import torch
    from tensordict import TensorDict

    rng = np.random.default_rng(int(seed))
    task.env.reset(rng)
    task.env.set_ee_radius(float(task.cfg.scene.ee_radius) * task.ee_scale)
    goal = task.sample_goal(rng, task.observe())
    task.goal = goal
    sg, bg, gth = task.stem_goal, task.bar_goal, task.goal_theta
    fmax = float(task.cfg.collect.force_max)
    pol_dev = next(policy.parameters()).device
    budget = int(task.budget) if max_t is None else min(int(task.budget), int(max_t))

    frames = []
    trail = []
    t = 0
    ok = False
    while t < budget:
        cur = task.observe()
        st = cur["obj_states"]
        trail.append((float(st[0, 0]), float(st[0, 1])))
        if task.succeeded(cur, goal):
            ok = True
            break
        z, _, _, _ = wm.encode_one(cur, use_tactile=False)
        pose = torch.as_tensor(
            st[:, :3], dtype=torch.float32, device=wm.device).unsqueeze(0)
        feat = task_features(pose, sg, bg, gth)
        x = torch.cat([z.reshape(1, -1), feat], dim=-1)
        td = TensorDict({"policy": x.to(pol_dev)}, batch_size=[1])
        a = policy(td, stochastic_output=False)
        if isinstance(a, tuple):
            a = a[0]
        a = a.detach().float().cpu().numpy().reshape(-1)
        a = np.clip(a, -1.0, 1.0) * fmax
        for _ in range(hold):
            task.env.step(a.astype(np.float32))
            t += 1
            if t % 2 == 0:
                cur2 = task.observe()
                frames.append(_frame(
                    cur2["obj_states"], cur2["obj_geom"], sg, bg, gth, trail,
                    lines, status=f"t={t}"))
            if t >= budget:
                break

    cur = task.observe()
    tag = "SUCCESS" if ok else "FAIL"
    for _ in range(int(fps * settle_s)):
        frames.append(_frame(
            cur["obj_states"], cur["obj_geom"], sg, bg, gth, trail,
            lines, status=tag))
    return ok, frames, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="learning/checkpoints/push_t")
    ap.add_argument(
        "--policy",
        default="learning/checkpoints/push_t_wm_rsl/best_tol012_90pct.pt")
    ap.add_argument("--out", default="docs/media/pusht_promo.mp4",
                    help="Combined promo mp4 (title + success clips + end)")
    ap.add_argument("--out-dir", default="docs/media",
                    help="Directory for per-success clips pusht_success_XX.mp4")
    ap.add_argument("--need-ok", type=int, default=5,
                    help="Number of successful episodes to export")
    ap.add_argument("--max-tries", type=int, default=40,
                    help="Max episode attempts while hunting successes")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.12,
                    help="Success pose tolerance (m); use 0.12 for 90% ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--en-only", action="store_true",
                    help="English-only overlay (skip Chinese labels)")
    ap.add_argument("--no-promo", action="store_true",
                    help="Skip combined promo; only write per-success clips")
    args = ap.parse_args()

    import torch
    from learning.tasks.push_to_goal import load_ensemble
    from learning.rl.imagine import WMBundle
    from learning.tasks.push_t import PushTTask, push_t_config
    from learning.train_rsl_wm_push_t import WMObsVecEnv
    from learning.rl.push_t_vec_env import PushTVecEnv
    from learning.train_rsl_push_t import make_train_cfg
    from rsl_rl.runners import OnPolicyRunner

    _pick_font()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir or ".", exist_ok=True)
    if not args.no_promo:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    policy_path = args.policy
    if not os.path.isfile(policy_path):
        alt = "learning/checkpoints/push_t_wm_rsl/best.pt"
        if os.path.isfile(alt):
            print(f"missing {policy_path}; falling back to {alt}", flush=True)
            policy_path = alt
        else:
            raise SystemExit(f"policy not found: {policy_path}")

    print(f"loading wm={args.wm} policy={policy_path} device={device}",
          flush=True)
    models, norm, stride = load_ensemble(args.wm, device)
    wm = WMBundle(models[0], norm, device, stride=stride)
    for p in wm.model.parameters():
        p.requires_grad_(False)
    wm.model.eval()
    hold = max(5, int(stride))

    base = PushTVecEnv(
        num_envs=1, hold=hold, device="cpu", seed=0,
        num_workers=0, max_ee_t_dist=None)
    env = WMObsVecEnv(base, wm, device=device)
    ns = argparse.Namespace(
        num_steps=32, max_iterations=1, save_interval=100,
        entropy_coef=0.01, lr=1e-3)
    cfg = make_train_cfg(ns)
    cfg["run_name"] = "pusht_promo"
    log_dir = os.path.join(os.path.dirname(policy_path), "_promo_logs")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy(device=device)

    task_cfg = push_t_config()
    task_cfg.task.tol = float(args.tol)
    task = PushTTask(task_cfg, planner_kwargs=dict(use_servo=True),
                     ee_spawn="support")
    lines = _setup_lines(float(args.tol), bilingual=not args.en_only)

    success_clips = []
    n_ok = 0
    for attempt in range(int(args.max_tries)):
        if n_ok >= int(args.need_ok):
            break
        seed = int(args.seed) + attempt
        print(f"try {attempt} seed={seed} …", flush=True)
        ok, frames, t = _rollout_episode(
            task, wm, policy, hold, seed, lines, args.fps)
        tag = "OK" if ok else "FAIL"
        print(f"  → {tag} t={t} frames={len(frames)}", flush=True)
        if not ok:
            continue
        n_ok += 1
        clip_path = os.path.join(
            args.out_dir, f"pusht_success_{n_ok:02d}.mp4")
        imageio.mimsave(clip_path, frames, fps=args.fps)
        print(f"  wrote {clip_path}", flush=True)
        success_clips.append((clip_path, frames, seed))

    if n_ok < int(args.need_ok):
        print(f"warning: only {n_ok}/{args.need_ok} successes "
              f"in {args.max_tries} tries", flush=True)

    if not args.no_promo and success_clips:
        frames = []
        title = _card([
            "FlatWorld  ·  StateLeWM",
            "WM + PPO  ·  Push-T",
            "position random · unrestricted · simple reward",
            f"tol={float(args.tol):.2f} m  ·  ghost-T goal",
        ])
        for _ in range(int(args.fps * 1.8)):
            frames.append(title)
        for i, (_path, clip_frames, seed) in enumerate(success_clips):
            # Brief seed card between clips.
            bridge = _card([
                f"Success {i + 1}/{len(success_clips)}",
                f"seed={seed}",
                "position random · unrestricted · simple reward",
            ], bg="#142033")
            for _ in range(int(args.fps * 0.7)):
                frames.append(bridge)
            frames.extend(clip_frames)
        end = _card([
            f"Push-T  ·  {len(success_clips)} success rollouts",
            "WM+PPO on real physics  ·  frozen encoder",
            "github.com/swaggyliu/FlatWorld",
        ])
        for _ in range(int(args.fps * 2.0)):
            frames.append(end)
        imageio.mimsave(args.out, frames, fps=args.fps)
        print(f"wrote promo {args.out}  ({len(frames)} frames @ {args.fps} fps)",
              flush=True)

    env.close()
    print(f"done: {n_ok} success clip(s) in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
