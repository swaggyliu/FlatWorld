"""Render a short WM+PPO Push-T promo clip (title → rollouts → end card).

  python -m learning.render_pusht_promo \\
    --policy learning/checkpoints/push_t_wm_rsl/best.pt \\
    --wm learning/checkpoints/push_t \\
    --out learning/results/wm_rl/pusht_promo.mp4 \\
    --episodes 4 --seconds 20
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
import numpy as np
import imageio.v2 as imageio


def _obb(x, y, th, hw, hh):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    corners = (R @ np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]]).T).T
    return corners + np.array([x, y])


def _frame(states, geom, sg, bg, trail, title, subtitle, w=960, h=540):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#f4f6f8")
    fig.patch.set_facecolor("#f4f6f8")
    ax.axhline(0.0, color="#222", lw=2.0, zorder=1)

    if trail:
        tr = np.asarray(trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#2a9d8f", lw=1.5, alpha=0.55, zorder=2)

    ax.plot(sg[0], sg[1], "*", color="#e63946", ms=16, zorder=5)
    ax.plot(bg[0], bg[1], "o", color="#e63946", ms=6, zorder=5)

    for i in range(1, len(states)):
        x, y, th = states[i, 0], states[i, 1], states[i, 2]
        hw, hh = float(geom[i, 0]), float(geom[i, 1])
        face = "#6c757d" if i == 1 else "#adb5bd"
        ax.add_patch(Polygon(
            _obb(x, y, th, hw, hh), closed=True, fc=face, ec="k", lw=1.0, zorder=3))
    ax.add_patch(Circle(
        (states[0, 0], states[0, 1]), float(geom[0, 0]),
        fc="#2a9d8f", ec="k", lw=1.0, zorder=4))

    ax.text(0.02, 1.08, title, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color="#1d3557", va="top")
    ax.text(0.02, 1.00, subtitle, transform=ax.transAxes, fontsize=10,
            color="#457b9d", va="top")
    fig.tight_layout(pad=0.2)
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
        (0.08, 0.25), 0.84, 0.5, boxstyle="round,pad=0.02",
        facecolor="#457b9d", edgecolor="none"))
    y = 0.62
    for i, line in enumerate(lines):
        ax.text(0.5, y, line, ha="center", va="center",
                fontsize=22 if i == 0 else 14,
                fontweight="bold" if i == 0 else "normal",
                color="white")
        y -= 0.10
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="learning/checkpoints/push_t")
    ap.add_argument("--policy",
                    default="learning/checkpoints/push_t_wm_rsl/best.pt")
    ap.add_argument("--out", default="learning/results/wm_rl/pusht_promo.mp4")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from tensordict import TensorDict
    from learning.tasks.push_to_goal import load_ensemble
    from learning.rl.imagine import WMBundle, task_features
    from learning.tasks.push_t import PushTTask, push_t_config
    from learning.train_rsl_wm_push_t import WMObsVecEnv
    from learning.rl.push_t_vec_env import PushTVecEnv
    from learning.train_rsl_push_t import make_train_cfg
    from rsl_rl.runners import OnPolicyRunner

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"loading wm={args.wm} policy={args.policy}", flush=True)
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
    log_dir = os.path.join(os.path.dirname(args.policy), "_promo_logs")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)
    runner.load(args.policy)
    policy = runner.get_inference_policy(device=device)
    pol_dev = next(policy.parameters()).device

    task = PushTTask(push_t_config(), planner_kwargs=dict(use_servo=True),
                     ee_spawn="support")
    fmax = float(task.cfg.collect.force_max)
    sg, bg, gth = task.stem_goal, task.bar_goal, task.goal_theta

    frames = []
    title = _card([
        "FlatWorld  ·  StateLeWM",
        "WM + PPO  ·  Push-T",
        "frozen world-model encoder  ·  real physics RL",
    ])
    for _ in range(int(args.fps * 2.0)):
        frames.append(title)

    n_ok = 0
    for ep in range(args.episodes):
        rng = np.random.default_rng(args.seed + ep)
        task.env.reset(rng)
        task.env.set_ee_radius(float(task.cfg.scene.ee_radius) * task.ee_scale)
        goal = task.sample_goal(rng, task.observe())
        task.goal = goal
        trail = []
        t = 0
        ok = False
        while t < min(int(task.budget), 400):
            cur = task.observe()
            st = cur["obj_states"]
            geom = cur["obj_geom"]
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
                        cur2["obj_states"], cur2["obj_geom"], sg, bg, trail,
                        title="Push-T  ·  WM-obs PPO",
                        subtitle=f"ep {ep}   t={t}   {'OK' if ok else '…'}"))
                if t >= task.budget:
                    break
        n_ok += int(ok)
        # settle hold
        cur = task.observe()
        tag = "OK" if ok else "FAIL"
        for _ in range(int(args.fps * 0.6)):
            frames.append(_frame(
                cur["obj_states"], cur["obj_geom"], sg, bg, trail,
                title="Push-T  ·  WM-obs PPO",
                subtitle=f"ep {ep}   {tag}"))
        print(f"  ep{ep} {tag}", flush=True)

    end = _card([
        f"Push-T demo  {n_ok}/{args.episodes} succeeded",
        "paradigms: WM+CEM · WM+BPTT · WM+PPO",
        "github.com/swaggyliu/FlatWorld",
    ])
    for _ in range(int(args.fps * 2.5)):
        frames.append(end)

    # trim to requested length
    max_n = int(args.seconds * args.fps)
    if len(frames) > max_n:
        frames = frames[:max_n]
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out}  ({len(frames)} frames @ {args.fps} fps)",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
