"""Generate final report figures: training curves, success statistics and
rendered task scenes.

Usage (repo root):
    python -m learning.make_report

Inputs:  learning/results/train_log.csv
         learning/results/task_eval.json
         learning/results/task_trajectories.npz
Outputs: learning/results/training_curves.png
         learning/results/task_success_summary.png
         learning/results/scene_rollouts.png
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, FancyBboxPatch

from learning.configs.default import Config

RESULTS = "learning/results"


def plot_training_curves(path, out):
    epochs, tr = [], {k: [] for k in ("total", "dynamics", "recon_states")}
    vd, vr, ol10, ol25, ol50, zs = [], [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr["total"].append(float(row["train_total"]))
            tr["dynamics"].append(float(row["train_dynamics"]))
            tr["recon_states"].append(float(row["train_recon_states"]))
            vd.append(float(row["val_dynamics"]))
            vr.append(float(row["val_recon_states"]))
            ol10.append(float(row.get("val_openloop_10", "nan")))
            ol25.append(float(row.get("val_openloop_25", "nan")))
            ol50.append(float(row.get("val_openloop_50", "nan")))
            zs.append(float(row["z_std"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(epochs, tr["total"], label="train total")
    ax.plot(epochs, tr["dynamics"], label="train dynamics")
    ax.plot(epochs, vd, "--", label="val dynamics")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("World model losses"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, tr["recon_states"], label="train recon (states)")
    ax.plot(epochs, vr, "--", label="val recon (states)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("Reconstruction (anti-collapse)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, ol10, color="tab:red", label="10-step open-loop MSE")
    ax.plot(epochs, ol25, color="tab:orange", label="25-step")
    ax.plot(epochs, ol50, color="tab:pink", label="50-step")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(epochs, zs, color="tab:green", label="latent std")
    ax.set_xlabel("epoch"); ax.set_ylabel("open-loop MSE (log)", color="tab:red")
    ax2.set_ylabel("latent std", color="tab:green")
    ax.set_title("Long-horizon rollout & latent health")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("StateLeWM Phase 1 training", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_success_summary(eval_json, out):
    with open(eval_json, encoding="utf-8") as f:
        data = json.load(f)
    s = data["summary"]
    eps = data["episodes"]
    dists = np.array([e["final_dist"] for e in eps])
    frames = np.array([e["settle_frame"] for e in eps])
    succ = np.array([e["success"] for e in eps], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.bar(["success", "failure"],
           [s["n_success"], s["episodes"] - s["n_success"]],
           color=["#2a9d8f", "#e76f51"])
    for i, (v, n) in enumerate(zip([s["n_success"], s["episodes"] - s["n_success"]],
                                   [s["n_success"], s["episodes"] - s["n_success"]])):
        ax.text(i, v + 0.5, f"{n}\n({100 * n / s['episodes']:.0f}%)",
                ha="center", fontsize=11)
    ax.set_title(f"PushToGoal task result  (tol={s['tol']}, budget={s['budget']}f)")
    ax.set_ylabel("episodes")

    ax = axes[1]
    bins = np.linspace(0, max(0.35, dists.max() + 0.02), 18)
    ax.hist(dists[succ], bins=bins, color="#2a9d8f", alpha=0.8, label="success")
    ax.hist(dists[~succ], bins=bins, color="#e76f51", alpha=0.8, label="failure")
    ax.axvline(s["tol"], color="k", ls="--", label=f"tolerance {s['tol']}")
    ax.set_xlabel("final distance to goal (m)")
    ax.set_title("Final distance distribution")
    ax.legend()

    ax = axes[2]
    ax.hist(frames[succ], bins=20, color="#264653", alpha=0.85)
    ax.set_xlabel("settle frame")
    ax.set_ylabel("episodes")
    ax.set_title(f"Time to success (mean {s['mean_settle_frame']:.0f} frames)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _draw_scene(ax, states, cfg, target_idx=None, alpha=1.0):
    """Draw one scene frame. states: (N, 6)."""
    sc = cfg.scene
    types = [0] + [1] * sc.num_boxes + [2] * sc.num_balls
    ax.axhline(0.0, color="k", lw=2, alpha=alpha)  # ground
    for i, st in enumerate(states):
        x, y, th = st[0], st[1], st[2]
        is_target = (target_idx is not None and i == target_idx)
        if types[i] == 1:  # box: rotated rectangle
            hw, hh = sc.box_ext
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])
            corners = R @ np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]]).T
            corners = corners.T + np.array([x, y])
            face = "#e9c46a" if is_target else "#b8b8b8"
            ax.add_patch(Polygon(corners, closed=True, fc=face, ec="k",
                                 lw=1.2, alpha=alpha))
        else:  # circles: EE or ball
            r = sc.ee_radius if i == 0 else sc.ball_radius
            face = "#2a9d8f" if i == 0 else ("#e9c46a" if is_target else "#cbb3d6")
            ax.add_patch(Circle((x, y), r, fc=face, ec="k", lw=1.2, alpha=alpha))
        if is_target:
            ax.plot(x, y, "k*", ms=14, alpha=alpha)


def plot_scenes(traj_npz, out, cfg, n_show=4):
    d = np.load(traj_npz, allow_pickle=True)
    states = d["states"]
    goals = d["goal"]
    succ = d["success"]
    tgt_idx = d["target_idx"] if "target_idx" in d else [1] * len(states)
    n = min(n_show, len(states))
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        traj = states[i]                      # (T+1, N, 6)
        ti = int(tgt_idx[i])
        _draw_scene(ax, traj[0], cfg, ti, alpha=0.30)   # initial poses (light)
        _draw_scene(ax, traj[-1], cfg, ti)              # final poses (bold)
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], "-", color="#2a9d8f", lw=1.5,
                alpha=0.8, label="EE path")
        ax.plot(traj[:, ti, 0], traj[:, ti, 1], "-", color="#b8860b", lw=1.5,
                alpha=0.9, label="target path")
        g = goals[i]
        ax.plot(g[0], g[1], "r*", ms=18, mec="k", label="goal")
        tag = "OK" if succ[i] else "FAIL"
        final_d = np.linalg.norm(traj[-1, ti, :2] - g)
        ax.set_title(f"episode {i}: {tag} (final dist {final_d:.3f} m)")
        xmax = max(1.6, float(np.nanmax(traj[..., 0])) + 0.05)
        ax.set_xlim(0.0, xmax); ax.set_ylim(-0.08, 0.55)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("World-model MPC pushing: initial (light) / final (bold) scenes",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    cfg = Config()
    os.makedirs(RESULTS, exist_ok=True)
    curves = os.path.join(RESULTS, "training_curves.png")
    summary = os.path.join(RESULTS, "task_success_summary.png")
    scenes = os.path.join(RESULTS, "scene_rollouts.png")
    plot_training_curves(os.path.join(RESULTS, "train_log.csv"), curves)
    print(f"wrote {curves}")
    plot_success_summary(os.path.join(RESULTS, "task_eval.json"), summary)
    print(f"wrote {summary}")
    traj = os.path.join(RESULTS, "task_trajectories.npz")
    if os.path.exists(traj):
        plot_scenes(traj, scenes, cfg)
        print(f"wrote {scenes}")


if __name__ == "__main__":
    main()
