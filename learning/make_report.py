"""Generate final report figures: training curves, success statistics and
rendered task scenes.

Usage (repo root):
    python -m learning.make_report
    python -m learning.make_report --pair19
    python -m learning.make_report --tag pair19ens_H32_random

Inputs:  learning/results/train_log.csv
         learning/results/task_eval{tag}.json
         learning/results/task_trajectories{tag}.npz
Outputs: learning/results/...png  (or learning/results/pair15/ with --pair15)
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from learning.configs.default import Config

RESULTS = "learning/results"

PAIR19_TAGS = (
    "pair19ens_H32_random",
    "pair19ens_H32_leftmost",
    "pair19ens_H32_rightmost",
)


def plot_training_curves(path, out, title_suffix=""):
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
    title = "StateLeWM training (GNN + contact head, ensemble member 0)"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_success_summary(eval_json, out, title_extra=""):
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
    tag = s.get("tag") or title_extra
    mode = s.get("target_mode", "")
    cem = s.get("cem") or {}
    h = cem.get("horizon")
    extra = f"  [{tag}]" if tag else ""
    if mode:
        extra += f"  target={mode}"
    if h is not None:
        extra += f"  H={h}"
    ax.set_title(f"PushToGoal  (tol={s['tol']}, budget={s['budget']}f){extra}")
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


def _draw_scene(ax, states, cfg, target_idx=None, alpha=1.0, geom=None):
    """Draw one scene frame. states: (N, 6). geom: (N, 2) half-extents."""
    sc = cfg.scene
    types = [0] + [1] * sc.num_boxes + [2] * sc.num_balls
    ax.axhline(0.0, color="k", lw=2, alpha=alpha)  # ground
    for i, st in enumerate(states):
        x, y, th = st[0], st[1], st[2]
        is_target = (target_idx is not None and i == target_idx)
        if geom is not None:
            hw, hh = float(geom[i, 0]), float(geom[i, 1])
        elif types[i] == 1:
            hw, hh = sc.box_ext
        else:
            hw = hh = sc.ee_radius if i == 0 else sc.ball_radius
        if types[i] == 1:  # box: rotated rectangle
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])
            corners = R @ np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]]).T
            corners = corners.T + np.array([x, y])
            face = "#e9c46a" if is_target else "#b8b8b8"
            ax.add_patch(Polygon(corners, closed=True, fc=face, ec="k",
                                 lw=1.2, alpha=alpha))
        else:  # circles: EE or ball
            r = hw
            face = "#2a9d8f" if i == 0 else ("#e9c46a" if is_target else "#cbb3d6")
            ax.add_patch(Circle((x, y), r, fc=face, ec="k", lw=1.2, alpha=alpha))
        if is_target:
            ax.plot(x, y, "k*", ms=14, alpha=alpha)


def plot_scenes(traj_npz, out, cfg, n_show=None, ncols=5):
    d = np.load(traj_npz, allow_pickle=True)
    states = d["states"]
    goals = d["goal"]
    succ = d["success"]
    tgt_idx = d["target_idx"] if "target_idx" in d else [1] * len(states)
    geoms = d["geom"] if "geom" in d.files else None
    n = len(states) if n_show is None else min(n_show, len(states))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        traj = states[i]
        ti = int(tgt_idx[i])
        gi = geoms[i] if geoms is not None else None
        _draw_scene(ax, traj[0], cfg, ti, alpha=0.30, geom=gi)
        _draw_scene(ax, traj[-1], cfg, ti, geom=gi)
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], "-", color="#2a9d8f", lw=1.2,
                alpha=0.8, label="EE path")
        ax.plot(traj[:, ti, 0], traj[:, ti, 1], "-", color="#b8860b", lw=1.2,
                alpha=0.9, label="target path")
        g = goals[i]
        ax.plot(g[0], g[1], "r*", ms=12, mec="k", label="goal")
        tag = "OK" if succ[i] else "FAIL"
        color = "#2a9d8f" if succ[i] else "#e76f51"
        final_d = np.linalg.norm(traj[-1, ti, :2] - g)
        ax.set_title(f"{i}: {tag}  d={final_d:.3f}", color=color, fontsize=9)
        xmax = max(1.6, float(np.nanmax(traj[..., 0])) + 0.05)
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(-0.08, 0.55)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(loc="upper right", fontsize=6)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("PushToGoal: all recorded episodes  (light=init, bold=final)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _tag_paths(results_dir, tag):
    suffix = f"_{tag}" if tag else ""
    return {
        "eval": os.path.join(results_dir, f"task_eval{suffix}.json"),
        "traj": os.path.join(results_dir, f"task_trajectories{suffix}.npz"),
        "summary": os.path.join(results_dir, f"task_success_summary{suffix}.png"),
        "scenes": os.path.join(results_dir, f"scene_rollouts{suffix}.png"),
    }


def _maybe_training_curves(results_dir, out_dir, title_suffix="", min_epochs=5):
    log_path = os.path.join(results_dir, "train_log.csv")
    out = os.path.join(out_dir, "training_curves.png")
    if not os.path.exists(log_path):
        print(f"skip training curves: missing {log_path}")
        return
    with open(log_path, encoding="utf-8") as f:
        n_epochs = sum(1 for _ in csv.DictReader(f))
    if n_epochs < min_epochs:
        print(f"skip training curves: {log_path} has only {n_epochs} epoch(s)")
        return
    plot_training_curves(log_path, out, title_suffix=title_suffix)
    print(f"wrote {out}")


def generate_report(cfg, results_dir, out_dir, tag="", title_suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    paths = _tag_paths(results_dir, tag)
    out_paths = _tag_paths(out_dir, tag)
    if os.path.exists(paths["eval"]):
        plot_success_summary(paths["eval"], out_paths["summary"])
        print(f"wrote {out_paths['summary']}")
    else:
        print(f"skip summary: missing {paths['eval']}")
    if os.path.exists(paths["traj"]):
        plot_scenes(paths["traj"], out_paths["scenes"], cfg)
        print(f"wrote {out_paths['scenes']}")
    else:
        print(f"skip scenes: missing {paths['traj']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=RESULTS)
    parser.add_argument("--out-dir", type=str, default="",
                        help="output directory (default: --results or pair19_ens/)")
    parser.add_argument("--tag", type=str, default="",
                        help="eval tag suffix, e.g. pair19ens_H32_random")
    parser.add_argument("--pair19", action="store_true",
                        help="render pair19_ens eval JSONs into results/pair19_ens/")
    args = parser.parse_args()

    cfg = Config()
    results_dir = args.results
    os.makedirs(results_dir, exist_ok=True)

    if args.pair19:
        out_dir = args.out_dir or os.path.join(results_dir, "pair19_ens")
        _maybe_training_curves(results_dir, out_dir, title_suffix="pair19_ens")
        for tag in PAIR19_TAGS:
            generate_report(cfg, results_dir, out_dir, tag=tag)
        main = _tag_paths(out_dir, "pair19ens_H32_random")
        for src, dst in (
            (main["summary"], os.path.join(out_dir, "task_success_summary.png")),
            (main["scenes"], os.path.join(out_dir, "scene_rollouts.png")),
        ):
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                print(f"copied {dst}")
        left = _tag_paths(out_dir, "pair19ens_H32_leftmost")
        if os.path.exists(left["scenes"]):
            import shutil
            shutil.copy2(left["scenes"],
                         os.path.join(out_dir, "scene_rollouts_leftmost.png"))
            print(f"copied {out_dir}/scene_rollouts_leftmost.png")
        print(f"pair19_ens report -> {out_dir}")
        return

    if args.tag:
        out_dir = args.out_dir or results_dir
        generate_report(cfg, results_dir, out_dir, tag=args.tag)
        if not args.out_dir:
            _maybe_training_curves(results_dir, out_dir)
        return

    # Legacy default paths (no tag).
    out_dir = args.out_dir or results_dir
    _maybe_training_curves(results_dir, out_dir)
    legacy_eval = os.path.join(results_dir, "task_eval.json")
    if os.path.exists(legacy_eval):
        plot_success_summary(legacy_eval, os.path.join(out_dir, "task_success_summary.png"))
        print(f"wrote {os.path.join(out_dir, 'task_success_summary.png')}")
    traj_left = os.path.join(results_dir, "task_trajectories.npz")
    traj_random = os.path.join(results_dir, "task_trajectories_random.npz")
    if os.path.exists(traj_left):
        plot_scenes(traj_left, os.path.join(out_dir, "scene_rollouts_leftmost.png"), cfg)
        print(f"wrote {os.path.join(out_dir, 'scene_rollouts_leftmost.png')}")
        if not os.path.exists(traj_random):
            plot_scenes(traj_left, os.path.join(out_dir, "scene_rollouts.png"), cfg)
            print(f"wrote {os.path.join(out_dir, 'scene_rollouts.png')} from leftmost")
    if os.path.exists(traj_random):
        plot_scenes(traj_random, os.path.join(out_dir, "scene_rollouts.png"), cfg)
        print(f"wrote {os.path.join(out_dir, 'scene_rollouts.png')} from random")
    eval_r = os.path.join(results_dir, "task_eval_random.json")
    if os.path.exists(eval_r):
        plot_success_summary(eval_r,
                             os.path.join(out_dir, "task_success_summary_random.png"))
        print(f"wrote {os.path.join(out_dir, 'task_success_summary_random.png')}")


if __name__ == "__main__":
    main()
