"""Combined figures for two-room / push-T / reacher StateLeWM runs.

Usage (repo root):
    python -m learning.make_lewm_report
    python -m learning.make_lewm_report --root learning/results
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning.configs.default import Config
from learning.make_report import (
    generate_report, plot_success_summary, plot_scenes, plot_training_curves,
)

TASKS = (
    ("two_room", "Two-room"),
    ("push_t", "Push-T"),
    ("reacher", "Reacher"),
)


def _read_eval(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_log(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows


def plot_overview(root, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for col, (key, title) in enumerate(TASKS):
        res = os.path.join(root, key)
        log_path = os.path.join(res, "train_log.csv")
        eval_path = os.path.join(res, "task_eval.json")
        ax = axes[0, col]
        if os.path.exists(log_path):
            rows = _read_log(log_path)
            ep = [int(r["epoch"]) for r in rows]
            ax.plot(ep, [float(r["train_dynamics"]) for r in rows], label="train dyn")
            ax.plot(ep, [float(r["val_dynamics"]) for r in rows], "--", label="val dyn")
            ax.plot(ep, [float(r.get("val_openloop_10", "nan")) for r in rows],
                    color="tab:red", alpha=0.8, label="open-loop 10")
            ax.set_yscale("log")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{title} training")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        if os.path.exists(eval_path):
            data = _read_eval(eval_path)
            s = data["summary"]
            n_ok = int(s["n_success"])
            n = int(s["episodes"])
            ax.bar(["success", "fail"], [n_ok, n - n_ok], color=["#2a9d8f", "#e76f51"])
            ax.set_title(
                f"{title}  {n_ok}/{n} = {100 * s['success_rate']:.0f}%\n"
                f"mean dist {s['mean_final_dist']:.3f} m, settle {s['mean_settle_frame']:.0f}f"
            )
        else:
            ax.set_title(f"{title} (no eval)")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("StateLeWM  ·  two-room / push-T / reacher", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="learning/results")
    args = parser.parse_args()
    cfg = Config()
    os.makedirs(args.root, exist_ok=True)
    for key, title in TASKS:
        res = os.path.join(args.root, key)
        if not os.path.isdir(res):
            print(f"skip {key}: missing {res}")
            continue
        generate_report(cfg, res, res, tag="")
        log = os.path.join(res, "train_log.csv")
        if os.path.exists(log):
            plot_training_curves(log, os.path.join(res, "training_curves.png"),
                                 title_suffix=title)
            print(f"wrote {res}/training_curves.png")
        eval_p = os.path.join(res, "task_eval.json")
        if os.path.exists(eval_p):
            plot_success_summary(eval_p, os.path.join(res, "task_success_summary.png"))
        traj = os.path.join(res, "task_trajectories.npz")
        if os.path.exists(traj):
            plot_scenes(traj, os.path.join(res, "scene_rollouts.png"), cfg,
                        title=f"{title}: recorded episodes  (light=init, bold=final)")
    overview = os.path.join(args.root, "lewm_three_tasks.png")
    plot_overview(args.root, overview)
    print(f"wrote {overview}")


if __name__ == "__main__":
    main()
