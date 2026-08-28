"""Evaluate PushToGoal with the shipped CEM controller.

Usage (repo root):
    python -m learning.eval_task --checkpoint learning/checkpoints_pair19_ens \\
        --episodes 50 --target-mode random
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from learning.configs.default import Config
from learning.tasks.push_to_goal import PushToGoalTask, load_ensemble


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="learning/checkpoints_pair19_ens")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--budget", type=int, default=None,
                        help="sim frames (default: Config.task.budget = 400)")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--vel-tol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--target-mode", type=str, default="leftmost",
                        choices=("leftmost", "rightmost", "random"))
    parser.add_argument("--record", type=int, default=50,
                        help="episodes to record trajectories")
    parser.add_argument("--tag", type=str, default="",
                        help="optional suffix for output files, e.g. random")
    parser.add_argument("--ee-scale", type=float, default=1.0,
                        help="EE radius multiplier (game slider is 0.5-2.0)")
    parser.add_argument("--results", type=str, default="learning/results")
    parser.add_argument("--uncert-cost", type=float, default=None,
                        help="override ensemble disagreement cost (default 0.1)")
    return parser


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    model, norm, stride = load_ensemble(args.checkpoint, device)
    n_models = len(model) if isinstance(model, list) else 1
    pk = dict(horizon=args.horizon, population=args.population,
              iterations=args.iterations, seed=args.seed)
    if args.uncert_cost is not None:
        pk["uncert_cost"] = float(args.uncert_cost)
    task = PushToGoalTask(
        cfg, model, norm, device=device, tol=args.tol,
        vel_tol=args.vel_tol, budget=args.budget,
        stride=stride, target_mode=args.target_mode,
        ee_scale=args.ee_scale, planner_kwargs=pk)

    records = []
    recorded = {"states": [], "masks": [], "actions": [], "goal": [],
                "success": [], "target_idx": [], "geom": []}
    t0 = time.time()
    n_success = 0
    for i in range(args.episodes):
        rng = np.random.default_rng(args.seed + i)
        rec = i < args.record
        r = task.run_episode(rng, record=rec)
        n_success += r["success"]
        if rec:
            states, masks, actions = r["frames"]
            recorded["states"].append(states)
            recorded["masks"].append(masks)
            recorded["actions"].append(actions)
            recorded["goal"].append(r["goal"])
            recorded["success"].append(r["success"])
            recorded["target_idx"].append(r["target_idx"])
            recorded["geom"].append(task._geom().copy())
        records.append({k: v for k, v in r.items() if k != "frames"})
        if (i + 1) % 10 == 0 or i == 0:
            rate = n_success / (i + 1)
            print(f"[{i + 1}/{args.episodes}] success so far {n_success}/{i + 1} "
                  f"({100 * rate:.1f}%), last final_dist {r['final_dist']:.3f}")

    rate = n_success / args.episodes
    elapsed = time.time() - t0
    dists = [r["final_dist"] for r in records]
    frames = [r["settle_frame"] for r in records]
    summary = {
        "checkpoint": args.checkpoint,
        "tag": args.tag,
        "episodes": args.episodes,
        "budget": task.budget,
        "tol": task.tol,
        "vel_tol": task.vel_tol,
        "success_rate": rate,
        "n_success": n_success,
        "mean_final_dist": float(np.mean(dists)),
        "std_final_dist": float(np.std(dists)),
        "mean_settle_frame": float(np.mean(frames)),
        "elapsed_s": elapsed,
        "n_models": n_models,
        "target_mode": args.target_mode,
        "ee_scale": args.ee_scale,
        "cem": {"horizon": args.horizon, "population": args.population,
                "iterations": args.iterations,
                "uncert_cost": args.uncert_cost},
    }
    os.makedirs(args.results, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    eval_path = os.path.join(args.results, f"task_eval{suffix}.json")
    traj_path = os.path.join(args.results, f"task_trajectories{suffix}.npz")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": records}, f, indent=2)
    if recorded["states"]:
        np.savez_compressed(
            traj_path,
            states=np.array(recorded["states"], dtype=object),
            masks=np.array(recorded["masks"], dtype=object),
            actions=np.array(recorded["actions"], dtype=object),
            goal=np.asarray(recorded["goal"], dtype=np.float32),
            success=np.asarray(recorded["success"]),
            target_idx=np.asarray(recorded["target_idx"]),
            geom=np.stack(recorded["geom"]).astype(np.float32),
        )

    print("-" * 60)
    print(f"SUCCESS RATE: {n_success}/{args.episodes} = {100 * rate:.1f}%")
    print(f"target={args.target_mode}")
    print(f"Mean final distance: {summary['mean_final_dist']:.3f} "
          f"(tol {task.tol}, vel {task.vel_tol}), "
          f"mean settle frame {summary['mean_settle_frame']:.0f}")
    print(f"Elapsed {elapsed:.1f}s ({elapsed / args.episodes:.2f}s/episode)")
    print(f"Wrote {eval_path}")
    return summary


def main():
    run_eval(build_parser().parse_args())


if __name__ == "__main__":
    main()
