"""Evaluate the PushToGoal task success rate with the learned world model.

Usage (repo root):
    python -m learning.eval_task --checkpoint learning/checkpoints/best.pt \
        --episodes 50 --budget 150

Writes learning/results/task_eval.json (+ task_trajectories.npz) for
reporting / plotting.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from learning.configs.default import Config
from learning.tasks.push_to_goal import PushToGoalTask, load_ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="learning/checkpoints")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--tol", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--target-mode", type=str, default="leftmost",
                        choices=("leftmost", "random"))
    parser.add_argument("--record", type=int, default=50,
                        help="episodes to record trajectories")
    parser.add_argument("--tag", type=str, default="",
                        help="optional suffix for output files, e.g. random")
    parser.add_argument("--results", type=str, default="learning/results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    cfg.collect.force_max = 6.0
    model, norm, stride = load_ensemble(args.checkpoint, device)
    n_models = len(model) if isinstance(model, list) else 1
    task = PushToGoalTask(cfg, model, norm, device=device, tol=args.tol,
                          budget=args.budget, stride=stride,
                          target_mode=args.target_mode,
                          planner_kwargs=dict(horizon=args.horizon,
                                              population=args.population,
                                              iterations=args.iterations,
                                              seed=args.seed))

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
        "episodes": args.episodes,
        "budget": args.budget,
        "tol": args.tol,
        "success_rate": rate,
        "n_success": n_success,
        "mean_final_dist": float(np.mean(dists)),
        "std_final_dist": float(np.std(dists)),
        "mean_settle_frame": float(np.mean(frames)),
        "elapsed_s": elapsed,
        "n_models": n_models,
        "target_mode": args.target_mode,
        "planner": {"horizon": args.horizon, "population": args.population,
                    "iterations": args.iterations},
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
    print(f"Mean final distance: {summary['mean_final_dist']:.3f} "
          f"(tol {args.tol}), mean settle frame {summary['mean_settle_frame']:.0f}")
    print(f"Elapsed {elapsed:.1f}s ({elapsed / args.episodes:.2f}s/episode)")


if __name__ == "__main__":
    main()
