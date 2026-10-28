"""Re-run world-model CEM failures with solver-forward CEM.

Loads fail indices from ``task_eval_500_nofreeze_H32_{mode}.json`` and
replans those seeds with the same cost (reach + catch) but FlatWorld
physics as the CEM dynamics. Physics is serial, so the default CEM is H=16, P=12, iters=2 with
4 executed CEM steps per replan. Every episode is recorded to npz and
replayed (OK and FAIL).

Usage (repo root)::

    python -u -m learning.eval_solver_cem
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from learning.configs.default import Config
from learning.tasks.push_to_goal import PushToGoalTask

MODES = ("leftmost", "rightmost", "random")
JSON_TMPL = "learning/results/task_eval_500_nofreeze_H32_{mode}.json"


def _fails(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [e for e in data["episodes"] if not e["success"]]
    return data["summary"], rows


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="learning/results")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--elites", type=int, default=4)
    p.add_argument("--exec-horizon", type=int, default=4,
                   help="solver CEM steps to execute before replanning")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--mode", type=str, default="all",
                   choices=("all",) + MODES)
    p.add_argument("--no-replay", action="store_true",
                   help="skip MP4s after eval (trajectories are still saved)")
    return p


def run_mode(mode: str, args) -> dict:
    from learning.replay import Episode, Pack, dump_mp4s, save_traj_npz

    src = JSON_TMPL.format(mode=mode)
    wm_summary, fails = _fails(src)
    idx = [int(e["episode_idx"]) for e in fails]
    wm_by_ep = {int(e["episode_idx"]): e for e in fails}
    tag = f"500_solvercem_H{args.horizon}_{mode}"
    print(f"\n=== {mode}: {len(idx)} WM-CEM fails {idx} ===")

    cfg = Config()
    pk = dict(
        use_solver_cem=True,
        horizon=args.horizon,
        population=args.population,
        iterations=args.iterations,
        elites=args.elites,
        seed=args.seed,
        exec_horizon=args.exec_horizon,
    )
    task = PushToGoalTask(
        cfg, None, None, device="cpu", budget=args.budget,
        stride=args.stride, target_mode=mode, planner_kwargs=pk)

    records = []
    episodes = []
    n_ok = 0
    t0 = time.time()
    for k, i in enumerate(idx):
        rng = np.random.default_rng(args.seed + i)
        r = task.run_episode(rng, record=True)
        r = dict(r)
        r["episode_idx"] = i
        wm = wm_by_ep[i]
        r["wm_final_dist"] = wm["final_dist"]
        r["wm_settle_frame"] = wm["settle_frame"]
        r["wm_target_idx"] = wm["target_idx"]
        n_ok += int(r["success"])
        states, masks, actions = r.pop("frames")
        episodes.append(Episode(
            states=np.asarray(states, dtype=np.float32),
            geom=task._geom().copy(),
            types=np.asarray(task.env.obj_types_np, dtype=np.int64),
            goal=np.asarray(r["goal"], dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            masks=np.asarray(masks),
            success=bool(r["success"]),
            target_idx=int(r["target_idx"]),
            episode_idx=int(i),
            final_dist=float(r["final_dist"]),
            settle_frame=int(r["settle_frame"]),
            tag=tag,
        ))
        records.append(r)
        mark = "OK" if r["success"] else "FAIL"
        print(
            f"[{mode} {k + 1}/{len(idx)} ep{i:02d}] {mark}  "
            f"solver fin={r['final_dist']:.3f} settle={r['settle_frame']}  "
            f"WM fin={wm['final_dist']:.3f}  "
            f"tid {r['target_idx']} (wm {wm['target_idx']})"
        )
    elapsed = time.time() - t0
    summary = {
        "planner": "solver_cem",
        "source": src,
        "seed": args.seed,
        "tag": tag,
        "mode": mode,
        "target_mode": mode,
        "n_fail_wm": len(idx),
        "n_success_solver": n_ok,
        "success_rate": n_ok / max(len(idx), 1),
        "horizon": args.horizon,
        "population": args.population,
        "iterations": args.iterations,
        "exec_horizon": args.exec_horizon,
        "stride": args.stride,
        "budget": args.budget,
        "elapsed_s": elapsed,
        "wm_success_rate": wm_summary.get("success_rate"),
        "episode_idx": idx,
    }
    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, f"task_eval_{tag}.json")
    traj = os.path.join(args.results, f"task_trajectories_{tag}.npz")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": records}, f, indent=2)
    save_traj_npz(traj, episodes)
    print(f"{mode}: solver recovered {n_ok}/{len(idx)}  ({elapsed:.0f}s)  "
          f"wrote {out} + {traj}")
    if not args.no_replay:
        dump_mp4s(
            [Pack(tag=tag, eval_path=out, traj_path=traj,
                  summary=summary, episodes=episodes)],
            out_dir=os.path.join(args.results, "replays"),
            size="1280x720", fps=30, skip=2, hold=20, vel_ticks=True)
    return summary


def main():
    args = build_parser().parse_args()
    modes = MODES if args.mode == "all" else (args.mode,)
    print(
        f"solver CEM H={args.horizon} P={args.population} "
        f"iters={args.iterations} stride={args.stride} "
        f"exec={args.exec_horizon}"
    )
    summaries = [run_mode(m, args) for m in modes]
    print("\n" + "-" * 60)
    tot_ok = sum(s["n_success_solver"] for s in summaries)
    tot = sum(s["n_fail_wm"] for s in summaries)
    print(f"solver recovered {tot_ok}/{tot} world-model CEM failures")
    for s in summaries:
        print(
            f"  {s['mode']:10s}  {s['n_success_solver']}/{s['n_fail_wm']}  "
            f"{100 * s['success_rate']:.0f}%"
        )


if __name__ == "__main__":
    main()
