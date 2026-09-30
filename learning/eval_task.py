"""Evaluate PushToGoal with the shipped CEM controller.

Usage (repo root):
    python -m learning.eval_task --checkpoint learning/checkpoints \\
        --episodes 50 --target-mode random
    python -m learning.eval_task --task push_t --solver-cem --episodes 10
"""

import argparse
import json
import os
import time

import numpy as np

from learning.configs.default import Config


def _task_bundle(name: str):
    if name in ("two_room", "two-room"):
        from learning.tasks.two_room import TwoRoomTask, two_room_config
        return TwoRoomTask, two_room_config()
    if name in ("push_t", "push-t"):
        from learning.tasks.push_t import PushTTask, push_t_config
        return PushTTask, push_t_config()
    if name == "reacher":
        from learning.tasks.reacher import ReacherTask, reacher_config
        return ReacherTask, reacher_config()
    from learning.tasks.push_to_goal import PushToGoalTask
    return PushToGoalTask, Config()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", type=str,
                        default=["learning/checkpoints"],
                        help="checkpoint dir(s) or file(s); multiple = ensemble")
    parser.add_argument("--task", type=str, default="push",
                        help="push | two_room | push_t | reacher")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--budget", type=int, default=None,
                        help="sim frames (default: Config.task.budget = 400)")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--vel-tol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=None,
                        help="CEM horizon (default WM 32 / push_t 12 / solver 16)")
    parser.add_argument("--population", type=int, default=None,
                        help="CEM population (default WM 96 / push_t 192 / solver 12)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="CEM iterations (default WM 5 / push_t 10 / solver 2)")
    parser.add_argument("--elites", type=int, default=None,
                        help="CEM elites (default WM 16 / push_t 32 / solver 4)")
    parser.add_argument("--stride", type=int, default=None,
                        help="action hold frames (default: ckpt WM / 1 solver=collect)")
    parser.add_argument("--solver-cem", action="store_true",
                        help="CEM over FlatWorld solver (same physics as collect); no WM")
    parser.add_argument("--servo", action="store_true",
                        help="closed-loop hint servo (no CEM, no world model)")
    parser.add_argument("--exec-horizon", type=int, default=4,
                        help="solver-CEM steps to execute before replanning")
    parser.add_argument("--target-mode", type=str, default="leftmost",
                        choices=("leftmost", "rightmost", "random"))
    parser.add_argument("--record", type=int, default=50,
                        help="episodes to record trajectories")
    parser.add_argument("--tag", type=str, default="",
                        help="optional suffix for output files, e.g. random")
    parser.add_argument("--ee-scale", type=float, default=1.0,
                        help="EE radius multiplier (game slider is 0.5-2.0)")
    parser.add_argument("--ee-spawn", type=str, default="default",
                        choices=("default", "support"),
                        help="push-T EE reset: default random standoff, or "
                             "support = 1-2 cm behind the T (goal-opposite face)")
    parser.add_argument("--results", type=str, default="learning/results")
    parser.add_argument("--uncert-cost", type=float, default=None,
                        help="override ensemble disagreement cost (default 0.1)")
    parser.add_argument("--reduce", type=str, default="mean",
                        choices=("mean", "min"),
                        help="how to combine ensemble member costs")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated episode indices (seed+i); "
                             "skip all other episodes")
    parser.add_argument("--no-reach", action="store_true",
                        help="ablate reach (contact-face) cost")
    parser.add_argument("--no-settle", action="store_true",
                        help="ablate in-tolerance speed penalty")
    parser.add_argument("--settle-w", type=float, default=None,
                        help="in-tolerance imagined-speed weight (default 0)")
    parser.add_argument("--tactile", type=str, default="gate",
                        choices=("off", "gate", "on"),
                        help="EE tactile residual: never / miss-gated / always")
    parser.add_argument("--replay-fails", action="store_true", default=True,
                        help="write MP4s for failed episodes after eval")
    parser.add_argument("--no-replay-fails", action="store_true",
                        help="skip fail MP4s")
    parser.add_argument("--simple-cost", action="store_true",
                        help="use simplified PushTCost (task decomposition, no state machine)")
    return parser


def _cem_defaults(task_name: str, solver: bool):
    """Shorter H + denser CEM on Push-T (indirect contact); other tasks keep prior defaults."""
    is_pt = task_name in ("push_t", "push-t")
    if solver:
        return (16, 16, 3, 6) if is_pt else (16, 12, 2, 4)
    if is_pt:
        return 12, 192, 10, 32
    return 32, 96, 5, 16


def run_eval(args):
    solver = bool(getattr(args, "solver_cem", False))
    servo = bool(getattr(args, "servo", False))
    task_name = getattr(args, "task", "push")
    TaskCls, cfg = _task_bundle(task_name)
    dH, dP, dI, dE = _cem_defaults(task_name, solver)
    H = args.horizon if args.horizon is not None else dH
    P = args.population if args.population is not None else dP
    I = args.iterations if args.iterations is not None else dI
    elites = args.elites if args.elites is not None else dE
    if servo:
        device = "cpu"
        model, norm = None, None
        stride = int(args.stride) if args.stride is not None else 1
        n_models = 0
        pk = dict(use_servo=True)
        print("hint servo (closed-loop face/spin, no CEM)", flush=True)
    elif solver:
        device = "cpu"
        model, norm = None, None
        stride = int(args.stride) if args.stride is not None else 1
        n_models = 0
        pk = dict(
            use_solver_cem=True,
            horizon=H, population=P, iterations=I, elites=elites,
            seed=args.seed,
            exec_horizon=int(getattr(args, "exec_horizon", 4)),
        )
        if args.no_reach:
            pk["reach_cost"] = 0.0
        print(f"solver CEM H={H} P={P} iters={I} elites={elites} "
              f"stride={stride} exec={pk['exec_horizon']} "
              f"(same ExplicitLoop as collect)", flush=True)
    else:
        import torch
        from learning.tasks.push_to_goal import load_ensemble
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, norm, ckpt_stride = load_ensemble(args.checkpoint, device)
        n_models = len(model) if isinstance(model, list) else 1
        stride = int(args.stride) if args.stride is not None else int(ckpt_stride)
        pk = dict(horizon=H, population=P, iterations=I, seed=args.seed,
                  elites=elites)
        if args.uncert_cost is not None:
            pk["uncert_cost"] = float(args.uncert_cost)
        pk["ensemble_reduce"] = getattr(args, "reduce", "mean")
        if args.no_reach:
            pk["reach_cost"] = 0.0
        if getattr(args, "no_settle", False):
            pk["settle_w"] = 0.0
        if getattr(args, "settle_w", None) is not None:
            pk["settle_w"] = float(args.settle_w)
        pk["tactile_mode"] = args.tactile
    task = TaskCls(
        cfg, model, norm, device=device, tol=args.tol,
        vel_tol=args.vel_tol, budget=args.budget,
        stride=stride, target_mode=args.target_mode,
        ee_scale=args.ee_scale, planner_kwargs=pk,
        ee_spawn=getattr(args, "ee_spawn", "default"),
        simple_cost=bool(getattr(args, "simple_cost", False)))
    print(f"ee_spawn={getattr(args, 'ee_spawn', 'default')}", flush=True)

    records = []
    recorded = {"states": [], "masks": [], "actions": [], "goal": [],
                "success": [], "target_idx": [], "geom": [], "types": [],
                "episode_idx": [], "final_dist": [], "settle_frame": []}
    t0 = time.time()
    n_success = 0
    only = getattr(args, "only", "") or ""
    if only.strip():
        indices = [int(x) for x in only.split(",") if x.strip()]
    else:
        indices = list(range(args.episodes))
    n_run = len(indices)
    for k, i in enumerate(indices):
        rng = np.random.default_rng(args.seed + i)
        rec = k < args.record
        r = task.run_episode(rng, record=rec)
        n_success += r["success"]
        r = dict(r)
        r["episode_idx"] = i
        if rec:
            states, masks, actions = r["frames"]
            recorded["states"].append(states)
            recorded["masks"].append(masks)
            recorded["actions"].append(actions)
            recorded["goal"].append(r["goal"])
            recorded["success"].append(r["success"])
            recorded["target_idx"].append(r["target_idx"])
            recorded["geom"].append(task._geom().copy())
            recorded["types"].append(np.asarray(task.env.obj_types_np, dtype=np.int64))
            recorded["episode_idx"].append(i)
            recorded["final_dist"].append(r["final_dist"])
            recorded["settle_frame"].append(r["settle_frame"])
        records.append({key: v for key, v in r.items() if key != "frames"})
        if only.strip() or (k + 1) % 10 == 0 or k == 0 or n_run <= 15:
            rate = n_success / (k + 1)
            extra = ""
            if "init_ee_gap" in r:
                extra = f" init_ee_gap {r['init_ee_gap']:.3f}"
            print(f"[ep {i} {k + 1}/{n_run}] success so far {n_success}/{k + 1} "
                  f"({100 * rate:.1f}%), last final_dist {r['final_dist']:.3f}"
                  f"{extra}",
                  flush=True)
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    rate = n_success / n_run
    elapsed = time.time() - t0
    dists = [r["final_dist"] for r in records]
    frames = [r["settle_frame"] for r in records]
    summary = {
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "tag": args.tag,
        "episodes": n_run,
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
        "planner": "servo" if servo else ("solver_cem" if solver else "wm_cem"),
        "target_mode": getattr(args, "target_mode", ""),
        "task": getattr(args, "task", "push"),
        "ee_scale": args.ee_scale,
        "ee_spawn": getattr(args, "ee_spawn", "default"),
        "cem": {"horizon": H, "population": P,
                "iterations": I, "elites": elites,
                "stride": stride,
                "exec_horizon": int(getattr(args, "exec_horizon", 4)) if solver else 1,
                "uncert_cost": args.uncert_cost,
                "score": getattr(args, "score", "xy"),
                "only": getattr(args, "only", "") or None},
        "cost_ablate": {"reach": not args.no_reach},
        "tactile": args.tactile,
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
            types=np.stack(recorded["types"]).astype(np.int64),
            episode_idx=np.asarray(recorded["episode_idx"], dtype=np.int32),
            final_dist=np.asarray(recorded["final_dist"], dtype=np.float32),
            settle_frame=np.asarray(recorded["settle_frame"], dtype=np.int32),
        )

    print("-" * 60)
    print(f"SUCCESS RATE: {n_success}/{n_run} = {100 * rate:.1f}%")
    print(f"target={args.target_mode}")
    print(f"Mean final distance: {summary['mean_final_dist']:.3f} "
          f"(tol {task.tol}, vel {task.vel_tol}), "
          f"mean settle frame {summary['mean_settle_frame']:.0f}")
    print(f"Elapsed {elapsed:.1f}s ({elapsed / n_run:.2f}s/episode)")
    print(f"Wrote {eval_path}")
    if recorded["states"] and getattr(args, "replay_fails", True) \
            and not getattr(args, "no_replay_fails", False):
        from learning.replay import dump_eval_fails
        dump_eval_fails(eval_path)
    return summary


def main():
    run_eval(build_parser().parse_args())


if __name__ == "__main__":
    main()
