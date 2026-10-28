"""Train Push-T with rsl-rl PPO on real FlatWorld physics (no world model).

Uses the library PPO / MLP / Gaussian policy instead of a hand-rolled SAC/PPO.
This is the physics-side ablation: if rsl-rl succeeds here, the previous
Dreamer-style WM imagination stack was the bottleneck (or sample inefficiency).

Usage:
  python -m learning.train_rsl_push_t \\
    --num-envs 128 --num-workers 22 --max-iterations 2000 \\
    --out learning/checkpoints/push_t_rsl
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch


def make_train_cfg(args) -> dict:
    return {
        "num_steps_per_env": args.num_steps,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "run_name": "push_t_phys_rsl",
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
                "learn_std": True,
                "std_range": [0.05, 1.5],
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 8,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": args.entropy_coef,
            "learning_rate": args.lr,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": True,
        },
    }


@torch.no_grad()
def eval_policy(env, policy, n_eps: int, seed: int = 1000):
    """Deterministic-ish eval: use mean action (policy may still sample)."""
    # Recreate single-task eval for fair metric vs prior scripts.
    from learning.tasks.push_t import PushTTask, push_t_config
    from learning.rl.imagine import task_features

    device = env.device
    cfg = push_t_config()
    task = PushTTask(cfg, planner_kwargs=dict(use_servo=True), ee_spawn="support")
    fmax = float(cfg.collect.force_max)
    hold = env.hold
    sg, bg, gth = task.stem_goal, task.bar_goal, task.goal_theta
    ok_n, dists = 0, []

    for ep in range(n_eps):
        rng = np.random.default_rng(seed + ep)
        obs = task.env.reset(rng)
        task.env.set_ee_radius(float(task.cfg.scene.ee_radius) * task.ee_scale)
        obs = task._observe()
        goal = task.sample_goal(rng, obs)
        task.goal = goal
        t = 0
        success = False
        while t < task.budget:
            cur = task._observe()
            if task.succeeded(cur, goal):
                success = True
                break
            st = torch.as_tensor(
                cur["obj_states"], dtype=torch.float32).reshape(1, -1)
            pose = torch.as_tensor(
                cur["obj_states"][:, :3], dtype=torch.float32).unsqueeze(0)
            feat = task_features(pose, sg, bg, gth).reshape(1, -1)
            x = torch.cat([st, feat], dim=-1)
            from tensordict import TensorDict
            pol_device = next(policy.parameters()).device
            td = TensorDict({"policy": x.to(pol_device)}, batch_size=[1])
            # MLPModel: stochastic_output=False → deterministic mean action
            a = policy(td, stochastic_output=False)
            if isinstance(a, tuple):
                a = a[0]
            a = a.detach().float().cpu().numpy().reshape(-1)
            a = np.clip(a, -1.0, 1.0) * fmax
            for _ in range(hold):
                cur = task.env.step(a.astype(np.float32))
                t += 1
                if t >= task.budget:
                    break
        dists.append(float(task.metric(task._observe(), goal)))
        ok_n += int(success)
    return ok_n / max(n_eps, 1), float(np.mean(dists)), ok_n, n_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="learning/checkpoints/push_t_rsl")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="physics worker processes (default: cpu_count-2)")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=64,
                    help="steps per env per PPO iteration")
    ap.add_argument("--max-iterations", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-eps", type=int, default=20)
    ap.add_argument("--final-eval-eps", type=int, default=40)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--target-rate", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--max-ee-t-dist", type=float, default=-1.0,
        help="terminate if EE–T dist exceeds this; <=0 disables (recommended)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Physics (Warp) stays on CPU; policy on CUDA.
    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    from learning.rl.push_t_vec_env import PushTVecEnv
    from rsl_rl.runners import OnPolicyRunner

    ee_cap = None if args.max_ee_t_dist <= 0 else args.max_ee_t_dist
    print(f"rsl-rl PPO on physics Push-T  num_envs={args.num_envs} "
          f"workers={args.num_workers} hold={args.hold} "
          f"max_ee_t_dist={ee_cap} device={device}",
          flush=True)

    env = PushTVecEnv(
        num_envs=args.num_envs, hold=args.hold, device="cpu", seed=args.seed,
        num_workers=args.num_workers, max_ee_t_dist=ee_cap)
    cfg = make_train_cfg(args)
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)

    best_rate = -1.0
    history = []
    t0 = time.time()
    remaining = args.max_iterations
    done_iters = 0

    while remaining > 0:
        chunk = min(args.eval_every, remaining)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=False)
        done_iters += chunk
        remaining -= chunk

        policy = runner.get_inference_policy(device=device)
        rate, mdist, n_ok, n_ep = eval_policy(
            env, policy, args.eval_eps, seed=1000)
        print(f"  eval@{done_iters}: {n_ok}/{n_ep} = {100 * rate:.1f}% "
              f"mean_dist={mdist:.3f}", flush=True)
        history.append({
            "iter": done_iters,
            "success_rate": rate,
            "mean_dist": mdist,
            "elapsed_s": time.time() - t0,
        })
        ckpt_path = os.path.join(args.out, "last.pt")
        runner.save(ckpt_path)
        if rate >= best_rate:
            best_rate = rate
            runner.save(os.path.join(args.out, "best.pt"))
            print(f"  saved best.pt rate={100 * rate:.1f}%", flush=True)
        with open(os.path.join(args.out, "train_rsl_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "method": "rsl_rl_ppo_physics",
                "best_rate": best_rate,
                "history": history,
                "num_envs": args.num_envs,
                "num_workers": args.num_workers,
                "hold": args.hold,
            }, f, indent=2)
        if rate >= args.target_rate:
            print(f"TARGET {100 * args.target_rate:.0f}% reached", flush=True)
            break

    policy = runner.get_inference_policy(device=device)
    rate, mdist, n_ok, n_ep = eval_policy(
        env, policy, args.final_eval_eps, seed=1000)
    # Prefer best checkpoint for final if better
    best_path = os.path.join(args.out, "best.pt")
    if os.path.exists(best_path) and best_rate >= rate:
        runner.load(best_path)
        policy = runner.get_inference_policy(device=device)
        rate, mdist, n_ok, n_ep = eval_policy(
            env, policy, args.final_eval_eps, seed=1000)

    summary = {
        "method": "rsl_rl_ppo_physics",
        "final_success_rate": rate,
        "final_n_success": n_ok,
        "final_episodes": n_ep,
        "final_mean_dist": mdist,
        "best_train_eval_rate": best_rate,
        "wm_rl_best_ref": 0.45,
        "handrolled_phys_best_ref": 0.10,
        "num_envs": args.num_envs,
        "num_workers": args.num_workers,
        "hold": args.hold,
        "history": history,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(args.out, "train_rsl_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"FINAL rsl-rl physics: {n_ok}/{n_ep} = {100 * rate:.1f}% "
          f"mean_dist={mdist:.3f} (best_eval={100 * best_rate:.1f}%)",
          flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
