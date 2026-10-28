"""Evaluate WM-obs rsl-rl Push-T policy in sim and write report figures.

  python -m learning.eval_rl_push_t_report \\
    --policy learning/checkpoints/push_t_wm_rsl/best.pt \\
    --wm learning/checkpoints/push_t \\
    --episodes 40 --seed 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict


@torch.no_grad()
def eval_and_record(task, wm, policy, n_eps, seed, hold=5, deterministic=True):
    from learning.rl.imagine import task_features

    device = wm.device
    pol_device = next(policy.parameters()).device
    fmax = float(task.cfg.collect.force_max)
    sg, bg, gth = task.stem_goal, task.bar_goal, task.goal_theta
    records = []
    traj = {
        "states": [], "masks": [], "actions": [], "goal": [],
        "success": [], "target_idx": [], "geom": [], "types": [],
        "episode_idx": [], "final_dist": [], "settle_frame": [],
    }

    for ep in range(n_eps):
        rng = np.random.default_rng(seed + ep)
        task.env.reset(rng)
        task.env.set_ee_radius(float(task.cfg.scene.ee_radius) * task.ee_scale)
        goal = task.sample_goal(rng, task._observe())
        task.goal = goal

        states, actions, masks = [], [], []
        t = 0
        ok = False
        settle = int(task.budget)
        while t < task.budget:
            cur = task._observe()
            states.append(np.asarray(cur["obj_states"], dtype=np.float32).copy())
            masks.append(np.ones(cur["obj_states"].shape[0], dtype=np.float32))
            if task.succeeded(cur, goal):
                ok = True
                settle = t
                actions.append(np.zeros(2, dtype=np.float32))
                break
            z, _, _, _ = wm.encode_one(cur, use_tactile=False)
            pose = torch.as_tensor(
                cur["obj_states"][:, :3], device=device,
                dtype=torch.float32).unsqueeze(0)
            feat = task_features(pose, sg, bg, gth)
            x = torch.cat([z.reshape(1, -1), feat], dim=-1)
            td = TensorDict({"policy": x.to(pol_device)}, batch_size=[1])
            a = policy(td, stochastic_output=not deterministic)
            if isinstance(a, tuple):
                a = a[0]
            a = a.detach().float().cpu().numpy().reshape(-1)
            a = np.clip(a, -1.0, 1.0) * fmax
            actions.append(a.astype(np.float32).copy())
            for _ in range(hold):
                cur = task.env.step(a.astype(np.float32))
                t += 1
                if t >= task.budget:
                    break

        final = task._observe()
        fd = float(task.metric(final, goal))
        records.append({
            "success": bool(ok),
            "final_dist": fd,
            "settle_frame": int(settle),
            "episode_idx": ep,
            "target_idx": int(getattr(task, "target_idx", 1)),
        })
        traj["states"].append(np.stack(states, axis=0))
        traj["masks"].append(np.stack(masks, axis=0))
        traj["actions"].append(
            np.stack(actions, axis=0) if actions else np.zeros((1, 2), np.float32))
        traj["goal"].append(np.asarray(goal, dtype=np.float32))
        traj["success"].append(bool(ok))
        traj["target_idx"].append(int(getattr(task, "target_idx", 1)))
        traj["geom"].append(np.asarray(task._geom(), dtype=np.float32).copy())
        traj["types"].append(np.asarray(task.env.obj_types_np, dtype=np.int64).copy())
        traj["episode_idx"].append(ep)
        traj["final_dist"].append(fd)
        traj["settle_frame"].append(int(settle))
        print(f"  ep{ep:02d} {'OK' if ok else 'FAIL'} dist={fd:.3f} t={settle}",
              flush=True)

    n_ok = sum(1 for r in records if r["success"])
    return records, traj, n_ok / max(n_eps, 1), float(np.mean(traj["final_dist"])), n_ok


def plot_rl_curve(summary_json, out):
    with open(summary_json, encoding="utf-8") as f:
        d = json.load(f)
    hist = d.get("history") or []
    if not hist:
        return
    iters = [h["iter"] for h in hist]
    rates = [100 * float(h.get("success_rate", h.get("eval_success_rate", 0)))
             for h in hist]
    dists = [float(h["mean_dist"]) for h in hist]
    best = d.get("best_train_eval_rate", d.get("best_rate", 0))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(iters, rates, "-o", color="#2a9d8f", ms=4)
    ax[0].axhline(90, color="k", ls="--", lw=1, alpha=0.5, label="90% target")
    ax[0].set_xlabel("train iter")
    ax[0].set_ylabel("eval success (%)")
    ax[0].set_title(f"WM-obs rsl-rl train eval  (best={100 * float(best):.0f}%)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(iters, dists, "-o", color="#e76f51", ms=4)
    ax[1].set_xlabel("train iter")
    ax[1].set_ylabel("mean final dist (m)")
    ax[1].set_title("WM-obs rsl-rl mean distance")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _load_rsl_wm_policy(policy_path, wm_path, device, hold=5):
    """Rebuild a 1-env WMObsVecEnv + OnPolicyRunner and load checkpoint."""
    from learning.tasks.push_to_goal import load_ensemble
    from learning.rl.imagine import WMBundle
    from learning.rl.push_t_vec_env import PushTVecEnv
    from learning.train_rsl_wm_push_t import WMObsVecEnv
    from learning.train_rsl_push_t import make_train_cfg
    from rsl_rl.runners import OnPolicyRunner

    models, norm, stride = load_ensemble(wm_path, device)
    wm = WMBundle(models[0], norm, device, stride=stride)
    for p in wm.model.parameters():
        p.requires_grad_(False)
    wm.model.eval()

    base = PushTVecEnv(
        num_envs=1, hold=hold, device="cpu", seed=0,
        num_workers=0, max_ee_t_dist=None)
    env = WMObsVecEnv(base, wm, device=device)
    args = argparse.Namespace(
        num_steps=32, max_iterations=1, save_interval=100,
        entropy_coef=0.01, lr=1e-3)
    cfg = make_train_cfg(args)
    cfg["run_name"] = "push_t_wm_rsl_eval"
    log_dir = os.path.join(os.path.dirname(policy_path), "_eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy(device=device)
    return wm, policy, env, stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="learning/checkpoints/push_t_wm_rsl/best.pt")
    ap.add_argument("--wm", default="learning/checkpoints/push_t")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="learning/results/wm_rl")
    ap.add_argument("--tag", default="wm_rsl")
    ap.add_argument("--n-show", type=int, default=20)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    from learning.tasks.push_t import PushTTask, push_t_config
    from learning.make_report import plot_success_summary, plot_scenes

    print(f"loading WM={args.wm} policy={args.policy} device={device}", flush=True)
    wm, policy, train_env, stride = _load_rsl_wm_policy(
        args.policy, args.wm, device, hold=args.hold)
    hold = max(int(args.hold), int(stride), 1)

    cfg = push_t_config()
    task = PushTTask(cfg, planner_kwargs=dict(use_servo=True), ee_spawn="support")
    t0 = time.time()
    print(f"eval {args.episodes} eps seed={args.seed} hold={hold} ...", flush=True)
    records, traj, rate, mdist, n_ok = eval_and_record(
        task, wm, policy, args.episodes, seed=args.seed, hold=hold)

    summary = {
        "task": "PushT",
        "policy": args.policy,
        "wm": args.wm,
        "episodes": args.episodes,
        "n_success": n_ok,
        "success_rate": rate,
        "mean_final_dist": mdist,
        "mean_settle_frame": float(np.mean([
            r["settle_frame"] for r in records if r["success"]] or [0])),
        "tol": float(task.tol),
        "budget": int(task.budget),
        "seed": args.seed,
        "ee_spawn": "support",
        "planner": "wm_obs_rsl_rl",
        "tag": args.tag,
        "elapsed_s": time.time() - t0,
        "method": "rsl_rl_ppo_wm_obs",
        "best_train_eval_rate": None,
    }
    ckpt_dir = os.path.dirname(args.policy)
    train_sum = os.path.join(ckpt_dir, "train_wm_rsl_summary.json")
    if not os.path.exists(train_sum):
        train_sum = os.path.join(ckpt_dir, "train_rl_summary.json")
    if os.path.exists(train_sum):
        with open(train_sum, encoding="utf-8") as f:
            ts = json.load(f)
        summary["best_train_eval_rate"] = ts.get(
            "best_train_eval_rate", ts.get("best_rate"))
        summary["train_final_success_rate"] = ts.get("final_success_rate")

    eval_json = os.path.join(args.out, f"task_eval_{args.tag}.json")
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": records}, f, indent=2)

    traj_npz = os.path.join(args.out, f"task_trajectories_{args.tag}.npz")
    np.savez_compressed(
        traj_npz,
        states=np.array(traj["states"], dtype=object),
        masks=np.array(traj["masks"], dtype=object),
        actions=np.array(traj["actions"], dtype=object),
        goal=np.asarray(traj["goal"], dtype=np.float32),
        success=np.asarray(traj["success"]),
        target_idx=np.asarray(traj["target_idx"]),
        geom=np.stack(traj["geom"]).astype(np.float32),
        types=np.stack(traj["types"]).astype(np.int64),
        episode_idx=np.asarray(traj["episode_idx"], dtype=np.int32),
        final_dist=np.asarray(traj["final_dist"], dtype=np.float32),
        settle_frame=np.asarray(traj["settle_frame"], dtype=np.int32),
    )

    succ_png = os.path.join(args.out, f"task_success_summary_{args.tag}.png")
    scenes_png = os.path.join(args.out, f"task_scenes_{args.tag}.png")
    curve_png = os.path.join(args.out, f"train_curve_{args.tag}.png")
    plot_success_summary(eval_json, succ_png, title_extra=args.tag)
    plot_scenes(traj_npz, scenes_png, cfg, n_show=args.n_show,
                title=f"WM-obs rsl-rl Push-T  {n_ok}/{args.episodes} = {100 * rate:.0f}%")
    if os.path.exists(train_sum):
        plot_rl_curve(train_sum, curve_png)

    print(f"SUCCESS RATE: {n_ok}/{args.episodes} = {100 * rate:.1f}%", flush=True)
    print(f"mean_dist={mdist:.3f}", flush=True)
    print(f"wrote {eval_json}", flush=True)
    print(f"wrote {traj_npz}", flush=True)
    print(f"wrote {succ_png}", flush=True)
    print(f"wrote {scenes_png}", flush=True)
    if os.path.exists(curve_png):
        print(f"wrote {curve_png}", flush=True)

    if hasattr(train_env, "close"):
        train_env.close()


if __name__ == "__main__":
    main()
