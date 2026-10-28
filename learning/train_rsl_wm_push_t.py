"""Push-T PPO with frozen WM latents + real FlatWorld physics (rsl-rl).

World model only encodes observations; dynamics/reward stay on the real
simulator — the recipe that already hit 90% on raw-state PPO, but the
policy now conditions on WM ``z`` + task features.

Usage:
  python -m learning.train_rsl_wm_push_t \\
    --wm learning/checkpoints/push_t \\
    --num-envs 128 --num-workers 6 \\
    --out learning/checkpoints/push_t_wm_rsl --target-rate 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tensordict import TensorDict

from learning.train_rsl_push_t import make_train_cfg


class WMObsVecEnv:
    """Wrap ``PushTVecEnv``: replace policy obs with ``[z_flat | feat]``."""

    def __init__(self, base, wm, device: str = "cuda"):
        from learning.rl.imagine import task_features, FEAT_DIM

        self.base = base
        self.wm = wm
        self.device = torch.device(device)
        self._task_features = task_features
        self.num_envs = base.num_envs
        self.num_actions = base.num_actions
        self.max_episode_length = base.max_episode_length
        self.episode_length_buf = base.episode_length_buf
        self.cfg = dict(base.cfg)
        self.cfg["wm_obs"] = True
        self.hold = base.hold
        self.sg = base.sg
        self.bg = base.bg
        self.gth = base.gth

        # Minimal observe template (tactile dropped — WM trained with drop).
        from learning.models.relations import empty_rel
        self._tmpl = {
            "obj_types": np.asarray([0, 1, 1], dtype=np.int64),
            "obj_geom": np.asarray([[0.05, 0.05], [0.04, 0.14], [0.16, 0.04]],
                                   dtype=np.float32),
            "rel_type": empty_rel(3),
        }
        st0 = base.get_observations()["policy"][0, :18].cpu().numpy().reshape(3, 6)
        z, _, _, _ = wm.encode_one(self._states_to_obs(st0), use_tactile=False)
        self.z_dim = int(z.numel())
        self.feat_dim = FEAT_DIM
        self._obs_dim = self.z_dim + self.feat_dim
        self._obs = self._encode_td(base.get_observations()["policy"])

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _states_to_obs(self, states: np.ndarray) -> dict:
        n = states.shape[0]
        geom = self._tmpl.get("obj_geom")
        if geom is None or len(geom) != n:
            geom = np.zeros((n, 2), dtype=np.float32)
            geom[0] = 0.05
            geom[1:] = 0.04
        types = self._tmpl.get("obj_types")
        if types is None or len(types) != n:
            types = np.asarray([0, 1, 1], dtype=np.int64)
        k = 8
        return {
            "obj_states": np.asarray(states, dtype=np.float32),
            "obj_types": np.asarray(types, dtype=np.int64),
            "obj_geom": np.asarray(geom, dtype=np.float32),
            "contact_feat": np.zeros((k, 7), dtype=np.float32),
            "contact_mask": np.zeros((k,), dtype=np.float32),
            "tactile_summary": np.zeros((4,), dtype=np.float32),
            "rel_type": self._tmpl.get("rel_type"),
        }

    @torch.no_grad()
    def _encode_td(self, policy_row: torch.Tensor) -> TensorDict:
        """``policy_row``: (N, 18+feat) raw-state obs from base env.

        States stay on device for task features; only the WM encode path
        materializes a small numpy obs list (frozen GNN batch).
        """
        x = policy_row.detach().float()
        states_t = x[:, :18].reshape(self.num_envs, 3, 6)
        states_np = states_t.detach().cpu().numpy()
        obs_list = [self._states_to_obs(states_np[i]) for i in range(self.num_envs)]
        z, _, _, _ = self.wm.encode_batch(obs_list, use_tactile=False)
        z_flat = z.reshape(self.num_envs, -1)
        pose = states_t[:, :, :3].to(device=self.wm.device, dtype=torch.float32)
        feat = self._task_features(pose, self.sg, self.bg, self.gth)
        pol = torch.cat([z_flat, feat], dim=-1).to(self.device)
        return TensorDict({"policy": pol}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._obs.clone()

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = self.base.step(actions)
        self.episode_length_buf = self.base.episode_length_buf
        self._obs = self._encode_td(obs["policy"])
        return self._obs.clone(), rew.to(self.device), dones.to(self.device), extras

    def close(self):
        if hasattr(self.base, "close"):
            self.base.close()


@torch.no_grad()
def eval_policy(wm, policy, n_eps: int, seed: int = 1000, hold: int = 5):
    from learning.tasks.push_t import PushTTask, push_t_config
    from learning.rl.imagine import task_features
    from tensordict import TensorDict

    device = wm.device
    cfg = push_t_config()
    task = PushTTask(cfg, planner_kwargs=dict(use_servo=True), ee_spawn="support")
    fmax = float(cfg.collect.force_max)
    sg, bg, gth = task.stem_goal, task.bar_goal, task.goal_theta
    ok_n, dists = 0, []
    pol_device = next(policy.parameters()).device

    for ep in range(n_eps):
        rng = np.random.default_rng(seed + ep)
        task.env.reset(rng)
        task.env.set_ee_radius(float(task.cfg.scene.ee_radius) * task.ee_scale)
        goal = task.sample_goal(rng, task._observe())
        task.goal = goal
        t = 0
        success = False
        while t < task.budget:
            cur = task._observe()
            if task.succeeded(cur, goal):
                success = True
                break
            z, _, _, _ = wm.encode_one(cur, use_tactile=False)
            pose = torch.as_tensor(
                cur["obj_states"][:, :3], dtype=torch.float32,
                device=device).unsqueeze(0)
            feat = task_features(pose, sg, bg, gth)
            x = torch.cat([z.reshape(1, -1), feat], dim=-1)
            td = TensorDict({"policy": x.to(pol_device)}, batch_size=[1])
            a = policy(td, stochastic_output=False)
            if isinstance(a, tuple):
                a = a[0]
            a = a.detach().float().cpu().numpy().reshape(-1)
            a = np.clip(a, -1.0, 1.0) * fmax
            for _ in range(hold):
                task.env.step(a.astype(np.float32))
                t += 1
                if t >= task.budget:
                    break
        dists.append(float(task.metric(task._observe(), goal)))
        ok_n += int(success)
    return ok_n / max(n_eps, 1), float(np.mean(dists)), ok_n, n_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="learning/checkpoints/push_t")
    ap.add_argument("--out", default="learning/checkpoints/push_t_wm_rsl")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-eps", type=int, default=20)
    ap.add_argument("--final-eval-eps", type=int, default=40)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--target-rate", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    from learning.tasks.push_to_goal import load_ensemble
    from learning.rl.imagine import WMBundle
    from learning.rl.push_t_vec_env import PushTVecEnv
    from rsl_rl.runners import OnPolicyRunner

    print(f"WM-obs rsl-rl PPO  wm={args.wm} envs={args.num_envs} "
          f"workers={args.num_workers} device={device}", flush=True)
    models, norm, stride = load_ensemble(args.wm, device)
    wm = WMBundle(models[0], norm, device, stride=stride)
    for p in wm.model.parameters():
        p.requires_grad_(False)
    wm.model.eval()

    base = PushTVecEnv(
        num_envs=args.num_envs, hold=args.hold, device="cpu", seed=args.seed,
        num_workers=args.num_workers, max_ee_t_dist=None)
    env = WMObsVecEnv(base, wm, device=device)
    print(f"obs_dim={env._obs_dim} (z={env.z_dim}+feat={env.feat_dim}) "
          f"stride={stride}", flush=True)

    cfg = make_train_cfg(args)
    cfg["run_name"] = "push_t_wm_rsl"
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
            wm, policy, args.eval_eps, seed=1000, hold=args.hold)
        print(f"  eval@{done_iters}: {n_ok}/{n_ep} = {100 * rate:.1f}% "
              f"mean_dist={mdist:.3f}", flush=True)
        history.append({
            "iter": done_iters,
            "success_rate": rate,
            "mean_dist": mdist,
            "elapsed_s": time.time() - t0,
        })
        runner.save(os.path.join(args.out, "last.pt"))
        if rate >= best_rate:
            best_rate = rate
            runner.save(os.path.join(args.out, "best.pt"))
            print(f"  saved best.pt rate={100 * rate:.1f}%", flush=True)
        with open(os.path.join(args.out, "train_wm_rsl_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "method": "rsl_rl_ppo_wm_obs",
                "wm": args.wm,
                "best_rate": best_rate,
                "history": history,
                "num_envs": args.num_envs,
                "hold": args.hold,
            }, f, indent=2)
        if rate >= args.target_rate:
            print(f"TARGET {100 * args.target_rate:.0f}% reached", flush=True)
            break

    policy = runner.get_inference_policy(device=device)
    rate, mdist, n_ok, n_ep = eval_policy(
        wm, policy, args.final_eval_eps, seed=1000, hold=args.hold)
    best_path = os.path.join(args.out, "best.pt")
    if os.path.exists(best_path) and best_rate >= rate:
        runner.load(best_path)
        policy = runner.get_inference_policy(device=device)
        rate, mdist, n_ok, n_ep = eval_policy(
            wm, policy, args.final_eval_eps, seed=1000, hold=args.hold)

    summary = {
        "method": "rsl_rl_ppo_wm_obs",
        "wm": args.wm,
        "final_success_rate": rate,
        "final_n_success": n_ok,
        "final_episodes": n_ep,
        "final_mean_dist": mdist,
        "best_train_eval_rate": best_rate,
        "num_envs": args.num_envs,
        "num_workers": args.num_workers,
        "hold": args.hold,
        "history": history,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(args.out, "train_wm_rsl_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"FINAL WM-obs rsl: {n_ok}/{n_ep} = {100 * rate:.1f}% "
          f"mean_dist={mdist:.3f} (best_eval={100 * best_rate:.1f}%)",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
