"""Compare world-model rollouts vs FlatWorld ground truth.

This checks the SuperMimic hypothesis that early tracking error can come from
f_ω ≠ simulator dynamics. Reports k-step open-loop state / reward error on
policy (or reference) transitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .character import CharacterConfig, CharacterEnv
from .models import Policy, WorldModel
from .motion import make_clip, make_lafan2d_library


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def collect_traj(env: CharacterEnv, policy: Policy | None, device, n_steps: int):
    obs = env.reset(phase=0.0)
    states, actions, goals, rewards, next_states = [], [], [], [], []
    for _ in range(n_steps):
        s = env.full_state()
        g = env.goal()
        if policy is None:
            a, _ = env.clip.sample(env.phase)
        else:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = policy(o).squeeze(0).cpu().numpy()
        obs, r, done, info = env.step(a, action_noise=0.0)
        states.append(s)
        actions.append(info.get("action", a))
        goals.append(g)
        rewards.append(r)
        next_states.append(env.full_state())
        if done:
            obs = env.reset()
    return (
        np.asarray(states, np.float32),
        np.asarray(actions, np.float32),
        np.asarray(goals, np.float32),
        np.asarray(rewards, np.float32),
        np.asarray(next_states, np.float32),
    )


@torch.no_grad()
def wm_open_loop_error(wm: WorldModel, states, actions, goals, next_states, rewards, k: int, device):
    """Autoregressive k-step error starting at every feasible index."""
    T = states.shape[0]
    errs_s, errs_r = [], []
    for t0 in range(0, T - k):
        s = torch.as_tensor(states[t0], device=device).unsqueeze(0)
        for j in range(k):
            a = torch.as_tensor(actions[t0 + j], device=device).unsqueeze(0)
            g = torch.as_tensor(goals[t0 + j], device=device).unsqueeze(0)
            s, r_hat, _ = wm(s, a, g)
            s_tgt = next_states[t0 + j]
            errs_s.append(float(np.mean((s.squeeze(0).cpu().numpy() - s_tgt) ** 2)))
            errs_r.append(float((float(r_hat.item()) - float(rewards[t0 + j])) ** 2))
    return float(np.mean(errs_s)), float(np.mean(errs_r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="supermimic2d/runs/dance/best.pt")
    p.add_argument("--clip", type=str, default="dance")
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--root-kp", type=float, default=90.0)
    p.add_argument("--all", action="store_true", help="Diagnose every LAFAN-2D clip")
    args = p.parse_args()

    device = _device()
    cfg = CharacterConfig(
        control_hz=30.0, max_episode_steps=args.steps + 5,
        root_kp=args.root_kp, action_noise_std=0.0, obs_noise_std=0.0,
    )
    names = list(make_lafan2d_library().keys()) if args.all else [args.clip]

    ckpt = None
    policy = None
    wm = None
    path = Path(args.ckpt)
    if path.exists():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        wm = WorldModel(ckpt["state_dim"], ckpt["action_dim"], ckpt["goal_dim"],
                        hidden=int(ckpt.get("cfg", {}).get("hidden", 256))).to(device)
        wm.load_state_dict(ckpt["wm"])
        wm.eval()
        policy = Policy(
            ckpt["obs_dim"], ckpt["action_dim"],
            proprio_dim=int(ckpt.get("proprio_dim", ckpt["state_dim"])),
            hidden=int(ckpt.get("cfg", {}).get("hidden", 256)),
            delta_scale=float(ckpt.get("cfg", {}).get("delta_scale", 0.45)),
        ).to(device)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        print(f"loaded {path}")
    else:
        print(f"no ckpt at {path}; measuring reference-PD vs untrained WM skipped")
        return

    print(f"{'clip':12s}  {'1-step MSE':>10s}  {'{0}-step MSE'.format(args.k):>12s}  "
          f"{'r MSE':>10s}  {'ref 1-step':>10s}")
    for name in names:
        clip = make_clip(name, fps=30.0)
        env = CharacterEnv(cfg, clip)
        # policy traj
        S, A, G, R, Sp = collect_traj(env, policy, device, args.steps)
        e1, _ = wm_open_loop_error(wm, S, A, G, Sp, R, k=1, device=device)
        ek, er = wm_open_loop_error(wm, S, A, G, Sp, R, k=args.k, device=device)
        # reference-PD traj (distribution WM may have seen less of lately)
        env2 = CharacterEnv(cfg, clip)
        S2, A2, G2, R2, Sp2 = collect_traj(env2, None, device, args.steps)
        e1_ref, _ = wm_open_loop_error(wm, S2, A2, G2, Sp2, R2, k=1, device=device)
        print(f"{name:12s}  {e1:10.4f}  {ek:12.4f}  {er:10.4f}  {e1_ref:10.4f}")


if __name__ == "__main__":
    main()
