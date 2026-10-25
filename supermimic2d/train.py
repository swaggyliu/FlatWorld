"""SuperMimic-2D training loop.

Alternates:
  (1) collect transitions in FlatWorld with πθ (+ action noise)
  (2) fit differentiable world model on replay
  (3) BPTT policy + critic through H imagined steps (world model frozen)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .character import CharacterConfig, CharacterEnv
from .models import Critic, Policy, WorldModel
from .motion import MotionLibrary, make_clip, make_dance_clip, make_lafan2d_library


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _pick_clip(env: CharacterEnv, library: MotionLibrary | None, rng: np.random.Generator | None):
    if library is not None and rng is not None:
        env.set_clip(library.sample_clip(rng))


@torch.no_grad()
def collect_rollouts(env: CharacterEnv, policy: Policy, buffer: ReplayBuffer,
                     n_steps: int, device, action_noise: float,
                     library: MotionLibrary | None = None,
                     rng: np.random.Generator | None = None):
    """Phase (1): simulator data collection."""
    policy.eval()
    _pick_clip(env, library, rng)
    obs = env.reset()
    ep_ret = 0.0
    returns = []
    for _ in range(n_steps):
        state = env.full_state()
        goal = env.goal()
        obs_t = _to_t(obs, device).unsqueeze(0)
        action = policy(obs_t).squeeze(0).cpu().numpy()
        next_obs, reward, done, info = env.step(action, action_noise=action_noise)
        next_state = env.full_state()
        next_goal = env.goal()
        a_exec = info.get("action", action)
        buffer.add(state, a_exec, reward, next_state, done, goal, obs, next_goal)
        ep_ret += float(reward)
        obs = next_obs
        if done:
            returns.append(ep_ret)
            _pick_clip(env, library, rng)
            obs = env.reset()
            ep_ret = 0.0
    return float(np.mean(returns)) if returns else ep_ret


def fit_world_model(wm: WorldModel, opt, buffer: ReplayBuffer, steps: int,
                    batch_size: int, k: int, device, rng: np.random.Generator):
    """Phase (2): k-step autoregressive fit on replay windows."""
    wm.train()
    logs = []
    for _ in range(steps):
        batch = buffer.sample_windows(batch_size, k, rng)
        state = _to_t(batch["state"], device)       # (B, k, S)
        action = _to_t(batch["action"], device)
        goal = _to_t(batch["goal"], device)
        next_state = _to_t(batch["next_state"], device)
        reward = _to_t(batch["reward"], device)
        done = _to_t(batch["done"], device)

        B = state.shape[0]
        s_hat = state[:, 0]
        loss = 0.0
        metrics = {"wm_state": 0.0, "wm_reward": 0.0, "wm_done": 0.0}
        for t in range(k):
            a_t = action[:, t]
            g_t = goal[:, t]
            s_tgt = next_state[:, t]
            r_tgt = reward[:, t]
            d_tgt = done[:, t]
            # Teacher-force mix: mostly use predicted state (autoregressive)
            if t > 0:
                s_in = s_hat
            else:
                s_in = state[:, 0]
            # Update running norms on real transitions (no grad).
            if t == 0:
                wm.state_norm.update(s_in)
                wm.action_norm.update(a_t)
                wm.goal_norm.update(g_t)
                wm.state_norm.update(s_tgt)
            pred_s, pred_r, pred_d = wm(s_in, a_t, g_t)
            # Compare in normalized space for stable scales.
            ls = F.mse_loss(wm.state_norm.normalize(pred_s), wm.state_norm.normalize(s_tgt))
            lr = F.mse_loss(pred_r, r_tgt)
            ld = F.binary_cross_entropy_with_logits(pred_d, d_tgt)
            loss = loss + ls + 0.5 * lr + 0.1 * ld
            metrics["wm_state"] += float(ls.detach()) / k
            metrics["wm_reward"] += float(lr.detach()) / k
            metrics["wm_done"] += float(ld.detach()) / k
            s_hat = pred_s
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 5.0)
        opt.step()
        logs.append({"loss": float(loss.detach()) / k, **metrics})
    # average
    keys = logs[0].keys()
    return {k: float(np.mean([d[k] for d in logs])) for k in keys}


def improve_policy(wm: WorldModel, policy: Policy, critic: Critic,
                   opt_pi, opt_v, buffer: ReplayBuffer, *,
                   batch_size: int, H: int, gamma: float, lam: float,
                   device, rng: np.random.Generator, obs_noise_std: float,
                   bc_weight: float = 0.5):
    """Phase (3): BPTT through frozen world model + critic bootstrap."""
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    policy.train()
    critic.train()

    batch = buffer.sample_batch(batch_size, rng)
    s = _to_t(batch["state"], device)
    g = _to_t(batch["goal"], device)
    a_demo = _to_t(batch["action"], device)  # often near reference (seed / PD)
    # Reconstruct obs = proprio(state) + goal  — state IS proprio here
    # obs used by policy: concatenate state and goal (matches CharacterEnv.observation)
    def make_obs(state, goal):
        if obs_noise_std > 0:
            noise = torch.randn_like(state) * obs_noise_std
            # only noise proprio part
            return torch.cat([state + noise, goal], dim=-1)
        return torch.cat([state, goal], dim=-1)

    rewards = []
    dones = []
    values = []
    surv = torch.ones(batch_size, device=device)
    survs = [surv]
    total_return = 0.0

    s_t = s
    g_t = g
    # BC toward reference joints in the goal window (and logged near-ref actions)
    obs0 = make_obs(s, g)
    a0 = policy(obs0)
    q_ref = g[..., : policy.action_dim]
    loss_bc = F.mse_loss(a0, q_ref) + 0.25 * F.mse_loss(a0, a_demo)

    for t in range(H):
        obs_t = make_obs(s_t, g_t)
        a_t = policy(obs_t)
        v_t = critic(s_t, g_t)
        values.append(v_t)
        s_next, r_hat, d_logit = wm(s_t, a_t, g_t)
        d_hat = torch.sigmoid(d_logit)
        rewards.append(r_hat)
        dones.append(d_hat)
        surv = surv * (1.0 - d_hat)
        survs.append(surv)
        total_return = total_return + (gamma ** t) * survs[t] * r_hat
        s_t = s_next
        # Advance goal mildly: blend toward next_goal from seed transition
        if t == 0:
            g_t = 0.7 * g_t + 0.3 * _to_t(batch["next_goal"], device)

    # Bootstrap
    with torch.no_grad():
        v_H = critic(s_t.detach(), g_t.detach())
    total_return = total_return + (gamma ** H) * survs[H] * v_H

    # Maximize imagined return + stay close to reference setpoints
    loss_pi = -total_return.mean() + float(bc_weight) * loss_bc
    opt_pi.zero_grad(set_to_none=True)
    loss_pi.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    opt_pi.step()

    # Critic: GAE-style targets on imagined trajectory (detach)
    with torch.no_grad():
        R = v_H
        targets = []
        for t in reversed(range(H)):
            r = rewards[t].detach()
            d = dones[t].detach()
            v = values[t].detach()
            v_next = R if t == H - 1 else values[t + 1].detach()
            delta = r + gamma * (1.0 - d) * v_next - v
            R = v + delta  # λ=1 style; simplified
            # Proper GAE:
            # We'll recompute with λ
        # Recompute GAE properly
        adv = torch.zeros(batch_size, device=device)
        targets = [None] * H
        next_v = v_H
        for t in reversed(range(H)):
            r = rewards[t].detach()
            d = dones[t].detach()
            v = values[t].detach()
            delta = r + gamma * (1.0 - d) * next_v - v
            adv = delta + gamma * lam * (1.0 - d) * adv
            targets[t] = (adv + v).detach()
            next_v = v

    # Fresh forward for critic loss (avoid double-backward through shared graph)
    loss_v = 0.0
    s_t = _to_t(batch["state"], device)
    g_t = _to_t(batch["goal"], device)
    for t in range(H):
        v = critic(s_t, g_t)
        loss_v = loss_v + F.mse_loss(v, targets[t])
        with torch.no_grad():
            obs_t = make_obs(s_t, g_t)
            a_t = policy(obs_t)
            s_t, _, _ = wm(s_t, a_t, g_t)
            if t == 0:
                g_t = 0.7 * g_t + 0.3 * _to_t(batch["next_goal"], device)
    loss_v = loss_v / H
    opt_v.zero_grad(set_to_none=True)
    loss_v.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
    opt_v.step()

    for p in wm.parameters():
        p.requires_grad_(True)

    return {
        "J": float(total_return.mean().detach()),
        "loss_pi": float(loss_pi.detach()),
        "loss_v": float(loss_v.detach()),
        "loss_bc": float(loss_bc.detach()),
        "imag_r": float(torch.stack(rewards).mean().detach()),
    }


@torch.no_grad()
def evaluate(env: CharacterEnv, policy: Policy, device, n_episodes: int = 5,
             library: MotionLibrary | None = None):
    policy.eval()
    scores = []
    pose = []
    per_clip = {}
    names = library.names if library is not None else [getattr(env.clip, "name", "clip")]
    # At least one episode per clip when using a library.
    plan = []
    if library is not None:
        for name in names:
            plan.extend([name] * max(1, n_episodes // max(len(names), 1)))
        while len(plan) < n_episodes:
            plan.append(names[len(plan) % len(names)])
        plan = plan[: max(n_episodes, len(names))]
    else:
        plan = [names[0]] * n_episodes

    for ep, name in enumerate(plan):
        if library is not None:
            env.set_clip(library.get(name))
        # Always start at rest (phase 0); clip ease-in handles the ramp.
        obs = env.reset(phase=0.0)
        ret = 0.0
        pose_acc = []
        done = False
        while not done:
            a = policy(_to_t(obs, device).unsqueeze(0)).squeeze(0).cpu().numpy()
            obs, r, done, info = env.step(a, action_noise=0.0)
            ret += float(r)
            pose_acc.append(float(info["r_pose"]))
        scores.append(ret)
        pose.append(float(np.mean(pose_acc)))
        per_clip.setdefault(name, []).append(ret)
    out = {"return": float(np.mean(scores)), "pose": float(np.mean(pose))}
    for n, vals in per_clip.items():
        out[f"ret_{n}"] = float(np.mean(vals))
    return out


def train(args):
    device = _device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = CharacterConfig(
        control_hz=args.fps,
        max_episode_steps=args.ep_len,
        root_kp=args.root_kp,
        action_noise_std=args.action_noise,
        obs_noise_std=args.obs_noise,
    )
    library = None
    if args.library:
        # LAFAN-like multi-skill suite (shared world model amortizes dynamics).
        names = [n.strip() for n in args.library.split(",") if n.strip()]
        if names == ["all"] or names == ["lafan2d"]:
            library = MotionLibrary(make_lafan2d_library(fps=args.fps))
        else:
            library = MotionLibrary({n: make_clip(n, fps=args.fps) for n in names})
        clip = library.sample_clip(np.random.default_rng(args.seed))
        print(f"[supermimic2d] motion library: {library.names}", flush=True)
    else:
        clip = make_clip(args.clip, fps=args.fps) if args.clip != "dance" \
            else make_dance_clip(fps=args.fps, duration=args.clip_duration)
    env = CharacterEnv(cfg, clip)

    state_dim = env.proprio_dim
    action_dim = env.action_dim
    goal_dim = env.goal_dim
    obs_dim = env.obs_dim

    wm = WorldModel(state_dim, action_dim, goal_dim, hidden=args.hidden).to(device)
    policy = Policy(
        obs_dim, action_dim, proprio_dim=state_dim,
        hidden=args.hidden, delta_scale=args.delta_scale,
    ).to(device)
    critic = Critic(state_dim, goal_dim, hidden=args.hidden).to(device)

    opt_wm = torch.optim.AdamW(
        [p for n, p in wm.named_parameters() if "norm" not in n],
        lr=args.lr_wm, weight_decay=1e-4,
    )
    opt_pi = torch.optim.AdamW(policy.parameters(), lr=args.lr_pi, weight_decay=1e-4)
    opt_v = torch.optim.AdamW(critic.parameters(), lr=args.lr_v, weight_decay=1e-4)

    buffer = ReplayBuffer(args.buffer, state_dim, action_dim, goal_dim, obs_dim)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    history = []
    t0 = time.time()
    best = -1e9
    start_it = 1
    print(f"[supermimic2d] device={device}  state={state_dim} act={action_dim} "
          f"goal={goal_dim}  clip_frames={clip.n_frames}", flush=True)

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and args.resume_best:
        cand = out / "best.pt"
        resume_path = cand if cand.exists() else None
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        critic.load_state_dict(ckpt["critic"])
        wm.load_state_dict(ckpt["wm"])
        policy.delta_scale = float(args.delta_scale)
        best = float(ckpt.get("eval", {}).get("return", -1e9))
        hist_path = out / "history.json"
        if hist_path.exists():
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)
            if history:
                start_it = int(history[-1].get("iter", 0)) + 1
        print(f"[supermimic2d] resumed {resume_path}  best={best:.2f}  "
              f"continue from iter {start_it}", flush=True)
        # Refresh buffer with current policy + reference mix
        print("[supermimic2d] refreshing replay buffer ...", flush=True)
        _pick_clip(env, library, rng)
        obs = env.reset(seed=args.seed)
        for i in range(args.seed_steps):
            state = env.full_state()
            goal = env.goal()
            if i % 3 == 0:
                q_ref, _ = env.clip.sample(env.phase)
                a = q_ref + rng.normal(0, 0.08, size=q_ref.shape).astype(np.float32)
            else:
                with torch.no_grad():
                    a = policy(_to_t(obs, device).unsqueeze(0)).squeeze(0).cpu().numpy()
            next_obs, reward, done, info = env.step(a, action_noise=0.0)
            buffer.add(state, info["action"], reward, env.full_state(), done,
                       goal, obs, env.goal())
            obs = next_obs
            if done:
                _pick_clip(env, library, rng)
                obs = env.reset()
            if (i + 1) % 500 == 0:
                print(f"  refreshed {i + 1}/{args.seed_steps}", flush=True)
        for _ in range(max(2, args.wm_warmup // 2)):
            fit_world_model(
                wm, opt_wm, buffer, steps=args.wm_steps, batch_size=args.batch,
                k=args.k, device=device, rng=rng,
            )
    else:
        # Bootstrap buffer with noisy reference actions
        print("[supermimic2d] seeding buffer with reference PD tracking ...", flush=True)
        _pick_clip(env, library, rng)
        obs = env.reset(seed=args.seed)
        for i in range(args.seed_steps):
            state = env.full_state()
            goal = env.goal()
            q_ref, _ = env.clip.sample(env.phase)
            a = q_ref + rng.normal(0, 0.15, size=q_ref.shape).astype(np.float32)
            next_obs, reward, done, info = env.step(a, action_noise=0.0)
            buffer.add(state, info["action"], reward, env.full_state(), done, goal, obs, env.goal())
            obs = next_obs
            if done:
                _pick_clip(env, library, rng)
                obs = env.reset()
            if (i + 1) % 500 == 0:
                print(f"  seeded {i + 1}/{args.seed_steps}", flush=True)
        print(f"[supermimic2d] buffer size={buffer.size}, pretraining world model ...", flush=True)
        for _ in range(args.wm_warmup):
            fit_world_model(
                wm, opt_wm, buffer, steps=args.wm_steps, batch_size=args.batch,
                k=args.k, device=device, rng=rng,
            )

    end_it = start_it + args.iters - 1
    print(f"[supermimic2d] starting SuperMimic loop  iters {start_it}..{end_it}", flush=True)

    for it in range(start_it, end_it + 1):
        # Mild root-assist anneal — keep tracking tight (large decay hurt late evals).
        frac = min(1.0, (it - start_it + 1) / max(args.iters * 0.8, 1))
        env.cfg.root_kp = args.root_kp * (1.0 - 0.12 * frac)

        col_ret = collect_rollouts(
            env, policy, buffer, n_steps=args.collect_steps, device=device,
            action_noise=args.action_noise, library=library, rng=rng,
        )
        wm_stats = fit_world_model(
            wm, opt_wm, buffer, steps=args.wm_steps, batch_size=args.batch,
            k=args.k, device=device, rng=rng,
        )
        pi_stats = {"J": 0.0, "loss_pi": 0.0, "loss_v": 0.0, "loss_bc": 0.0, "imag_r": 0.0}
        # Delay policy BPTT until the world model has seen enough updates.
        # When resuming, start improving immediately.
        do_pi = (it - start_it + 1) > args.pi_delay or resume_path is not None
        if do_pi:
            for _ in range(args.pi_steps):
                s = improve_policy(
                    wm, policy, critic, opt_pi, opt_v, buffer,
                    batch_size=args.batch, H=args.H, gamma=args.gamma, lam=args.lam,
                    device=device, rng=rng, obs_noise_std=args.obs_noise,
                    bc_weight=args.bc_weight,
                )
                for k, v in s.items():
                    pi_stats[k] += v / args.pi_steps

        row = {"iter": it, "collect_ret": col_ret, **wm_stats, **pi_stats,
               "buffer": buffer.size, "sec": time.time() - t0}
        if it % args.eval_every == 0 or it == 1:
            ev = evaluate(env, policy, device, n_episodes=args.eval_episodes, library=library)
            row.update({f"eval_{k}": v for k, v in ev.items()})
            extra = ""
            if library is not None:
                bits = [f"{n}:{ev.get(f'ret_{n}', float('nan')):.1f}" for n in library.names[:4]]
                extra = "  [" + " ".join(bits) + (" …]" if len(library.names) > 4 else "]")
            print(
                f"iter {it:4d}  collect={col_ret:6.2f}  eval_ret={ev['return']:6.2f}  "
                f"pose={ev['pose']:.3f}  wm={wm_stats['loss']:.4f}  "
                f"J={pi_stats['J']:.3f}  t={row['sec']:.0f}s{extra}",
                flush=True,
            )
            if ev["return"] > best:
                best = ev["return"]
                ckpt = {
                    "policy": policy.state_dict(),
                    "critic": critic.state_dict(),
                    "wm": wm.state_dict(),
                    "cfg": vars(args),
                    "obs_dim": obs_dim,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "goal_dim": goal_dim,
                    "proprio_dim": state_dim,
                    "eval": ev,
                }
                torch.save(ckpt, out / "best.pt")
        else:
            print(
                f"iter {it:4d}  collect={col_ret:6.2f}  wm={wm_stats['loss']:.4f}  "
                f"J={pi_stats['J']:.3f}  t={row['sec']:.0f}s",
                flush=True,
            )
        history.append(row)

    torch.save(
        {
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "wm": wm.state_dict(),
            "cfg": vars(args),
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "goal_dim": goal_dim,
            "proprio_dim": state_dim,
        },
        out / "last.pt",
    )
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[supermimic2d] done. best_eval={best:.2f}  checkpoints → {out}", flush=True)
    return out / "best.pt"


def build_argparser():
    p = argparse.ArgumentParser(description="Train SuperMimic-2D dance tracker")
    p.add_argument("--out", type=str, default="supermimic2d/runs/dance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--clip", type=str, default="dance",
                   help="Single clip name (ignored if --library is set)")
    p.add_argument("--library", type=str, default="",
                   help="Comma-separated clips, or 'all'/'lafan2d' for the full suite")
    p.add_argument("--clip-duration", type=float, default=4.0)
    p.add_argument("--ep-len", type=int, default=90)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--seed-steps", type=int, default=2000)
    p.add_argument("--collect-steps", type=int, default=400)
    p.add_argument("--wm-steps", type=int, default=20)
    p.add_argument("--pi-steps", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=8, help="WM autoregressive window")
    p.add_argument("--H", type=int, default=16, help="imagination horizon")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr-wm", type=float, default=3e-4)
    p.add_argument("--lr-pi", type=float, default=5e-5)
    p.add_argument("--lr-v", type=float, default=3e-4)
    p.add_argument("--action-noise", type=float, default=0.05)
    p.add_argument("--obs-noise", type=float, default=0.01)
    p.add_argument("--root-kp", type=float, default=90.0)
    p.add_argument("--delta-scale", type=float, default=0.6)
    p.add_argument("--wm-warmup", type=int, default=8,
                   help="WM-only fit passes before the main loop")
    p.add_argument("--pi-delay", type=int, default=3,
                   help="Skip policy BPTT for the first N iters")
    p.add_argument("--resume", type=str, default="",
                   help="Checkpoint path to resume from")
    p.add_argument("--resume-best", action="store_true",
                   help="Resume from <out>/best.pt if present")
    p.add_argument("--bc-weight", type=float, default=0.75,
                   help="Behavior-cloning weight toward reference joint targets")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=4)
    return p


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
