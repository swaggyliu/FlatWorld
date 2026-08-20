"""Phase 1 random data collection: random-force rollouts -> npz.

Timing convention: one rollout stores T+1 state/tactile frames
(s_0..s_T) and T actions (a_0..a_{T-1}); the supervision signal is
(s_t, a_t) -> s_{t+1}.

Usage (from the repo root d:/FlatWorld):
    python -m learning.data.collect --num-rollouts 2 --episode-len 80
"""

import argparse
import os
import time

import numpy as np

from learning.configs.default import Config
from learning.env.flatworld_wrapper import PushSceneEnv


def apply_barriers(a: np.ndarray, cfg: Config, ee_pos: np.ndarray) -> np.ndarray:
    """Soft workspace walls shared by ALL collection modes.

    With the real (wide) box geometry the EE gets blocked by the pile and
    ejected upward by the contact solver; an unbounded collector then spends
    episodes at y > 1 m being yanked back down, which both biases the action
    distribution (mean Fy ~ -2 N) and buries the force -> motion signal.
    The barrier force is part of the stored action, so the model still
    learns the true applied-force-to-motion mapping.
    """
    c = cfg.collect
    k = c.barrier_k
    if ee_pos[0] < c.x_lo:
        a[0] += k * (c.x_lo - ee_pos[0])
    elif ee_pos[0] > c.x_hi:
        a[0] -= k * (ee_pos[0] - c.x_hi)
    if ee_pos[1] < c.y_lo:
        a[1] += k * (c.y_lo - ee_pos[1])
    elif ee_pos[1] > c.y_hi:
        a[1] -= k * (ee_pos[1] - c.y_hi)
    return a


def sample_action(rng: np.random.Generator, cfg: Config, prev: np.ndarray,
                  ee_pos: np.ndarray, obj_pos: np.ndarray,
                  mode: str = "attract", cmd: np.ndarray = None) -> np.ndarray:
    """Smooth random force. Three collection modes (see run_rollout):

    - "free": unbiased force sweep -- a constant random force held for a
      random interval (re-sampled by the caller). Teaches the true
      force -> EE-motion mapping in BOTH axes with no directional bias;
      the attraction-biased data alone makes the model believe force has
      almost no effect on the EE.
    - "attract": OU noise + capped attraction toward the nearest object
      with a vertical deadband near the ground band, so the EE approaches
      at push height instead of being pressed into the ground plane.
    - "push": constant commanded force toward/away from the nearest
      object, alternating on a timer (push ... withdraw ...). Teaches
      contact making/breaking and that released objects stop moving.
    """
    c = cfg.collect
    if mode in ("push", "free") and cmd is not None:
        a = cmd + 0.2 * c.ou_sigma * rng.standard_normal(2)
    else:
        a = c.ou_theta * prev + c.ou_sigma * rng.standard_normal(2)
        if mode == "attract":
            d = obj_pos - ee_pos
            dist = float(np.linalg.norm(d))
            u = d / max(dist, 1e-6)
            mag = np.clip(dist * c.attract_gain, 0.0, c.attract_cap)
            # vertical deadband: near the ground the attraction is
            # horizontal-only (the EE must not dig into the ground plane
            # while approaching an object resting on it)
            if ee_pos[1] < 0.16:
                u[1] = 0.0
                u = u / max(float(np.linalg.norm(u)), 1e-6)
            a = a + u * mag
    a = apply_barriers(a, cfg, ee_pos)
    return np.clip(a, -c.force_max, c.force_max).astype(np.float32)


def _front_idx(states: np.ndarray, target_idx: int) -> int:
    """Nearest object on the EE → target x-interval (inclusive)."""
    ee_x = float(states[0, 0])
    tx = float(states[target_idx, 0])
    lo, hi = min(ee_x, tx), max(ee_x, tx)
    best, best_x = target_idx, tx
    for j in range(1, len(states)):
        x = float(states[j, 0])
        if lo - 0.03 <= x <= hi + 0.03 and x < best_x - 1e-4:
            best, best_x = j, x
    return best


def _clip_action(a, cfg: Config, env: PushSceneEnv) -> np.ndarray:
    fmax = float(cfg.collect.force_max)
    override = getattr(env, "force_cap_override", None)
    if override is not None:
        fmax = max(fmax, float(override))
    return np.clip(a, -fmax, fmax).astype(np.float32)


def pile_action(env: PushSceneEnv, obs: dict, target_idx: int,
                rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Shove the object in front of a (possibly buried) random target.

    Covers the Random-task gap the 3-mode mix almost never visits:
    chain contact, clearing a corridor, mass-aware +Fx above 6 N.
    """
    states = obs["obj_states"]
    geom = obs["obj_geom"]
    front = _front_idx(states, target_idx)
    ee = states[0]
    fr = states[front]
    standoff = float(geom[front, 0]) + float(geom[0, 0])
    contact_y = max(float(fr[1]), float(geom[0, 1]) + 0.005)
    ee_x, tx = float(ee[0]), float(states[target_idx, 0])
    lo, hi = min(ee_x, tx) - 0.04, max(ee_x, tx) + 0.04
    g = float(cfg.scene.gravity)
    need = 0.0
    for j in range(1, len(states)):
        if lo <= float(states[j, 0]) <= hi:
            need += float(env.obj_mu[j]) * float(env.obj_mass[j]) * g
    fcap = float(np.clip(1.2 * need, 4.0, 8.5))
    env.force_cap_override = fcap
    fx = fcap if float(ee[0]) >= float(fr[0]) - standoff - 0.05 else min(fcap, 5.0)
    fy = float(np.clip(12.0 * (contact_y - float(ee[1])), -2.5, 2.2))
    a = np.array([fx, fy], dtype=np.float64)
    a = a + 0.25 * cfg.collect.ou_sigma * rng.standard_normal(2)
    a = apply_barriers(a, cfg, ee[:2])
    return _clip_action(a, cfg, env)


def run_rollout(env: PushSceneEnv, rng: np.random.Generator, cfg: Config) -> dict:
    c = cfg.collect
    obs = env.reset(rng)

    states = [obs["obj_states"]]
    feats = [obs["contact_feat"]]
    masks = [obs["contact_mask"]]
    sums = [obs["tactile_summary"]]
    actions = []
    attract_idx = 1 + int(rng.integers(0, obs["obj_states"].shape[0] - 1))
    pile_target = 1 + int(rng.integers(0, obs["obj_states"].shape[0] - 1))

    # mode mix: 30% free / 25% attract / 15% push / 30% pile-through.
    # pile-through is the Random-task gap (buried target, chain contact).
    u = rng.random()
    if u < 0.30:
        mode = "free"
    elif u < 0.55:
        mode = "attract"
    elif u < 0.70:
        mode = "push"
    else:
        mode = "pile"
    cmd = None
    phase_left = 0
    pushing = True

    a = np.zeros(2, dtype=np.float32)
    for _ in range(c.episode_len):
        env.force_cap_override = None
        ee_pos = obs["obj_states"][0, :2]
        obj_pos_all = obs["obj_states"][1:, :2]
        if rng.random() < 0.15:
            attract_idx = 1 + int(rng.integers(0, obj_pos_all.shape[0]))
        attract_pos = obs["obj_states"][attract_idx, :2]
        nearest = obj_pos_all[np.argmin(
            np.linalg.norm(obj_pos_all - ee_pos, axis=1))]
        # push toward a random object (not always the nearest) so the EE
        # learns to drive into a blocked corridor.
        focus = attract_pos if mode in ("attract", "push") else nearest
        if mode == "pile":
            a = pile_action(env, obs, pile_target, rng, cfg)
        else:
            if phase_left <= 0:
                if mode == "free":
                    phase_left = int(rng.integers(12, 25))
                    ang = rng.uniform(0.0, 2.0 * np.pi)
                    mag = float(rng.uniform(c.sweep_min, c.sweep_max) * c.force_max)
                    cmd = np.array([np.cos(ang), np.sin(ang)]) * mag
                elif mode == "push":
                    pushing = not pushing
                    phase_left = int(rng.integers(20, 50))
                    d = focus - ee_pos
                    dist = max(float(np.linalg.norm(d)), 1e-6)
                    mag = float(rng.uniform(0.35, 0.95) * c.force_max)
                    cmd = (d / dist) * (mag if pushing else -mag)
            if mode != "attract":
                phase_left -= 1
            a = sample_action(rng, cfg, a, ee_pos, focus, mode, cmd)
        obs = env.step(a)
        actions.append(a)
        states.append(obs["obj_states"])
        feats.append(obs["contact_feat"])
        masks.append(obs["contact_mask"])
        sums.append(obs["tactile_summary"])

    return {
        "obj_types": obs["obj_types"],                    # (N,)
        "obj_geom": obs["obj_geom"],                      # (N, 2)
        "obj_states": np.stack(states),                   # (T+1, N, 6)
        "actions": np.stack(actions),                     # (T, 2)
        "contact_feat": np.stack(feats),                  # (T+1, K, 7)
        "contact_mask": np.stack(masks),                  # (T+1, K)
        "tactile_summary": np.stack(sums),                # (T+1, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="StateLeWM Phase 1 data collection")
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--episode-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force-max", type=float, default=None)
    parser.add_argument("--out", type=str, default=None, help="output directory")
    parser.add_argument("--start-index", type=int, default=0,
                        help="first rollout index (resume after interruption)")
    args = parser.parse_args()

    cfg = Config()
    if args.num_rollouts is not None:
        cfg.collect.num_rollouts = args.num_rollouts
    if args.episode_len is not None:
        cfg.collect.episode_len = args.episode_len
    if args.seed is not None:
        cfg.collect.seed = args.seed
    if args.force_max is not None:
        cfg.collect.force_max = args.force_max
    out_dir = args.out or cfg.collect.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(cfg.collect.seed)
    env = PushSceneEnv(cfg)

    t_start = time.time()
    tot_frames = 0
    tot_contact_frames = 0
    for i in range(args.start_index, cfg.collect.num_rollouts):
        data = run_rollout(env, rng, cfg)
        path = os.path.join(out_dir, f"rollout_{i:04d}.npz")
        np.savez_compressed(
            path,
            **data,
            frame_dt=np.float32(cfg.scene.frame_dt),
            config_json=np.str_(cfg.dump()),
        )
        contact_frames = int((data["contact_mask"].sum(axis=1) > 0).sum())
        tot_frames += data["contact_mask"].shape[0]
        tot_contact_frames += contact_frames
        if i < 3 or (i + 1) % 10 == 0:
            print(
                f"[{i + 1}/{cfg.collect.num_rollouts}] {os.path.basename(path)} "
                f"states{data['obj_states'].shape} contact frames {contact_frames}/{data['contact_mask'].shape[0]}"
            )

    dt = time.time() - t_start
    print("-" * 60)
    print(f"Done: {cfg.collect.num_rollouts} rollouts in {dt:.1f}s "
          f"({dt / cfg.collect.num_rollouts:.2f}s per rollout)")
    print(f"Frames with contacts: {tot_contact_frames}/{tot_frames} "
          f"({100.0 * tot_contact_frames / max(tot_frames, 1):.1f}%)")
    print(f"Output directory: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
