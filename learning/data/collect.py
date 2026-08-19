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


def sample_action(rng: np.random.Generator, cfg: Config, prev: np.ndarray,
                  ee_pos: np.ndarray, obj_pos: np.ndarray,
                  mode: str = "attract", push_cmd: np.ndarray = None) -> np.ndarray:
    """Smooth random force. Three collection modes (see run_rollout):

    - "attract": OU noise + weak attraction toward the nearest object.
      Guarantees rich contact data.
    - "free": symmetric OU noise only (plus soft workspace barriers).
      Teaches the true force -> EE-motion mapping and the "objects stay
      still without contact" gating; the attraction-biased data alone
      makes the model believe the EE drifts right regardless of force.
    - "push": constant commanded force toward/away from the nearest
      object, alternating on a timer (push ... withdraw ...). Teaches
      contact making/breaking and that released objects stop moving.
    """
    c = cfg.collect
    if mode == "push" and push_cmd is not None:
        a = push_cmd + 0.3 * c.ou_sigma * rng.standard_normal(2)
    else:
        a = c.ou_theta * prev + c.ou_sigma * rng.standard_normal(2)
        if mode == "attract":
            d = obj_pos - ee_pos
            dist = float(np.linalg.norm(d))
            u = d / max(dist, 1e-6)
            mag = np.clip(dist * c.attract_gain, 0.0, c.force_max * 0.7)
            a = a + u * mag
        else:
            # soft barriers keep the free-space walk inside the working
            # volume [x 0.15..1.70, y 0.12..0.30] without biasing the
            # force distribution inside it
            k = 12.0
            if ee_pos[0] < 0.15:
                a[0] += k * (0.15 - ee_pos[0])
            elif ee_pos[0] > 1.70:
                a[0] -= k * (ee_pos[0] - 1.70)
            if ee_pos[1] < 0.12:
                a[1] += k * (0.12 - ee_pos[1])
            elif ee_pos[1] > 0.30:
                a[1] -= k * (ee_pos[1] - 0.30)
    return np.clip(a, -c.force_max, c.force_max).astype(np.float32)


def run_rollout(env: PushSceneEnv, rng: np.random.Generator, cfg: Config) -> dict:
    c = cfg.collect
    obs = env.reset(rng)

    states = [obs["obj_states"]]
    feats = [obs["contact_feat"]]
    masks = [obs["contact_mask"]]
    sums = [obs["tactile_summary"]]
    actions = []

    # mode mix: 30% free / 50% attract / 20% push-withdraw
    u = rng.random()
    mode = "free" if u < 0.30 else ("attract" if u < 0.80 else "push")
    push_cmd = None
    phase_left = 0
    pushing = True

    a = np.zeros(2, dtype=np.float32)
    for _ in range(c.episode_len):
        ee_pos = obs["obj_states"][0, :2]
        obj_pos_all = obs["obj_states"][1:, :2]
        nearest = obj_pos_all[np.argmin(
            np.linalg.norm(obj_pos_all - ee_pos, axis=1))]
        if mode == "push":
            if phase_left <= 0:
                # switch phase: push toward the object, then withdraw
                pushing = not pushing
                phase_left = int(rng.integers(20, 50))
                d = nearest - ee_pos
                dist = max(float(np.linalg.norm(d)), 1e-6)
                mag = float(rng.uniform(0.35, 0.95) * c.force_max)
                push_cmd = (d / dist) * (mag if pushing else -mag)
            phase_left -= 1
        a = sample_action(rng, cfg, a, ee_pos, nearest, mode, push_cmd)
        obs = env.step(a)
        actions.append(a)
        states.append(obs["obj_states"])
        feats.append(obs["contact_feat"])
        masks.append(obs["contact_mask"])
        sums.append(obs["tactile_summary"])

    return {
        "obj_types": obs["obj_types"],                    # (N,)
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
