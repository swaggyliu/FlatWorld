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
from learning.data.sane import rollout_is_physical
from learning.env.flatworld_wrapper import PushSceneEnv


def apply_barriers(a: np.ndarray, cfg: Config, ee_pos: np.ndarray,
                   ee_r: float = None) -> np.ndarray:
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
    y_lo = max(float(c.y_lo), float(ee_r) + 0.005) if ee_r is not None else float(c.y_lo)
    if ee_pos[1] < y_lo:
        a[1] += k * (y_lo - ee_pos[1])
    elif ee_pos[1] > c.y_hi:
        a[1] -= k * (ee_pos[1] - c.y_hi)
    return a


def scaled_fmax(cfg: Config, ee_r: float = None) -> float:
    """Larger EE needs less force; tiny EE needs a bit more. Caps stay in-range."""
    base = float(cfg.collect.force_max)
    if ee_r is None:
        return base
    r0 = float(cfg.scene.ee_radius)
    s = (r0 / max(float(ee_r), 0.04)) ** 0.5
    return float(np.clip(base * s, 0.5 * base, 1.2 * base))


def sample_action(rng: np.random.Generator, cfg: Config, prev: np.ndarray,
                  ee_pos: np.ndarray, obj_pos: np.ndarray,
                  mode: str = "attract", cmd: np.ndarray = None,
                  ee_r: float = None) -> np.ndarray:
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
    if mode in ("push", "free", "release") and cmd is not None:
        a = cmd + 0.2 * c.ou_sigma * rng.standard_normal(2)
    else:
        a = c.ou_theta * prev + c.ou_sigma * rng.standard_normal(2)
        if mode == "attract":
            d = obj_pos - ee_pos
            dist = float(np.linalg.norm(d))
            u = d / max(dist, 1e-6)
            mag = np.clip(dist * c.attract_gain, 0.0, c.attract_cap)
            # vertical deadband: near the ground band the attraction is
            # horizontal-only (do not dig into the floor while approaching)
            dead_y = max(c.y_lo, float(ee_r if ee_r is not None else cfg.scene.ee_radius) + 0.06)
            if ee_pos[1] < dead_y:
                u[1] = 0.0
                u = u / max(float(np.linalg.norm(u)), 1e-6)
            a = a + u * mag
    a = apply_barriers(a, cfg, ee_pos, ee_r)
    fmax = scaled_fmax(cfg, ee_r)
    return np.clip(a, -fmax, fmax).astype(np.float32)


def _front_idx(states: np.ndarray, target_idx: int, geom: np.ndarray = None) -> int:
    """Object nearest the EE on the EE → target x-interval (inclusive)."""
    ee_x = float(states[0, 0])
    tx = float(states[target_idx, 0])
    pad = 0.0
    if geom is not None:
        pad = float(geom[0, 0])
    lo, hi = min(ee_x, tx) - pad, max(ee_x, tx) + pad
    best, best_d = target_idx, abs(tx - ee_x)
    for j in range(1, len(states)):
        x = float(states[j, 0])
        if lo <= x <= hi:
            d = abs(x - ee_x)
            if d < best_d - 1e-4:
                best, best_d = j, d
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

    Sign of Fx follows EE → target so left-side and right-side chain
    contact both appear in the dataset.
    """
    states = obs["obj_states"]
    geom = obs["obj_geom"]
    front = _front_idx(states, target_idx, geom)
    ee = states[0]
    fr = states[front]
    standoff = float(geom[front, 0]) + float(geom[0, 0])
    contact_y = max(float(fr[1]), float(geom[0, 1]) + 0.005)
    ee_x, tx = float(ee[0]), float(states[target_idx, 0])
    pad = float(geom[0, 0])
    lo, hi = min(ee_x, tx) - pad, max(ee_x, tx) + pad
    g = float(cfg.scene.gravity)
    need = 0.0
    for j in range(1, len(states)):
        if lo <= float(states[j, 0]) <= hi:
            need += float(env.obj_mu[j]) * float(env.obj_mass[j]) * g
    # Stay inside the action range the world model is trained on.
    fmax = float(cfg.collect.force_max)
    fcap = float(np.clip(1.2 * need, 0.25 * fmax, fmax))
    env.force_cap_override = None
    sign = 1.0 if tx >= ee_x else -1.0
    gap = sign * (float(fr[0]) - float(ee[0])) - standoff
    approach = min(fcap, 0.85 * fmax)
    fx = sign * (fcap if gap <= 0.25 * standoff else approach)
    fy = float(np.clip(12.0 * (contact_y - float(ee[1])), -fmax, fmax))
    a = np.array([fx, fy], dtype=np.float64)
    a = a + 0.25 * cfg.collect.ou_sigma * rng.standard_normal(2)
    a = apply_barriers(a, cfg, ee[:2], float(geom[0, 0]))
    return _clip_action(a, cfg, env)


def _neighbor_pair(states: np.ndarray, geom: np.ndarray):
    """Closest neighbouring object pair along x (indices, signed gap)."""
    order = 1 + np.argsort(states[1:, 0])
    best = (1, min(2, len(states) - 1), 1.0)
    best_gap = 1e9
    for a, b in zip(order, order[1:]):
        gap = float(states[b, 0] - states[a, 0]
                    - geom[a, 0] - geom[b, 0])
        if gap < best_gap:
            best_gap = gap
            best = (int(a), int(b), gap)
    return best


def contrast_action(obs: dict, rng: np.random.Generator, cfg: Config,
                    mode: str, mem: dict) -> np.ndarray:
    """Demonstration for gap / kiss / around (not used at plan time).

    gap:    push an isolated face so the neighbour should stay put
    kiss:   push into a touching (or nearly touching) pair
    around: approach the far face, then push back
    """
    fmax = float(cfg.collect.force_max)
    states, geom = obs["obj_states"], obs["obj_geom"]
    ee = states[0]
    if "i" not in mem:
        i, j, gap = _neighbor_pair(states, geom)
        mem["i"], mem["j"] = i, j
        mem["phase"] = "approach"
        xi, xj = float(states[i, 0]), float(states[j, 0])
        toward = 1.0 if xj >= xi else -1.0
        if mode == "kiss":
            mem["sign"] = toward
        elif mode == "gap":
            mem["sign"] = -toward
        else:
            mem["sign"] = 1.0 if rng.random() < 0.5 else -1.0
    i, sign = int(mem["i"]), float(mem["sign"])
    standoff = float(geom[i, 0] + geom[0, 0])
    cx = float(states[i, 0]) - sign * standoff
    cy = max(float(states[i, 1]), float(geom[0, 1]) + 0.005)
    if mem["phase"] == "approach":
        if mode == "around" and float(ee[1]) < cy + 0.10 and abs(float(ee[0]) - cx) > 0.07:
            a = np.array([0.15 * np.sign(cx - ee[0]) * fmax, 0.9 * fmax])
        else:
            dx, dy = cx - float(ee[0]), cy - float(ee[1])
            nrm = max(float(np.hypot(dx, dy)), 1e-6)
            a = np.array([dx / nrm, dy / nrm]) * 0.75 * fmax
        if abs(float(ee[0]) - cx) < 0.05 and abs(float(ee[1]) - cy) < 0.07:
            mem["phase"] = "push"
            mem["left"] = int(rng.integers(22, 45))
    else:
        mem["left"] = int(mem.get("left", 0)) - 1
        if mem["left"] <= 0:
            a = np.zeros(2, dtype=np.float64)
        else:
            fy = float(np.clip(10.0 * (cy - float(ee[1])), -0.4 * fmax, 0.4 * fmax))
            a = np.array([sign * 0.8 * fmax, fy], dtype=np.float64)
    a = a + 0.2 * cfg.collect.ou_sigma * rng.standard_normal(2)
    a = apply_barriers(a, cfg, ee[:2], float(geom[0, 0]))
    return np.clip(a, -fmax, fmax).astype(np.float32)


def run_rollout(env: PushSceneEnv, rng: np.random.Generator, cfg: Config) -> dict:
    c = cfg.collect
    obs = env.reset(rng)
    for _ in range(7):
        if env.min_separating_gap() >= -0.002:
            break
        obs = env.reset(rng)

    states = [obs["obj_states"]]
    feats = [obs["contact_feat"]]
    masks = [obs["contact_mask"]]
    sums = [obs["tactile_summary"]]
    pair0, ground0 = env.refresh_solver_contacts()
    pairs = [pair0]
    grounds = [ground0]
    actions = []
    attract_idx = 1 + int(rng.integers(0, obs["obj_states"].shape[0] - 1))
    pile_target = 1 + int(rng.integers(0, obs["obj_states"].shape[0] - 1))

    u = rng.random()
    if getattr(c, "contrast", False):
        if u < 0.25:
            mode = "around"
        elif u < 0.50:
            mode = "kiss"
        elif u < 0.70:
            mode = "gap"
        elif u < 0.90:
            mode = "release"
        else:
            mode = "push"
    else:
        t_free = c.mode_free
        t_attr = t_free + c.mode_attract
        t_push = t_attr + c.mode_push
        t_rel = t_push + getattr(c, "mode_release", 0.0)
        if u < t_free:
            mode = "free"
        elif u < t_attr:
            mode = "attract"
        elif u < t_push:
            mode = "push"
        elif u < t_rel:
            mode = "release"
        else:
            mode = "pile"
    cmd = None
    phase_left = 0
    pushing = True
    contrast_mem = {}

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
        focus = attract_pos if mode in ("attract", "push", "release") else nearest
        if mode in ("gap", "kiss", "around"):
            a = contrast_action(obs, rng, cfg, mode, contrast_mem)
        elif mode == "pile":
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
                elif mode == "release":
                    pushing = not pushing
                    phase_left = int(rng.integers(16, 32))
                    if pushing:
                        d = focus - ee_pos
                        dist = max(float(np.linalg.norm(d)), 1e-6)
                        mag = float(rng.uniform(0.4, 0.95) * c.force_max)
                        cmd = (d / dist) * mag
                    else:
                        cmd = np.zeros(2, dtype=np.float64)
            if mode != "attract":
                phase_left -= 1
            a = sample_action(rng, cfg, a, ee_pos, focus, mode, cmd,
                              ee_r=float(obs["obj_geom"][0, 0]))
        obs = env.step(a)
        actions.append(a)
        states.append(obs["obj_states"])
        feats.append(obs["contact_feat"])
        masks.append(obs["contact_mask"])
        sums.append(obs["tactile_summary"])
        p, g = env.solver_contacts()
        pairs.append(p)
        grounds.append(g)

    return {
        "obj_types": obs["obj_types"],                    # (N,)
        "obj_geom": obs["obj_geom"],                      # (N, 2)
        "obj_states": np.stack(states),                   # (T+1, N, 6)
        "actions": np.stack(actions),                     # (T, 2)
        "contact_feat": np.stack(feats),                  # (T+1, K, 7)
        "contact_mask": np.stack(masks),                  # (T+1, K)
        "tactile_summary": np.stack(sums),                # (T+1, 4)
        "pair_contact": np.stack(pairs),                  # (T+1, N, N) solver
        "ground_contact": np.stack(grounds),              # (T+1, N) solver
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
    parser.add_argument("--contrast", action="store_true",
                        help="collect gap/kiss/around/release contrast mix")
    args = parser.parse_args()

    cfg = Config()
    if args.contrast:
        cfg.collect.contrast = True
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
        for _ in range(5):
            if rollout_is_physical(data["obj_states"], data["obj_geom"]):
                break
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
