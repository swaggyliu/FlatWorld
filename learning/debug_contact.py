"""Probe the world model's IN-CONTACT imagination vs. ground truth.

Drives the real EE into contact with the leftmost object, encodes that
state, then compares (a) the model's imagined target trajectory under
constant forces with (b) the real simulator's trajectory under the same
forces (re-simulated from the contact state for each candidate).
"""

import numpy as np
import torch

from learning.configs.default import Config
from learning.env.flatworld_wrapper import PushSceneEnv
from learning.tasks.push_to_goal import load_model


def drive_to_contact(env, target_idx, fmax=6.0, max_frames=240):
    """PD-drive the EE to the left surface of the target object."""
    for _ in range(max_frames):
        obs = env._observe()
        ee = obs["obj_states"][0, :2]
        tgt = obs["obj_states"][target_idx, :2]
        contact_x = tgt[0] - (0.18 + 0.02)   # standoff + approach margin
        fx = 8.0 * (contact_x - ee[0]) - 2.0 * obs["obj_states"][0, 3]
        fy = 6.0 * (tgt[1] - ee[1]) - 1.5 * obs["obj_states"][0, 4]
        a = np.array([np.clip(fx, -fmax, fmax), np.clip(fy, -0.8, 0.8)],
                     dtype=np.float32)
        env.step(a)
        if obs["contact_mask"].sum() > 0 and ee[0] > contact_x - 0.03:
            break
    return env._observe()


def main():
    device = "cpu"
    model, norm, stride = load_model("learning/checkpoints/best.pt", device)
    cfg = Config()
    env = PushSceneEnv(cfg)
    rng = np.random.default_rng(1000)
    obs = env.reset(rng)
    target = 1 + int(np.argmin(obs["obj_states"][1:, 0]))   # leftmost object
    obs = drive_to_contact(env, target)
    st = norm.to_tensors(device)

    def nrm(key, x):
        return (x - st[key][0]) / st[key][1]

    def dn(key, x):
        return x * st[key][1] + st[key][0]

    n_contacts = int(obs["contact_mask"].sum())
    print(f"target {target} type {obs['obj_types'][target]} "
          f"at {obs['obj_states'][target, :2].round(decimals=3)}, "
          f"EE at {obs['obj_states'][0, :2].round(decimals=3)}, "
          f"contacts {n_contacts}")

    H = 8  # model steps = H * stride frames

    # ---------- imagined rollouts from the contact state ----------
    states = nrm("obj_states", torch.as_tensor(obs["obj_states"], dtype=torch.float32))
    cf = nrm("contact_feat", torch.as_tensor(obs["contact_feat"]))
    cm = torch.as_tensor(obs["contact_mask"])
    ts = nrm("tactile_summary", torch.as_tensor(obs["tactile_summary"]))
    types = torch.as_tensor(obs["obj_types"])
    with torch.no_grad():
        z0 = model.encode(types.unsqueeze(0), states.unsqueeze(0),
                          cf.unsqueeze(0), cm.unsqueeze(0), ts.unsqueeze(0))
        # zero-action baseline
        a0 = nrm("actions", torch.zeros(1, 2))
        zb = z0
        hb = model.predictor.init_hidden(zb)
        base = []
        for _ in range(H):
            zb, hb = model.predictor.step(zb, a0, hb)
            base.append(dn("obj_states", model.decoder(zb)[0])[0])
        base = torch.stack(base)
        now = torch.as_tensor(obs["obj_states"], dtype=torch.float32)
        print("imagined (baseline-corrected) target dx after "
              f"{H} model steps:")
        for name, a_raw in [("right+6", (6.0, 0.0)), ("right+2", (2.0, 0.0)),
                            ("zero", (0.0, 0.0)), ("left-2", (-2.0, 0.0)),
                            ("left-6", (-6.0, 0.0))]:
            a_n = nrm("actions", torch.tensor(a_raw))
            z = z0
            h = model.predictor.init_hidden(z)
            for t in range(H):
                z, h = model.predictor.step(z, a_n.unsqueeze(0), h)
            s = dn("obj_states", model.decoder(z)[0])[0]
            eff = s - base[-1] + now
            ee = eff[0, :2].numpy()
            tg = eff[target, :2].numpy()
            print(f"  {name:8s} EE=({ee[0]:+.3f},{ee[1]:+.3f}) "
                  f"tgt=({tg[0]:+.3f},{tg[1]:+.3f})")

    # ---------- ground truth: re-simulate from the contact state ----------
    # snapshot the state by simply continuing the sim with each action;
    # approximate comparison (state not exactly restored between candidates)
    print("real target dx after applying the same constant force "
          f"({H * stride} frames):")
    for name, a_raw in [("right+6", (6.0, 0.0)), ("right+2", (2.0, 0.0)),
                        ("zero", (0.0, 0.0)), ("left-2", (-2.0, 0.0))]:
        # re-drive to a fresh contact state for a fair comparison
        env2 = PushSceneEnv(cfg)
        obs0 = env2.reset(np.random.default_rng(1000))
        obs_c = drive_to_contact(env2, target)
        start = obs_c["obj_states"][target, :2].copy()
        start_ee = obs_c["obj_states"][0, :2].copy()
        a = np.asarray(a_raw, dtype=np.float32)
        for _ in range(H * stride):
            o = env2.step(a)
        tg = o["obj_states"][target, :2]
        ee = o["obj_states"][0, :2]
        print(f"  {name:8s} EE=({ee[0] - start_ee[0]:+.3f},{ee[1] - start_ee[1]:+.3f}) "
              f"tgt=({tg[0] - start[0]:+.3f},{tg[1] - start[1]:+.3f})")


if __name__ == "__main__":
    main()
