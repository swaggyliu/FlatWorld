"""Probe the planner's imagined cost for constant action sequences.

Replicates the CEMPlanner.plan() cost exactly (baseline-corrected EE
rollout + monotone push prior) and prints the per-horizon cost of a few
constant candidate actions, so cost-landscape pathologies are visible.
"""

import numpy as np
import torch

from learning.configs.default import Config
from learning.env.flatworld_wrapper import PushSceneEnv
from learning.tasks.push_to_goal import load_model


def main():
    device = "cpu"
    model, norm, stride = load_model("learning/checkpoints/best.pt", device)
    cfg = Config()
    env = PushSceneEnv(cfg)
    rng = np.random.default_rng(1000)
    obs = env.reset(rng)
    st = norm.to_tensors(device)

    def nrm(key, x):
        return (x - st[key][0]) / st[key][1]

    def dn(key, x):
        return x * st[key][1] + st[key][0]

    target = 4  # leftmost object in this layout (a ball)
    standoff = cfg.scene.ball_radius + cfg.scene.ee_radius
    goal = obs["obj_states"][target, :2] + np.array([0.10, 0.0])
    goal_x = float(goal[0])
    tgt_y = max(float(goal[1]), cfg.scene.ee_radius + 0.005)
    contact_x = float(obs["obj_states"][target, 0]) - standoff

    states = nrm("obj_states", torch.as_tensor(obs["obj_states"], dtype=torch.float32))
    cf = nrm("contact_feat", torch.as_tensor(obs["contact_feat"]))
    cm = torch.as_tensor(obs["contact_mask"])
    ts = nrm("tactile_summary", torch.as_tensor(obs["tactile_summary"]))
    types = torch.as_tensor(obs["obj_types"])
    H = 8
    with torch.no_grad():
        z0 = model.encode(types.unsqueeze(0), states.unsqueeze(0),
                          cf.unsqueeze(0), cm.unsqueeze(0), ts.unsqueeze(0))

        # zero-action baseline (drift reference)
        a0 = nrm("actions", torch.zeros(1, 2))
        zb = z0
        hb = model.predictor.init_hidden(zb)
        base = []
        for _ in range(H):
            zb, hb = model.predictor.step(zb, a0, hb)
            base.append(dn("obj_states", model.decoder(zb)[0])[0])
        base = torch.stack(base)

        now = torch.as_tensor(obs["obj_states"], dtype=torch.float32)
        print(f"true: EE {obs['obj_states'][0, :2].round(decimals=3)} "
              f"tgt {obs['obj_states'][target, :2].round(decimals=3)} "
              f"goal_x {goal_x:.3f} contact_x {contact_x:.3f} tgt_y_eff {tgt_y:.3f}")
        print(f"baseline EE drift over {H} steps: "
              f"{(base[-1][0, :2] - now[0, :2]).numpy().round(decimals=3)}")

        for name, a_raw in [("right+6", (6.0, 0.0)), ("right+3", (3.0, 0.0)),
                            ("right+6 dn", (6.0, -0.5)), ("zero", (0.0, 0.0)),
                            ("left-3", (-3.0, 0.0)), ("left-6", (-6.0, 0.0))]:
            a_n = nrm("actions", torch.tensor(a_raw))
            z = z0
            h = model.predictor.init_hidden(z)
            cost = 0.0
            pushed = float(now[target, 0])
            ee_end = None
            for t in range(H):
                z, h = model.predictor.step(z, a_n.unsqueeze(0), h)
                s = dn("obj_states", model.decoder(z)[0])[0]
                eff = s - base[t] + now
                ee_x, ee_y = float(eff[0, 0]), float(eff[0, 1])
                ee_end = (ee_x, ee_y)
                d_app = max(contact_x - ee_x, 0.0)
                pushed = max(pushed, ee_x + standoff)
                d_goal = abs(pushed - goal_x)
                d_align = abs(ee_y - tgt_y)
                cost += d_goal + 0.5 * d_app + 0.5 * d_align \
                    + 0.002 * (a_raw[0] ** 2 + a_raw[1] ** 2) / 2
            cost += 2.0 * abs(pushed - goal_x) + 0.5 * max(contact_x - ee_end[0], 0.0)
            print(f"{name:10s} -> EE_end=({ee_end[0]:+.3f},{ee_end[1]:+.3f}) "
                  f"pushed={pushed:.3f} cost={cost:.3f} "
                  f"(d_goal={abs(pushed - goal_x):.3f} d_app={max(contact_x - ee_end[0], 0):.3f} "
                  f"d_align={abs(ee_end[1] - tgt_y):.3f})")


if __name__ == "__main__":
    main()
