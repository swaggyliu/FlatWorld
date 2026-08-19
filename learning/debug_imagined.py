"""Sanity check: model open-loop imagination vs real simulator for constant forces."""

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
    rng = np.random.default_rng(3)
    obs = env.reset(rng)
    st = norm.to_tensors(device)
    m, s = st["obj_states"][0], st["obj_states"][1]
    am, as_ = st["actions"][0], st["actions"][1]

    def norm_state(x):
        return (x - m) / s

    def denorm(x):
        return x * s + m

    def norm_action(a):
        return (a - am) / as_

    for name, a_raw in [("right(+3,0)", np.array([3.0, 0.0])),
                        ("up(0,+3)", np.array([0.0, 3.0])),
                        ("zero", np.array([0.0, 0.0]))]:
        # ---- model imagination ----
        with torch.no_grad():
            states = norm_state(torch.as_tensor(obs["obj_states"], dtype=torch.float32))
            cfeat = norm.to_tensors  # noop
            cf = (torch.as_tensor(obs["contact_feat"]) - st["contact_feat"][0]) / st["contact_feat"][1]
            cm = torch.as_tensor(obs["contact_mask"])
            ts = (torch.as_tensor(obs["tactile_summary"]) - st["tactile_summary"][0]) / st["tactile_summary"][1]
            types = torch.as_tensor(obs["obj_types"])
            z = model.encode(types.unsqueeze(0), states.unsqueeze(0),
                             cf.unsqueeze(0), cm.unsqueeze(0), ts.unsqueeze(0))
            h = model.predictor.init_hidden(z)
            a_n = norm_action(torch.as_tensor(a_raw, dtype=torch.float32))
            print(f"\n=== action {name} (raw {a_raw}, normalized {a_n.numpy()}) ===")
            print(f"      t= 0  EE_pred=({obs['obj_states'][0,0]:.3f},{obs['obj_states'][0,1]:.3f})  (true)")
            for t in range(1, 13):
                z, h = model.predictor.step(z, a_n.unsqueeze(0), h)
                ee = denorm(model.decoder(z)[0])[0, 0, :2].numpy()
                if t % 3 == 0:
                    print(f"      t={t:2d}  EE_pred=({ee[0]:.3f},{ee[1]:.3f})")

        # ---- real simulator (hold the force for stride frames per model step) ----
        env2 = PushSceneEnv(cfg)
        rng2 = np.random.default_rng(3)
        obs2 = env2.reset(rng2)
        for t in range(12 * stride):
            obs2 = env2.step(a_raw)
        print(f"      t=12  EE_true=({obs2['obj_states'][0,0]:.3f},{obs2['obj_states'][0,1]:.3f}) "
              f"tgt_true=({obs2['obj_states'][1,0]:.3f},{obs2['obj_states'][1,1]:.3f})")


if __name__ == "__main__":
    main()
