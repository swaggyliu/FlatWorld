"""Planner diagnostics: dump EE / target trajectories + CEM actions."""

import numpy as np

from learning.configs.default import Config
from learning.tasks.push_to_goal import PushToGoalTask, load_model


def main(seed: int = 1002):
    cfg = Config()
    model, norm, stride = load_model("learning/checkpoints_pair19_ens/ens_0.pt", "cpu")
    task = PushToGoalTask(cfg, model, norm, device="cpu",
                          stride=stride,
                          planner_kwargs=dict(horizon=8, population=96,
                                              iterations=4, seed=1000))
    rng = np.random.default_rng(seed)
    obs = task.env.reset(rng)
    task.target_idx = task._pick_target(obs)
    goal = task._sample_goal(rng, obs, task.target_idx)
    print("init EE", obs["obj_states"][0, :2],
          "target", task.target_idx, obs["obj_states"][task.target_idx, :2], "goal", goal)
    task.planner.reset()
    t = 0
    while t < task.budget:
        cur = task._observe()
        tgt_pos = cur["obj_states"][task.target_idx, :2]
        tgt_vel = float(np.linalg.norm(cur["obj_states"][task.target_idx, 3:5]))
        d = float(np.linalg.norm(tgt_pos - goal))
        if task._succeeded(cur["obj_states"], goal):
            print(f"t={t:3d} SUCCESS d={d:.3f} vel={tgt_vel:.2f}")
            break
        a = task.plan_action(cur, goal)
        for _ in range(stride):
            obs = task.env.step(a)
            t += 1
            if t >= task.budget:
                break
        ee = obs["obj_states"][0, :2]
        tg = obs["obj_states"][task.target_idx, :2]
        d = float(np.linalg.norm(tg - goal))
        print(f"t={t:3d} cem    tgt_vel={tgt_vel:.2f} "
              f"ee=({ee[0]:.3f},{ee[1]:.3f}) "
              f"tgt=({tg[0]:.3f},{tg[1]:.3f}) d_goal={d:.3f} "
              f"n_contact={obs['tactile_summary'][3]:.0f}")


if __name__ == "__main__":
    main()
