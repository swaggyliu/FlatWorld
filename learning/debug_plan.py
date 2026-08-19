"""Temporary planner diagnostics: dump EE / target trajectories + actions."""

import numpy as np

from learning.configs.default import Config
from learning.tasks.push_to_goal import PushToGoalTask, load_model


def main():
    cfg = Config()
    cfg.collect.force_max = 6.0
    model, norm, stride = load_model("learning/checkpoints/best.pt", "cpu")
    task = PushToGoalTask(cfg, model, norm, device="cpu", tol=0.08, budget=300,
                          stride=stride,
                          planner_kwargs=dict(horizon=8, population=96,
                                              iterations=4, seed=7))
    rng = np.random.default_rng(1002)   # a failing box-push episode
    obs = task.env.reset(rng)
    task.target_idx = task._pick_target(obs)
    goal = task._sample_goal(np.random.default_rng(7), obs, task.target_idx)
    print("init EE", obs["obj_states"][0, :2],
          "target", task.target_idx, obs["obj_states"][task.target_idx, :2], "goal", goal)
    task.planner.reset()
    t = 0
    while t < 300:
        cur = task.env._observe()
        z0 = task.planner.encode_obs(cur)
        standoff = task._half_width(task.target_idx) + task.cfg.scene.ee_radius
        ee_floor = task.cfg.scene.ee_radius + 0.005
        if int(task.env.obj_types[task.target_idx]) == 1:
            contact_y = ee_floor
            slip = 0.06
            coast = 0.0
            fcap = 2.9
        else:
            contact_y = max(float(goal[1]), ee_floor)
            slip = 0.0
            coast = 0.4
            fcap = 4.5
        a = task.planner.plan(z0, 0, task.target_idx, goal, standoff, contact_y,
                              slip, coast, fcap, states_now=cur["obj_states"])
        for _ in range(stride):
            obs = task.env.step(a)
            t += 1
        ee = obs["obj_states"][0, :2]
        tg = obs["obj_states"][task.target_idx, :2]
        d = float(np.linalg.norm(tg - goal))
        print(f"t={t:3d} a=({a[0]:+.2f},{a[1]:+.2f}) ee=({ee[0]:.3f},{ee[1]:.3f}) "
              f"tgt=({tg[0]:.3f},{tg[1]:.3f}) d_goal={d:.3f} "
              f"n_contact={obs['tactile_summary'][3]:.0f}")


if __name__ == "__main__":
    main()
