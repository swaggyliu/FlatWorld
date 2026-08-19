"""Definitive physics probe: constant force on the EE in free space.

Applies +-Fx for 60 frames from rest and prints the EE x trajectory;
compares with the ideal F=ma prediction (mass from the config).
"""

import numpy as np

from learning.configs.default import Config
from learning.env.flatworld_wrapper import PushSceneEnv


def main():
    cfg = Config()
    env = PushSceneEnv(cfg)
    rng = np.random.default_rng(7)
    env.reset(rng)  # layout irrelevant: EE spawns left of the pile
    ee_mass = float(env.base_mass[0])

    for fx in (-6.0, -3.0, 3.0, 6.0):
        env.reset(rng)
        xs, vs = [], []
        for t in range(60):
            o = env.step(np.array([fx, 0.0], dtype=np.float32))
            xs.append(float(o["obj_states"][0, 0]))
            vs.append(float(o["obj_states"][0, 3]))
        dt = cfg.scene.frame_dt
        t_end = 60 * dt
        ideal_v = fx / ee_mass * t_end
        ideal_x = 0.5 * fx / ee_mass * t_end ** 2
        print(f"Fx={fx:+.0f}N  EE x: 0 -> {xs[-1]:+.4f} (ideal {ideal_x:+.4f})  "
              f"v_end {vs[-1]:+.3f} (ideal {ideal_v:+.3f})")
        print(f"          x every 15f: "
              f"{[round(xs[i], 4) for i in (14, 29, 44, 59)]}")


if __name__ == "__main__":
    main()
