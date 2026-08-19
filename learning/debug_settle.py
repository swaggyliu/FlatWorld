"""Check whether objects tip on their own during spawn settling.

Runs zero-force episodes across many seeds and reports how often any
box exceeds a small tilt, which would mean the spawn/layout itself is
unstable (bad data + unreliable evaluation).
"""

import numpy as np

from learning.configs.default import Config
from learning.env.flatworld_wrapper import PushSceneEnv


def main(n_seeds: int = 30, frames: int = 60):
    cfg = Config()
    env = PushSceneEnv(cfg)
    n_bad = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        obs = env.reset(rng)
        y0 = obs["obj_states"][1:4, 1].copy()
        max_tilt = 0.0
        for _ in range(frames):
            obs = env.step(np.zeros(2, dtype=np.float32))
            tilt = np.abs(obs["obj_states"][1:4, 2]).max()
            max_tilt = max(max_tilt, float(tilt))
        rest = obs["obj_states"][1:4, 1]
        drop = float(np.max(np.abs(y0 - rest)))
        if seed == 0:
            print(f"seed 1000: box spawn y {y0.round(3).tolist()} "
                  f"-> rest y {rest.round(3).tolist()} (drop {drop:.3f}); "
                  f"expected rest = half_h {cfg.scene.box_ext[1]}")
        # boxes are indices 1..3; flat boxes at rest should stay level
        # (0.05 rad tolerates solver jitter only)
        bad = max_tilt > 0.05
        n_bad += bad
        if bad or seed == 0:
            print(f"seed {1000 + seed}: max box tilt {max_tilt:.3f} rad"
                  f"{'  <-- UNSTABLE' if bad else ''}")
    print(f"\nunstable seeds: {n_bad}/{n_seeds}")


if __name__ == "__main__":
    main()
