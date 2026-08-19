"""Check whether objects tip on their own during spawn settling."""

import numpy as np

from learning.configs.default import Config
from learning.env.flatworld_wrapper import PushSceneEnv


def main():
    cfg = Config()
    env = PushSceneEnv(cfg)
    rng = np.random.default_rng(1007)
    obs = env.reset(rng)
    print("t= 0:", obs["obj_states"][:, :3].round(3).tolist())
    for t in range(1, 21):
        obs = env.step(np.zeros(2, dtype=np.float32))
        if t % 2 == 0:
            print(f"t={t:2d}:", obs["obj_states"][:, :3].round(3).tolist())


if __name__ == "__main__":
    main()
