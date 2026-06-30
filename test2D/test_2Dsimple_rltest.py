"""
Quick test to measure actual step time
"""

import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# from learning.robotRL import RobotLocomotionEnv
# import numpy as np


# def test_simple_rl():

#     print("Creating environment...")
#     env = RobotLocomotionEnv(dim=2, max_steps=10, render_mode=None)

#     print("Resetting...")
#     start = time.time()
#     obs = env.reset()
#     print(f"Reset took: {time.time() - start:.3f}s")

#     print("\nRunning 10 steps...")
#     for i in range(10):
#         action = env.action_space.sample().reshape(1, -1)  # Sample a random action and reshape for batch dimension

#         start = time.time()
#         obs, reward, dones, info = env.step(action)
#         step_time = time.time() - start

#         print(f"Step {i+1}: {step_time:.4f}s, reward={reward[0]:.3f}")

#         if np.any(dones):
#             print("Episode ended")
#             break

#     env.close()
#     print("Done!")


# if __name__ == "__main__":
#     test_simple_rl()
