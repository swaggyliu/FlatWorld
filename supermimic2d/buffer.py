"""Fixed-size replay buffer of simulator transitions."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, goal_dim: int,
                 obs_dim: int):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_goal = np.zeros((capacity, goal_dim), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, goal, obs, next_goal):
        i = self.ptr
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = reward
        self.next_state[i] = next_state
        self.done[i] = float(done)
        self.goal[i] = goal
        self.obs[i] = obs
        self.next_goal[i] = next_goal
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: getattr(self, k)[idx] for k in
                ("state", "action", "reward", "next_state", "done", "goal", "obs", "next_goal")}

    def sample_windows(self, batch_size: int, k: int, rng: np.random.Generator):
        """Sample k-step contiguous windows (for autoregressive WM fit).

        Falls back to single-step batches when the buffer is small.
        """
        if self.size < k + 1:
            b = self.sample_batch(batch_size, rng)
            # Expand to fake windows of length 1
            return {key: val[:, None, ...] if val.ndim >= 1 and key != "done" else val[:, None]
                    for key, val in b.items()}  # unused path; train uses single-step

        # Prefer contiguous indices (simple: random starts)
        max_start = self.size - k
        starts = rng.integers(0, max_start, size=batch_size)
        out = {}
        for key in ("state", "action", "reward", "next_state", "done", "goal", "obs", "next_goal"):
            arr = getattr(self, key)
            # Gather windows; handle ring buffer wrap by rejecting wrap starts
            windows = []
            for s in starts:
                s = int(s)
                # Ensure physical contiguity in ring buffer
                if self.size < self.capacity:
                    windows.append(arr[s:s + k])
                else:
                    # Avoid wrapping windows
                    if s + k <= self.capacity:
                        windows.append(arr[s:s + k])
                    else:
                        s2 = int(rng.integers(0, self.capacity - k))
                        windows.append(arr[s2:s2 + k])
            out[key] = np.stack(windows, axis=0)
        return out
