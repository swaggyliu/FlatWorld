"""ICEM (Sample-Efficient CEM) over the FlatWorld solver.

Based on Pinneri et al., "Sample-Efficient Cross-Entropy Method for Real-Time
Planning". Key improvements over vanilla CEM:

1. Quasi-random Sobol sampling — better coverage than pseudo-random at small P
2. Temporally-smooth noise — action sequences are correlated in time, reducing
   jitter and producing more physically plausible pushes
3. Multi-variance groups — population split into high/low std groups for
   explore+exploit balance
4. Hint as soft prior — hint biases the mean but does NOT hardcode any sample,
   so the optimiser can still deviate when the hint is wrong

Each candidate is rolled with ``snapshot`` / ``restore`` and scored with
the task cost's ``step`` / ``terminal``. Physics is serial, so this
planner uses a smaller population than GPU latent CEM.
"""

import numpy as np
from scipy.stats import qmc


def _scalar(c) -> float:
    if hasattr(c, "detach"):
        return float(c.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(c).reshape(-1)[0])


# 1-D smoothing kernel for temporally-correlated action noise
_SMOOTH_KERNEL = np.array([0.10, 0.20, 0.40, 0.20, 0.10], dtype=np.float32)


def _smooth_noise(noise):
    """Apply temporal smoothing along the horizon axis. noise: (P, H, 2)."""
    pad = len(_SMOOTH_KERNEL) // 2
    out = np.empty_like(noise)
    for ch in range(noise.shape[2]):
        for p in range(noise.shape[0]):
            padded = np.pad(noise[p, :, ch], pad, mode="edge")
            out[p, :, ch] = np.convolve(padded, _SMOOTH_KERNEL, mode="valid")
    return out


class SolverCEMPlanner:
    """Receding-horizon ICEM whose forward model is ``PushSceneEnv``."""

    def __init__(self, env, stride: int = 1, horizon: int = 16,
                 population: int = 12, elites: int = 4, iterations: int = 2,
                 force_max: float = 6.0, seed: int = 0, **_ignore):
        self.env = env
        self.stride = max(int(stride), 1)
        self.H = int(horizon)
        self.P = int(population)
        self.E = min(int(elites), self.P)
        self.iters = int(iterations)
        self.fmax = float(force_max)
        self.rng = np.random.default_rng(int(seed))
        span = 2.0 * self.fmax
        self.mean = np.zeros((self.H, 2), dtype=np.float32)
        self.std = np.full((self.H, 2), 0.25 * span, dtype=np.float32)
        # Sobol sampler for quasi-random low-discrepancy sequences
        self._sobol = qmc.Sobol(d=self.H * 2, seed=int(seed))
        self._sobol_count = 0

    def _sobol_normal(self, n):
        """Draw n quasi-random samples and map to standard normal via ppf."""
        # Sobol draws in [0,1); skip the first point (all zeros) on first call
        if self._sobol_count == 0:
            self._sobol.fast_forward(1)
            self._sobol_count = 1
        draws = self._sobol.random(n)
        self._sobol_count += n
        # Avoid exact 0/1 for ppf; clip to [1e-6, 1-1e-6]
        draws = np.clip(draws, 1e-6, 1.0 - 1e-6)
        from scipy.special import ndtri
        return ndtri(draws).astype(np.float32).reshape(n, self.H, 2)

    def reset(self, init=None):
        span = 2.0 * self.fmax
        self.mean = np.zeros((self.H, 2), dtype=np.float32)
        self.std = np.full((self.H, 2), 0.25 * span, dtype=np.float32)
        if init is not None:
            a0 = np.asarray(init, dtype=np.float32).reshape(-1)[:2]
            self.mean[:] = a0
        # Reset Sobol engine for reproducibility
        self._sobol = qmc.Sobol(d=self.H * 2, seed=0)
        self._sobol_count = 0

    def _pose_batch(self, st, cost):
        goal = getattr(cost, "goal_t", None)
        if goal is not None and hasattr(goal, "device"):
            import torch
            return torch.as_tensor(st, device=goal.device, dtype=torch.float32).unsqueeze(0)
        return np.asarray(st, dtype=np.float32)[None]

    def _action_batch(self, a, pose):
        if hasattr(pose, "device"):
            import torch
            return torch.as_tensor(a, device=pose.device, dtype=torch.float32).unsqueeze(0)
        return np.asarray(a, dtype=np.float32)[None]

    def _score_seq(self, seq, cost):
        env = self.env
        prev = None
        total = 0.0
        pose = None
        for t in range(self.H):
            a = seq[t]
            for _s in range(self.stride):
                env.advance_action(a)
            st = env.raw_states()
            pose = self._pose_batch(st, cost)
            a_t = self._action_batch(a, pose)
            total += _scalar(cost.step(pose, a_t, prev))
            prev = pose
        total += _scalar(cost.terminal(pose))
        return total

    def plan_sequence(self, cost):
        """Return best action sequence ``(H, 2)`` in Newtons (or joint targets)."""
        env = self.env
        snap = env.snapshot()
        mean, std = self.mean, self.std
        hint = getattr(cost, "hint_action", None)

        # Hint as soft prior (not hardcoded samples). ICEM_HINT_W overrides
        # the mix weight (0 = pure cost optimisation, no hint prior).
        import os as _os
        hint_w = float(_os.environ.get("ICEM_HINT_W", "0.45"))
        if hint is not None and hint_w > 0.0:
            h = np.asarray(hint, dtype=np.float32).reshape(2)
            mean = ((1.0 - hint_w) * mean + hint_w * h).astype(np.float32)

        low, high = -self.fmax, self.fmax
        best = mean.copy()
        span = 2.0 * self.fmax

        # Split population into two variance groups: half explore, half exploit
        p_hi = max(self.P // 2, 2)       # high-variance (explorers)
        p_lo = self.P - p_hi              # low-variance (exploiters)

        for it in range(self.iters):
            # Adaptive std for this iteration
            decay = 0.85 ** it
            std_hi = (std * 1.5 * decay).clip(min=0.02 * span)
            std_lo = (std * 0.6 * decay).clip(min=0.01 * span)

            # Sobol quasi-random noise, temporally smoothed
            noise_hi = _smooth_noise(self._sobol_normal(p_hi))
            noise_lo = _smooth_noise(self._sobol_normal(p_lo))

            seqs_hi = np.clip(mean + std_hi * noise_hi, low, high).astype(np.float32)
            seqs_lo = np.clip(mean + std_lo * noise_lo, low, high).astype(np.float32)
            seqs = np.concatenate([seqs_hi, seqs_lo], axis=0)

            # Always include the current mean (pure exploitation baseline)
            seqs[-1] = np.clip(mean, low, high).astype(np.float32)

            scores = np.empty(self.P, dtype=np.float64)
            for p in range(self.P):
                env.restore(snap)
                scores[p] = self._score_seq(seqs[p], cost)

            elite_idx = np.argpartition(scores, self.E - 1)[:self.E]
            elite_idx = elite_idx[np.argsort(scores[elite_idx])]
            elite = seqs[elite_idx]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0).clip(min=1e-3)
            best = elite[0].copy()

        env.restore(snap)
        rolled = np.roll(mean, -1, axis=0)
        rolled[-1] = mean[-1]
        self.mean = rolled.astype(np.float32)
        floor = 0.08 * 2.0 * self.fmax
        self.std = np.maximum(std * 0.9, floor).astype(np.float32)
        return best.astype(np.float32)

    def plan(self, cost, **_unused):
        return self.plan_sequence(cost)[0]
