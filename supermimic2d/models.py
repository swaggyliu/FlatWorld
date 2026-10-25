"""Differentiable world model + tracking policy + critic (SuperMimic-style)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.SiLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Welford running mean/std for world-model inputs."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        if int(self.count) == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var.clamp_min(1e-4))
            self.count.fill_(batch_count)
            return
        total = int(self.count) + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * int(self.count)
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * int(self.count) * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_((m2 / total).clamp_min(1e-4))
        self.count.fill_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.var.sqrt().clamp_min(1e-3)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.var.sqrt().clamp_min(1e-3) + self.mean


class WorldModel(nn.Module):
    """Deterministic one-step dynamics: ŝ', r̂, d̂ = f(s, a, g)."""

    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.state_norm = RunningNorm(state_dim)
        self.action_norm = RunningNorm(action_dim)
        self.goal_norm = RunningNorm(goal_dim)
        in_dim = state_dim + action_dim + goal_dim
        self.trunk = mlp([in_dim, hidden, hidden, hidden])
        self.delta_head = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)
        # Predict small normalized state deltas
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, state, action, goal):
        sn = self.state_norm.normalize(state)
        an = self.action_norm.normalize(action)
        gn = self.goal_norm.normalize(goal)
        h = self.trunk(torch.cat([sn, an, gn], dim=-1))
        delta_n = self.delta_head(h)
        next_n = sn + delta_n
        next_state = self.state_norm.denormalize(next_n)
        reward = self.reward_head(h).squeeze(-1)
        done_logit = self.done_head(h).squeeze(-1)
        return next_state, reward, done_logit


class Policy(nn.Module):
    """Residual goal-conditioned tracker: a = q_ref + δ·tanh(net(o)).

    ``q_ref`` is the current-frame reference joint vector stored at the head
    of the goal window inside the observation.
    """

    def __init__(self, obs_dim: int, action_dim: int, proprio_dim: int,
                 hidden: int = 256, delta_scale: float = 0.6):
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = int(proprio_dim)
        self.delta_scale = float(delta_scale)
        self.net = mlp([obs_dim, hidden, hidden, action_dim])
        # Start near pure reference tracking
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        q_ref = obs[..., self.proprio_dim: self.proprio_dim + self.action_dim]
        delta = self.delta_scale * torch.tanh(self.net(obs))
        return q_ref + delta


class Critic(nn.Module):
    def __init__(self, state_dim: int, goal_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([state_dim + goal_dim, hidden, hidden, 1])

    def forward(self, state, goal):
        return self.net(torch.cat([state, goal], dim=-1)).squeeze(-1)
