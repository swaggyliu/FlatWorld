"""Actor / critic for Push-T MBRL (latent z + task features)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian π(a | z, feat) with tanh squash into [-fmax, fmax]."""

    def __init__(self, z_dim: int, feat_dim: int = 13, hidden: int = 256,
                 fmax: float = 6.0, log_std_min: float = -3.0,
                 log_std_max: float = 0.0):
        # feat_dim default matches learning.rl.imagine.FEAT_DIM
        super().__init__()
        self.fmax = float(fmax)
        self.feat_dim = int(feat_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = mlp([z_dim + feat_dim, hidden, hidden, hidden])
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Linear(hidden, 2)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.constant_(self.log_std.bias, -1.0)

    def _dist_params(self, z_flat, feat):
        h = self.net(torch.cat([z_flat, feat], dim=-1))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def forward(self, z_flat, feat, deterministic: bool = False):
        mu, log_std = self._dist_params(z_flat, feat)
        pre = mu if deterministic else mu + torch.randn_like(mu) * log_std.exp()
        return torch.tanh(pre) * self.fmax

    def act(self, z_flat, feat, deterministic: bool = False):
        mu, log_std = self._dist_params(z_flat, feat)
        std = log_std.exp()
        if deterministic:
            pre = mu
            log_prob = _gauss_logprob(pre, mu, log_std)
        else:
            pre = mu + torch.randn_like(mu) * std
            log_prob = _gauss_logprob(pre, mu, log_std)
        a = torch.tanh(pre) * self.fmax
        log_prob = log_prob - _tanh_correction(pre)
        ent = (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1)
        return a, log_prob, ent

    def evaluate(self, z_flat, feat, a):
        mu, log_std = self._dist_params(z_flat, feat)
        y = (a / self.fmax).clamp(-0.999, 0.999)
        pre = 0.5 * (torch.log1p(y) - torch.log1p(-y))
        log_prob = _gauss_logprob(pre, mu, log_std) - _tanh_correction(pre)
        ent = (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1)
        return log_prob, ent


def _gauss_logprob(x, mu, log_std):
    var = (2.0 * log_std).exp()
    return (-0.5 * (((x - mu) ** 2) / var + 2.0 * log_std
                    + math.log(2.0 * math.pi))).sum(dim=-1)


def _tanh_correction(pre):
    return (2.0 * (math.log(2.0) - pre - F.softplus(-2.0 * pre))).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, z_dim: int, feat_dim: int = 13, hidden: int = 256):
        super().__init__()
        self.net = mlp([z_dim + feat_dim, hidden, hidden, 1])

    def forward(self, z_flat, feat):
        return self.net(torch.cat([z_flat, feat], dim=-1)).squeeze(-1)
