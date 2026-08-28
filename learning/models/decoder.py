"""Per-object decoder: each z_i maps to that body's state.

Tactile summary is read from the EE node (index 0). Shared heads, so N
is not baked into the weight shapes.
"""

from __future__ import annotations

import torch.nn as nn

from .encoder import mlp


class StateTactileDecoder(nn.Module):
    def __init__(self, n_obj: int = None, state_dim: int = 6, summary_dim: int = 4,
                 latent_dim: int = 128):
        super().__init__()
        self.n_obj = n_obj
        self.state_dim = state_dim
        self.trunk = mlp([latent_dim, 128, 128])
        self.state_head = nn.Linear(128, state_dim)
        self.summary_head = mlp([latent_dim, 64, summary_dim])

    def forward(self, z):
        """z (B, N, L) -> states (B, N, 6), summary (B, 4)."""
        if z.dim() == 2:
            z = z.unsqueeze(1)
        states = self.state_head(self.trunk(z))
        summary = self.summary_head(z[:, 0])
        return states, summary
