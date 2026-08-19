"""StateTactileDecoder: latent z_t -> reconstructed (obj_states, tactile_summary).

Mandatory component: without the reconstruction loss the encoder can
collapse to a constant latent (dynamics loss alone is trivially zeroed
by a constant predictor output).
"""

import torch
import torch.nn as nn

from .encoder import mlp


class StateTactileDecoder(nn.Module):
    def __init__(self, n_obj: int, state_dim: int = 6, summary_dim: int = 4,
                 latent_dim: int = 128):
        super().__init__()
        self.n_obj = n_obj
        self.state_dim = state_dim
        self.trunk = mlp([latent_dim, 128, 128])
        self.state_head = nn.Linear(128, n_obj * state_dim)
        self.summary_head = nn.Linear(128, summary_dim)

    def forward(self, z):
        """z: (B, latent_dim) -> states (B, N, 6), summary (B, 4) (normalized units)."""
        h = self.trunk(z)
        states = self.state_head(h).view(-1, self.n_obj, self.state_dim)
        summary = self.summary_head(h)
        return states, summary
