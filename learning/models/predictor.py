"""LatentPredictor: latent dynamics z_t + a_t -> z_{t+1}.

A single-layer GRU keeps memory across frames (contact discontinuities
are not perfectly Markovian in latent space). The output head maps the
recurrent state back to the latent space.
"""

import torch
import torch.nn as nn

from .encoder import mlp


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2,
                 hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        self.cell = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.out = mlp([hidden_dim, latent_dim], out_norm=True)

    def init_hidden(self, z, h=None):
        """Seeding trick: initialize the GRU hidden state from z_0."""
        if h is None:
            h = torch.tanh(z)
        return h

    def step(self, z, a, h):
        """One-step transition. z (B,L), a (B,A), h (B,H) -> z' (B,L), h' (B,H)."""
        h = self.cell(torch.cat([z, a], dim=-1), h)
        return self.out(h), h

    def forward(self, z_seq, a_seq, h0=None):
        """Roll out over a sequence.

        z_seq: (B, T, L)  latents z_0..z_{T-1}
        a_seq: (B, T, A)  actions a_0..a_{T-1}
        returns z_pred: (B, T, L)  predictions of z_1..z_T
        """
        B, T, _ = z_seq.shape
        h = self.init_hidden(z_seq[:, 0]) if h0 is None else h0
        preds = []
        z = z_seq[:, 0]
        for t in range(T):
            z, h = self.step(z, a_seq[:, t], h)
            preds.append(z)
        return torch.stack(preds, dim=1)
