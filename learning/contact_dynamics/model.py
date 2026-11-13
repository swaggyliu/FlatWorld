"""Small physics-constrained local contact impulse predictor."""

import torch
from torch import nn
from torch.nn import functional as F


class ContactImpulseNet(nn.Module):
    """Predict impulses for an already detected contact.

    Normal impulse is non-negative and tangential impulse is bounded by the
    Coulomb cone |lambda_t| <= mu * lambda_n by construction. Broadphase and
    contact candidate detection remain the responsibility of the simulator.
    """

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.impulse_head = nn.Linear(hidden_dim, 2)
        self.mode_head = nn.Linear(hidden_dim, 2)

    def forward(self, features, friction):
        h = self.encoder(features)
        raw = self.impulse_head(h)
        normal = F.softplus(raw[..., 0])
        tangent = torch.tanh(raw[..., 1]) * friction.clamp_min(0.0) * normal
        return torch.stack((normal, tangent), dim=-1), self.mode_head(h)
