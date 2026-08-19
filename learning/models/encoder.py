"""StateTactileEncoder: per-frame (states + tactile set) -> latent z_t.

Architecture:
- object tokens: type embedding + state MLP, mean-pooled over objects
- tactile: Deep Sets over variable-length contacts
  (shared MLP + mask-aware sum pooling, permutation invariant)
- fusion: concat + MLP + LayerNorm -> z_t
"""

import torch
import torch.nn as nn


def mlp(sizes, act=nn.GELU, out_norm=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


class StateTactileEncoder(nn.Module):
    def __init__(self, num_obj_types: int = 3, state_dim: int = 6,
                 contact_dim: int = 7, summary_dim: int = 4,
                 latent_dim: int = 128, obj_embed_dim: int = 64,
                 tactile_embed_dim: int = 64):
        super().__init__()
        self.type_emb = nn.Embedding(num_obj_types, 32)
        self.state_mlp = mlp([state_dim, 64, 64])
        self.obj_proj = nn.Linear(64 + 32, obj_embed_dim)

        # Deep Sets: shared per-contact MLP + masked sum pooling
        self.contact_mlp = mlp([contact_dim, 64, 64])
        self.tactile_proj = nn.Linear(64, tactile_embed_dim)

        self.summary_mlp = mlp([summary_dim, 32])

        self.fusion = mlp([obj_embed_dim + tactile_embed_dim + 32, 128, latent_dim],
                          out_norm=True)

    def forward(self, obj_types, obj_states, contact_feat, contact_mask,
                tactile_summary):
        """
        obj_types:       (B, N) long
        obj_states:      (B, N, 6) float (normalized)
        contact_feat:    (B, K, 7) float (normalized, padding rows zero)
        contact_mask:    (B, K) float, 1 = real contact
        tactile_summary: (B, 4) float (normalized)
        returns z: (B, latent_dim)
        """
        # --- object tokens ---
        B, N, _ = obj_states.shape
        tok = torch.cat([
            self.type_emb(obj_types),              # (B, N, 32)
            self.state_mlp(obj_states),            # (B, N, 64)
        ], dim=-1)                                 # (B, N, 96)
        obj_pool = self.obj_proj(tok).mean(dim=1)  # (B, obj_embed_dim)

        # --- tactile Deep Sets ---
        h = self.contact_mlp(contact_feat)               # (B, K, 64)
        h = (h * contact_mask.unsqueeze(-1)).sum(dim=1)  # masked sum -> (B, 64)
        tactile = self.tactile_proj(h)                   # (B, tactile_embed_dim)

        # --- summary + fusion ---
        s = self.summary_mlp(tactile_summary)            # (B, 32)
        z = self.fusion(torch.cat([obj_pool, tactile, s], dim=-1))
        return z
