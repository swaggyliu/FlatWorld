"""StateTactileEncoder: per-frame (states + tactile set) -> latent z_t.

Objects are a fully-connected relation graph (GNN): node = type + state,
edge = relative pose/velocity + distance. Two message-passing layers
capture object-object coupling (chain pushes, blockers) that a mean-pool
MLP cannot. Tactile contacts stay a Deep Set over the EE contact list.
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
                 tactile_embed_dim: int = 64, n_mp: int = 2):
        super().__init__()
        self.obj_embed_dim = obj_embed_dim
        self.n_mp = n_mp

        self.type_emb = nn.Embedding(num_obj_types, 32)
        self.state_mlp = mlp([state_dim, 64, 64])
        self.geom_mlp = mlp([2, 32, 32])
        self.node_proj = mlp([64 + 32 + 32, obj_embed_dim], out_norm=True)

        # relative state (6) + Euclidean pos distance (1)
        self.edge_mlp = mlp([state_dim + 1, 64, 64])
        self.msg_mlp = mlp([obj_embed_dim * 2 + 64, 64, obj_embed_dim])
        self.node_upd = nn.GRUCell(obj_embed_dim, obj_embed_dim)

        self.contact_mlp = mlp([contact_dim, 64, 64])
        self.tactile_proj = nn.Linear(64, tactile_embed_dim)
        self.summary_mlp = mlp([summary_dim, 32])

        # mean + max pool of nodes
        self.fusion = mlp(
            [obj_embed_dim * 2 + tactile_embed_dim + 32, 128, latent_dim],
            out_norm=True,
        )

    def _message_pass(self, h, obj_states):
        """One fully-connected MP step. h, obj_states: (B, N, *)."""
        B, N, D = h.shape
        rel = obj_states.unsqueeze(2) - obj_states.unsqueeze(1)   # (B, N, N, 6)
        dist = rel[..., :2].norm(dim=-1, keepdim=True)            # (B, N, N, 1)
        e = self.edge_mlp(torch.cat([rel, dist], dim=-1))         # (B, N, N, 64)

        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        msg = self.msg_mlp(torch.cat([hi, hj, e], dim=-1))        # (B, N, N, D)

        eye = torch.eye(N, device=h.device, dtype=torch.bool)
        msg = msg.masked_fill(eye.view(1, N, N, 1), 0.0)
        agg = msg.sum(dim=2) / float(max(N - 1, 1))               # (B, N, D)

        h_flat = h.reshape(B * N, D)
        h = self.node_upd(agg.reshape(B * N, D), h_flat).view(B, N, D)
        return h

    def forward(self, obj_types, obj_states, contact_feat, contact_mask,
                tactile_summary, obj_geom=None):
        """
        obj_types:       (B, N) long
        obj_states:      (B, N, 6) float (normalized)
        contact_feat:    (B, K, 7) float (normalized, padding rows zero)
        contact_mask:    (B, K) float, 1 = real contact
        tactile_summary: (B, 4) float (normalized)
        obj_geom:        (B, N, 2) half-extents, optional
        returns z: (B, latent_dim)
        """
        B, N, _ = obj_states.shape
        parts = [self.type_emb(obj_types), self.state_mlp(obj_states)]
        if obj_geom is None:
            obj_geom = obj_states.new_zeros(B, N, 2)
        parts.append(self.geom_mlp(obj_geom))
        h = self.node_proj(torch.cat(parts, dim=-1))
        for _ in range(self.n_mp):
            h = self._message_pass(h, obj_states)

        obj_mean = h.mean(dim=1)
        obj_max = h.max(dim=1).values
        obj_pool = torch.cat([obj_mean, obj_max], dim=-1)

        tc = self.contact_mlp(contact_feat)
        tc = (tc * contact_mask.unsqueeze(-1)).sum(dim=1)
        tactile = self.tactile_proj(tc)
        s = self.summary_mlp(tactile_summary)
        return self.fusion(torch.cat([obj_pool, tactile, s], dim=-1))
