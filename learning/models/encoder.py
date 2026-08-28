"""StateTactileEncoder: per-frame states + EE tactile -> per-object latents.

Sparse near-contact graph (k-NN + signed-gap). Tactile is injected into
the EE node only. Output is (B, N, L) — one latent per body, shared
weights, so N can change without new parameters.
"""

from __future__ import annotations

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


def pairwise_gap(xy, obj_geom):
    """xy (B,N,2), geom (B,N,2) -> rel (B,N,N,2), dist (B,N,N), gap (B,N,N)."""
    rel = xy.unsqueeze(2) - xy.unsqueeze(1)
    dist = rel.norm(dim=-1)
    rad = obj_geom.abs().amax(dim=-1)
    gap = dist - rad.unsqueeze(2) - rad.unsqueeze(1)
    return rel, dist, gap


def pair_touch(gap, gap_eps: float = 0.015):
    """Boolean (..., N, N) near-contact from signed gap; diagonal False."""
    n = gap.shape[-1]
    eye = torch.eye(n, device=gap.device, dtype=torch.bool)
    while eye.dim() < gap.dim():
        eye = eye.unsqueeze(0)
    return (gap < gap_eps) & ~eye


def knn_adjacency(dist, gap, k_nn: int = 2, gap_cut: float = 0.12):
    """Keep k nearest neighbours and any pair with signed gap below cut.

    ``gap`` and ``gap_cut`` share the units of the xy / geom tensors
    passed in (standardized during train and CEM). 0.12 matches the
    current checkpoints; retrain if you change the normalizer or cut.
    """
    B, N, _ = dist.shape
    k = min(int(k_nn), max(N - 1, 1))
    eye = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
    dist_nn = dist.masked_fill(eye, 1e6)
    knn = dist_nn.topk(k, dim=-1, largest=False).indices
    adj = torch.zeros(B, N, N, dtype=torch.bool, device=dist.device)
    adj.scatter_(2, knn, True)
    adj = adj | (gap < gap_cut)
    return adj & ~eye.expand(B, -1, -1)


def _normal_tangent(rel, dist):
    """rel (...,2), dist (...) -> unit normal / tangent (...,2)."""
    n = rel / dist.clamp(min=1e-6).unsqueeze(-1)
    t = torch.stack([-n[..., 1], n[..., 0]], dim=-1)
    return n, t


class StateTactileEncoder(nn.Module):
    def __init__(self, num_obj_types: int = 3, state_dim: int = 6,
                 contact_dim: int = 7, summary_dim: int = 4,
                 latent_dim: int = 128, obj_embed_dim: int = 64,
                 tactile_embed_dim: int = 64, n_mp: int = 2,
                 k_nn: int = 2, gap_cut: float = 0.12,
                 rich_edges: bool = True):
        super().__init__()
        self.obj_embed_dim = obj_embed_dim
        self.latent_dim = latent_dim
        self.n_mp = n_mp
        self.k_nn = k_nn
        self.gap_cut = gap_cut
        self.rich_edges = bool(rich_edges)

        self.type_emb = nn.Embedding(num_obj_types, 32)
        self.state_mlp = mlp([state_dim, 64, 64])
        self.geom_mlp = mlp([2, 32, 32])
        self.node_proj = mlp([64 + 32 + 32, obj_embed_dim], out_norm=True)

        if self.rich_edges:
            # rel is the 6-d state delta (xy already includes vx,vy,ω).
            # Extra collision scalars: dist, gap, contact, vn, vt → 11.
            # Not the predictor's 10-d pack (xy-rel + rel_v + prev_contact).
            edge_in = state_dim + 3 + 2
            msg_in = obj_embed_dim * 2 + 64
            self.node_upd = nn.GRUCell(obj_embed_dim + 1, obj_embed_dim)
        else:
            edge_in = state_dim + 3
            msg_in = obj_embed_dim * 2 + 64
            self.node_upd = nn.GRUCell(obj_embed_dim, obj_embed_dim)
        self.edge_mlp = mlp([edge_in, 64, 64])
        self.msg_mlp = mlp([msg_in, 64, obj_embed_dim])

        self.contact_mlp = mlp([contact_dim, 64, 64])
        self.tactile_proj = nn.Linear(64, tactile_embed_dim)
        self.summary_mlp = mlp([summary_dim, 32])
        self.ee_inject = mlp([tactile_embed_dim + 32, obj_embed_dim, obj_embed_dim])
        self.node_out = mlp([obj_embed_dim, latent_dim], out_norm=True)

    def _message_pass(self, h, obj_states, obj_geom):
        B, N, D = h.shape
        rel = obj_states.unsqueeze(2) - obj_states.unsqueeze(1)
        _, dist, gap = pairwise_gap(obj_states[..., :2], obj_geom)
        adj = knn_adjacency(dist, gap, self.k_nn, self.gap_cut)
        contact_flag = (gap < 0.0).to(h.dtype).unsqueeze(-1)
        if self.rich_edges:
            # state layout: (x, y, theta, vx, vy, omega) -> linear vel is [3:5]
            vel = obj_states[..., 3:5]
            rel_v = vel.unsqueeze(2) - vel.unsqueeze(1)
            n, t = _normal_tangent(rel[..., :2], dist)
            vn = (rel_v * n).sum(-1, keepdim=True)
            vt = (rel_v * t).sum(-1, keepdim=True)
            e = self.edge_mlp(torch.cat([rel, dist.unsqueeze(-1),
                                         gap.unsqueeze(-1), contact_flag,
                                         vn, vt], dim=-1))
        else:
            e = self.edge_mlp(torch.cat([rel, dist.unsqueeze(-1), gap.unsqueeze(-1),
                                         contact_flag], dim=-1))
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        msg = self.msg_mlp(torch.cat([hi, hj, e], dim=-1))
        msg = msg.masked_fill(~adj.unsqueeze(-1), 0.0)
        if self.rich_edges:
            agg = msg.sum(dim=2)
            deg = adj.sum(dim=2, keepdim=True).to(h.dtype)
            agg = torch.cat([agg, deg / 4.0], dim=-1)
        else:
            deg = adj.sum(dim=2, keepdim=True).clamp(min=1).to(h.dtype)
            agg = msg.sum(dim=2) / deg
        h = self.node_upd(agg.reshape(B * N, -1), h.reshape(B * N, D)).view(B, N, D)
        return h

    def forward(self, obj_types, obj_states, contact_feat, contact_mask,
                tactile_summary, obj_geom=None):
        """Returns z: (B, N, latent_dim)."""
        B, N, _ = obj_states.shape
        if obj_geom is None:
            obj_geom = obj_states.new_zeros(B, N, 2)
        h = self.node_proj(torch.cat([
            self.type_emb(obj_types), self.state_mlp(obj_states),
            self.geom_mlp(obj_geom),
        ], dim=-1))

        tc = self.contact_mlp(contact_feat)
        tc = (tc * contact_mask.unsqueeze(-1)).sum(dim=1)
        tactile = self.tactile_proj(tc)
        summary = self.summary_mlp(tactile_summary)
        h = h.clone()
        h[:, 0] = h[:, 0] + self.ee_inject(torch.cat([tactile, summary], dim=-1))

        for _ in range(self.n_mp):
            h = self._message_pass(h, obj_states, obj_geom)
        return self.node_out(h)
