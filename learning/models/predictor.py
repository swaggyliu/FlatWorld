"""Per-object Interaction-Network dynamics.

z_i' = f(z_i, sum_{j in N(i)} g(z_i, z_j, e_ij), a * 1_{i=EE})
A GRU per node (shared weights) keeps short contact memory.
Neighbourhoods are rebuilt each step from a cheap xy readout + geom.

Edges carry relative xy, distance, signed gap, contact flag, relative
velocity ``rel_v = (Δvx, Δvy)`` from vel_head, and the previous-step
contact flag (persistence).

Messages sum (impulses add through a contact chain); neighbour degree is
appended as a node feature. Ground contact is concatenated into the GRU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import knn_adjacency, mlp, pairwise_gap


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2,
                 hidden_dim: int = None, k_nn: int = 2, gap_cut: float = 0.12):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.k_nn = k_nn
        self.gap_cut = gap_cut
        self.xy_head = nn.Linear(latent_dim, 2)
        self.vel_head = nn.Linear(latent_dim, 2)
        # xy-rel, dist, gap, contact, rel_v, prev_c
        edge_in = 2 + 1 + 1 + 1 + 2 + 1
        # z + (agg, deg) + ground + action
        cell_in = latent_dim + latent_dim + 1 + 1 + action_dim
        self.edge_mlp = mlp([edge_in, 64, 64])
        self.msg_mlp = mlp([latent_dim * 2 + 64, 64, latent_dim])
        self.cell = nn.GRUCell(cell_in, hidden_dim)
        self.out = mlp([hidden_dim, latent_dim], out_norm=True)

    def init_hidden(self, z, h=None):
        if h is None:
            h = torch.tanh(z)
        return h

    def _contact(self, xy, geom):
        """(B,N,2) xy -> (B,N,N) float contact flag (gap < 0)."""
        _, _, gap = pairwise_gap(xy, geom)
        return (gap < 0.0).to(xy.dtype)

    def _aggregate(self, z, geom, xy=None, prev_contact=None, pair_contact=None):
        B, N, L = z.shape
        if xy is None:
            xy = self.xy_head(z)
        elif xy.dim() == 2:
            xy = xy.unsqueeze(0).expand(B, -1, -1)
        elif xy.shape[0] == 1 and B > 1:
            xy = xy.expand(B, -1, -1)
        rel, dist, gap = pairwise_gap(xy, geom)
        adj = knn_adjacency(dist, gap, self.k_nn, self.gap_cut)
        contact_flag = (gap < 0.0).to(z.dtype).unsqueeze(-1)
        if pair_contact is not None:
            pc_now = pair_contact.to(z.dtype)
            if pc_now.dim() == 2:
                pc_now = pc_now.unsqueeze(0)
            if pc_now.dim() == 3:
                pc_now = pc_now.unsqueeze(-1)
            contact_flag = pc_now
        vel = self.vel_head(z)                          # (B,N,2)
        rel_v = vel.unsqueeze(2) - vel.unsqueeze(1)     # (B,N,N,2)
        if prev_contact is None:
            prev_contact = self._contact(xy, geom)
        pc = prev_contact.to(z.dtype)
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        pc = pc.unsqueeze(-1)                # (B,N,N,1)
        e = self.edge_mlp(torch.cat(
            [rel, dist.unsqueeze(-1), gap.unsqueeze(-1),
             contact_flag, rel_v, pc], dim=-1))
        hi = z.unsqueeze(2).expand(B, N, N, L)
        hj = z.unsqueeze(1).expand(B, N, N, L)
        msg = self.msg_mlp(torch.cat([hi, hj, e], dim=-1))
        msg = msg.masked_fill(~adj.unsqueeze(-1), 0.0)
        agg = msg.sum(dim=2)
        deg = adj.sum(dim=2, keepdim=True).to(z.dtype)
        return torch.cat([agg, deg / 4.0], dim=-1)

    def step(self, z, a, h, geom=None, xy=None, prev_contact=None,
             pair_contact=None, ground=None):
        """z (B,N,L), a (B,A), h (B,N,H) -> z', h', contact_now (B,N,N).

        ``ground`` is a per-node floor-contact flag (B,N) or (B,N,1). CEM
        and scheduled sampling feed ``sigmoid(ground_head(z))``; teacher
        forcing feeds solver labels.
        """
        if z.dim() == 2:
            z = z.unsqueeze(1)
            h = h.unsqueeze(1)
        B, N, L = z.shape
        if geom is None:
            geom = z.new_ones(B, N, 2) * 0.1
        elif geom.dim() == 2:
            geom = geom.unsqueeze(0).expand(B, -1, -1)
        elif geom.shape[0] == 1 and B > 1:
            geom = geom.expand(B, -1, -1)
        xy_now = xy if xy is not None else self.xy_head(z)
        if xy_now.dim() == 2:
            xy_now = xy_now.unsqueeze(0).expand(B, -1, -1)
        elif xy_now.shape[0] == 1 and B > 1:
            xy_now = xy_now.expand(B, -1, -1)
        agg = self._aggregate(z, geom, xy=xy_now, prev_contact=prev_contact,
                              pair_contact=pair_contact)
        contact_now = self._contact(xy_now, geom)
        a_full = z.new_zeros(B, N, self.action_dim)
        a_full[:, 0] = a
        if ground is None:
            g = z.new_zeros(B, N, 1)
        else:
            g = ground.to(dtype=z.dtype)
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.dim() == 2:
                g = g.unsqueeze(-1)
            if g.shape[0] == 1 and B > 1:
                g = g.expand(B, -1, -1)
        inp = torch.cat([z, agg, g, a_full], dim=-1).reshape(B * N, -1)
        h = self.cell(inp, h.reshape(B * N, -1)).view(B, N, -1)
        return self.out(h), h, contact_now

    def forward(self, z_seq, a_seq, h0=None, geom=None, xy_seq=None):
        """z_seq (B,T,N,L), a_seq (B,T,A) -> z_pred (B,T,N,L)."""
        B, T, N, _ = z_seq.shape
        h = self.init_hidden(z_seq[:, 0]) if h0 is None else h0
        z = z_seq[:, 0]
        preds = []
        prev_c = None
        for t in range(T):
            xy = None if xy_seq is None else xy_seq[:, t]
            z, h, prev_c = self.step(z, a_seq[:, t], h, geom=geom, xy=xy,
                                     prev_contact=prev_c)
            preds.append(z)
        return torch.stack(preds, dim=1)
