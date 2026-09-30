"""Per-object Interaction-Network dynamics.

z_i' = f(z_i, sum_{j in N(i)} g(z_i, z_j, e_ij), a_i)
Default: the 2-vector action lands on node 0 (force-controlled EE).
Joint scenes pass ``action_nodes`` so channel k hits body action_nodes[k]
(reacher: θ1 → link1, θ2 → link2).
A GRU per node (shared weights) keeps short contact memory.
Neighbourhoods are rebuilt each step from pose_head (x, y, θ) + geom.
Typed circle/OBB signed gap (denormed θ) drives contact edges. `pose_head`
reads (x, y, θ).

Edges carry relative xy, distance, signed gap, contact flag, relative
velocity ``rel_v = (Δvx, Δvy)`` from vel_head, previous-step contact
persistence, and a relation-type embedding (contact / revolute / weld /
prismatic / none). Joint pairs are always in the adjacency.

Messages sum (impulses add through a contact chain); neighbour degree is
appended as a node feature. Ground contact is concatenated into the GRU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import knn_adjacency, mlp, pairwise_gap
from .relations import (
    NUM_REL, REL_EMB_DIM, REL_NONE, as_rel_tensor, is_joint, overlay_contact,
)


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2,
                 hidden_dim: int = None, k_nn: int = 2, gap_cut: float = 0.12):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.k_nn = k_nn
        self.gap_cut = gap_cut
        self.pose_head = nn.Linear(latent_dim, 3)  # x, y, theta (z-scored)
        self.vel_head = nn.Linear(latent_dim, 2)
        # xy-rel, dist, gap, contact, rel_v, prev_c, relation-type emb
        edge_in = 2 + 1 + 1 + 1 + 2 + 1 + REL_EMB_DIM
        # z + (agg, deg) + ground + action
        cell_in = latent_dim + latent_dim + 1 + 1 + action_dim
        self.rel_emb = nn.Embedding(NUM_REL, REL_EMB_DIM)
        self.edge_mlp = mlp([edge_in, 64, 64])
        self.msg_mlp = mlp([latent_dim * 2 + 64, 64, latent_dim])
        self.cell = nn.GRUCell(cell_in, hidden_dim)
        self.out = mlp([hidden_dim, latent_dim], out_norm=True)

    def _gap_kwargs(self, gap_stats):
        gs = gap_stats or {}
        return dict(
            st_mean=gs.get("st_mean"), st_std=gs.get("st_std"),
            geom_mean=gs.get("geom_mean"), geom_std=gs.get("geom_std"),
        )

    def _expand_types(self, obj_types, B, device):
        if obj_types is None:
            return None
        t = obj_types if torch.is_tensor(obj_types) else torch.as_tensor(
            obj_types, device=device, dtype=torch.long)
        t = t.to(device=device, dtype=torch.long)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, -1)
        return t

    def _pose_xy_th(self, z, pose=None):
        """xy (B,N,2) and θ (B,N) for the graph; teacher-forced pose may be 2 or 3-d."""
        if pose is None:
            pose = self.pose_head(z)
        elif pose.dim() == 2:
            pose = pose.unsqueeze(0).expand(z.shape[0], -1, -1)
        elif pose.shape[0] == 1 and z.shape[0] > 1:
            pose = pose.expand(z.shape[0], -1, -1)
        xy = pose[..., :2]
        th = pose[..., 2] if pose.shape[-1] >= 3 else self.pose_head(z)[..., 2]
        return xy, th

    def xy_head(self, z):
        """xy channels of pose_head (graph / tactile-gate)."""
        return self.pose_head(z)[..., :2]

    def init_hidden(self, z, h=None):
        if h is None:
            h = torch.tanh(z)
        return h

    def _contact(self, xy, geom, theta=None, obj_types=None, gap_stats=None):
        """(B,N,2) xy -> (B,N,N) float contact flag (gap < 0)."""
        _, _, gap = pairwise_gap(
            xy, geom, theta=theta, obj_types=obj_types,
            **self._gap_kwargs(gap_stats))
        return (gap < 0.0).to(xy.dtype)

    def _aggregate(self, z, geom, xy=None, prev_contact=None, pair_contact=None,
                   rel_type=None, obj_types=None, gap_stats=None):
        B, N, L = z.shape
        xy_now, th = self._pose_xy_th(z, xy)
        _, dist, gap = pairwise_gap(
            xy_now, geom, theta=th, obj_types=obj_types,
            **self._gap_kwargs(gap_stats))
        rel = xy_now.unsqueeze(2) - xy_now.unsqueeze(1)
        rel_ids = as_rel_tensor(rel_type, B, N, z.device)
        joint = is_joint(rel_ids)
        adj = knn_adjacency(dist, gap, self.k_nn, self.gap_cut) | joint
        contact_bool = gap < 0.0
        if pair_contact is not None:
            pc_now = pair_contact
            if pc_now.dim() == 2:
                pc_now = pc_now.unsqueeze(0)
            if pc_now.dim() == 4:
                pc_now = pc_now.squeeze(-1)
            contact_bool = pc_now.to(dtype=torch.bool)
        contact_bool = contact_bool & ~joint
        contact_flag = contact_bool.to(z.dtype).unsqueeze(-1)
        vel = self.vel_head(z)                          # (B,N,2)
        rel_v = vel.unsqueeze(2) - vel.unsqueeze(1)     # (B,N,N,2)
        if prev_contact is None:
            prev_contact = self._contact(
                xy_now, geom, theta=th, obj_types=obj_types, gap_stats=gap_stats)
        pc = prev_contact.to(z.dtype)
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        if pc.dim() == 3:
            pc = pc.unsqueeze(-1)                # (B,N,N,1)
        pc = pc * (~joint).unsqueeze(-1).to(pc.dtype)
        edge_ids = overlay_contact(rel_ids, contact_bool).clamp(0, REL_NONE)
        e = self.edge_mlp(torch.cat(
            [rel, dist.unsqueeze(-1), gap.unsqueeze(-1),
             contact_flag, rel_v, pc, self.rel_emb(edge_ids)], dim=-1))
        hi = z.unsqueeze(2).expand(B, N, N, L)
        hj = z.unsqueeze(1).expand(B, N, N, L)
        msg = self.msg_mlp(torch.cat([hi, hj, e], dim=-1))
        msg = msg.masked_fill(~adj.unsqueeze(-1), 0.0)
        agg = msg.sum(dim=2)
        deg = adj.sum(dim=2, keepdim=True).to(z.dtype)
        return torch.cat([agg, deg / 4.0], dim=-1)

    def _scatter_action(self, a, B, N, action_nodes=None):
        """(B, A) -> (B, N, A). Default: full vector on node 0.

        ``action_nodes[k]`` is the body index that receives channel k
        (other channels on that body stay 0).
        """
        a = a.reshape(B, -1)
        a_full = a.new_zeros(B, N, self.action_dim)
        if action_nodes is None:
            a_full[:, 0, :a.shape[-1]] = a[:, :self.action_dim]
            return a_full
        for k, idx in enumerate(action_nodes):
            if k >= a.shape[-1]:
                break
            a_full[:, int(idx), k] = a[:, k]
        return a_full

    def step(self, z, a, h, geom=None, xy=None, prev_contact=None,
             pair_contact=None, ground=None, rel_type=None, action_nodes=None,
             obj_types=None, gap_stats=None):
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
        types = self._expand_types(obj_types, B, z.device)
        xy_now, th = self._pose_xy_th(z, xy)
        agg = self._aggregate(z, geom, xy=xy, prev_contact=prev_contact,
                              pair_contact=pair_contact, rel_type=rel_type,
                              obj_types=types, gap_stats=gap_stats)
        contact_now = self._contact(
            xy_now, geom, theta=th, obj_types=types, gap_stats=gap_stats)
        a_full = self._scatter_action(a, B, N, action_nodes=action_nodes)
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
