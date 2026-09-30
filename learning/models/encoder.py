"""StateTactileEncoder: per-frame states; EE tactile is an EE-node residual.

Sparse near-contact graph (k-NN + typed signed-gap). Circle–circle,
circle–OBB, and OBB–OBB gaps use denormed (x, y, θ, geom). Joint pairs
(revolute / weld / prismatic) are always connected and carry a relation-type
embedding so contact physics is not mixed with kinematic constraints.
Tactile is added only to the EE node and is zeroed for a state-only
encode (CEM default, or train-time dropout). Output is (B, N, L).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .relations import (
    NUM_REL, REL_EMB_DIM, REL_NONE, as_rel_tensor, is_joint, overlay_contact,
)

# Matches learning.env.flatworld_wrapper.OBJ_TYPE_BOX (avoid env import here).
OBJ_TYPE_BOX = 1


def mlp(sizes, act=nn.GELU, out_norm=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


def _circle_obb_gap(pc, r, pb, hw, hh, th):
    """Signed circle–OBB gap. Negative = penetration. All (...,) / (..., 2)."""
    d = pc - pb
    c, s = th.cos(), th.sin()
    lx = c * d[..., 0] + s * d[..., 1]
    ly = -s * d[..., 0] + c * d[..., 1]
    qx = lx.clamp(-hw, hw)
    qy = ly.clamp(-hh, hh)
    # hypot(0,0) backward is NaN; inside the OBB both residuals are 0.
    dist = ((lx - qx).square() + (ly - qy).square() + 1e-12).sqrt()
    inside = (lx.abs() <= hw) & (ly.abs() <= hh)
    pen = torch.minimum(hw - lx.abs(), hh - ly.abs())
    return torch.where(inside, -pen - r, dist - r)


def _obb_obb_gap(p1, hw1, hh1, th1, p2, hw2, hh2, th2):
    """SAT signed gap along the four OBB axes. Negative = overlap."""

    def ext(hw, hh, th, ux, uy):
        c, s = th.cos(), th.sin()
        return (ux * c + uy * s).abs() * hw + (-ux * s + uy * c).abs() * hh

    c1, s1 = th1.cos(), th1.sin()
    c2, s2 = th2.cos(), th2.sin()
    overlaps = []
    for ux, uy in ((c1, s1), (-s1, c1), (c2, s2), (-s2, c2)):
        e1 = ext(hw1, hh1, th1, ux, uy)
        e2 = ext(hw2, hh2, th2, ux, uy)
        a = p1[..., 0] * ux + p1[..., 1] * uy
        b = p2[..., 0] * ux + p2[..., 1] * uy
        overlaps.append(torch.minimum(a + e1, b + e2) - torch.maximum(a - e1, b - e2))
    return -torch.stack(overlaps, dim=-1).min(dim=-1).values


def _typed_signed_gap(p, geom, th, is_box):
    """Physical signed gap (metres). p (B,N,2), geom (B,N,2), th/is_box (B,N)."""
    hw = geom[..., 0].abs()
    hh = geom[..., 1].abs()
    rad = hw
    p_i, p_j = p.unsqueeze(2), p.unsqueeze(1)
    hw_i, hw_j = hw.unsqueeze(2), hw.unsqueeze(1)
    hh_i, hh_j = hh.unsqueeze(2), hh.unsqueeze(1)
    th_i, th_j = th.unsqueeze(2), th.unsqueeze(1)
    rad_i, rad_j = rad.unsqueeze(2), rad.unsqueeze(1)
    box_i, box_j = is_box.unsqueeze(2), is_box.unsqueeze(1)
    # eps: .norm() at 0 (self-pairs) has undefined backward.
    g_cc = ((p_i - p_j).square().sum(dim=-1) + 1e-12).sqrt() - rad_i - rad_j
    g_cb = _circle_obb_gap(p_i, rad_i, p_j, hw_j, hh_j, th_j)
    g_bc = _circle_obb_gap(p_j, rad_j, p_i, hw_i, hh_i, th_i)
    g_bb = _obb_obb_gap(p_i, hw_i, hh_i, th_i, p_j, hw_j, hh_j, th_j)
    return torch.where(
        box_i & box_j, g_bb,
        torch.where(box_i, g_bc, torch.where(box_j, g_cb, g_cc)))


def pairwise_gap(xy, obj_geom, theta=None, obj_types=None,
                 st_mean=None, st_std=None, geom_mean=None, geom_std=None):
    """xy (B,N,2), geom (B,N,2) -> rel, dist, gap (B,N,N[,2]).

    ``rel`` / ``dist`` stay in the space of ``xy`` (z-scored in train/CEM).
    With ``theta`` and ``obj_types``, ``gap`` is a typed circle/OBB signed
    distance in metres, then divided by ``st_std[0]`` so ``gap_cut`` stays
    in xy-std units. z-scored θ cannot rotate a box — denorm first.
    """
    if xy.dim() == 2:
        xy = xy.unsqueeze(0)
    if obj_geom.dim() == 2:
        obj_geom = obj_geom.unsqueeze(0)
    rel = xy.unsqueeze(2) - xy.unsqueeze(1)
    dist = (rel.square().sum(dim=-1) + 1e-12).sqrt()
    if theta is None or obj_types is None:
        rad = obj_geom.abs().amax(dim=-1)
        gap = dist - rad.unsqueeze(2) - rad.unsqueeze(1)
        return rel, dist, gap

    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    types = obj_types if torch.is_tensor(obj_types) else torch.as_tensor(
        obj_types, device=xy.device)
    types = types.to(device=xy.device, dtype=torch.long)
    if types.dim() == 1:
        types = types.unsqueeze(0)
    if types.shape[0] == 1 and xy.shape[0] > 1:
        types = types.expand(xy.shape[0], -1)

    xy_r, geom_r, th_r = xy, obj_geom, theta
    scale = xy.new_ones(())
    if st_mean is not None and st_std is not None:
        sm = st_mean.to(device=xy.device, dtype=xy.dtype)
        ss = st_std.to(device=xy.device, dtype=xy.dtype).clamp_min(1e-6)
        xy_r = xy * ss[:2] + sm[:2]
        th_r = theta * ss[2] + sm[2]
        scale = ss[0]
    if geom_mean is not None and geom_std is not None:
        gm = geom_mean.to(device=xy.device, dtype=xy.dtype)
        gs = geom_std.to(device=xy.device, dtype=xy.dtype).clamp_min(1e-6)
        geom_r = obj_geom * gs + gm
    if th_r.shape[0] == 1 and xy_r.shape[0] > 1:
        th_r = th_r.expand(xy_r.shape[0], -1)

    gap = _typed_signed_gap(xy_r, geom_r, th_r, types == OBJ_TYPE_BOX) / scale
    return rel, dist, gap


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


class StateTactileEncoder(nn.Module):
    def __init__(self, num_obj_types: int = 3, state_dim: int = 6,
                 contact_dim: int = 7, summary_dim: int = 4,
                 latent_dim: int = 128, obj_embed_dim: int = 64,
                 tactile_embed_dim: int = 64, n_mp: int = 3,
                 k_nn: int = 2, gap_cut: float = 0.12):
        super().__init__()
        self.obj_embed_dim = obj_embed_dim
        self.latent_dim = latent_dim
        self.n_mp = n_mp
        self.k_nn = k_nn
        self.gap_cut = gap_cut

        self.type_emb = nn.Embedding(num_obj_types, 32)
        self.state_mlp = mlp([state_dim, 64, 64])
        self.geom_mlp = mlp([2, 32, 32])
        self.node_proj = mlp([64 + 32 + 32, obj_embed_dim], out_norm=True)

        # rel is 6-d state delta (includes Δvx, Δvy, Δω) plus dist, gap,
        # contact, explicit rel_v (Δvx, Δvy), and a relation-type embedding
        # (contact / revolute / weld / prismatic / none).
        edge_in = state_dim + 3 + 2 + REL_EMB_DIM
        self.rel_emb = nn.Embedding(NUM_REL, REL_EMB_DIM)
        self.node_upd = nn.GRUCell(obj_embed_dim + 1, obj_embed_dim)
        self.edge_mlp = mlp([edge_in, 64, 64])
        self.msg_mlp = mlp([obj_embed_dim * 2 + 64, 64, obj_embed_dim])

        self.contact_mlp = mlp([contact_dim, 64, 64])
        self.tactile_proj = nn.Linear(64, tactile_embed_dim)
        self.summary_mlp = mlp([summary_dim, 32])
        self.ee_inject = mlp([tactile_embed_dim + 32, obj_embed_dim, obj_embed_dim])
        self.node_out = mlp([obj_embed_dim, latent_dim], out_norm=True)

    def _message_pass(self, h, obj_states, obj_geom, rel_type=None,
                      obj_types=None, gap_stats=None):
        B, N, D = h.shape
        rel = obj_states.unsqueeze(2) - obj_states.unsqueeze(1)
        gs = gap_stats or {}
        _, dist, gap = pairwise_gap(
            obj_states[..., :2], obj_geom, theta=obj_states[..., 2],
            obj_types=obj_types, **gs)
        rel_ids = as_rel_tensor(rel_type, B, N, h.device)
        joint = is_joint(rel_ids)
        adj = knn_adjacency(dist, gap, self.k_nn, self.gap_cut) | joint
        contact_flag = ((gap < 0.0) & ~joint).to(h.dtype).unsqueeze(-1)
        vel = obj_states[..., 3:5]
        rel_v = vel.unsqueeze(2) - vel.unsqueeze(1)
        edge_ids = overlay_contact(rel_ids, gap < 0.0).clamp(0, REL_NONE)
        e = self.edge_mlp(torch.cat([rel, dist.unsqueeze(-1),
                                     gap.unsqueeze(-1), contact_flag,
                                     rel_v, self.rel_emb(edge_ids)], dim=-1))
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        msg = self.msg_mlp(torch.cat([hi, hj, e], dim=-1))
        msg = msg.masked_fill(~adj.unsqueeze(-1), 0.0)
        agg = msg.sum(dim=2)
        deg = adj.sum(dim=2, keepdim=True).to(h.dtype)
        agg = torch.cat([agg, deg / 4.0], dim=-1)
        h = self.node_upd(agg.reshape(B * N, -1), h.reshape(B * N, D)).view(B, N, D)
        return h

    def forward(self, obj_types, obj_states, contact_feat, contact_mask,
                tactile_summary, obj_geom=None, rel_type=None, gap_stats=None):
        """Returns z: (B, N, latent_dim)."""
        B, N, _ = obj_states.shape
        if obj_geom is None:
            obj_geom = obj_states.new_zeros(B, N, 2)
        h = self.node_proj(torch.cat([
            self.type_emb(obj_types), self.state_mlp(obj_states),
            self.geom_mlp(obj_geom),
        ], dim=-1))

        # Tactile is an EE residual on top of the state embedding, not a
        # second backbone. Zero feat/mask/summary (no contact, or CEM
        # state-only encode) → residual ≈ 0 and z is state-driven.
        tc = self.contact_mlp(contact_feat)
        tc = (tc * contact_mask.unsqueeze(-1)).sum(dim=1)
        tactile = self.tactile_proj(tc)
        summary = self.summary_mlp(tactile_summary)
        h = h.clone()
        h[:, 0] = h[:, 0] + self.ee_inject(torch.cat([tactile, summary], dim=-1))

        for _ in range(self.n_mp):
            h = self._message_pass(
                h, obj_states, obj_geom, rel_type=rel_type,
                obj_types=obj_types, gap_stats=gap_stats)
        return self.node_out(h)
