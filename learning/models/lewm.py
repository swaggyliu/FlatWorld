"""StateLeWM: assembled state + tactile latent world model.

    encoder:   sparse graph + EE-only tactile -> z_t  (B, N, L)
    predictor: Interaction Network + per-node GRU
    decoder:   z_i -> state_i; EE node -> tactile summary
    contact:   z_EE -> P(EE in contact)

Loss:
    Total = Latent_Pred_Loss
          + w_rec  * Recon(z_enc) + Recon(z_pred)
          + w_var  * SIGReg
          + w_c    * BCE(contact now) + BCE(contact next | z_pred)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import StateTactileEncoder, mlp
from .decoder import StateTactileDecoder
from .predictor import LatentPredictor


class StateLeWM(nn.Module):
    def __init__(self, n_obj: int, num_obj_types: int = 3, latent_dim: int = 128,
                 n_mp: int = 2, drop_tactile: bool = False, drop_geom: bool = False,
                 rich_edges: bool = True):
        super().__init__()
        self.n_obj = n_obj
        self.latent_dim = latent_dim
        self.n_mp = int(n_mp)
        self.drop_tactile = bool(drop_tactile)
        self.drop_geom = bool(drop_geom)
        self.rich_edges = bool(rich_edges)
        self.encoder = StateTactileEncoder(num_obj_types=num_obj_types,
                                           latent_dim=latent_dim,
                                           n_mp=self.n_mp,
                                           rich_edges=self.rich_edges)
        self.predictor = LatentPredictor(latent_dim=latent_dim,
                                         rich_edges=self.rich_edges)
        self.decoder = StateTactileDecoder(n_obj=n_obj, latent_dim=latent_dim)
        self.contact_head = mlp([latent_dim, 64, 1])
        # Privileged pair / ground heads: train-only labels from states+geom.
        # Inference still sees only EE tactile.
        self.pair_head = mlp([latent_dim * 3, 64, 1])
        self.ground_head = mlp([latent_dim, 32, 1])
        self.ss_prob = 0.6

    def encode(self, obj_types, obj_states, contact_feat, contact_mask,
               tactile_summary, obj_geom=None):
        """Encode a batch of frames (any leading dims are flattened).

        obj_states: (..., N, 6); contact_feat: (..., K, 7);
        contact_mask: (..., K); tactile_summary: (..., 4);
        obj_types: (N,) or (..., N)
        obj_geom: (..., N, 2) half-extents, optional
        returns z: (..., N, latent_dim)
        """
        n_obj = obj_states.shape[-2]
        lead = obj_states.shape[:-2]
        flat_states = obj_states.reshape(-1, n_obj, obj_states.shape[-1])
        flat_feat = contact_feat.reshape(-1, contact_feat.shape[-2],
                                         contact_feat.shape[-1])
        flat_mask = contact_mask.reshape(-1, contact_mask.shape[-1])
        flat_sum = tactile_summary.reshape(-1, tactile_summary.shape[-1])
        if obj_types.dim() == 1:
            flat_types = obj_types.unsqueeze(0).expand(flat_states.shape[0], -1)
        else:
            flat_types = obj_types.reshape(-1, n_obj)
        if self.drop_tactile:
            flat_feat = flat_feat * 0
            flat_mask = flat_mask * 0
            flat_sum = flat_sum * 0
        flat_geom = None
        if obj_geom is not None:
            flat_geom = obj_geom.reshape(-1, n_obj, obj_geom.shape[-1])
        if self.drop_geom:
            if flat_geom is None:
                flat_geom = flat_states.new_zeros(flat_states.shape[0], n_obj, 2)
            else:
                flat_geom = flat_geom * 0

        z = self.encoder(flat_types, flat_states, flat_feat, flat_mask, flat_sum,
                         obj_geom=flat_geom)
        return z.reshape(*lead, n_obj, self.latent_dim)

    def contact_logit(self, z):
        """z: (..., N, L) or (..., L) -> logits (...). Uses the EE node."""
        if z.dim() >= 3 and z.shape[-1] == self.latent_dim:
            z = z[..., 0, :]
        lead = z.shape[:-1]
        return self.contact_head(z.reshape(-1, self.latent_dim)).squeeze(-1).reshape(*lead)

    def pair_logit(self, z):
        """z (..., N, L) -> pair logits (..., N, N)."""
        lead = z.shape[:-2]
        n_obj, lat = z.shape[-2], z.shape[-1]
        zf = z.reshape(-1, n_obj, lat)
        hi = zf.unsqueeze(2).expand(-1, n_obj, n_obj, lat)
        hj = zf.unsqueeze(1).expand(-1, n_obj, n_obj, lat)
        logit = self.pair_head(torch.cat([hi, hj, (hi - hj).abs()], dim=-1))
        return logit.squeeze(-1).reshape(*lead, n_obj, n_obj)

    def ground_logit(self, z):
        """z (..., N, L) -> per-node ground-contact logits (..., N)."""
        lead = z.shape[:-1]
        return self.ground_head(z.reshape(-1, self.latent_dim)).squeeze(-1).reshape(*lead)

    def forward(self, batch):
        """batch dict of (B, T, ...) tensors. Returns everything compute_loss needs."""
        obj_states = batch["obj_states"]            # (B, T, N, 6)
        actions = batch["actions"]                  # (B, T-1, 2)
        contact_feat = batch["contact_feat"]        # (B, T, K, 7)
        contact_mask = batch["contact_mask"]        # (B, T, K)
        summary = batch["tactile_summary"]          # (B, T, 4)
        obj_types = batch["obj_types"]              # (B, N)

        B, T, n_obj = obj_states.shape[:3]

        geom = batch.get("obj_geom")
        geom_t = None
        if geom is not None:
            geom_t = geom.unsqueeze(1).expand(B, T, n_obj, geom.shape[-1])
        z = self.encode(obj_types.unsqueeze(1).expand(B, T, n_obj),
                        obj_states, contact_feat, contact_mask,
                        summary, obj_geom=geom_t)     # (B, T, N, L)

        # Scheduled sampling: CEM rebuilds the graph from predicted xy, so
        # training must sometimes do the same (not always teacher-forced).
        use_tf = (not self.training) or (float(torch.rand(1)) >= self.ss_prob)
        xy_seq = obj_states[:, :-1, :, :2] if use_tf else None
        z_pred = self.predictor(z[:, :-1], actions, geom=geom,
                                xy_seq=xy_seq)  # (B, T-1, N, L)
        z_target = z[:, 1:].detach()

        rec_s, rec_t = self.decoder(z.reshape(B * T, n_obj, self.latent_dim))
        recon_states = rec_s.view(B, T, n_obj, -1)
        recon_summary = rec_t.view(B, T, -1)

        rec_ps, rec_pt = self.decoder(z_pred.reshape(B * (T - 1), n_obj, self.latent_dim))
        pred_recon_states = rec_ps.view(B, T - 1, n_obj, -1)
        pred_recon_summary = rec_pt.view(B, T - 1, -1)

        contact_now = (contact_mask.sum(dim=-1) > 0).float()     # (B, T)
        pair = batch.get("pair_contact")
        ground = batch.get("ground_contact")
        return {
            "z": z,
            "z_pred": z_pred,
            "z_target": z_target,
            "recon_states": recon_states,
            "recon_summary": recon_summary,
            "pred_recon_states": pred_recon_states,
            "pred_recon_summary": pred_recon_summary,
            "states": obj_states,
            "summary": summary,
            "contact_now": contact_now,
            "contact_logit": self.contact_logit(z),
            "contact_logit_pred": self.contact_logit(z_pred),
            "xy_enc": self.predictor.xy_head(z),
            "xy_pred": self.predictor.xy_head(z_pred),
            "vel_enc": self.predictor.vel_head(z) if self.rich_edges else None,
            "vel_pred": self.predictor.vel_head(z_pred) if self.rich_edges else None,
            "pair_contact": pair,
            "ground_contact": ground,
            "actions": actions,
            "pair_logit": self.pair_logit(z) if pair is not None else None,
            "pair_logit_pred": self.pair_logit(z_pred) if pair is not None else None,
            "ground_logit": self.ground_logit(z) if ground is not None else None,
            "ground_logit_pred": self.ground_logit(z_pred) if ground is not None else None,
        }

    @staticmethod
    def sigreg(z, target_std: float = 1.0):
        """SIGReg: hinge max(0, target_std - std(z_d)) against latent collapse."""
        std = z.flatten(0, -2).std(dim=0)
        return F.relu(target_std - std).mean()

    @staticmethod
    def focal_bce_with_logits(logits, targets, gamma: float = 2.0):
        """Focal BCE: down-weights easy (high-confidence) examples so the
        model focuses on rare contact frames instead of coasting on the
        majority 'no-contact' label."""
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = ce * ((1 - p_t).clamp(min=1e-6) ** gamma)
        return loss.mean()

    def set_ss_prob(self, prob: float):
        self.ss_prob = float(prob)

    def compute_loss(self, out, w_dyn: float = 1.0, w_rec: float = 1.0,
                     w_pred_rec: float = 1.0, w_var: float = 0.1,
                     w_contact: float = 1.0, w_xy: float = 0.5,
                     w_pair: float = 1.5, w_ground: float = 0.5,
                     w_drift: float = 0.3, w_vel: float = 0.5):
        loss_dyn = F.mse_loss(out["z_pred"], out["z_target"])
        loss_rec_s = F.mse_loss(out["recon_states"], out["states"])
        loss_rec_t = F.mse_loss(out["recon_summary"], out["summary"])
        loss_prt = F.mse_loss(out["pred_recon_summary"], out["summary"][:, 1:])
        loss_var = self.sigreg(out["z"])
        loss_xy = F.mse_loss(out["xy_enc"], out["states"][..., :2])
        loss_xy = loss_xy + F.mse_loss(out["xy_pred"], out["states"][:, 1:, :, :2])
        d_true = out["states"][:, 1:, :, :2] - out["states"][:, :-1, :, :2]
        d_pred = out["xy_pred"] - out["states"][:, :-1, :, :2]
        move_w = 1.0 + 4.0 * (d_true.norm(dim=-1) > 0.01).to(d_true.dtype)
        loss_xy = loss_xy + (move_w.unsqueeze(-1) * (d_pred - d_true).pow(2)).mean()

        err = (out["pred_recon_states"] - out["states"][:, 1:]) ** 2
        dim_w = err.new_tensor([2.0, 2.0, 1.0, 3.0, 3.0, 1.0])
        pair = out.get("pair_contact")
        ground = out.get("ground_contact")
        node_w = err.new_ones(err.shape[:3])
        if pair is not None:
            node_w = node_w + 2.0 * pair[:, 1:].sum(dim=-1).clamp(max=3.0)
        if ground is not None:
            node_w = node_w + 1.0 * ground[:, 1:]
        loss_prs = (err * node_w.unsqueeze(-1) * dim_w).mean()

        # Zero-force drift: when EE action is small, non-EE objects should
        # not move. Penalize predicted displacement under near-rest.
        a_mag = out["actions"].norm(dim=-1, keepdim=True)   # (B, T-1, 1)
        rest_mask = (a_mag < 0.3).to(err.dtype)             # (B, T-1, 1)
        d_rest = out["xy_pred"][:, :, 1:] - out["xy_enc"][:, :-1, 1:]
        # d_rest: (B, T-1, N-1, 2) -> norm: (B, T-1, N-1)
        # rest_mask broadcasts over the last (object) dim.
        loss_drift = (rest_mask * d_rest.norm(dim=-1)).mean()

        # Velocity readout: the rich-edge features consume vel_head(z), so
        # the readout must actually track true velocity or the collision
        # features (approach speed) are fiction during imagined rollouts.
        loss_vel = err.new_zeros(())
        if out.get("vel_pred") is not None:
            # state layout: (x, y, theta, vx, vy, omega) -> linear vel is [3:5]
            v_true = out["states"][..., 3:5]
            loss_vel = F.mse_loss(out["vel_enc"], v_true)
            loss_vel = loss_vel + F.mse_loss(out["vel_pred"], v_true[:, 1:])

        c_now = out["contact_now"]
        c_smooth = c_now * 0.9 + 0.05
        loss_c = self.focal_bce_with_logits(out["contact_logit"], c_smooth)
        loss_c = loss_c + self.focal_bce_with_logits(
            out["contact_logit_pred"], c_smooth[:, 1:])

        loss_pair = err.new_zeros(())
        if pair is not None and out.get("pair_logit") is not None:
            n = pair.shape[-1]
            off = ~torch.eye(n, dtype=torch.bool, device=pair.device)
            tgt = (pair * 0.9 + 0.05)[..., off]
            loss_pair = self.focal_bce_with_logits(
                out["pair_logit"][..., off], tgt)
            loss_pair = loss_pair + self.focal_bce_with_logits(
                out["pair_logit_pred"][..., off], (pair[:, 1:] * 0.9 + 0.05)[..., off])

        loss_g = err.new_zeros(())
        if ground is not None and out.get("ground_logit") is not None:
            g = ground * 0.9 + 0.05
            loss_g = self.focal_bce_with_logits(out["ground_logit"], g)
            loss_g = loss_g + self.focal_bce_with_logits(
                out["ground_logit_pred"], g[:, 1:])

        total = (w_dyn * loss_dyn
                 + w_rec * (loss_rec_s + loss_rec_t)
                 + w_pred_rec * (loss_prs + loss_prt)
                 + w_var * loss_var
                 + w_contact * loss_c
                 + w_xy * loss_xy
                 + w_pair * loss_pair
                 + w_ground * loss_g
                 + w_drift * loss_drift
                 + w_vel * loss_vel)
        return {
            "total": total,
            "dynamics": loss_dyn,
            "recon_states": loss_rec_s,
            "recon_summary": loss_rec_t,
            "pred_recon_states": loss_prs,
            "pred_recon_summary": loss_prt,
            "sigreg": loss_var,
            "contact": loss_c,
            "pair": loss_pair,
            "ground": loss_g,
            "drift": loss_drift,
            "vel": loss_vel,
        }
