"""StateLeWM: assembled state + tactile latent world model.

    encoder:   sparse graph on states; EE tactile is a residual on the EE node
    predictor: Interaction Net + per-node GRU; edges carry rel_v from vel_head
    xy_head:   z_i -> (x, y)   (CEM scores this)
    pair/ground heads: privileged contacts, fed back during imagination

Loss (defaults):
    Total = Latent_Pred_Loss + SIGReg + xy + pair + ground + drift + vel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import StateTactileEncoder, mlp
from .predictor import LatentPredictor


class StateLeWM(nn.Module):
    def __init__(self, n_obj: int, num_obj_types: int = 3, latent_dim: int = 128,
                 n_mp: int = 3, tactile_drop_prob: float = 0.5):
        super().__init__()
        self.n_obj = n_obj
        self.latent_dim = latent_dim
        self.n_mp = int(n_mp)
        self.tactile_drop_prob = float(tactile_drop_prob)
        self.encoder = StateTactileEncoder(num_obj_types=num_obj_types,
                                           latent_dim=latent_dim,
                                           n_mp=self.n_mp)
        self.predictor = LatentPredictor(latent_dim=latent_dim)
        self.pair_head = mlp([latent_dim * 3, 64, 1])
        self.ground_head = mlp([latent_dim, 32, 1])
        self.ss_prob = 0.6

    def encode(self, obj_types, obj_states, contact_feat, contact_mask,
               tactile_summary, obj_geom=None, use_tactile: bool = True):
        """Encode a batch of frames (any leading dims are flattened).

        use_tactile: if False, zero the EE residual (state-only encode).
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
        if not use_tactile:
            flat_feat = flat_feat * 0
            flat_mask = flat_mask * 0
            flat_sum = flat_sum * 0
        flat_geom = None
        if obj_geom is not None:
            flat_geom = obj_geom.reshape(-1, n_obj, obj_geom.shape[-1])

        z = self.encoder(flat_types, flat_states, flat_feat, flat_mask, flat_sum,
                         obj_geom=flat_geom)
        return z.reshape(*lead, n_obj, self.latent_dim)

    def pair_prob(self, z):
        """Soft symmetric pair-contact matrix."""
        p = torch.sigmoid(self.pair_logit(z))
        p = 0.5 * (p + p.transpose(-1, -2))
        n = p.shape[-1]
        eye = torch.eye(n, device=p.device, dtype=p.dtype)
        while eye.dim() < p.dim():
            eye = eye.unsqueeze(0)
        return p * (1.0 - eye)

    def ground_prob(self, z):
        """Per-node ground-contact probability."""
        return torch.sigmoid(self.ground_logit(z))

    def roll_step(self, z, a, h, geom, prev_c):
        """One imagined step: pair on edges, ground on the GRU node input."""
        pair_now = self.pair_prob(z)
        ground_now = self.ground_prob(z)
        z, h, _ = self.predictor.step(
            z, a, h, geom=geom, prev_contact=prev_c, pair_contact=pair_now,
            ground=ground_now)
        return z, h, pair_now

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
        use_tac = True
        if self.training and self.tactile_drop_prob > 0:
            use_tac = float(torch.rand(1)) >= self.tactile_drop_prob
        z = self.encode(obj_types.unsqueeze(1).expand(B, T, n_obj),
                        obj_states, contact_feat, contact_mask,
                        summary, obj_geom=geom_t,
                        use_tactile=use_tac)

        use_tf = (not self.training) or (float(torch.rand(1)) >= self.ss_prob)
        z_target = z[:, 1:].detach()

        pair = batch.get("pair_contact")
        ground = batch.get("ground_contact")
        h = self.predictor.init_hidden(z[:, 0])
        zt = z[:, 0]
        preds = []
        if use_tf and pair is not None:
            prev_c = pair[:, 0]
        else:
            prev_c = self.pair_prob(zt)
        for t in range(T - 1):
            xy = obj_states[:, t, :, :2] if use_tf else None
            if use_tf and pair is not None:
                pair_now = pair[:, t]
            else:
                pair_now = self.pair_prob(zt)
            if use_tf and ground is not None:
                g_now = ground[:, t]
            else:
                g_now = self.ground_prob(zt)
            zt, h, _ = self.predictor.step(
                zt, actions[:, t], h, geom=geom, xy=xy,
                prev_contact=prev_c, pair_contact=pair_now, ground=g_now)
            preds.append(zt)
            prev_c = pair_now
        z_pred = torch.stack(preds, dim=1)

        return {
            "z": z,
            "z_pred": z_pred,
            "z_target": z_target,
            "states": obj_states,
            "summary": summary,
            "xy_enc": self.predictor.xy_head(z),
            "xy_pred": self.predictor.xy_head(z_pred),
            "vel_enc": self.predictor.vel_head(z),
            "vel_pred": self.predictor.vel_head(z_pred),
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

    def compute_loss(self, out, w_dyn: float = 1.0, w_var: float = 0.1,
                     w_xy: float = 0.5, w_pair: float = 1.5, w_ground: float = 0.5,
                     w_drift: float = 0.3, w_vel: float = 0.5):
        loss_dyn = F.mse_loss(out["z_pred"], out["z_target"])
        loss_var = self.sigreg(out["z"])
        loss_xy = F.mse_loss(out["xy_enc"], out["states"][..., :2])
        loss_xy = loss_xy + F.mse_loss(out["xy_pred"], out["states"][:, 1:, :, :2])
        d_true = out["states"][:, 1:, :, :2] - out["states"][:, :-1, :, :2]
        d_pred = out["xy_pred"] - out["states"][:, :-1, :, :2]
        move_w = 1.0 + 4.0 * (d_true.norm(dim=-1) > 0.01).to(d_true.dtype)
        loss_xy = loss_xy + (move_w.unsqueeze(-1) * (d_pred - d_true).pow(2)).mean()

        pair = out.get("pair_contact")
        ground = out.get("ground_contact")
        zero = d_true.new_zeros(())

        a_mag = out["actions"].norm(dim=-1, keepdim=True)
        rest_mask = (a_mag < 0.3).to(d_true.dtype)
        d_rest = out["xy_pred"][:, :, 1:] - out["xy_enc"][:, :-1, 1:]
        loss_drift = (rest_mask * d_rest.norm(dim=-1)).mean()

        v_true = out["states"][..., 3:5]
        loss_vel = F.mse_loss(out["vel_enc"], v_true)
        loss_vel = loss_vel + F.mse_loss(out["vel_pred"], v_true[:, 1:])

        loss_pair = zero
        if pair is not None and out.get("pair_logit") is not None:
            n = pair.shape[-1]
            off = ~torch.eye(n, dtype=torch.bool, device=pair.device)
            tgt = (pair * 0.9 + 0.05)[..., off]
            loss_pair = self.focal_bce_with_logits(
                out["pair_logit"][..., off], tgt)
            loss_pair = loss_pair + self.focal_bce_with_logits(
                out["pair_logit_pred"][..., off], (pair[:, 1:] * 0.9 + 0.05)[..., off])

        loss_g = zero
        if ground is not None and out.get("ground_logit") is not None:
            g = ground * 0.9 + 0.05
            loss_g = self.focal_bce_with_logits(out["ground_logit"], g)
            loss_g = loss_g + self.focal_bce_with_logits(
                out["ground_logit_pred"], g[:, 1:])

        total = (w_dyn * loss_dyn
                 + w_var * loss_var
                 + w_xy * loss_xy
                 + w_pair * loss_pair
                 + w_ground * loss_g
                 + w_drift * loss_drift
                 + w_vel * loss_vel)
        return {
            "total": total,
            "dynamics": loss_dyn,
            "sigreg": loss_var,
            "pair": loss_pair,
            "ground": loss_g,
            "drift": loss_drift,
            "vel": loss_vel,
        }
