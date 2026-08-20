"""StateLeWM: assembled state + tactile latent world model.

    encoder:   relation-graph + tactile Deep Sets -> z_t
    predictor: (z_t, a_t) -> z_{t+1}        (GRU)
    decoder:   z -> (states, tactile summary)
    contact:   z -> P(EE in contact)         (auxiliary gating head)

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
    def __init__(self, n_obj: int, num_obj_types: int = 3, latent_dim: int = 128):
        super().__init__()
        self.n_obj = n_obj
        self.latent_dim = latent_dim
        self.encoder = StateTactileEncoder(num_obj_types=num_obj_types,
                                           latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim)
        self.decoder = StateTactileDecoder(n_obj=n_obj, latent_dim=latent_dim)
        self.contact_head = mlp([latent_dim, 64, 1])

    def encode(self, obj_types, obj_states, contact_feat, contact_mask,
               tactile_summary, obj_geom=None):
        """Encode a batch of frames (any leading dims are flattened).

        obj_states: (..., N, 6); contact_feat: (..., K, 7);
        contact_mask: (..., K); tactile_summary: (..., 4);
        obj_types: (N,) or (..., N)
        obj_geom: (..., N, 2) half-extents, optional
        returns z: (..., latent_dim)
        """
        lead = obj_states.shape[:-2]
        flat_states = obj_states.reshape(-1, self.n_obj, obj_states.shape[-1])
        flat_feat = contact_feat.reshape(-1, contact_feat.shape[-2],
                                         contact_feat.shape[-1])
        flat_mask = contact_mask.reshape(-1, contact_mask.shape[-1])
        flat_sum = tactile_summary.reshape(-1, tactile_summary.shape[-1])
        if obj_types.dim() == 1:
            flat_types = obj_types.unsqueeze(0).expand(flat_states.shape[0], -1)
        else:
            flat_types = obj_types.reshape(-1, self.n_obj)
        flat_geom = None
        if obj_geom is not None:
            flat_geom = obj_geom.reshape(-1, self.n_obj, obj_geom.shape[-1])

        z = self.encoder(flat_types, flat_states, flat_feat, flat_mask, flat_sum,
                         obj_geom=flat_geom)
        return z.reshape(*lead, -1)

    def contact_logit(self, z):
        """z: (..., L) -> logits (...,)."""
        lead = z.shape[:-1]
        return self.contact_head(z.reshape(-1, self.latent_dim)).squeeze(-1).reshape(*lead)

    def forward(self, batch):
        """batch dict of (B, T, ...) tensors. Returns everything compute_loss needs."""
        obj_states = batch["obj_states"]            # (B, T, N, 6)
        actions = batch["actions"]                  # (B, T-1, 2)
        contact_feat = batch["contact_feat"]        # (B, T, K, 7)
        contact_mask = batch["contact_mask"]        # (B, T, K)
        summary = batch["tactile_summary"]          # (B, T, 4)
        obj_types = batch["obj_types"]              # (B, N)

        B, T = obj_states.shape[:2]

        geom = batch.get("obj_geom")
        if geom is not None:
            geom = geom.unsqueeze(1).expand(B, T, self.n_obj, geom.shape[-1])
        z = self.encode(obj_types.unsqueeze(1).expand(B, T, self.n_obj),
                        obj_states, contact_feat, contact_mask,
                        summary, obj_geom=geom)     # (B, T, L)

        z_pred = self.predictor(z[:, :-1], actions)  # (B, T-1, L)
        z_target = z[:, 1:].detach()

        rec_s, rec_t = self.decoder(z.reshape(-1, self.latent_dim))
        recon_states = rec_s.view(B, T, self.n_obj, -1)
        recon_summary = rec_t.view(B, T, -1)

        rec_ps, rec_pt = self.decoder(z_pred.reshape(-1, self.latent_dim))
        pred_recon_states = rec_ps.view(B, T - 1, self.n_obj, -1)
        pred_recon_summary = rec_pt.view(B, T - 1, -1)

        contact_now = (contact_mask.sum(dim=-1) > 0).float()     # (B, T)
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
        }

    @staticmethod
    def sigreg(z, target_std: float = 1.0):
        """SIGReg: hinge max(0, target_std - std(z_d)) against latent collapse."""
        std = z.flatten(0, -2).std(dim=0)
        return F.relu(target_std - std).mean()

    def compute_loss(self, out, w_dyn: float = 1.0, w_rec: float = 1.0,
                     w_pred_rec: float = 1.0, w_var: float = 0.1,
                     w_contact: float = 0.5):
        loss_dyn = F.mse_loss(out["z_pred"], out["z_target"])
        loss_rec_s = F.mse_loss(out["recon_states"], out["states"])
        loss_rec_t = F.mse_loss(out["recon_summary"], out["summary"])
        loss_prs = F.mse_loss(out["pred_recon_states"], out["states"][:, 1:])
        loss_prt = F.mse_loss(out["pred_recon_summary"], out["summary"][:, 1:])
        loss_var = self.sigreg(out["z"])

        c_now = out["contact_now"]
        loss_c = F.binary_cross_entropy_with_logits(out["contact_logit"], c_now)
        loss_c = loss_c + F.binary_cross_entropy_with_logits(
            out["contact_logit_pred"], c_now[:, 1:])

        total = (w_dyn * loss_dyn
                 + w_rec * (loss_rec_s + loss_rec_t)
                 + w_pred_rec * (loss_prs + loss_prt)
                 + w_var * loss_var
                 + w_contact * loss_c)
        return {
            "total": total,
            "dynamics": loss_dyn,
            "recon_states": loss_rec_s,
            "recon_summary": loss_rec_t,
            "pred_recon_states": loss_prs,
            "pred_recon_summary": loss_prt,
            "sigreg": loss_var,
            "contact": loss_c,
        }
