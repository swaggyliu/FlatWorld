"""StateLeWM: assembled state + tactile latent world model.

    encoder:   (states, tactile set) -> z_t
    predictor: (z_t, a_t) -> z_{t+1}        (GRU, teacher-forced training)
    decoder:   z -> (states, tactile summary) (anti-collapse)

Loss (Phase 2 formulation):
    Total = Latent_Pred_Loss                        [MSE(z_pred, z_target)]
          + w_rs * State_Recon_Loss                  [decode(z) vs states,
                                                     applied to encoded AND
                                                     predicted latents]
          + w_rt * Tactile_Recon_Loss                [decode(z) vs summary,
                                                     same dual application]
          + w_var * SIGReg                           [variance regularization
                                                     to prevent latent
                                                     dimension collapse]

All inputs/outputs are in normalized units (see data/normalizer.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import StateTactileEncoder
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

    # ------------------------------------------------------------------ #
    def encode(self, obj_types, obj_states, contact_feat, contact_mask,
               tactile_summary):
        """Encode a batch of frames (any leading dims are flattened).

        obj_states: (..., N, 6); contact_feat: (..., K, 7);
        contact_mask: (..., K); tactile_summary: (..., 4);
        obj_types: (N,) or (..., N)
        returns z: (..., latent_dim)
        """
        lead = obj_states.shape[:-2]
        flat_states = obj_states.reshape(-1, self.n_obj, obj_states.shape[-1])
        flat_feat = contact_feat.reshape(-1, contact_feat.shape[-2],
                                         contact_feat.shape[-1])
        flat_mask = contact_mask.reshape(-1, contact_mask.shape[-1])
        flat_sum = tactile_summary.reshape(-1, tactile_summary.shape[-1])
        if obj_types.dim() == 1:  # (N,) shared across the batch
            flat_types = obj_types.unsqueeze(0).expand(flat_states.shape[0], -1)
        else:
            flat_types = obj_types.reshape(-1, self.n_obj)

        z = self.encoder(flat_types, flat_states, flat_feat, flat_mask, flat_sum)
        return z.reshape(*lead, -1)

    # ------------------------------------------------------------------ #
    def forward(self, batch):
        """batch dict of (B, T, ...) tensors. Returns everything compute_loss needs."""
        obj_states = batch["obj_states"]            # (B, T, N, 6)
        actions = batch["actions"]                  # (B, T-1, 2)
        contact_feat = batch["contact_feat"]        # (B, T, K, 7)
        contact_mask = batch["contact_mask"]        # (B, T, K)
        summary = batch["tactile_summary"]          # (B, T, 4)
        obj_types = batch["obj_types"]              # (B, N)

        B, T = obj_states.shape[:2]

        z = self.encode(obj_types.unsqueeze(1).expand(B, T, self.n_obj),
                        obj_states, contact_feat, contact_mask,
                        summary)                    # (B, T, L)

        z_pred = self.predictor(z[:, :-1], actions)  # (B, T-1, L)
        z_target = z[:, 1:].detach()

        # Reconstruction from encoded latents (identity supervision)
        rec_s, rec_t = self.decoder(z.reshape(-1, self.latent_dim))
        recon_states = rec_s.view(B, T, self.n_obj, -1)
        recon_summary = rec_t.view(B, T, -1)

        # Reconstruction from PREDICTED latents z_{t+1} against ground truth
        # s_{t+1} (Phase 2 auxiliary loss: forces the latent transition to be
        # physically meaningful, not merely geometrically close in z space)
        rec_ps, rec_pt = self.decoder(z_pred.reshape(-1, self.latent_dim))
        pred_recon_states = rec_ps.view(B, T - 1, self.n_obj, -1)
        pred_recon_summary = rec_pt.view(B, T - 1, -1)

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
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def sigreg(z, target_std: float = 1.0):
        """SIGReg: variance regularization -var(z) against latent collapse.

        Implemented in hinge form max(0, target_std - std(z_d)) so the
        pressure stops once every latent dimension is sufficiently spread
        (plain -var(z) would push spreads to infinity).
        z: (B, T, L) or (B, L).
        """
        std = z.flatten(0, -2).std(dim=0)          # per-dimension std (L,)
        return F.relu(target_std - std).mean()

    def compute_loss(self, out, w_dyn: float = 1.0, w_rec: float = 1.0,
                     w_pred_rec: float = 1.0, w_var: float = 0.1):
        loss_dyn = F.mse_loss(out["z_pred"], out["z_target"])
        loss_rec_s = F.mse_loss(out["recon_states"], out["states"])
        loss_rec_t = F.mse_loss(out["recon_summary"], out["summary"])
        # predicted-latent decoding vs the true next-frame quantities
        loss_prs = F.mse_loss(out["pred_recon_states"], out["states"][:, 1:])
        loss_prt = F.mse_loss(out["pred_recon_summary"], out["summary"][:, 1:])
        loss_var = self.sigreg(out["z"])

        total = (w_dyn * loss_dyn
                 + w_rec * (loss_rec_s + loss_rec_t)
                 + w_pred_rec * (loss_prs + loss_prt)
                 + w_var * loss_var)
        return {
            "total": total,
            "dynamics": loss_dyn,
            "recon_states": loss_rec_s,
            "recon_summary": loss_rec_t,
            "pred_recon_states": loss_prs,
            "pred_recon_summary": loss_prt,
            "sigreg": loss_var,
        }
