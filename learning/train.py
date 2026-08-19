"""Train the StateLeWM world model on collected rollouts.

Usage (repo root):
    python -m learning.train --data learning/data/rollouts --epochs 120

Outputs:
    learning/checkpoints/best.pt   best validation checkpoint (+ normalizer)
    learning/results/train_log.csv per-epoch metrics (for plotting)
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from learning.configs.default import Config
from learning.data.dataset import PushWindowDataset, train_val_split
from learning.data.normalizer import Normalizer
from learning.models.lewm import StateLeWM


@torch.no_grad()
def evaluate(model, loader, device):
    """Validation losses + latent statistics + long-horizon open-loop error."""
    model.eval()
    agg = {"dynamics": 0.0, "recon_states": 0.0, "recon_summary": 0.0,
           "pred_recon_states": 0.0, "sigreg": 0.0}
    z_std_sum, z_batches = 0.0, 0
    k_err = {10: [], 25: [], 50: []}
    n_win = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        losses = model.compute_loss(out)
        for k in agg:
            agg[k] += losses[k].item() * batch["obj_states"].shape[0]
        z_std_sum += out["z"].std(dim=0).mean().item()
        z_batches += 1
        n_win += batch["obj_states"].shape[0]

        # Open-loop K-step rollout from z_0 with ground-truth actions
        z0 = out["z"][:, 0]
        h = model.predictor.init_hidden(z0)
        z = z0
        T = batch["actions"].shape[1]
        for t in range(T):
            z, h = model.predictor.step(z, batch["actions"][:, t], h)
            if (t + 1) in k_err:
                err = torch.nn.functional.mse_loss(z, out["z_target"][:, t]).item()
                k_err[t + 1].append(err)
            elif t + 1 == T and T < 50:
                # shorter rollouts (frame-skipped data): report the final
                # horizon in the 50-step bucket so long-horizon trends stay
                # visible in the same CSV column
                err = torch.nn.functional.mse_loss(z, out["z_target"][:, t]).item()
                k_err[50].append(err)
    model.train()
    res = {k: v / n_win for k, v in agg.items()}
    res["z_std"] = z_std_sum / max(z_batches, 1)
    for k, v in k_err.items():
        if v:
            res[f"openloop_{k}"] = float(np.mean(v))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="learning/data/rollouts")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-window", type=int, default=24,
                        help="teacher-forcing window length during training "
                             "(validation always uses 50 for long rollouts)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularization; diverse-scene datasets are "
                             "small and overfit without it")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--stride", type=int, default=5,
                        help="frame-skip / action repeat: one model step "
                             "covers this many simulation frames")
    parser.add_argument("--windows-per-rollout", type=int, default=16)
    parser.add_argument("--out", type=str, default="learning/checkpoints")
    parser.add_argument("--results", type=str, default="learning/results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----- data -----
    norm = Normalizer.fit_from_dir(args.data)
    # number of model steps per rollout after frame skipping
    n_steps = len(np.load(sorted(glob.glob(
        os.path.join(args.data, "*.npz")))[0])["actions"]) // args.stride
    full = PushWindowDataset(args.data, window=n_steps, normalizer=norm,
                             stride=args.stride)
    n_obj = full[0]["obj_types"].shape[0]
    train_ds, val_ds = train_val_split(full)
    # Short training windows (fast teacher forcing) but full-length validation
    # rollouts for open-loop diagnostics
    train_ds.window = min(args.train_window, n_steps)
    # Sample multiple random windows per rollout per epoch
    train_ds_rollout_len = len(train_ds)

    class Windowed(torch.utils.data.Dataset):
        def __init__(self, base, mult):
            self.base, self.mult = base, mult

        def __len__(self):
            return len(self.base) * self.mult

        def __getitem__(self, idx):
            return self.base[idx % len(self.base)]

    train_loader = DataLoader(Windowed(train_ds, args.windows_per_rollout),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ----- model / optimizer -----
    model = StateLeWM(n_obj=n_obj, latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_obj={n_obj}, params={n_params / 1e3:.1f}K, "
          f"train rollouts={len(train_ds)}, val rollouts={len(val_ds)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    norm.save(os.path.join(args.out, "normalizer.json"))
    log_path = os.path.join(args.results, "train_log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    logger.writerow(["epoch", "train_total", "train_dynamics", "train_recon_states",
                     "train_pred_recon_states", "train_sigreg", "val_dynamics",
                     "val_recon_states", "val_pred_recon_states",
                     "val_openloop_10", "val_openloop_25", "val_openloop_50",
                     "z_std", "lr", "elapsed_s"])

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep = {"total": 0.0, "dynamics": 0.0, "recon_states": 0.0,
              "pred_recon_states": 0.0, "sigreg": 0.0}
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            losses = model.compute_loss(out)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = batch["obj_states"].shape[0]
            for k in ep:
                ep[k] += losses[k].item() * bs
            n += bs
        sched.step()
        ep = {k: v / n for k, v in ep.items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val = evaluate(model, val_loader, device)
            score = val["dynamics"] + val.get("openloop_10", 0.0)
            logger.writerow([epoch, f"{ep['total']:.5f}", f"{ep['dynamics']:.5f}",
                             f"{ep['recon_states']:.5f}",
                             f"{ep['pred_recon_states']:.5f}", f"{ep['sigreg']:.5f}",
                             f"{val['dynamics']:.5f}", f"{val['recon_states']:.5f}",
                             f"{val['pred_recon_states']:.5f}",
                             f"{val.get('openloop_10', float('nan')):.5f}",
                             f"{val.get('openloop_25', float('nan')):.5f}",
                             f"{val.get('openloop_50', float('nan')):.5f}",
                             f"{val['z_std']:.4f}", f"{sched.get_last_lr()[0]:.2e}",
                             f"{time.time() - t0:.1f}"])
            log_file.flush()
            print(f"[{epoch:3d}/{args.epochs}] total {ep['total']:.4f} "
                  f"dyn {ep['dynamics']:.4f} recS {ep['recon_states']:.4f} "
                  f"predRecS {ep['pred_recon_states']:.4f} | "
                  f"val dyn {val['dynamics']:.4f} "
                  f"ol {val.get('openloop_10', float('nan')):.3f}/"
                  f"{val.get('openloop_25', float('nan')):.3f}/"
                  f"{val.get('openloop_50', float('nan')):.3f} "
                  f"z_std {val['z_std']:.3f}")
            if score < best_val:
                best_val = score
                torch.save({"model": model.state_dict(),
                            "n_obj": n_obj,
                            "latent_dim": args.latent_dim,
                            "stride": args.stride,
                            "normalizer": "normalizer.json",
                            "epoch": epoch,
                            "val": {k: float(v) for k, v in val.items()}},
                           os.path.join(args.out, "best.pt"))

    torch.save({"model": model.state_dict(), "n_obj": n_obj,
                "latent_dim": args.latent_dim,
                "stride": args.stride,
                "normalizer": "normalizer.json", "epoch": args.epochs},
               os.path.join(args.out, "last.pt"))
    log_file.close()
    print(f"Training done. Best val score {best_val:.5f}. "
          f"Checkpoints in {os.path.abspath(args.out)}, log {log_path}")


if __name__ == "__main__":
    main()
