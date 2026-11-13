"""Train/evaluate ContactImpulseNet on recorded FlatWorld PGS samples.

Example:
  python -m learning.contact_dynamics.train --data learning/data/contact_dynamics \
      --out learning/checkpoints/contact_dynamics --epochs 40
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import ContactImpulseNet
from .schema import FEATURE_NAMES, SCHEMA_VERSION


def load_samples(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(files) < 2:
        raise ValueError("need at least two rollout npz files for rollout-level train/validation split")
    xs, ys, modes, groups = [], [], [], []
    for file_index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            for prefix in ("pair", "ground"):
                fk, yk, mk, maskk = (
                    "contact_" + prefix + "_features",
                    "contact_" + prefix + "_impulses",
                    "contact_" + prefix + "_modes",
                    "contact_" + prefix + "_mask",
                )
                if not all(k in data.files for k in (fk, yk, mk, maskk)):
                    continue
                mask = data[maskk].reshape(-1) > 0.5
                if not mask.any():
                    continue
                x = data[fk].reshape(-1, len(FEATURE_NAMES))[mask].astype(np.float32)
                y = data[yk].reshape(-1, 2)[mask].astype(np.float32)
                m = data[mk].reshape(-1)[mask].astype(np.int64)
                finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
                xs.append(x[finite])
                ys.append(y[finite])
                modes.append(m[finite])
                groups.extend([file_index] * int(finite.sum()))
    if not xs:
        raise ValueError("no contact samples found; recollect with --record-contact-transitions")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(modes), np.asarray(groups)


def main():
    parser = argparse.ArgumentParser(description="Train Contact Dynamics AI impulse predictor")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="learning/checkpoints/contact_dynamics")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x, y, mode, groups = load_samples(args.data)
    unique = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(0.15 * len(unique))))
    val_groups = set(unique[:n_val].tolist())
    val_mask = np.asarray([g in val_groups for g in groups], dtype=bool)
    train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError("rollout split produced an empty train or validation set")

    mean = x[train_mask].mean(axis=0)
    std = np.maximum(x[train_mask].std(axis=0), 1e-6)
    x_norm = ((x - mean) / std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor_set(mask):
        return TensorDataset(
            torch.from_numpy(x_norm[mask]),
            torch.from_numpy(x[mask, 3]),
            torch.from_numpy(y[mask]),
            torch.from_numpy(mode[mask]),
        )

    train_loader = DataLoader(tensor_set(train_mask), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tensor_set(val_mask), batch_size=args.batch_size, shuffle=False)
    model = ContactImpulseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    best_metrics = {}
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, mu, yb, mb in train_loader:
            xb, mu, yb, mb = xb.to(device), mu.to(device), yb.to(device), mb.to(device)
            pred, logits = model(xb, mu)
            loss = F.smooth_l1_loss(pred, yb) + 0.1 * F.cross_entropy(logits, mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)

        model.eval()
        pred_all, target_all, mode_pred, mode_true = [], [], [], []
        with torch.no_grad():
            for xb, mu, yb, mb in val_loader:
                pred, logits = model(xb.to(device), mu.to(device))
                pred_all.append(pred.cpu().numpy())
                target_all.append(yb.numpy())
                mode_pred.append(logits.argmax(-1).cpu().numpy())
                mode_true.append(mb.numpy())
        pred_all = np.concatenate(pred_all)
        target_all = np.concatenate(target_all)
        mode_pred = np.concatenate(mode_pred)
        mode_true = np.concatenate(mode_true)
        impulse_mae = float(np.abs(pred_all - target_all).mean())
        normal_mae = float(np.abs(pred_all[:, 0] - target_all[:, 0]).mean())
        tangent_mae = float(np.abs(pred_all[:, 1] - target_all[:, 1]).mean())
        mode_acc = float((mode_pred == mode_true).mean())
        score = impulse_mae + 0.1 * (1.0 - mode_acc)
        print(
            f"epoch {epoch:03d} train_loss={total / max(count, 1):.6f} "
            f"val_impulse_mae={impulse_mae:.6f} normal_mae={normal_mae:.6f} "
            f"tangent_mae={tangent_mae:.6f} mode_acc={mode_acc:.4f}"
        )
        if score < best:
            best = score
            best_metrics = {
                "impulse_mae": impulse_mae,
                "normal_impulse_mae": normal_mae,
                "tangent_impulse_mae": tangent_mae,
                "mode_accuracy": mode_acc,
            }
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": FEATURE_NAMES,
                    "metrics": best_metrics,
                },
                os.path.join(args.out, "best.pt"),
            )

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "num_samples": int(len(x)),
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "metrics": best_metrics,
                "note": "Validation measures imitation of FlatWorld PGS labels, not physical accuracy against external ground truth.",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
