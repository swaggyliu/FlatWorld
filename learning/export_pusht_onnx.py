"""Export fused Push-T inference graph (WM encode + PPO actor) to ONNX.

Usage:
  python -m learning.export_pusht_onnx \\
    --wm learning/checkpoints/push_t \\
    --policy learning/checkpoints/push_t_wm_rsl/best.pt \\
    --out learning/checkpoints/push_t_wm_rsl/pusht_infer.onnx

Writes ``pusht_infer.onnx`` + ``pusht_infer.json`` (norm stats + defaults).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from learning.models.relations import empty_rel
from learning.onnx_pusht import FEAT_DIM, load_torch_actor
from learning.rl.imagine import WMBundle, task_features
from learning.tasks.push_to_goal import load_ensemble


class FusedPushTInfer(nn.Module):
    """Encoder (tactile off) + rsl-rl ONNX actor MLP. ``feat`` is an input."""

    def __init__(self, wm_model: nn.Module, actor_onnx: nn.Module):
        super().__init__()
        self.wm = wm_model
        self.actor = actor_onnx

    def forward(
        self,
        obj_types,
        obj_states_n,
        contact_feat_n,
        contact_mask,
        tactile_summary_n,
        obj_geom_n,
        rel_type,
        feat,
    ):
        z = self.wm.encode(
            obj_types, obj_states_n, contact_feat_n, contact_mask,
            tactile_summary_n, obj_geom=obj_geom_n, use_tactile=False,
            rel_type=rel_type)
        x = torch.cat([z.reshape(z.shape[0], -1), feat], dim=-1)
        return self.actor(x)


def _norm_entry(stats: dict, key: str) -> dict | None:
    if key not in stats:
        return None
    s = stats[key]
    return {
        "mean": np.asarray(s["mean"], dtype=np.float32).tolist(),
        "std": np.asarray(s["std"], dtype=np.float32).tolist(),
    }


def export(
    wm_path: str,
    policy_path: str,
    out_path: str,
    hold: int = 5,
    contact_k: int = 8,
    opset: int = 17,
    device: str = "cpu",
) -> str:
    from rsl_rl.models.mlp_model import _OnnxMLPModel

    models, norm, stride = load_ensemble(wm_path, device)
    wm = WMBundle(models[0], norm, device, stride=stride)
    actor = load_torch_actor(policy_path, device)
    actor_onnx = _OnnxMLPModel(actor, verbose=False)

    n_obj = int(wm.model.n_obj)
    latent = int(wm.model.latent_dim)
    obs_dim = n_obj * latent + FEAT_DIM
    if int(actor_onnx.input_size) != obs_dim:
        raise RuntimeError(
            f"actor obs_dim {actor_onnx.input_size} != {obs_dim} "
            f"(n_obj={n_obj}, latent={latent}, feat={FEAT_DIM})")

    default_types = np.asarray([0, 1, 1], dtype=np.int64)
    default_geom = np.asarray(
        [[0.05, 0.05], [0.04, 0.14], [0.16, 0.04]], dtype=np.float32)
    default_rel = empty_rel(n_obj)

    # Representative non-zero trace inputs (fixed Push-T shapes).
    types = torch.as_tensor(default_types, dtype=torch.long).unsqueeze(0)
    states = torch.randn(1, n_obj, 6, dtype=torch.float32) * 0.05
    cfeat = torch.zeros(1, contact_k, 7, dtype=torch.float32)
    cmask = torch.zeros(1, contact_k, dtype=torch.float32)
    tsum = torch.zeros(1, 4, dtype=torch.float32)
    geom = torch.as_tensor(default_geom, dtype=torch.float32).unsqueeze(0)
    rel = torch.as_tensor(default_rel, dtype=torch.long).unsqueeze(0)
    st = wm.st
    states_n = (states - st["obj_states"][0]) / st["obj_states"][1]
    cfeat_n = (cfeat - st["contact_feat"][0]) / st["contact_feat"][1]
    tsum_n = (tsum - st["tactile_summary"][0]) / st["tactile_summary"][1]
    geom_n = (geom - st["obj_geom"][0]) / st["obj_geom"][1] if "obj_geom" in st else geom
    feat = task_features(states[..., :3], (0.0, 0.0), (0.2, 0.0), 0.0)

    fused = FusedPushTInfer(wm.model, actor_onnx).eval()
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        fused,
        (types, states_n, cfeat_n, cmask, tsum_n, geom_n, rel, feat),
        out_path,
        input_names=[
            "obj_types", "obj_states_n", "contact_feat_n", "contact_mask",
            "tactile_summary_n", "obj_geom_n", "rel_type", "feat",
        ],
        output_names=["actions"],
        dynamo=False,
        opset_version=int(opset),
    )

    hold_i = max(int(hold), int(stride), 1)
    meta = {
        "format": "pusht_infer_v1",
        "onnx": os.path.basename(out_path),
        "wm": os.path.abspath(wm_path),
        "policy": os.path.abspath(policy_path),
        "hold": hold_i,
        "stride": int(stride),
        "n_obj": n_obj,
        "latent_dim": latent,
        "feat_dim": FEAT_DIM,
        "action_dim": 2,
        "contact_k": int(contact_k),
        "obs_dim": obs_dim,
        "default_types": default_types.tolist(),
        "default_geom": default_geom.tolist(),
        "default_rel": np.asarray(default_rel, dtype=np.int64).tolist(),
        "norm": {
            k: v for k, v in {
                "obj_states": _norm_entry(norm.stats, "obj_states"),
                "contact_feat": _norm_entry(norm.stats, "contact_feat"),
                "tactile_summary": _norm_entry(norm.stats, "tactile_summary"),
                "obj_geom": _norm_entry(norm.stats, "obj_geom"),
            }.items() if v is not None
        },
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    print(f"wrote {meta_path}")
    return out_path


def _verify(wm_path: str, policy_path: str, onnx_path: str, device: str = "cpu"):
    from learning.onnx_pusht import OnnxPushTPolicy, TorchPushTPolicy

    torch_pol = TorchPushTPolicy(wm_path, policy_path, device=device)
    onnx_pol = OnnxPushTPolicy(onnx_path)
    rng = np.random.default_rng(0)
    n_obj = 3
    max_abs = 0.0
    for i in range(8):
        states = rng.normal(0, 0.05, size=(n_obj, 6)).astype(np.float32)
        cur = {
            "obj_states": states,
            "obj_types": np.asarray([0, 1, 1], dtype=np.int64),
            "obj_geom": np.asarray(
                [[0.05, 0.05], [0.04, 0.14], [0.16, 0.04]], dtype=np.float32),
            "contact_feat": np.zeros((8, 7), np.float32),
            "contact_mask": np.zeros((8,), np.float32),
            "tactile_summary": np.zeros((4,), np.float32),
            "rel_type": empty_rel(n_obj),
        }
        sg, bg, th = (0.1, -0.05), (0.25, 0.0), 0.15
        a_t = torch_pol.act(cur, sg, bg, th)
        a_o = onnx_pol.act(cur, sg, bg, th)
        err = float(np.max(np.abs(a_t - a_o)))
        max_abs = max(max_abs, err)
        print(f"  sample {i}: torch={a_t} onnx={a_o} max|d|={err:.3e}")
    print(f"verify max|torch-onnx|={max_abs:.3e}")
    if max_abs > 1e-3:
        raise SystemExit(f"ONNX mismatch too large: {max_abs}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wm", default="learning/checkpoints/push_t")
    p.add_argument("--policy", default="learning/checkpoints/push_t_wm_rsl/best.pt")
    p.add_argument(
        "--out", default="learning/checkpoints/push_t_wm_rsl/pusht_infer.onnx")
    p.add_argument("--hold", type=int, default=5)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--skip-verify", action="store_true")
    args = p.parse_args()
    path = export(
        args.wm, args.policy, args.out, hold=args.hold, opset=args.opset)
    if not args.skip_verify:
        _verify(args.wm, args.policy, path)


if __name__ == "__main__":
    main()
