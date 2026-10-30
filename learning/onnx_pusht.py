"""CPU-friendly Push-T inference: ONNX Runtime (WM encode + PPO actor).

Game path builds goal features in NumPy, feeds normalized encoder tensors into
``pusht_infer.onnx``, and returns deterministic actions in ``[-1, 1]``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

FEAT_DIM = 13
EE, STEM, BAR = 0, 1, 2


def _task_features_np(
    pose: np.ndarray, stem_goal, bar_goal, goal_theta: float,
) -> np.ndarray:
    """Match ``learning.rl.imagine.task_features`` (B=1, FEAT_DIM)."""
    stem = pose[STEM, :2]
    bar = pose[BAR, :2]
    ee = pose[EE, :2]
    th = float(pose[STEM, 2])
    sg = np.asarray(stem_goal, dtype=np.float32)
    bg = np.asarray(bar_goal, dtype=np.float32)
    stem_err = stem - sg
    bar_err = bar - bg
    ang = th - float(goal_theta)
    d_s = float(np.linalg.norm(stem_err))
    d_b = float(np.linalg.norm(bar_err))
    ang_abs = float(abs(np.arctan2(np.sin(ang), np.cos(ang))))
    ee_stem = ee - stem
    ee_dist = float(np.linalg.norm(ee_stem))
    feat = np.array([
        stem_err[0], stem_err[1],
        bar_err[0], bar_err[1],
        np.sin(ang), np.cos(ang),
        d_s, d_b, ang_abs,
        ee_stem[0], ee_stem[1],
        ee_dist,
        d_s + d_b,
    ], dtype=np.float32)
    assert feat.shape[0] == FEAT_DIM
    return feat[None, :]


def _nrm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def resolve_infer_onnx(policy_path: str) -> tuple[str, str] | None:
    """Return ``(onnx_path, meta_path)`` if a fused export sits next to the .pt."""
    if policy_path.endswith(".onnx"):
        meta = os.path.splitext(policy_path)[0] + ".json"
        if os.path.isfile(policy_path) and os.path.isfile(meta):
            return policy_path, meta
        return None
    root = policy_path if os.path.isdir(policy_path) else os.path.dirname(policy_path)
    if not root:
        root = "."
    # Prefer fused graph next to the PPO checkpoint.
    for name in ("pusht_infer.onnx", "policy_infer.onnx"):
        onnx = os.path.join(root, name)
        meta = os.path.splitext(onnx)[0] + ".json"
        if os.path.isfile(onnx) and os.path.isfile(meta):
            return onnx, meta
    return None


class OnnxPushTPolicy:
    """ORT session for fused WM-encode + actor. No PyTorch at inference time."""

    def __init__(self, onnx_path: str, meta_path: str | None = None):
        import onnxruntime as ort

        if meta_path is None:
            meta_path = os.path.splitext(onnx_path)[0] + ".json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != "pusht_infer_v1":
            raise ValueError(f"unsupported ONNX meta format: {meta.get('format')}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.meta = meta
        self.hold = int(meta.get("hold", meta.get("stride", 5)))
        self.n_obj = int(meta["n_obj"])
        self.contact_k = int(meta.get("contact_k", 8))
        self.feat_dim = int(meta.get("feat_dim", FEAT_DIM))
        self._types = np.asarray(meta["default_types"], dtype=np.int64)
        self._geom = np.asarray(meta["default_geom"], dtype=np.float32)
        self._rel = np.asarray(meta["default_rel"], dtype=np.int64)
        n = meta["norm"]
        self._st_m = {k: np.asarray(v["mean"], dtype=np.float32) for k, v in n.items()}
        self._st_s = {k: np.asarray(v["std"], dtype=np.float32) for k, v in n.items()}

    def act(self, cur: dict, stem_goal, bar_goal, goal_theta: float) -> np.ndarray:
        """Deterministic unit action in ``[-1, 1]^2`` (caller scales by force_max)."""
        states = np.asarray(cur["obj_states"], dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != self.n_obj:
            raise ValueError(f"expected obj_states ({self.n_obj}, 6), got {states.shape}")

        types = np.asarray(cur.get("obj_types", self._types), dtype=np.int64)
        geom = np.asarray(cur.get("obj_geom", self._geom), dtype=np.float32)
        rel = cur.get("rel_type", self._rel)
        if rel is None:
            rel = self._rel
        rel = np.asarray(rel, dtype=np.int64)

        k = self.contact_k
        # Game / WM-obs path always drops tactile; keep the exported contact_k.
        cfeat = np.zeros((k, 7), dtype=np.float32)
        cmask = np.zeros((k,), dtype=np.float32)
        tsum = np.zeros((4,), dtype=np.float32)

        states_n = _nrm(states, self._st_m["obj_states"], self._st_s["obj_states"])
        cfeat_n = _nrm(cfeat, self._st_m["contact_feat"], self._st_s["contact_feat"])
        tsum_n = _nrm(tsum, self._st_m["tactile_summary"], self._st_s["tactile_summary"])
        if "obj_geom" in self._st_m:
            geom_n = _nrm(geom, self._st_m["obj_geom"], self._st_s["obj_geom"])
        else:
            geom_n = geom

        feat = _task_features_np(states[:, :3], stem_goal, bar_goal, goal_theta)
        feeds = {
            "obj_types": types.reshape(1, self.n_obj),
            "obj_states_n": states_n.reshape(1, self.n_obj, -1),
            "contact_feat_n": cfeat_n.reshape(1, k, -1),
            "contact_mask": cmask.reshape(1, k),
            "tactile_summary_n": tsum_n.reshape(1, -1),
            "obj_geom_n": geom_n.reshape(1, self.n_obj, -1),
            "rel_type": rel.reshape(1, self.n_obj, self.n_obj),
            "feat": feat.astype(np.float32),
        }
        out = self.sess.run(["actions"], feeds)[0]
        a = np.asarray(out, dtype=np.float32).reshape(-1)
        return np.clip(a, -1.0, 1.0).astype(np.float32)


def load_torch_actor(policy_path: str, device: str = "cpu"):
    """Build rsl-rl MLPModel from a PPO ``.pt`` without OnPolicyRunner."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.models.mlp_model import MLPModel

    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    sd = ckpt["actor_state_dict"]
    obs_dim = int(sd["mlp.0.weight"].shape[1])
    obs = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    actor = MLPModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=2,
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "rsl_rl.modules.GaussianDistribution",
            "init_std": 0.8,
            "std_type": "scalar",
            "learn_std": True,
            "std_range": [0.05, 1.5],
        },
    )
    actor.load_state_dict(sd, strict=True)
    actor.eval()
    actor.to(device)
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor


class TorchPushTPolicy:
    """Thin torch WM+actor path (no rsl-rl runner / train env)."""

    def __init__(self, wm_path: str, policy_path: str, device: str = "cpu", hold: int = 5):
        import torch
        from learning.tasks.push_to_goal import load_ensemble
        from learning.rl.imagine import WMBundle, task_features

        models, norm, stride = load_ensemble(wm_path, device)
        self.wm = WMBundle(models[0], norm, device, stride=stride)
        self.policy = load_torch_actor(policy_path, device)
        self.hold = max(int(hold), int(stride), 1)
        self.device = device
        self._task_features = task_features
        self._torch = torch

    def act(self, cur: dict, stem_goal, bar_goal, goal_theta: float) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            z, _, _, _ = self.wm.encode_one(cur, use_tactile=False)
            pose = torch.as_tensor(
                cur["obj_states"][:, :3], dtype=torch.float32,
                device=self.wm.device).unsqueeze(0)
            feat = self._task_features(pose, stem_goal, bar_goal, goal_theta)
            x = torch.cat([z.reshape(1, -1), feat], dim=-1)
            from tensordict import TensorDict
            td = TensorDict({"policy": x}, batch_size=[1])
            a = self.policy(td, stochastic_output=False)
            if isinstance(a, tuple):
                a = a[0]
            a = a.detach().float().cpu().numpy().reshape(-1)
        return np.clip(a, -1.0, 1.0).astype(np.float32)


def load_pusht_policy(
    wm_path: str,
    policy_path: str,
    device: str = "cpu",
    hold: int = 5,
    prefer_onnx: bool = True,
) -> tuple[Any, str]:
    """Return ``(policy, backend_name)`` with ``policy.act`` / ``policy.hold``."""
    if prefer_onnx:
        resolved = resolve_infer_onnx(policy_path)
        if resolved is not None:
            onnx_path, meta_path = resolved
            return OnnxPushTPolicy(onnx_path, meta_path), "onnx"
    return TorchPushTPolicy(wm_path, policy_path, device=device, hold=hold), "torch"
