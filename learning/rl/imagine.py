"""Frozen-WM encode / roll / Push-T success reward + task features."""

from __future__ import annotations

import numpy as np
import torch


EE, STEM, BAR = 0, 1, 2
FEAT_DIM = 13


class WMBundle:
    """Thin wrapper: normalize ↔ encode ↔ roll_step ↔ denorm pose."""

    def __init__(self, model, norm, device: str, stride: int = 5):
        self.model = model
        self.norm = norm
        self.device = device
        self.stride = int(stride)
        self.st = norm.to_tensors(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def nrm(self, key, x):
        m, s = self.st[key]
        return (x - m) / s

    def denorm_pose(self, pose_n):
        m, s = self.st["obj_states"]
        return pose_n * s[..., :3] + m[..., :3]

    def encode_batch(self, obs_list, use_tactile: bool = False):
        B = len(obs_list)
        device = self.device
        types = torch.stack([
            torch.as_tensor(o["obj_types"], device=device) for o in obs_list])
        states = torch.stack([
            torch.as_tensor(o["obj_states"], device=device, dtype=torch.float32)
            for o in obs_list])
        cfeat = torch.stack([
            torch.as_tensor(o["contact_feat"], device=device, dtype=torch.float32)
            for o in obs_list])
        cmask = torch.stack([
            torch.as_tensor(o["contact_mask"], device=device, dtype=torch.float32)
            for o in obs_list])
        tsum = torch.stack([
            torch.as_tensor(o["tactile_summary"], device=device, dtype=torch.float32)
            for o in obs_list])
        geom = torch.stack([
            torch.as_tensor(o["obj_geom"], device=device, dtype=torch.float32)
            for o in obs_list])
        rel = None
        if obs_list[0].get("rel_type") is not None:
            rel = torch.stack([
                torch.as_tensor(o["rel_type"], device=device, dtype=torch.long)
                for o in obs_list])

        states_n = self.nrm("obj_states", states)
        cfeat_n = self.nrm("contact_feat", cfeat)
        tsum_n = self.nrm("tactile_summary", tsum)
        geom_n = self.nrm("obj_geom", geom) if "obj_geom" in self.st else geom

        with torch.inference_mode():
            z = self.model.encode(
                types, states_n, cfeat_n, cmask, tsum_n,
                obj_geom=geom_n, use_tactile=use_tactile, rel_type=rel)
        return z, geom_n, rel, types

    def encode_one(self, obs, use_tactile: bool = False):
        return self.encode_batch([obs], use_tactile=use_tactile)

    def roll(self, z, a_raw, h, geom, prev_c, rel, types):
        a_n = self.nrm("actions", a_raw)
        return self.model.roll_step(
            z, a_n, h, geom, prev_c, rel_type=rel, obj_types=types)

    def pose(self, z):
        return self.denorm_pose(self.model.predictor.pose_head(z))


def goal_tensor(stem_goal, bar_goal, theta: float, device, batch: int = 1):
    g = torch.tensor(
        [float(stem_goal[0]), float(stem_goal[1]),
         float(bar_goal[0]), float(bar_goal[1]), float(theta)],
        device=device, dtype=torch.float32)
    return g.unsqueeze(0).expand(batch, -1).contiguous()


def pose_errors(pose, stem_goal, bar_goal, goal_theta: float):
    stem = pose[:, STEM, :2]
    bar = pose[:, BAR, :2]
    th = pose[:, STEM, 2]
    sg = stem.new_tensor([float(stem_goal[0]), float(stem_goal[1])])
    bg = stem.new_tensor([float(bar_goal[0]), float(bar_goal[1])])
    d_s = (stem - sg).norm(dim=-1)
    d_b = (bar - bg).norm(dim=-1)
    ang = torch.atan2(torch.sin(th - goal_theta), torch.cos(th - goal_theta)).abs()
    d_xy = d_s + d_b
    # Angle-prioritized potential early; xy dominates near goal.
    phi = -(d_xy + 1.5 * ang)
    return d_s, d_b, ang, d_xy, phi


def task_features(pose, stem_goal, bar_goal, goal_theta: float):
    """Goal-relative features for the policy (B, FEAT_DIM).

    Gives π an explicit geometric view of Push-T beyond the WM latent.
    """
    stem = pose[:, STEM, :2]
    bar = pose[:, BAR, :2]
    ee = pose[:, EE, :2]
    th = pose[:, STEM, 2]
    sg = stem.new_tensor([float(stem_goal[0]), float(stem_goal[1])])
    bg = stem.new_tensor([float(bar_goal[0]), float(bar_goal[1])])
    stem_err = stem - sg
    bar_err = bar - bg
    ang = th - goal_theta
    d_s = stem_err.norm(dim=-1, keepdim=True)
    d_b = bar_err.norm(dim=-1, keepdim=True)
    ang_abs = torch.atan2(torch.sin(ang), torch.cos(ang)).abs().unsqueeze(-1)
    ee_stem = ee - stem
    ee_bar = ee - bar
    # Rough "on contact face" proxy: EE below stem (negative local y-ish).
    ee_dist = ee_stem.norm(dim=-1, keepdim=True)
    feat = torch.cat([
        stem_err,                  # 2
        bar_err,                   # 2
        torch.sin(ang).unsqueeze(-1),
        torch.cos(ang).unsqueeze(-1),  # 2
        d_s, d_b, ang_abs,         # 3
        ee_stem,                   # 2
        ee_dist,                   # 1
        (d_s + d_b),               # 1  total xy
    ], dim=-1)
    assert feat.shape[-1] == FEAT_DIM, feat.shape
    return feat


def _reach(pose):
    ee = pose[:, EE, :2]
    stem = pose[:, STEM, :2]
    bar = pose[:, BAR, :2]
    return torch.minimum((ee - stem).norm(dim=-1), (ee - bar).norm(dim=-1))


def push_t_reward(pose, stem_goal, bar_goal, goal_theta: float,
                  a_raw=None, prev_pose=None, *,
                  tol: float = 0.12, ang_tol: float = 0.25,
                  ang_w: float = 0.5, reach_w: float = 0.5,
                  success_bonus: float = 1.0, **_unused):
    """Push-T reward.

    If ``prev_pose`` is given → potential progress
        Φ = d_goal + reach_w·d_reach
        r = (Φ_prev − Φ) + success_bonus·1[tol]
    else → absolute shaping ``r = −Φ + bonus`` (legacy).
    """
    del a_raw, _unused
    d_s, d_b, ang, d_xy, _phi = pose_errors(
        pose, stem_goal, bar_goal, goal_theta)
    d_reach = _reach(pose)
    phi = d_xy + ang_w * ang + reach_w * d_reach
    ok = (d_s < tol) & (d_b < tol) & (ang < ang_tol)
    if prev_pose is not None:
        _ps, _pb, pang, pxy, _ = pose_errors(
            prev_pose, stem_goal, bar_goal, goal_theta)
        phi_prev = pxy + ang_w * pang + reach_w * _reach(prev_pose)
        r = (phi_prev - phi) + success_bonus * ok.float()
    else:
        r = -phi + success_bonus * ok.float()
    info = {"d_xy": d_xy, "ang": ang, "d_reach": d_reach, "ok": ok, "phi": phi}
    return r, ok, info


def dense_reward(pose, stem_goal, bar_goal, goal_theta, a_raw=None,
                 prev_pose=None, **kwargs):
    r, ok, _ = push_t_reward(
        pose, stem_goal, bar_goal, goal_theta,
        a_raw=a_raw, prev_pose=prev_pose, **kwargs)
    return r, ok


def reward_from_obs(obs, stem_goal, bar_goal, goal_theta, prev_obs=None,
                    a_raw=None, **kwargs):
    """Numpy obs → torch reward (for real-env PPO)."""
    st = torch.as_tensor(obs["obj_states"], dtype=torch.float32).unsqueeze(0)
    pose = st[..., :3]
    prev = None
    if prev_obs is not None:
        prev = torch.as_tensor(
            prev_obs["obj_states"], dtype=torch.float32).unsqueeze(0)[..., :3]
    a = None
    if a_raw is not None:
        a = torch.as_tensor(a_raw, dtype=torch.float32).view(1, -1)
    r, ok, info = push_t_reward(
        pose, stem_goal, bar_goal, goal_theta, a_raw=a, prev_pose=prev, **kwargs)
    return float(r[0]), bool(ok[0]), {k: (v[0].item() if torch.is_tensor(v) else v)
                                      for k, v in info.items()}
