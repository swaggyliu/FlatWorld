"""Procedural 2D motion clips inspired by LAFAN1 themes.

Self-contained analytic clips (no BVH download). Joint targets are radians
relative to the standing rest pose used by ``CharacterEnv``.

LAFAN1 themes mirrored here (side-view 2D):
  walk, run, sprint, dance, jump, crouch, fight, fall_getup
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Actuated joint order (must match CharacterEnv.JOINT_NAMES)
JOINT_NAMES = (
    "shoulder_L",
    "elbow_L",
    "shoulder_R",
    "elbow_R",
    "hip_L",
    "knee_L",
    "hip_R",
    "knee_R",
)
N_JOINTS = len(JOINT_NAMES)

# Geometry must match CharacterConfig defaults (character.py).
GROUND_Y = 0.05
TORSO_SIZE = (0.16, 0.26)  # (w, h)
HEAD_RADIUS = 0.055
ARM_SIZE = (0.14, 0.045)
THIGH_SIZE = (0.055, 0.15)
SHIN_SIZE = (0.05, 0.14)
# Standing torso y: shin sole sits just on the ground plane (matches CharacterEnv).
STAND_Y = float(GROUND_Y) + 0.5 * TORSO_SIZE[1] + THIGH_SIZE[1] + SHIN_SIZE[1] + 0.01
# Target sole height: a few mm above the ground line so feet read as planted
# (not submerged under the floor stroke, and not floating).
CONTACT_Y = float(GROUND_Y) + 0.008

# Body indices in fk_body_poses / CharacterEnv
_IDX_TORSO, _IDX_THIGH_L, _IDX_SHIN_L, _IDX_THIGH_R, _IDX_SHIN_R = 0, 6, 7, 8, 9


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def fk_body_poses(q: np.ndarray, root: np.ndarray) -> np.ndarray:
    """FK body centers/angles (N_bodies, 3): x, y, θ — same layout as CharacterEnv."""
    q = np.asarray(q, dtype=np.float64).reshape(N_JOINTS)
    cx, cy, th = float(root[0]), float(root[1]), float(root[2])
    tw, thh = TORSO_SIZE
    aw, _ah = ARM_SIZE
    _uw, uh = THIGH_SIZE
    _sw, sh = SHIN_SIZE
    hr = HEAD_RADIUS

    Rt = _rot(th)
    torso_xy = np.array([cx, cy], dtype=np.float64)
    head_xy = torso_xy + Rt @ np.array([0.0, 0.5 * thh + hr * 0.85])
    sh_l = torso_xy + Rt @ np.array([-0.5 * tw, 0.12])
    sh_r = torso_xy + Rt @ np.array([0.5 * tw, 0.12])
    hip_l = torso_xy + Rt @ np.array([-0.04, -0.5 * thh])
    hip_r = torso_xy + Rt @ np.array([0.04, -0.5 * thh])

    ual_ang = th + float(q[0])
    lal_ang = ual_ang + float(q[1])
    uar_ang = th + float(q[2])
    lar_ang = uar_ang + float(q[3])
    tl_ang = th + float(q[4])
    sl_ang = tl_ang + float(q[5])
    tr_ang = th + float(q[6])
    sr_ang = tr_ang + float(q[7])

    poses = [
        (torso_xy, th),
        (head_xy, th),
        (sh_l + _rot(ual_ang) @ np.array([-0.5 * aw, 0.0]), ual_ang),
        (sh_l + _rot(ual_ang) @ np.array([-aw, 0.0]) + _rot(lal_ang) @ np.array([-0.5 * aw, 0.0]), lal_ang),
        (sh_r + _rot(uar_ang) @ np.array([0.5 * aw, 0.0]), uar_ang),
        (sh_r + _rot(uar_ang) @ np.array([aw, 0.0]) + _rot(lar_ang) @ np.array([0.5 * aw, 0.0]), lar_ang),
        (hip_l + _rot(tl_ang) @ np.array([0.0, -0.5 * uh]), tl_ang),
        (hip_l + _rot(tl_ang) @ np.array([0.0, -uh]) + _rot(sl_ang) @ np.array([0.0, -0.5 * sh]), sl_ang),
        (hip_r + _rot(tr_ang) @ np.array([0.0, -0.5 * uh]), tr_ang),
        (hip_r + _rot(tr_ang) @ np.array([0.0, -uh]) + _rot(sr_ang) @ np.array([0.0, -0.5 * sh]), sr_ang),
    ]
    out = np.zeros((len(poses), 3), dtype=np.float64)
    for i, (xy, ang) in enumerate(poses):
        out[i, 0] = float(xy[0])
        out[i, 1] = float(xy[1])
        out[i, 2] = float(ang)
    return out


def _body_half_extents() -> list[tuple[str, float, float]]:
    tw, th = TORSO_SIZE
    aw, ah = ARM_SIZE
    uw, uh = THIGH_SIZE
    sw, sh = SHIN_SIZE
    return [
        ("box", 0.5 * tw, 0.5 * th),
        ("ball", HEAD_RADIUS, HEAD_RADIUS),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * aw, 0.5 * ah),
        ("box", 0.5 * uw, 0.5 * uh),
        ("box", 0.5 * sw, 0.5 * sh),
        ("box", 0.5 * uw, 0.5 * uh),
        ("box", 0.5 * sw, 0.5 * sh),
    ]


def _obb_min_y(x: float, y: float, th: float, hw: float, hh: float) -> float:
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    world = corners @ _rot(float(th)).T + np.array([float(x), float(y)])
    return float(world[:, 1].min())


def body_bbox_min_y(q: np.ndarray, root: np.ndarray) -> float:
    """Lowest world-Y of any body OBB corner (or ball bottom)."""
    poses = fk_body_poses(q, root)
    geoms = _body_half_extents()
    ymin = 1e9
    for (x, y, th), (kind, a, b) in zip(poses, geoms):
        if kind == "ball":
            ymin = min(ymin, float(y) - float(a))
        else:
            ymin = min(ymin, _obb_min_y(x, y, th, a, b))
    return float(ymin)


def foot_sole_heights(q: np.ndarray, root: np.ndarray) -> tuple[float, float]:
    """World-Y of L/R foot contact = lowest shin *box* corner (what you see).

    Ankle-tip planting left the thick shin rectangles visually through the
    ground whenever the leg angled; planting the shin OBB matches the overlay.
    Arms/torso are still ignored so a lean does not levitate the whole body.
    """
    poses = fk_body_poses(q, root)
    geoms = _body_half_extents()
    out = []
    for idx in (_IDX_SHIN_L, _IDX_SHIN_R):
        x, y, th = float(poses[idx, 0]), float(poses[idx, 1]), float(poses[idx, 2])
        _kind, hw, hh = geoms[idx]
        out.append(_obb_min_y(x, y, th, hw, hh))
    return out[0], out[1]


def support_contact_y(q: np.ndarray, root: np.ndarray) -> float:
    """Lowest Y of any body for ground rolls / falls (torso, limbs, head).

    Fall poses rest on the side; ignoring arms left fists through the floor in
    the overlay even after a torso/leg plant.
    """
    return body_bbox_min_y(q, root)


def _smoothstep01(a: float | np.ndarray) -> float | np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def _smooth_1d(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Reflect-padded Gaussian smooth (reduces non-physical jerk)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 or sigma <= 0:
        return x.astype(np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (k / float(sigma)) ** 2)
    ker /= ker.sum()
    pad = np.pad(x, radius, mode="reflect")
    return np.convolve(pad, ker, mode="valid")


def _smooth_traj(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.empty_like(arr)
    if arr.ndim == 1:
        return _smooth_1d(arr, sigma=sigma).astype(np.float32)
    for j in range(arr.shape[1]):
        out[:, j] = _smooth_1d(arr[:, j], sigma=sigma)
    return out.astype(np.float32)


def _penetration_deltas(
    q: np.ndarray,
    root: np.ndarray,
    *,
    contact: str,
    contact_y: float,
) -> np.ndarray:
    floor = float(contact_y)
    T = int(q.shape[0])
    deltas = np.zeros(T, dtype=np.float64)
    for i in range(T):
        if contact == "support":
            y = support_contact_y(q[i], root[i])
        else:
            fl, fr = foot_sole_heights(q[i], root[i])
            y = min(fl, fr)
        deltas[i] = max(0.0, floor - y)
    return deltas


def fit_root_to_contact(
    q: np.ndarray,
    root: np.ndarray,
    *,
    contact: str = "feet",
    aerial: np.ndarray | None = None,
    contact_y: float = CONTACT_Y,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Penetration-only root lift with temporally smooth lift amounts.

    Raw per-frame lifts create huge root jerks (esp. jump/crouch) that the
    world model cannot learn — J then oscillates. We:
      1) measure push-up deltas
      2) Gaussian-smooth those deltas (still ≥ 0 → never pull air down)
      3) apply, then a hard cleanup pass for residual penetration

    ``aerial`` kept for API compat (push-up-only already preserves flight).
    """
    del aerial
    q = np.asarray(q, dtype=np.float32)
    root = np.asarray(root, dtype=np.float32).copy()
    deltas = _penetration_deltas(q, root, contact=contact, contact_y=contact_y)
    if smooth_sigma > 0:
        deltas = np.maximum(0.0, _smooth_1d(deltas, sigma=smooth_sigma))
    root[:, 1] = root[:, 1] + deltas.astype(np.float32)
    # Hard cleanup (no smooth) so nothing remains below the floor.
    hard = _penetration_deltas(q, root, contact=contact, contact_y=contact_y)
    root[:, 1] = root[:, 1] + hard.astype(np.float32)
    return root


# Back-compat alias
def clamp_above_ground(
    q: np.ndarray,
    root: np.ndarray,
    ground_y: float = GROUND_Y,
    margin: float = 0.0,
    **kwargs,
) -> np.ndarray:
    return fit_root_to_contact(
        q, root, contact=kwargs.get("contact", "feet"),
        aerial=kwargs.get("aerial"),
        contact_y=float(ground_y) + float(margin),
    )


def clip_ground_stats(
    clip: "MotionClip",
    ground_y: float = GROUND_Y,
) -> dict[str, float]:
    """Per-clip foot / body contact stats."""
    foot_mins, body_mins = [], []
    for i in range(clip.n_frames):
        fl, fr = foot_sole_heights(clip.joint_q[i], clip.root_xy_theta[i])
        foot_mins.append(min(fl, fr))
        body_mins.append(body_bbox_min_y(clip.joint_q[i], clip.root_xy_theta[i]))
    feet = np.asarray(foot_mins, dtype=np.float64)
    body = np.asarray(body_mins, dtype=np.float64)
    return {
        "min_foot_y": float(feet.min()),
        "mean_foot_y": float(feet.mean()),
        "min_bbox_y": float(body.min()),
        "max_penetration": float(max(0.0, ground_y - feet.min())),
        "n_penetrating": int(np.sum(feet < ground_y - 1e-5)),
        "mean_float": float(np.mean(np.maximum(0.0, feet - ground_y))),
    }


@dataclass
class MotionClip:
    """Reference motion sampled at ``fps``."""

    fps: float
    duration: float
    joint_q: np.ndarray  # (T, 8)
    root_xy_theta: np.ndarray  # (T, 3) torso x, y, θ
    name: str = "clip"

    @property
    def n_frames(self) -> int:
        return int(self.joint_q.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    def sample(self, phase: float) -> tuple[np.ndarray, np.ndarray]:
        """phase in [0, 1) → (joint_q[8], root[3])."""
        t = float(phase) % 1.0
        idx = t * (self.n_frames - 1)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, self.n_frames - 1)
        a = float(idx - i0)
        q = (1.0 - a) * self.joint_q[i0] + a * self.joint_q[i1]
        r = (1.0 - a) * self.root_xy_theta[i0] + a * self.root_xy_theta[i1]
        return q.astype(np.float32), r.astype(np.float32)

    def frame(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        i = int(np.clip(i, 0, self.n_frames - 1))
        return self.joint_q[i].astype(np.float32), self.root_xy_theta[i].astype(np.float32)

    def goal_window(self, phase: float, n_future: int = 2) -> np.ndarray:
        """Flattened goal: current + future joint/root refs."""
        parts = []
        for k in range(n_future + 1):
            ph = (phase + k / max(self.n_frames - 1, 1)) % 1.0
            q, r = self.sample(ph)
            parts.append(q)
            parts.append(r)
        return np.concatenate(parts, axis=0).astype(np.float32)


# Back-compat alias
DanceClip = MotionClip


def _timeline(fps: float, duration: float) -> tuple[int, np.ndarray]:
    T = max(2, int(round(duration * fps)))
    t = np.linspace(0.0, 1.0, T, endpoint=False, dtype=np.float64)
    return T, t


def rest_root() -> np.ndarray:
    """Standing rest torso (x, y, θ) — matches CharacterEnv build pose."""
    return np.array([0.50, STAND_Y, 0.0], dtype=np.float32)


def rest_joints() -> np.ndarray:
    return np.zeros(N_JOINTS, dtype=np.float32)


def ease_from_rest(
    q: np.ndarray,
    root: np.ndarray,
    fps: float = 30.0,
    ease_sec: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Force frame 0 = standing rest; smoothstep into the motion.

    Avoids reference clips that begin mid-crouch / mid-fall (often look like
    ground penetration) so policy and ref both start from the same rest pose.
    """
    q = np.asarray(q, dtype=np.float32).copy()
    root = np.asarray(root, dtype=np.float32).copy()
    T = int(q.shape[0])
    n = max(2, int(round(ease_sec * fps)))
    n = min(n, max(2, T - 1))
    q0 = rest_joints()
    r0 = rest_root()
    # Keep a few pure-rest frames, then ease.
    hold = max(1, n // 4)
    for i in range(T):
        if i < hold:
            q[i] = q0
            root[i] = r0
            continue
        if i >= n:
            break
        a = (i - hold) / max(n - hold, 1)
        s = a * a * (3.0 - 2.0 * a)  # smoothstep
        q[i] = (1.0 - s) * q0 + s * q[i]
        root[i] = (1.0 - s) * r0 + s * root[i]
    q[0] = q0
    root[0] = r0
    # Soft clamp: never ask for torso below a near-standing floor early on
    # (true crouch/fall still happen after the ease window).
    return q, root


def _finalize(
    fps: float,
    duration: float,
    q: np.ndarray,
    root: np.ndarray,
    name: str,
    *,
    contact: str = "feet",
    aerial: np.ndarray | None = None,
) -> MotionClip:
    q, root = ease_from_rest(q, root, fps=fps)
    # Temporal smooth before contact so generators with piecewise phases
    # do not inject non-physical jerks into the world-model targets.
    q = _smooth_traj(q, sigma=1.0)
    root = _smooth_traj(root, sigma=1.2)
    q0, r0 = rest_joints(), rest_root()
    q[0], root[0] = q0, r0
    # Penetration-only lift with smooth lift curve (never pull air down).
    root = fit_root_to_contact(
        q, root, contact=contact, aerial=aerial, contact_y=CONTACT_Y,
        smooth_sigma=2.0,
    )
    root = _smooth_traj(root, sigma=0.8)
    root[0] = r0
    # Final hard un-penetrate after the last smooth.
    root = fit_root_to_contact(
        q, root, contact=contact, aerial=aerial, contact_y=CONTACT_Y,
        smooth_sigma=0.0,
    )
    return MotionClip(fps=fps, duration=duration, joint_q=q, root_xy_theta=root, name=name)


def make_walk_clip(fps: float = 30.0, duration: float = 2.0) -> MotionClip:
    """Side-view walk cycle (LAFAN Walk)."""
    T, u = _timeline(fps, duration)
    # one gait cycle over the clip
    g = 2.0 * np.pi * u
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    # arm swing opposite to legs
    q[:, 0] = 0.35 * np.sin(g + np.pi)          # shoulder L
    q[:, 1] = 0.55 + 0.15 * np.sin(g + np.pi)
    q[:, 2] = 0.35 * np.sin(g)                   # shoulder R
    q[:, 3] = 0.55 + 0.15 * np.sin(g)
    # hip / knee gait
    q[:, 4] = 0.35 * np.sin(g)                   # hip L
    q[:, 5] = 0.25 + 0.35 * np.maximum(0.0, np.sin(g))
    q[:, 6] = 0.35 * np.sin(g + np.pi)
    q[:, 7] = 0.25 + 0.35 * np.maximum(0.0, np.sin(g + np.pi))
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 0] = 0.50 + 0.06 * np.sin(g)
    root[:, 1] = STAND_Y + 0.015 * np.abs(np.sin(g))  # bounce kept; no pull-down
    root[:, 2] = 0.04 * np.sin(g)
    return _finalize(fps, duration, q, root, "walk")


def make_run_clip(fps: float = 30.0, duration: float = 1.5) -> MotionClip:
    """Jog / run cycle (LAFAN Run)."""
    T, u = _timeline(fps, duration)
    g = 2.0 * np.pi * u
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    q[:, 0] = 0.55 * np.sin(g + np.pi)
    q[:, 1] = 0.70 + 0.25 * np.sin(g + np.pi)
    q[:, 2] = 0.55 * np.sin(g)
    q[:, 3] = 0.70 + 0.25 * np.sin(g)
    q[:, 4] = 0.55 * np.sin(g)
    q[:, 5] = 0.35 + 0.55 * np.maximum(0.0, np.sin(g))
    q[:, 6] = 0.55 * np.sin(g + np.pi)
    q[:, 7] = 0.35 + 0.55 * np.maximum(0.0, np.sin(g + np.pi))
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 0] = 0.50 + 0.08 * np.sin(g)
    root[:, 1] = STAND_Y + 0.03 * np.abs(np.sin(g))
    root[:, 2] = 0.08 * np.sin(g) - 0.05  # forward lean
    return _finalize(fps, duration, q, root, "run")


def make_sprint_clip(fps: float = 30.0, duration: float = 1.2) -> MotionClip:
    """Sprint cycle (LAFAN Sprint) — lean / stride, amplitudes capped for trackability."""
    T, u = _timeline(fps, duration)
    g = 2.0 * np.pi * u
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    q[:, 0] = 0.60 * np.sin(g + np.pi)
    q[:, 1] = 0.75 + 0.22 * np.sin(g + np.pi)
    q[:, 2] = 0.60 * np.sin(g)
    q[:, 3] = 0.75 + 0.22 * np.sin(g)
    q[:, 4] = 0.55 * np.sin(g)
    q[:, 5] = 0.35 + 0.55 * np.maximum(0.0, np.sin(g))
    q[:, 6] = 0.55 * np.sin(g + np.pi)
    q[:, 7] = 0.35 + 0.55 * np.maximum(0.0, np.sin(g + np.pi))
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 0] = 0.50 + 0.08 * np.sin(g)
    root[:, 1] = STAND_Y + 0.03 * np.abs(np.sin(g))
    root[:, 2] = 0.08 * np.sin(g) - 0.10
    return _finalize(fps, duration, q, root, "sprint")


def make_dance_clip(fps: float = 30.0, duration: float = 4.0, seed: int = 0) -> MotionClip:
    """Free dance (LAFAN Dance)."""
    del seed
    T, t = _timeline(fps, duration)
    w1 = 2.0 * np.pi * 2.0
    w2 = 2.0 * np.pi * 1.0
    w3 = 2.0 * np.pi * 4.0
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    q[:, 0] = 0.50 * np.sin(w1 * t) + 0.28 * np.sin(w2 * t + 0.4)
    q[:, 1] = 0.85 + 0.32 * np.sin(w1 * t + 0.8)
    q[:, 2] = -0.50 * np.sin(w1 * t) - 0.28 * np.sin(w2 * t + 0.4)
    q[:, 3] = 0.85 + 0.32 * np.sin(w1 * t + 0.8 + np.pi)
    q[:, 4] = 0.18 * np.sin(w2 * t) + 0.10 * np.sin(w3 * t)
    q[:, 5] = 0.35 + 0.25 * np.maximum(0.0, np.sin(w2 * t))
    q[:, 6] = 0.18 * np.sin(w2 * t + np.pi) + 0.10 * np.sin(w3 * t + np.pi)
    q[:, 7] = 0.35 + 0.25 * np.maximum(0.0, np.sin(w2 * t + np.pi))
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 0] = 0.50 + 0.03 * np.sin(w2 * t)
    root[:, 1] = STAND_Y + 0.02 * np.abs(np.sin(w2 * t))
    root[:, 2] = 0.06 * np.sin(w2 * t + 0.3)
    return _finalize(fps, duration, q, root, "dance")


def make_jump_clip(fps: float = 30.0, duration: float = 2.0) -> MotionClip:
    """In-place jump with C0-smooth phases + parabolic flight (physical-ish).

    Old piecewise-linear root_y had |accel|~100+ and wrecked WM / J stability.
    """
    T, u = _timeline(fps, duration)
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    root = np.zeros((T, 3), dtype=np.float32)
    aerial = np.zeros(T, dtype=bool)
    # Phase knots (fraction of clip)
    t_crouch, t_extend, t_land = 0.28, 0.38, 0.72
    y_crouch = STAND_Y - 0.07
    y_takeoff = STAND_Y + 0.015
    h_peak = 0.14
    for i, p in enumerate(u):
        if p < t_crouch:
            a = float(_smoothstep01(p / t_crouch))
            crouch, air = a, 0.0
            root_y = STAND_Y + (y_crouch - STAND_Y) * a
        elif p < t_extend:
            a = float(_smoothstep01((p - t_crouch) / max(t_extend - t_crouch, 1e-6)))
            crouch, air = 1.0 - a, 0.35 * a
            root_y = y_crouch + (y_takeoff - y_crouch) * a
        elif p < t_land:
            a = (p - t_extend) / max(t_land - t_extend, 1e-6)  # 0..1
            crouch, air = 0.0, 1.0
            # Parabola: y(0)=y(1)=y_takeoff, peak at mid.
            root_y = y_takeoff + 4.0 * h_peak * a * (1.0 - a)
            aerial[i] = True
        else:
            a = float(_smoothstep01((p - t_land) / max(1.0 - t_land, 1e-6)))
            absorb = float(np.sin(np.pi * a))
            crouch, air = 0.40 * absorb, max(0.0, 1.0 - a) * 0.15
            root_y = y_takeoff + (STAND_Y - y_takeoff) * a - 0.04 * absorb
        q[i, 0] = 0.15 * crouch - 0.25 * air
        q[i, 1] = 0.55 + 0.35 * crouch
        q[i, 2] = 0.15 * crouch - 0.25 * air
        q[i, 3] = 0.55 + 0.35 * crouch
        q[i, 4] = 0.45 * crouch - 0.10 * air
        q[i, 5] = 0.18 + 0.95 * crouch
        q[i, 6] = 0.45 * crouch - 0.10 * air
        q[i, 7] = 0.18 + 0.95 * crouch
        root[i, 0] = 0.50
        root[i, 1] = root_y
        root[i, 2] = 0.04 * crouch
    return _finalize(fps, duration, q, root, "jump", aerial=aerial)


def make_crouch_clip(fps: float = 30.0, duration: float = 2.5) -> MotionClip:
    """Crouch / low locomotion — COM height tied to leg bend (less clamp jerk)."""
    T, u = _timeline(fps, duration)
    g = 2.0 * np.pi * u
    # Ease into crouch depth so the clip does not start mid-squat after rest.
    depth = 0.55 + 0.45 * _smoothstep01(np.clip((u - 0.08) / 0.20, 0.0, 1.0))
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    q[:, 0] = (0.20 * np.sin(g + np.pi) + 0.12) * depth
    q[:, 1] = 0.70 + 0.25 * depth + 0.08 * np.sin(g)
    q[:, 2] = (0.20 * np.sin(g) + 0.12) * depth
    q[:, 3] = 0.70 + 0.25 * depth + 0.08 * np.sin(g + np.pi)
    q[:, 4] = (0.55 + 0.12 * np.sin(g)) * depth
    q[:, 5] = (0.85 + 0.25 * np.maximum(0.0, np.sin(g))) * depth
    q[:, 6] = (0.55 + 0.12 * np.sin(g + np.pi)) * depth
    q[:, 7] = (0.85 + 0.25 * np.maximum(0.0, np.sin(g + np.pi))) * depth
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 0] = 0.50 + 0.03 * np.sin(g) * depth
    # Approximate COM drop from hip/knee bend (matches shin length scale).
    hip = 0.5 * (q[:, 4] + q[:, 6])
    knee = 0.5 * (q[:, 5] + q[:, 7])
    root[:, 1] = STAND_Y - 0.05 * hip - 0.06 * knee
    root[:, 2] = 0.08 * depth + 0.02 * np.sin(g)
    return _finalize(fps, duration, q, root, "crouch")


def make_fight_clip(fps: float = 30.0, duration: float = 2.0) -> MotionClip:
    """Punch combo (LAFAN Fight) — alternating jabs."""
    T, u = _timeline(fps, duration)
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    root = np.zeros((T, 3), dtype=np.float32)
    # two punches: L at 0.15-0.35, R at 0.55-0.75
    for i, p in enumerate(u):
        jab_l = float(np.exp(-((p - 0.25) / 0.08) ** 2))
        jab_r = float(np.exp(-((p - 0.65) / 0.08) ** 2))
        guard = 0.35
        q[i, 0] = guard - 0.9 * jab_l + 0.15 * jab_r   # extend L forward (neg in our frame ≈ outward)
        q[i, 1] = 1.1 - 0.7 * jab_l
        q[i, 2] = -guard + 0.9 * jab_r - 0.15 * jab_l
        q[i, 3] = 1.1 - 0.7 * jab_r
        q[i, 4] = 0.25 + 0.1 * jab_l
        q[i, 5] = 0.45
        q[i, 6] = 0.25 + 0.1 * jab_r
        q[i, 7] = 0.45
        root[i, 0] = 0.50 + 0.03 * jab_r - 0.03 * jab_l
        root[i, 1] = STAND_Y - 0.02
        root[i, 2] = 0.08 * jab_r - 0.08 * jab_l
    return _finalize(fps, duration, q, root, "fight")


def make_fall_getup_clip(fps: float = 30.0, duration: float = 3.0) -> MotionClip:
    """Fall and get up — smoothstep phase blends (no linear kink jerks)."""
    T, u = _timeline(fps, duration)
    q = np.zeros((T, N_JOINTS), dtype=np.float32)
    root = np.zeros((T, 3), dtype=np.float32)
    for i, p in enumerate(u):
        if p < 0.32:  # tipping
            a = float(_smoothstep01(p / 0.32))
            fall, down, up = a, 0.0, 0.0
        elif p < 0.55:  # on ground
            a = float(_smoothstep01((p - 0.32) / 0.23))
            fall, down, up = 1.0, a, 0.0
        else:  # get up
            a = float(_smoothstep01((p - 0.55) / 0.45))
            fall, down, up = 1.0 - a, 1.0 - a, a
        q[i, 0] = 0.35 * fall - 0.15 * up
        q[i, 1] = 0.45 + 0.50 * down
        q[i, 2] = 0.35 * fall - 0.15 * up
        q[i, 3] = 0.45 + 0.50 * down
        q[i, 4] = 0.65 * fall + 0.25 * down - 0.15 * up
        q[i, 5] = 0.25 + 1.0 * down
        q[i, 6] = 0.65 * fall + 0.25 * down - 0.15 * up
        q[i, 7] = 0.25 + 1.0 * down
        root[i, 0] = 0.50 + 0.06 * fall
        # Tip COM toward ground smoothly; get-up recovers standing height.
        root[i, 1] = STAND_Y * (1.0 - 0.45 * fall) + 0.10 * up
        root[i, 2] = 0.95 * fall * (1.0 - 0.75 * up)
    return _finalize(fps, duration, q, root, "fall_getup", contact="support")


CLIP_BUILDERS = {
    "walk": make_walk_clip,
    "run": make_run_clip,
    "sprint": make_sprint_clip,
    "dance": make_dance_clip,
    "jump": make_jump_clip,
    "crouch": make_crouch_clip,
    "fight": make_fight_clip,
    "fall_getup": make_fall_getup_clip,
}


def make_clip(name: str, fps: float = 30.0, **kwargs) -> MotionClip:
    key = name.lower().replace("-", "_")
    if key == "side_dance":
        key = "dance"
    if key not in CLIP_BUILDERS:
        raise KeyError(f"unknown clip {name!r}; choose from {sorted(CLIP_BUILDERS)}")
    return CLIP_BUILDERS[key](fps=fps, **kwargs)


def procedural_lafan2d_library(fps: float = 30.0) -> dict[str, MotionClip]:
    """Analytic LAFAN-like clips (pre-physics bake)."""
    return {
        "walk": make_walk_clip(fps=fps, duration=2.0),
        "run": make_run_clip(fps=fps, duration=1.5),
        "sprint": make_sprint_clip(fps=fps, duration=1.2),
        "dance": make_dance_clip(fps=fps, duration=4.0),
        "jump": make_jump_clip(fps=fps, duration=2.0),
        "crouch": make_crouch_clip(fps=fps, duration=2.5),
        "fight": make_fight_clip(fps=fps, duration=2.0),
        "fall_getup": make_fall_getup_clip(fps=fps, duration=3.0),
    }


def default_baked_dir() -> "Path":
    from pathlib import Path
    return Path(__file__).resolve().parent / "runs" / "lafan2d" / "baked_refs"


def save_clip(clip: MotionClip, path) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps=np.float32(clip.fps),
        duration=np.float32(clip.duration),
        joint_q=np.asarray(clip.joint_q, dtype=np.float32),
        root_xy_theta=np.asarray(clip.root_xy_theta, dtype=np.float32),
        name=np.asarray(clip.name),
    )


def load_clip(path) -> MotionClip:
    from pathlib import Path
    data = np.load(Path(path), allow_pickle=False)
    name = data["name"]
    name_s = str(name.item() if hasattr(name, "item") else name)
    return MotionClip(
        fps=float(data["fps"]),
        duration=float(data["duration"]),
        joint_q=np.asarray(data["joint_q"], dtype=np.float32),
        root_xy_theta=np.asarray(data["root_xy_theta"], dtype=np.float32),
        name=name_s,
    )


def make_lafan2d_library(
    fps: float = 30.0,
    *,
    use_baked: bool = True,
    baked_dir=None,
) -> dict[str, MotionClip]:
    """LAFAN-2D suite. Prefer physics-baked clips when present (LAFAN-like pipeline).

    Real LAFAN1 = mocap + retarget. Without mocap, the substitute is:
    procedural draft → PD-track in FlatWorld → save simulated trajectory as ref.
    """
    from pathlib import Path
    procedural = procedural_lafan2d_library(fps=fps)
    if not use_baked:
        return procedural
    bdir = Path(baked_dir) if baked_dir is not None else default_baked_dir()
    out: dict[str, MotionClip] = {}
    for name, proc in procedural.items():
        path = bdir / f"{name}.npz"
        if path.exists():
            clip = load_clip(path)
            # Re-stamp fps/name; keep baked joint/root.
            clip.name = name
            out[name] = clip
        else:
            out[name] = proc
    return out


class MotionLibrary:
    """Sample clips for multi-skill SuperMimic-style training."""

    def __init__(self, clips: dict[str, MotionClip] | None = None, fps: float = 30.0):
        self.clips = clips or make_lafan2d_library(fps=fps)
        self.names = list(self.clips.keys())

    def get(self, name: str) -> MotionClip:
        return self.clips[name]

    def sample_name(self, rng: np.random.Generator) -> str:
        return str(rng.choice(self.names))

    def sample_clip(self, rng: np.random.Generator) -> MotionClip:
        return self.clips[self.sample_name(rng)]
