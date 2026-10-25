"""2D humanoid character built on FlatWorld rigid bodies + PD joints.

Topology (side view):
  head ──weld── torso
                 ├── shoulder_L ── elbow_L
                 ├── shoulder_R ── elbow_R
                 ├── hip_L ── knee_L
                 └── hip_R ── knee_R

Actions are absolute joint-angle PD targets (``use_pd=2``), matching the
ReacherEnv pattern already used in ``learning/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from flatworld import (
    BallRigid,
    BoxRigid,
    ExplicitLoop,
    Gravity,
    GroundDomain,
    RigidBodyDomain,
)
from flatworld.joints import RevoluteJoint, WeldJoint
from flatworld.wp_init import ensure_warp
from rigidmanager import _patch_array

from .motion import JOINT_NAMES, N_JOINTS, MotionClip, make_dance_clip


BODY_NAMES = (
    "torso",
    "head",
    "upper_arm_L",
    "lower_arm_L",
    "upper_arm_R",
    "lower_arm_R",
    "thigh_L",
    "shin_L",
    "thigh_R",
    "shin_R",
)
N_BODIES = len(BODY_NAMES)


@dataclass
class CharacterConfig:
    control_hz: float = 30.0
    gravity: float = 9.8
    ground_y: float = 0.05
    friction: float = 0.9
    torso_size: tuple[float, float] = (0.16, 0.26)   # (width, height)
    head_radius: float = 0.055
    arm_size: tuple[float, float] = (0.14, 0.045)    # horizontal limbs
    thigh_size: tuple[float, float] = (0.055, 0.15)  # vertical limbs (w, h)
    shin_size: tuple[float, float] = (0.05, 0.14)
    mass_scale: float = 1.0
    kp: float = 80.0
    kd: float = 2.5
    effort_limit: float = 12.0
    velocity_limit: float = 20.0
    # Soft root assist toward reference (helps early learning; can be 0)
    root_kp: float = 90.0
    root_kd: float = 12.0
    fall_height: float = 0.32
    max_episode_steps: int = 120
    n_goal_future: int = 2
    action_noise_std: float = 0.05
    obs_noise_std: float = 0.01


class CharacterEnv:
    """Single FlatWorld humanoid for SuperMimic-style motion tracking."""

    JOINT_NAMES = JOINT_NAMES
    BODY_NAMES = BODY_NAMES

    def __init__(self, cfg: CharacterConfig | None = None, clip: MotionClip | None = None):
        ensure_warp()
        self.cfg = cfg or CharacterConfig()
        self.clip = clip or make_dance_clip(fps=self.cfg.control_hz)
        self.dt = 1.0 / float(self.cfg.control_hz)
        self.n_joints = N_JOINTS
        self.n_bodies = N_BODIES
        self.goal_dim = (self.cfg.n_goal_future + 1) * (N_JOINTS + 3)
        self.proprio_dim = N_JOINTS * 2 + 6  # q, qd, root(6)
        self.obs_dim = self.proprio_dim + self.goal_dim
        self.action_dim = N_JOINTS

        self._build()
        self.phase = 0.0
        self.step_count = 0
        self._rng = np.random.default_rng(0)
        self.clip_name = getattr(self.clip, "name", "clip")

    def set_clip(self, clip: MotionClip):
        """Swap reference motion (multi-skill SuperMimic library)."""
        self.clip = clip
        self.clip_name = getattr(clip, "name", "clip")

    # ------------------------------------------------------------------ build
    def _build(self):
        c = self.cfg
        g = Gravity([0.0, -float(c.gravity)])
        m = float(c.mass_scale)
        tw, th = c.torso_size
        aw, ah = c.arm_size
        uw, uh = c.thigh_size
        sw, sh = c.shin_size

        # Standing rest pose: feet just on the ground plane.
        # shin_bottom = cy - 0.5*th - uh - sh  == ground_y  → solve for cy
        cx = 0.50
        cy = float(c.ground_y) + 0.5 * th + uh + sh + 0.01
        torso = BoxRigid(2, [cx, cy], [tw, th], [0.0], 4.0 * m)
        head = BallRigid(2, [cx, cy + 0.5 * th + c.head_radius * 0.85], c.head_radius, 0.8 * m)

        # Arms slightly out
        ual = BoxRigid(2, [cx - 0.5 * tw - 0.5 * aw, cy + 0.12], [aw, ah], [0.0], 0.6 * m)
        lal = BoxRigid(2, [cx - 0.5 * tw - 1.4 * aw, cy + 0.12], [aw, ah], [0.0], 0.45 * m)
        uar = BoxRigid(2, [cx + 0.5 * tw + 0.5 * aw, cy + 0.12], [aw, ah], [0.0], 0.6 * m)
        lar = BoxRigid(2, [cx + 0.5 * tw + 1.4 * aw, cy + 0.12], [aw, ah], [0.0], 0.45 * m)

        # Legs under torso
        tl = BoxRigid(2, [cx - 0.04, cy - 0.5 * th - 0.5 * uh], [uw, uh], [0.0], 1.2 * m)
        sl = BoxRigid(2, [cx - 0.04, cy - 0.5 * th - uh - 0.5 * sh], [sw, sh], [0.0], 0.9 * m)
        tr = BoxRigid(2, [cx + 0.04, cy - 0.5 * th - 0.5 * uh], [uw, uh], [0.0], 1.2 * m)
        sr = BoxRigid(2, [cx + 0.04, cy - 0.5 * th - uh - 0.5 * sh], [sw, sh], [0.0], 0.9 * m)
        self.standing_height = cy

        friction = float(c.friction)
        bodies = [torso, head, ual, lal, uar, lar, tl, sl, tr, sr]
        # Ground contact only — skip rigid–rigid self-collisions between links.
        self.domains = [
            RigidBodyDomain(
                b, bcs=[g], friction=friction,
                considerContact=False, considerGroundContact=True,
            )
            for b in bodies
        ]
        self.rigids = bodies
        ground = GroundDomain(2, [0.0, c.ground_y], [0.0, 1.0], [], True)

        # Anchors at rest pose
        a_head = [cx, cy + 0.5 * th]
        a_sh_l = [cx - 0.5 * tw, cy + 0.12]
        a_el_l = [cx - 0.5 * tw - aw, cy + 0.12]
        a_sh_r = [cx + 0.5 * tw, cy + 0.12]
        a_el_r = [cx + 0.5 * tw + aw, cy + 0.12]
        a_hip_l = [cx - 0.04, cy - 0.5 * th]
        a_kn_l = [cx - 0.04, cy - 0.5 * th - uh]
        a_hip_r = [cx + 0.04, cy - 0.5 * th]
        a_kn_r = [cx + 0.04, cy - 0.5 * th - uh]

        kp, kd = float(c.kp), float(c.kd)
        el, vl = float(c.effort_limit), float(c.velocity_limit)

        def rev(i, j, anc, lim=None):
            return RevoluteJoint(
                i, j, anc, axis=[0, 0], bcs=[], stiff=kp, damping=kd,
                velocity_limit=vl, effort_limit=el,
                limits=lim if lim is not None else [-2.5, 2.5],
                motor_target_position=0.0,
            )

        weld_head = WeldJoint(0, 1, a_head, bcs=[])
        # Actuated joint order matches JOINT_NAMES / motion.py
        self.actuated = [
            rev(0, 2, a_sh_l, [-2.2, 2.2]),   # shoulder_L
            rev(2, 3, a_el_l, [0.0, 2.4]),    # elbow_L
            rev(0, 4, a_sh_r, [-2.2, 2.2]),   # shoulder_R
            rev(4, 5, a_el_r, [0.0, 2.4]),    # elbow_R
            rev(0, 6, a_hip_l, [-1.5, 1.5]),  # hip_L
            rev(6, 7, a_kn_l, [0.0, 2.2]),    # knee_L
            rev(0, 8, a_hip_r, [-1.5, 1.5]),  # hip_R
            rev(8, 9, a_kn_r, [0.0, 2.2]),    # knee_R
        ]
        joints = [weld_head] + self.actuated
        self.actuated_joint_indices = list(range(1, 1 + N_JOINTS))  # skip weld at 0

        domains = list(self.domains) + [ground]
        # Fixed substeps (faster than adaptive): 4 physics steps per 30 Hz control.
        phys_dt = 1.0 / 240.0
        self.looper = ExplicitLoop(
            phys_dt, domains, joints, useAdapativeDT=False, damping=0.02,
            considerRigidRigidContact=False, use_pd=2,
        )
        self.rm = self.looper.rigidManager
        # Arms share one blue; legs share one green.
        self.colors = [
            0xE8C4A0, 0xFFDD88,
            0x4A90D9, 0x4A90D9, 0x4A90D9, 0x4A90D9,
            0x6B8F71, 0x6B8F71, 0x6B8F71, 0x6B8F71,
        ]
        self.standing_height = float(cy)
        # Fall if torso drops well below the standing height.
        self.cfg.fall_height = min(float(self.cfg.fall_height), 0.65 * self.standing_height)
        self._rest_origins = [np.asarray(b.origin, dtype=np.float64).copy() for b in bodies]
        self._rest_angles = [float(np.asarray(b.angle).reshape(-1)[0]) for b in bodies]

    # ---------------------------------------------------------------- state
    def body_states(self) -> np.ndarray:
        """(N_bodies, 6): x, y, θ, vx, vy, ω."""
        out = np.zeros((N_BODIES, 6), dtype=np.float32)
        rp = self.rm.rigidParams.numpy()
        quat = self.rm.quat.numpy()
        V = self.rm.V.numpy()
        W = self.rm.RotV.numpy()
        for i, d in enumerate(self.domains):
            gid = int(d.ndOffset)
            out[i, 0] = float(rp[gid, 0][0])
            out[i, 1] = float(rp[gid, 0][1])
            out[i, 2] = float(quat[gid])
            out[i, 3] = float(V[gid][0])
            out[i, 4] = float(V[gid][1])
            out[i, 5] = float(W[gid])
        # wrap theta
        out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))
        return out

    def joint_angles(self) -> np.ndarray:
        quat = self.rm.quat.numpy()
        q0 = self.rm.quat_initial.numpy()
        ja = self.rm.joint_id_a.numpy()
        jb = self.rm.joint_id_b.numpy()
        out = np.zeros(N_JOINTS, dtype=np.float32)
        for k, j in enumerate(self.actuated_joint_indices):
            a, b = int(ja[j]), int(jb[j])
            ang = (float(quat[b]) - float(q0[b])) - (float(quat[a]) - float(q0[a]))
            out[k] = np.arctan2(np.sin(ang), np.cos(ang))
        return out

    def joint_vels(self) -> np.ndarray:
        W = self.rm.RotV.numpy()
        ja = self.rm.joint_id_a.numpy()
        jb = self.rm.joint_id_b.numpy()
        out = np.zeros(N_JOINTS, dtype=np.float32)
        for k, j in enumerate(self.actuated_joint_indices):
            a, b = int(ja[j]), int(jb[j])
            out[k] = float(W[b]) - float(W[a])
        return out

    def root_state(self) -> np.ndarray:
        """Torso (x, y, θ, vx, vy, ω)."""
        return self.body_states()[0].astype(np.float32)

    def proprio(self) -> np.ndarray:
        q = self.joint_angles()
        qd = self.joint_vels()
        root = self.root_state()
        return np.concatenate([q, qd, root], axis=0).astype(np.float32)

    def full_state(self) -> np.ndarray:
        """Compact dynamics state for the world model."""
        return self.proprio()

    def goal(self) -> np.ndarray:
        return self.clip.goal_window(self.phase, n_future=self.cfg.n_goal_future)

    def observation(self, noisy: bool = True) -> np.ndarray:
        o = np.concatenate([self.proprio(), self.goal()], axis=0).astype(np.float32)
        if noisy and self.cfg.obs_noise_std > 0:
            o = o + self._rng.normal(0.0, self.cfg.obs_noise_std, size=o.shape).astype(np.float32)
        return o

    # ------------------------------------------------------------- control
    def set_joint_targets(self, targets: np.ndarray):
        t = np.asarray(targets, dtype=np.float64).reshape(-1)
        assert t.size >= N_JOINTS
        for k, j in enumerate(self.actuated_joint_indices):
            _patch_array(self.rm.joint_control_target, j, float(t[k]))

    def _apply_root_assist(self, ref_root: np.ndarray):
        """PD force/torque on torso toward reference root (assistive)."""
        c = self.cfg
        if c.root_kp <= 0:
            return
        st = self.root_state()
        ex = float(ref_root[0] - st[0])
        ey = float(ref_root[1] - st[1])
        eth = float(np.arctan2(np.sin(ref_root[2] - st[2]), np.cos(ref_root[2] - st[2])))
        # Stronger vertical hold so the dancer stays upright while limbs track.
        fx = c.root_kp * ex - c.root_kd * float(st[3])
        fy = 1.8 * c.root_kp * ey - c.root_kd * float(st[4])
        tau = 0.6 * c.root_kp * eth - 0.3 * c.root_kd * float(st[5])
        gid = int(self.domains[0].ndOffset)
        # Accumulate into linear/angular velocity impulse-ish via Force BC path:
        # write temporary velocity correction (gentle).
        V = self.rm.V.numpy()
        W = self.rm.RotV.numpy()
        mass = float(self.rm.mass.numpy()[gid])
        inertia = float(self.rm.inertia.numpy()[gid])
        dt = self.dt
        vx = float(V[gid][0]) + (fx / max(mass, 1e-3)) * dt
        vy = float(V[gid][1]) + (fy / max(mass, 1e-3)) * dt
        w = float(W[gid]) + (tau / max(inertia, 1e-3)) * dt
        import warp as wp
        _patch_array(self.rm.V, gid, wp.vec2(vx, vy))
        _patch_array(self.rm.RotV, gid, w)

    def kinematic_bodies(self, q: np.ndarray, root: np.ndarray) -> np.ndarray:
        """FK body poses (N, 6) from joint angles + torso root — for ghost overlay."""
        q = np.asarray(q, dtype=np.float64).reshape(N_JOINTS)
        cx, cy, th = float(root[0]), float(root[1]), float(root[2])
        tw, thh = self.cfg.torso_size
        aw, _ah = self.cfg.arm_size
        _uw, uh = self.cfg.thigh_size
        _sw, sh = self.cfg.shin_size
        hr = self.cfg.head_radius

        def R(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s], [s, c]], dtype=np.float64)

        Rt = R(th)
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
            (sh_l + R(ual_ang) @ np.array([-0.5 * aw, 0.0]), ual_ang),
            (sh_l + R(ual_ang) @ np.array([-aw, 0.0]) + R(lal_ang) @ np.array([-0.5 * aw, 0.0]), lal_ang),
            (sh_r + R(uar_ang) @ np.array([0.5 * aw, 0.0]), uar_ang),
            (sh_r + R(uar_ang) @ np.array([aw, 0.0]) + R(lar_ang) @ np.array([0.5 * aw, 0.0]), lar_ang),
            (hip_l + R(tl_ang) @ np.array([0.0, -0.5 * uh]), tl_ang),
            (hip_l + R(tl_ang) @ np.array([0.0, -uh]) + R(sl_ang) @ np.array([0.0, -0.5 * sh]), sl_ang),
            (hip_r + R(tr_ang) @ np.array([0.0, -0.5 * uh]), tr_ang),
            (hip_r + R(tr_ang) @ np.array([0.0, -uh]) + R(sr_ang) @ np.array([0.0, -0.5 * sh]), sr_ang),
        ]
        out = np.zeros((N_BODIES, 6), dtype=np.float32)
        for i, (xy, ang) in enumerate(poses):
            out[i, 0] = float(xy[0])
            out[i, 1] = float(xy[1])
            out[i, 2] = float(np.arctan2(np.sin(ang), np.cos(ang)))
        return out

    def _reset_to_rest(self):
        """Snap bodies back to the build-time rest pose (joint anchors stay valid)."""
        import warp as wp
        for i, origin in enumerate(self._rest_origins):
            gid = int(self.domains[i].ndOffset)
            ang = float(self._rest_angles[i])
            _patch_array(self.rm.rigidParams, (gid, 0), [float(origin[0]), float(origin[1])])
            _patch_array(self.rm.quat, gid, ang)
            _patch_array(self.rm.quat_initial, gid, ang)
            _patch_array(self.rm.V, gid, wp.vec2(0.0, 0.0))
            _patch_array(self.rm.RotV, gid, 0.0)
        self.set_joint_targets(np.zeros(N_JOINTS, dtype=np.float32))

    # ------------------------------------------------------------- API
    def reset(self, phase: float | None = None, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Default: both sim and reference start at standing rest (clip frame 0
        # is forced to rest via motion.ease_from_rest). Random mid-clip resets
        # make fair overlay compares impossible when the ref begins in crouch/fall.
        if phase is None:
            phase = 0.0
        self.phase = float(phase) % 1.0
        self.step_count = 0
        self._reset_to_rest()
        # Hold rest targets matching clip[0]
        q0, _ = self.clip.sample(self.phase)
        self.set_joint_targets(q0)
        for _ in range(2):
            self.looper.advanceWithTime(self.dt)
        return self.observation(noisy=True)

    def tracking_reward(self) -> tuple[float, dict]:
        """BeyondMimic-style dense tracking reward (pose + root + vel)."""
        q_ref, root_ref = self.clip.sample(self.phase)
        q = self.joint_angles()
        qd = self.joint_vels()
        root = self.root_state()

        # joint pose
        dq = np.arctan2(np.sin(q - q_ref), np.cos(q - q_ref))
        r_pose = float(np.exp(-3.5 * np.mean(dq ** 2)))
        # joint vel (finite-diff ref)
        q_next, _ = self.clip.sample((self.phase + 1.0 / max(self.clip.n_frames - 1, 1)) % 1.0)
        qd_ref = (q_next - q_ref) / self.dt
        r_vel = float(np.exp(-0.08 * np.mean((qd - qd_ref) ** 2)))
        # root position / orientation
        dxy = root[:2] - root_ref[:2]
        dth = float(np.arctan2(np.sin(root[2] - root_ref[2]), np.cos(root[2] - root_ref[2])))
        r_root = float(np.exp(-35.0 * float(np.dot(dxy, dxy)) - 4.0 * dth * dth))
        # alive
        alive = 1.0 if float(root[1]) > self.cfg.fall_height else 0.0
        # Emphasize joint + root alignment (what the overlay comparison shows).
        reward = 0.55 * r_pose + 0.10 * r_vel + 0.30 * r_root + 0.05 * alive
        info = {
            "r_pose": r_pose,
            "r_vel": r_vel,
            "r_root": r_root,
            "alive": alive,
            "fall": 1.0 - alive,
        }
        return float(reward), info

    def terminated(self) -> bool:
        root = self.root_state()
        if float(root[1]) < self.cfg.fall_height:
            return True
        if abs(float(root[0]) - 0.5) > 0.55:
            return True
        return False

    def step(self, action: np.ndarray, action_noise: float | None = None):
        """Execute one control step. Returns obs, reward, done, info."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)[:N_JOINTS]
        noise_std = self.cfg.action_noise_std if action_noise is None else float(action_noise)
        if noise_std > 0:
            a = a + self._rng.normal(0.0, noise_std, size=a.shape).astype(np.float32)
        # Clip to joint limits roughly
        a = np.clip(a, -2.5, 2.5)

        q_ref, root_ref = self.clip.sample(self.phase)
        self.set_joint_targets(a)
        self._apply_root_assist(root_ref)
        self.looper.advanceWithTime(self.dt)

        reward, info = self.tracking_reward()
        done = self.terminated() or (self.step_count + 1 >= self.cfg.max_episode_steps)
        # advance phase along clip
        self.phase = (self.phase + 1.0 / max(self.clip.n_frames - 1, 1)) % 1.0
        self.step_count += 1
        obs = self.observation(noisy=True)
        info = dict(info)
        info["phase"] = self.phase
        info["action"] = a
        return obs, reward, bool(done), info
