"""FlatWorld push-scene wrapper (Phase 1 data collection).

Scene: a circular end-effector (EE, force-controlled via a Force BC,
gravity-free hover) randomly pushes through a pile of boxes/balls on
a ground plane.

Conventions (all verified against flatworld/rigidmanager.py):
- Rigid state: position rigidParams[idx, 0], 2D angle quat[idx] (scalar),
  linear velocity V[idx], angular velocity RotV[idx]
- Contact force signs: in the PGS solver impulse on A = +lambda * J_a,
  therefore contact_force is the force ON body A; the force on B is its
  negation. Pair contact normals point from A to B. ground_contact_force
  is the force applied by the ground ON the rigid body.
- Time-varying force control: the Force BC value lives in
  rm.bcTValues[idx] and is read every substep, so patching it once per
  frame implements a (Fx, Fy) action.
- The contact arrays hold contacts from the LAST substep inside
  advanceWithTime, sampled as the frame's tactile observation.

Tactile representation (Deep Sets input):
- contact_feat: (K, 7) = [rel_pos(2), ee_outward_normal(2), force_on_ee(2), penetration(1)]
- contact_mask: (K,) 1 = real contact, 0 = padding
- tactile_summary: (4,) = [sum_Fx, sum_Fy, sum_tau, num_contacts]
"""

import numpy as np
import warp as wp

from flatworld import (
    BallRigid,
    BoxRigid,
    ExplicitLoop,
    FixedAll,
    Force,
    Gravity,
    GroundDomain,
    RigidBodyDomain,
)
# NOTE: flatworld uses top-level import style internally (its __init__.py
# puts the package dir on sys.path). We follow the same style to avoid
# loading rigidmanager twice under two module names.
from rigidmanager import _patch_array
from wp_init import init_warp

from learning.configs.default import Config

OBJ_TYPE_EE = 0
OBJ_TYPE_BOX = 1
OBJ_TYPE_BALL = 2


class PushSceneEnv:
    """Gym-style (numpy dict obs) FlatWorld push environment."""

    def __init__(self, cfg: Config = None, device: str = None):
        self.cfg = cfg or Config()
        init_warp(device=device, prefer_cuda=False)  # CPU is faster for small scenes; GPU also works
        self._build()

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #
    def _build(self):
        sc = self.cfg.scene

        # End-effector: no gravity, controlled by a Force BC
        self.ee_rigid = BallRigid(2, [0.3, sc.ee_height], sc.ee_radius, sc.ee_mass)
        self.ee_domain = RigidBodyDomain(
            self.ee_rigid, bcs=[Force([0], [0.0, 0.0])], friction=sc.friction
        )

        # Objects: gravity, randomly placed (randomized in reset)
        self.obj_rigids = []
        self.obj_domains = []
        self.obj_types = [OBJ_TYPE_EE]
        self.obj_half_heights = [sc.ee_radius]

        for _ in range(sc.num_boxes):
            # BoxRigid.ext is the FULL extent (rigidmanager halves it for
            # collision), while sc.box_ext stores HALF extents (spawn
            # height, planner standoff, report drawing all use halves).
            r = BoxRigid(2, [0.8, sc.box_ext[1] + sc.spawn_drop],
                         [2.0 * sc.box_ext[0], 2.0 * sc.box_ext[1]],
                         [0, 0], sc.box_mass)
            self.obj_rigids.append(r)
            self.obj_types.append(OBJ_TYPE_BOX)
            self.obj_half_heights.append(sc.box_ext[1])
        for _ in range(sc.num_balls):
            r = BallRigid(2, [0.8, sc.ball_radius + sc.spawn_drop],
                          sc.ball_radius, sc.ball_mass)
            self.obj_rigids.append(r)
            self.obj_types.append(OBJ_TYPE_BALL)
            self.obj_half_heights.append(sc.ball_radius)

        for r in self.obj_rigids:
            self.obj_domains.append(
                RigidBodyDomain(r, bcs=[Gravity([0.0, -sc.gravity])], friction=sc.friction)
            )

        ground = GroundDomain(2, (0.0, 0.0), (0, 1), bcs=[FixedAll([0])])

        domains = [self.ee_domain] + self.obj_domains + [ground]
        self.looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)
        self.rm = self.looper.rigidManager

        # Global rigid indices: EE is the first rigid domain -> index 0
        self.ee_idx = self.ee_domain.ndOffset
        self.rigid_ids = [self.ee_idx] + [d.ndOffset for d in self.obj_domains]
        self.obj_types_np = np.asarray(self.obj_types, dtype=np.int64)
        self.n_obj = len(self.rigid_ids)
        self.obj_geom = np.zeros((self.n_obj, 2), dtype=np.float32)
        self.obj_geom[0] = (sc.ee_radius, sc.ee_radius)
        for i, t in enumerate(self.obj_types):
            if i == 0:
                continue
            if t == OBJ_TYPE_BOX:
                self.obj_geom[i] = sc.box_ext
            else:
                self.obj_geom[i] = (sc.ball_radius, sc.ball_radius)

        # Base dynamics for domain randomization (Phase 2)
        self.base_mass = [sc.ee_mass] + [r.mass for r in self.obj_rigids]
        self.obj_mass = np.asarray(self.base_mass, dtype=np.float32)
        self.obj_mu = np.full(self.n_obj, sc.friction, dtype=np.float32)
        self._noise_rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    # Randomization / lifecycle
    # ------------------------------------------------------------------ #
    def _randomize_sizes(self, rng: np.random.Generator):
        """Sample per-object half-extents and patch collision geometry.

        ``rm.reset()`` restores pose but not extents/radius, so we write
        ``rigidParams[idx, 1]`` and ``radius`` after reset.
        """
        sc = self.cfg.scene
        self.obj_geom[0] = (sc.ee_radius, sc.ee_radius)
        if not sc.dr_size_enabled:
            for i, t in enumerate(self.obj_types):
                if i == 0:
                    continue
                if t == OBJ_TYPE_BOX:
                    self.obj_geom[i] = sc.box_ext
                    self.obj_half_heights[i] = sc.box_ext[1]
                    self.obj_rigids[i - 1].ext = np.array(
                        [2.0 * sc.box_ext[0], 2.0 * sc.box_ext[1]], dtype=np.float32)
                else:
                    self.obj_geom[i] = (sc.ball_radius, sc.ball_radius)
                    self.obj_half_heights[i] = sc.ball_radius
                    self.obj_rigids[i - 1].radius = sc.ball_radius
            return
        for i, t in enumerate(self.obj_types):
            if i == 0:
                continue
            rigid = self.obj_rigids[i - 1]
            if t == OBJ_TYPE_BOX:
                hw = float(rng.uniform(*sc.dr_box_hw))
                hh = float(rng.uniform(*sc.dr_box_hh))
                if hw < hh:
                    hw, hh = hh, hw
                rigid.ext = np.array([2.0 * hw, 2.0 * hh], dtype=np.float32)
                self.obj_geom[i] = (hw, hh)
                self.obj_half_heights[i] = hh
            else:
                r = float(rng.uniform(*sc.dr_ball_r))
                rigid.radius = r
                self.obj_geom[i] = (r, r)
                self.obj_half_heights[i] = r

    def _rest_height(self, i: int) -> float:
        """Center y so the object's bottom sits on the ground + spawn_drop.

        Boxes: ``BoxRigid.ext`` is the FULL extent; collision uses half of
        ``rigidParams[idx, 1]``, so rest y = 0.5 * ext_y + drop.
        Balls: rest y = radius + drop.
        """
        drop = float(self.cfg.scene.spawn_drop)
        if self.obj_types[i] == OBJ_TYPE_BOX:
            ext_y = float(self.obj_rigids[i - 1].ext[1])
            return 0.5 * ext_y + drop
        if self.obj_types[i] == OBJ_TYPE_BALL:
            return float(self.obj_rigids[i - 1].radius) + drop
        return float(self.cfg.scene.ee_radius) + drop

    def _apply_geom_to_rm(self):
        """Write sampled sizes into the live RigidManager arrays."""
        for i, gid in enumerate(self.rigid_ids):
            if i == 0:
                continue
            t = self.obj_types[i]
            rigid = self.obj_rigids[i - 1]
            if t == OBJ_TYPE_BOX:
                _patch_array(self.rm.rigidParams, (gid, 1), rigid.ext)
            else:
                _patch_array(self.rm.radius, gid, float(rigid.radius))

    def _seat_on_ground(self):
        """Re-seat every object after size patch so bottoms do not go through y=0."""
        sc = self.cfg.scene
        self.ee_rigid.origin[1] = np.float32(
            max(float(self.ee_rigid.origin[1]), sc.ee_radius + sc.spawn_drop))
        _patch_array(self.rm.rigidParams, (self.ee_idx, 0), self.ee_rigid.origin)
        for i, gid in enumerate(self.rigid_ids):
            if i == 0:
                continue
            rigid = self.obj_rigids[i - 1]
            rest_y = self._rest_height(i)
            rigid.origin[1] = np.float32(rest_y)
            self.obj_half_heights[i] = rest_y - sc.spawn_drop
            if self.obj_types[i] == OBJ_TYPE_BOX:
                self.obj_geom[i, 1] = 0.5 * float(rigid.ext[1])
            else:
                r = float(rigid.radius)
                self.obj_geom[i] = (r, r)
            _patch_array(self.rm.rigidParams, (gid, 0), rigid.origin)

    def _randomize_layout(self, rng: np.random.Generator):
        sc = self.cfg.scene
        order = rng.permutation(len(self.obj_rigids))
        half_widths = [float(self.obj_geom[i + 1, 0]) for i in range(len(self.obj_rigids))]
        span = sc.area_x[1] - sc.area_x[0]
        total_half = sum(2 * half_widths[i] for i in order)
        slack = max(span - total_half - sc.min_gap * len(order), 0.0)
        x = sc.area_x[0]
        for k, i in enumerate(order):
            x += half_widths[i]
            self.obj_rigids[i].origin[0] = np.float32(x)
            self.obj_rigids[i].origin[1] = np.float32(self._rest_height(i + 1))
            gap = sc.min_gap + rng.random() * 0.5 * slack / max(len(order), 1)
            x += half_widths[i] + gap

        self.ee_rigid.origin[0] = np.float32(rng.uniform(*sc.ee_spawn_x))
        self.ee_rigid.origin[1] = np.float32(
            max(sc.ee_height, sc.ee_radius + sc.spawn_drop))

    def _randomize_dynamics(self, rng: np.random.Generator):
        """Phase 2 domain randomization: per-episode mass / inertia / friction.

        Patches the RigidManager arrays directly (rm.reset() does not touch
        mass / inertia / contactParams, so this runs after the reset).
        """
        sc = self.cfg.scene
        restitution = float(sc.restitution)
        for i, gid in enumerate(self.rigid_ids):
            if i == 0:
                factor = rng.uniform(*sc.dr_ee_mass_range)
            else:
                factor = rng.uniform(*sc.dr_mass_range)
            m = self.base_mass[i] * factor
            if i == 0:
                I = 0.5 * m * sc.ee_radius ** 2
            elif self.obj_types[i] == OBJ_TYPE_BOX:
                w, h = 2.0 * self.obj_geom[i, 0], 2.0 * self.obj_geom[i, 1]
                I = (1.0 / 12.0) * m * (w * w + h * h)
            else:
                r = float(self.obj_geom[i, 0])
                I = 0.5 * m * r * r
            mu = rng.uniform(*sc.dr_friction_range) if i > 0 else sc.friction
            _patch_array(self.rm.mass, gid, float(m))
            _patch_array(self.rm.inertia, gid, float(I))
            _patch_array(self.rm.contactParams, gid, [mu, restitution])

    def _cache_body_params(self):
        """Snapshot live mass / friction for the planner (index = object slot)."""
        mass_np = np.asarray(self.rm.mass.numpy())
        cp_np = np.asarray(self.rm.contactParams.numpy())
        self.obj_mass = np.array(
            [float(mass_np[gid]) for gid in self.rigid_ids], dtype=np.float32)
        self.obj_mu = np.array(
            [float(cp_np[gid, 0]) for gid in self.rigid_ids], dtype=np.float32)

    def reset(self, rng: np.random.Generator = None) -> dict:
        """Randomize the layout and reset. Returns the initial observation."""
        rng = rng or np.random.default_rng()
        self._randomize_sizes(rng)
        self._randomize_layout(rng)
        # Extents must be in RM *before* reset rebuilds bboxes, otherwise a
        # leftover taller box/ball is tested against the new lower rest pose.
        self._apply_geom_to_rm()
        self.looper.reset()  # pose from rigid.origin; extents already patched
        self._seat_on_ground()
        self.rm.updateBBox()
        if self.cfg.scene.dr_enabled:
            self._randomize_dynamics(rng)
        self._cache_body_params()
        self._noise_rng = np.random.default_rng(int(rng.integers(1 << 31)))
        self.set_force((0.0, 0.0))
        return self._observe()

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    def set_force(self, f):
        """Set the force (Fx, Fy) applied to the EE, in Newtons. Call once per frame."""
        _patch_array(self.rm.bcTValues, self.ee_idx,
                     wp.vec2(float(f[0]), float(f[1])))

    def step(self, action: np.ndarray) -> dict:
        """Apply action=(Fx,Fy), advance one visual frame, return the new observation.

        The EE is admittance-controlled: the commanded force acts against a
        viscous damping term (-c * v) and the EE speed is hard-clamped.
        Without damping the EE is an undamped double integrator -- PGS
        contact impulses launch it meters across the scene, which both
        wrecks the collected force -> motion statistics and makes the
        planning problem unstable.
        """
        sc = self.cfg.scene
        a = np.asarray(action, dtype=np.float64)
        if sc.ee_damping > 0.0:
            v = self.rm.V.numpy()[self.ee_idx]
            a = a - sc.ee_damping * v
            # actuator saturation: the net force stays within the same
            # range the actions were collected in, unless a caller
            # (pile-clear) raises the cap for this step.
            fmax = float(self.cfg.collect.force_max)
            override = getattr(self, "force_cap_override", None)
            if override is not None:
                fmax = max(fmax, float(override))
            a = np.clip(a, -fmax, fmax)
        self.set_force(a)
        self.looper.advanceWithTime(sc.frame_dt)
        if sc.ee_vel_max > 0.0:
            v = self.rm.V.numpy()[self.ee_idx].copy()
            speed = float(np.hypot(v[0], v[1]))
            if speed > sc.ee_vel_max:
                v = v * (sc.ee_vel_max / speed)
                _patch_array(self.rm.V, self.ee_idx, wp.vec2(float(v[0]), float(v[1])))
        return self._observe()

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _observe(self) -> dict:
        rm = self.rm
        ids = np.asarray(self.rigid_ids)

        params = rm.rigidParams.numpy()      # (MAX_NODES, 2, 2)
        quat = rm.quat.numpy()               # (MAX_NODES,)
        V = rm.V.numpy()
        RotV = rm.RotV.numpy()

        pos = params[ids, 0, :].astype(np.float64)          # (N, 2)
        theta = np.arctan2(np.sin(quat[ids]), np.cos(quat[ids]))  # wrap to [-pi, pi]
        vel = V[ids]
        om = RotV[ids]
        states = np.stack(
            [pos[:, 0], pos[:, 1], theta, vel[:, 0], vel[:, 1], om], axis=1
        ).astype(np.float32)  # (N, 6)

        cfeat, cmask, csum = self._tactile(pos[0])
        sc = self.cfg.scene
        if sc.noise_enabled:
            # Phase 2 sensor noise: positions, velocities, contact forces
            states[:, 0:2] += self._noise_rng.normal(0.0, sc.noise_pos, (len(ids), 2))
            states[:, 3:5] += self._noise_rng.normal(0.0, sc.noise_vel, (len(ids), 2))
            states[:, 5] += self._noise_rng.normal(0.0, sc.noise_vel * 0.5, len(ids))
            cfeat[:, 4:6] += self._noise_rng.normal(0.0, sc.noise_force, (cfeat.shape[0], 2))
            csum[0:3] += self._noise_rng.normal(0.0, sc.noise_force, 3)
        return {
            "obj_states": states,
            "obj_types": self.obj_types_np,
            "obj_geom": self.obj_geom.copy(),
            "contact_feat": cfeat,
            "contact_mask": cmask,
            "tactile_summary": csum,
        }

    def _tactile(self, ee_pos: np.ndarray):
        """Extract all EE contacts. Returns (feat (K,7), mask (K,), summary (4,)).

        Per-contact features:
          rel_pos      contact point offset from EE center (2)
          normal       EE outward surface normal (always pointing from EE
                       toward the other body) (2)
          force        contact force ON the EE (2)
          penetration  contact penetration depth, >= 0 (1)
        """
        rm = self.rm
        K = self.cfg.tactile.max_contacts
        rows = []  # (rel_pos, normal, force, pen)
        F_SAT = 30.0  # sensor saturation (N); raw PGS impulses can spike to
        #               hundreds of N for a single substep, which would
        #               otherwise wreck the normalization statistics

        def _saturate(f):
            mag = float(np.hypot(f[0], f[1]))
            if mag > F_SAT:
                return f * (F_SAT / mag)
            return f

        # --- rigid-rigid contacts ---
        n = int(rm.num_contacts.numpy()[0])
        if n > 0:
            n = min(n, rm.MAX_CONTACTS)
            ca = rm.contact_rigid_a.numpy()
            cb = rm.contact_rigid_b.numpy()
            cp = rm.contact_point.numpy()
            cn = rm.contact_normal.numpy()
            cf = rm.contact_force.numpy()
            cd = rm.contact_depth.numpy()
            for i in range(n):
                a, b = int(ca[i]), int(cb[i])
                if a == self.ee_idx:
                    f_on_ee, nrm = cf[i], cn[i]
                elif b == self.ee_idx:
                    f_on_ee, nrm = -cf[i], -cn[i]
                else:
                    continue
                rows.append((cp[i] - ee_pos, nrm, _saturate(f_on_ee),
                             max(0.0, -float(cd[i]))))

        # --- ground contacts (EE hovers, normally none; handled for safety) ---
        ng = int(rm.num_ground_contacts.numpy()[0])
        if ng > 0:
            ng = min(ng, rm.MAX_GROUND_CONTACTS)
            grid_ = rm.ground_contact_rigid.numpy()
            gcp = rm.ground_contact_point.numpy()
            gcn = rm.ground_contact_normal.numpy()
            gcf = rm.ground_contact_force.numpy()
            gcd = rm.ground_contact_depth.numpy()
            for j in range(ng):
                if int(grid_[j]) != self.ee_idx:
                    continue
                rows.append((gcp[j] - ee_pos, gcn[j], gcf[j],
                             max(0.0, -float(gcd[j]))))

        # Keep the top-K contacts by penetration depth (descending)
        rows.sort(key=lambda r: -r[3])
        rows = rows[:K]

        feat = np.zeros((K, 7), dtype=np.float32)
        mask = np.zeros((K,), dtype=np.float32)
        sum_f = np.zeros(2, dtype=np.float64)
        tau = 0.0
        for i, (rel, nrm, f, pen) in enumerate(rows):
            feat[i] = (rel[0], rel[1], nrm[0], nrm[1], f[0], f[1], pen)
            mask[i] = 1.0
            sum_f += f
            tau += rel[0] * f[1] - rel[1] * f[0]  # 2D cross product

        summary = np.array([sum_f[0], sum_f[1], tau, len(rows)], dtype=np.float32)
        return feat, mask, summary
