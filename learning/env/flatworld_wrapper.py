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
from rigidmanager import _assign_scalar, _patch_array
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
        lo, hi = getattr(sc, "dr_ee_r_scale", (1.0, 1.0))
        if sc.dr_size_enabled and hi > lo:
            u = float(rng.random())
            span = hi - lo
            # U-shaped: extra mass at the game slider ends (0.5 and 2.0).
            if u < 0.35:
                scale = float(rng.uniform(lo, lo + 0.25 * span))
            elif u < 0.70:
                scale = float(rng.uniform(hi - 0.25 * span, hi))
            else:
                scale = float(rng.uniform(lo, hi))
            r = float(sc.ee_radius) * scale
        else:
            r = float(sc.ee_radius)
        self.ee_rigid.radius = r
        self.obj_geom[0] = (r, r)
        self.obj_half_heights[0] = r
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
        r = float(self.obj_geom[0, 0]) if self.obj_geom is not None else float(self.cfg.scene.ee_radius)
        return max(float(self.cfg.scene.ee_height), r + drop)

    def _apply_geom_to_rm(self):
        """Write sampled sizes into the live RigidManager arrays."""
        for i, gid in enumerate(self.rigid_ids):
            if i == 0:
                _patch_array(self.rm.radius, gid, float(self.obj_geom[0, 0]))
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
        ee_r = float(self.obj_geom[0, 0])
        self.ee_rigid.origin[1] = np.float32(
            max(float(self.ee_rigid.origin[1]), ee_r + sc.spawn_drop))
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
        """Place EE + objects in one left-to-right lineup.

        The EE occupies a random rank in ``ee_rank_range`` (1-indexed,
        typically 1..5) so it is not glued to the far left. Objects fill
        the remaining slots in random order.
        """
        sc = self.cfg.scene
        n_obj = len(self.obj_rigids)
        n_slots = n_obj + 1
        lo, hi = int(sc.ee_rank_range[0]), int(sc.ee_rank_range[1])
        ee_rank = int(rng.integers(lo, hi + 1)) - 1
        ee_rank = int(np.clip(ee_rank, 0, n_slots - 1))

        obj_order = list(rng.permutation(n_obj))
        seq = []
        oi = 0
        for slot in range(n_slots):
            if slot == ee_rank:
                seq.append(("ee", 0))
            else:
                seq.append(("obj", int(obj_order[oi])))
                oi += 1

        half_ws = []
        for kind, i in seq:
            if kind == "ee":
                half_ws.append(float(self.obj_geom[0, 0]))
            else:
                half_ws.append(float(self.obj_geom[i + 1, 0]))
        span = sc.area_x[1] - sc.area_x[0]
        total = sum(2.0 * w for w in half_ws)
        n_gap = max(len(seq) - 1, 1)
        # Never go negative: fat DR bodies may overflow area_x, but they
        # must not start inside each other (that is what launches PGS).
        base_gap = float(sc.min_gap)
        if total + base_gap * n_gap > span:
            base_gap = max(0.01, (span - total) / n_gap)
        slack = max(span - total - base_gap * n_gap, 0.0)
        x = float(sc.area_x[0])
        for k, (kind, i) in enumerate(seq):
            hw = half_ws[k]
            x += hw
            if kind == "ee":
                self.ee_rigid.origin[0] = np.float32(x)
                self.ee_rigid.origin[1] = np.float32(
                    max(sc.ee_height, float(self.obj_geom[0, 0]) + sc.spawn_drop))
            else:
                self.obj_rigids[i].origin[0] = np.float32(x)
                self.obj_rigids[i].origin[1] = np.float32(self._rest_height(i + 1))
            if k < len(seq) - 1:
                x += hw + base_gap + rng.random() * slack / n_gap
            else:
                x += hw

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
                I = 0.5 * m * float(self.obj_geom[0, 0]) ** 2
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

    def _apply_base_dynamics(self):
        """Restore default mass / inertia / friction (no domain randomization)."""
        sc = self.cfg.scene
        restitution = float(sc.restitution)
        for i, gid in enumerate(self.rigid_ids):
            m = float(self.base_mass[i])
            if i == 0:
                r = float(self.obj_geom[0, 0])
                I = 0.5 * m * r * r
            elif self.obj_types[i] == OBJ_TYPE_BOX:
                w, h = 2.0 * self.obj_geom[i, 0], 2.0 * self.obj_geom[i, 1]
                I = (1.0 / 12.0) * m * (w * w + h * h)
            else:
                r = float(self.obj_geom[i, 0])
                I = 0.5 * m * r * r
            _patch_array(self.rm.mass, gid, m)
            _patch_array(self.rm.inertia, gid, float(I))
            _patch_array(self.rm.contactParams, gid, [float(sc.friction), restitution])

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
        for _ in range(8):
            self._reset_once(rng)
            if self.min_separating_gap() >= -0.002:
                return self._observe()
        return self._observe()

    def _reset_once(self, rng: np.random.Generator):
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
        else:
            self._apply_base_dynamics()
        self._cache_body_params()
        self._noise_rng = np.random.default_rng(int(rng.integers(1 << 31)))
        self.set_force((0.0, 0.0))

    def solver_contacts(self):
        """Object–object and ground contact flags from the live PGS cache.

        Pair entries are 1 iff the solver generated a rigid-rigid contact
        between those two bodies on the last detect (not a geometric gap).
        Ground is 1 iff that body had an analytical-plane contact.
        """
        n = self.n_obj
        pair = np.zeros((n, n), dtype=np.float32)
        ground = np.zeros(n, dtype=np.float32)
        id_to_i = {int(gid): i for i, gid in enumerate(self.rigid_ids)}
        rm = self.rm
        nc = int(rm.num_contacts.numpy()[0])
        if nc > 0:
            nc = min(nc, rm.MAX_CONTACTS)
            ca = rm.contact_rigid_a.numpy()[:nc]
            cb = rm.contact_rigid_b.numpy()[:nc]
            for a, b in zip(ca, cb):
                ia, ib = id_to_i.get(int(a)), id_to_i.get(int(b))
                if ia is None or ib is None or ia == ib:
                    continue
                pair[ia, ib] = pair[ib, ia] = 1.0
        ng = int(rm.num_ground_contacts.numpy()[0])
        if ng > 0:
            ng = min(ng, rm.MAX_GROUND_CONTACTS)
            for rid in rm.ground_contact_rigid.numpy()[:ng]:
                i = id_to_i.get(int(rid))
                if i is not None:
                    ground[i] = 1.0
        return pair, ground

    def _fill_all_rigid_pairs(self):
        """Put every rigid-rigid pair into the primitive detect buffer."""
        rm = self.rm
        ids = self.rigid_ids
        n = len(ids)
        buf = np.asarray(rm.primitive_pairs_buffer.numpy())
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                buf[k] = (int(ids[i]), int(ids[j]))
                k += 1
        rm.primitive_pairs_buffer.assign(buf)
        _assign_scalar(rm.num_primitive_pairs, k)

    def apply_saved_geom(self, geom: np.ndarray):
        """Write npz ``obj_geom`` (half-extents) into the live solver."""
        geom = np.asarray(geom, dtype=np.float32)
        self.obj_geom = geom.copy()
        for i, gid in enumerate(self.rigid_ids):
            hw, hh = float(geom[i, 0]), float(geom[i, 1])
            if i == 0 or int(self.obj_types[i]) == OBJ_TYPE_BALL:
                r = hw
                if i == 0:
                    self.ee_rigid.radius = r
                else:
                    self.obj_rigids[i - 1].radius = r
                _patch_array(self.rm.radius, gid, r)
            else:
                ext = np.array([2.0 * hw, 2.0 * hh], dtype=np.float32)
                self.obj_rigids[i - 1].ext = ext
                _patch_array(self.rm.rigidParams, (gid, 1), ext)
        self._apply_geom_to_rm()

    def set_states(self, states: np.ndarray):
        """Teleport every body to ``states`` (N, 6) = x y θ vx vy ω."""
        states = np.asarray(states, dtype=np.float32)
        for i, gid in enumerate(self.rigid_ids):
            x, y, th = float(states[i, 0]), float(states[i, 1]), float(states[i, 2])
            vx, vy, om = float(states[i, 3]), float(states[i, 4]), float(states[i, 5])
            if i == 0:
                self.ee_rigid.origin[0] = np.float32(x)
                self.ee_rigid.origin[1] = np.float32(y)
            else:
                self.obj_rigids[i - 1].origin[0] = np.float32(x)
                self.obj_rigids[i - 1].origin[1] = np.float32(y)
            _patch_array(self.rm.rigidParams, (gid, 0), [x, y])
            _patch_array(self.rm.quat, gid, th)
            _patch_array(self.rm.V, gid, wp.vec2(vx, vy))
            _patch_array(self.rm.RotV, gid, om)

    def refresh_solver_contacts(self):
        """Detect contacts at the current pose (no time integration)."""
        rm = self.rm
        rm.precompute_rigid_transforms()
        rm.updateBBox()
        rm.reset_contact_caches_kernel()
        rm._generate_ground_pairs_direct_kernel()
        self._fill_all_rigid_pairs()
        rm.detect_all_contacts()
        rm.detectRigidGroundContact()
        return self.solver_contacts()

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    def set_force(self, f):
        """Set the force (Fx, Fy) applied to the EE, in Newtons. Call once per frame."""
        _patch_array(self.rm.bcTValues, self.ee_idx,
                     wp.vec2(float(f[0]), float(f[1])))

    def set_ee_contact(self, friction: float, restitution: float):
        """Patch the spirit's Coulomb mu and restitution (live contactParams)."""
        mu = float(np.clip(friction, 0.0, 1.0))
        e = float(np.clip(restitution, 0.0, 1.0))
        _patch_array(self.rm.contactParams, self.ee_idx, [mu, e])
        if getattr(self, "obj_mu", None) is not None and len(self.obj_mu) > 0:
            self.obj_mu[0] = mu

    def set_ee_radius(self, radius: float):
        """Patch the spirit's collision radius, geom, and disk inertia."""
        r = float(max(1e-4, radius))
        self.ee_rigid.radius = r
        _patch_array(self.rm.radius, self.ee_idx, r)
        self.obj_geom[0] = (r, r)
        self.obj_half_heights[0] = r
        mass_np = np.asarray(self.rm.mass.numpy())
        m = float(mass_np[self.ee_idx])
        _patch_array(self.rm.inertia, self.ee_idx, 0.5 * m * r * r)
        pos = np.asarray(self.rm.rigidParams.numpy())[self.ee_idx, 0]
        min_y = r + float(self.cfg.scene.spawn_drop)
        if float(pos[1]) < min_y:
            _patch_array(self.rm.rigidParams, (self.ee_idx, 0),
                         [float(pos[0]), min_y])
            self.ee_rigid.origin[1] = np.float32(min_y)
        self.rm.updateBBox()

    def set_body_pose(self, i: int, x: float, y: float = None, theta: float = 0.0):
        """Teleport body ``i`` (0 = EE) and zero its velocity."""
        gid = self.rigid_ids[i]
        if y is None:
            y = self._rest_height(i)
        x, y = float(x), float(y)
        if i == 0:
            self.ee_rigid.origin[0] = np.float32(x)
            self.ee_rigid.origin[1] = np.float32(y)
        else:
            self.obj_rigids[i - 1].origin[0] = np.float32(x)
            self.obj_rigids[i - 1].origin[1] = np.float32(y)
        _patch_array(self.rm.rigidParams, (gid, 0), [x, y])
        _patch_array(self.rm.quat, gid, float(theta))
        _patch_array(self.rm.V, gid, wp.vec2(0.0, 0.0))
        _patch_array(self.rm.RotV, gid, 0.0)

    def place_lineup(self, xs):
        """Set every body x, rest-height y, then refresh AABBs."""
        for i, x in enumerate(xs):
            self.set_body_pose(i, float(x))
        self.rm.updateBBox()

    def raw_states(self) -> np.ndarray:
        """Kinematic state from the solver, without sensor noise."""
        rm = self.rm
        ids = np.asarray(self.rigid_ids)
        params = rm.rigidParams.numpy()
        quat = rm.quat.numpy()
        V = rm.V.numpy()
        RotV = rm.RotV.numpy()
        pos = params[ids, 0, :].astype(np.float64)
        theta = np.arctan2(np.sin(quat[ids]), np.cos(quat[ids]))
        vel = V[ids]
        om = RotV[ids]
        return np.stack(
            [pos[:, 0], pos[:, 1], theta, vel[:, 0], vel[:, 1], om], axis=1
        ).astype(np.float32)

    def min_separating_gap(self) -> float:
        """Most-overlapping pair, AABB separating-axis gap (meters).

        Negative means the two AABBs interpenetrate. Touching or kissing
        (gap ≈ 0) is allowed; deep illegal overlap is not.
        """
        s = self.raw_states()
        g = self.obj_geom
        worst = 1.0e9
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                gx = abs(float(s[i, 0] - s[j, 0])) - float(g[i, 0] + g[j, 0])
                gy = abs(float(s[i, 1] - s[j, 1])) - float(g[i, 1] + g[j, 1])
                worst = min(worst, max(gx, gy))
        return float(worst)

    def snapshot(self) -> dict:
        """Copy kinematic + dynamic state so a planner can roll and rewind."""
        ids = np.asarray(self.rigid_ids)
        rp = np.asarray(self.rm.rigidParams.numpy())
        return {
            "pos": rp[ids, 0].copy(),
            "ext": rp[ids, 1].copy(),
            "theta": np.asarray(self.rm.quat.numpy())[ids].copy(),
            "vel": np.asarray(self.rm.V.numpy())[ids].copy(),
            "om": np.asarray(self.rm.RotV.numpy())[ids].copy(),
            "mass": np.asarray(self.rm.mass.numpy())[ids].copy(),
            "inertia": np.asarray(self.rm.inertia.numpy())[ids].copy(),
            "contact": np.asarray(self.rm.contactParams.numpy())[ids].copy(),
            "radius": np.asarray(self.rm.radius.numpy())[ids].copy(),
            "obj_geom": np.asarray(self.obj_geom).copy(),
            "obj_half_heights": list(self.obj_half_heights),
            "obj_mass": np.asarray(self.obj_mass).copy(),
            "obj_mu": np.asarray(self.obj_mu).copy(),
            "ee_radius": float(self.ee_rigid.radius),
            "force_cap_override": getattr(self, "force_cap_override", None),
        }

    def restore(self, snap: dict):
        """Write ``snapshot()`` back onto the live RigidManager."""
        for i, gid in enumerate(self.rigid_ids):
            x, y = float(snap["pos"][i, 0]), float(snap["pos"][i, 1])
            if i == 0:
                self.ee_rigid.origin[0] = np.float32(x)
                self.ee_rigid.origin[1] = np.float32(y)
                self.ee_rigid.radius = float(snap["ee_radius"])
            else:
                self.obj_rigids[i - 1].origin[0] = np.float32(x)
                self.obj_rigids[i - 1].origin[1] = np.float32(y)
                if int(self.obj_types[i]) == OBJ_TYPE_BOX:
                    self.obj_rigids[i - 1].ext = np.asarray(
                        snap["ext"][i], dtype=np.float32)
                else:
                    self.obj_rigids[i - 1].radius = float(snap["radius"][i])
            _patch_array(self.rm.rigidParams, (gid, 0), [x, y])
            _patch_array(self.rm.rigidParams, (gid, 1), snap["ext"][i])
            _patch_array(self.rm.quat, gid, float(snap["theta"][i]))
            vx, vy = float(snap["vel"][i, 0]), float(snap["vel"][i, 1])
            _patch_array(self.rm.V, gid, wp.vec2(vx, vy))
            _patch_array(self.rm.RotV, gid, float(snap["om"][i]))
            _patch_array(self.rm.mass, gid, float(snap["mass"][i]))
            _patch_array(self.rm.inertia, gid, float(snap["inertia"][i]))
            _patch_array(self.rm.contactParams, gid, snap["contact"][i])
            _patch_array(self.rm.radius, gid, float(snap["radius"][i]))
        self.obj_geom = np.asarray(snap["obj_geom"], dtype=np.float32).copy()
        self.obj_half_heights = list(snap["obj_half_heights"])
        self.obj_mass = np.asarray(snap["obj_mass"], dtype=np.float32).copy()
        self.obj_mu = np.asarray(snap["obj_mu"], dtype=np.float32).copy()
        self.force_cap_override = snap.get("force_cap_override")
        self.rm.updateBBox()

    def _apply_action_advance(self, action: np.ndarray):
        """Admittance force + one visual frame. No observation."""
        sc = self.cfg.scene
        a = np.asarray(action, dtype=np.float64)
        if sc.ee_damping > 0.0:
            v = self.rm.V.numpy()[self.ee_idx]
            a = a - sc.ee_damping * v
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

    def advance_action(self, action: np.ndarray):
        """Force + one visual frame, no observation (solver-CEM inner loop)."""
        self._apply_action_advance(action)

    def step(self, action: np.ndarray) -> dict:
        """Apply action=(Fx,Fy), advance one visual frame, return the new observation.

        The EE is admittance-controlled: the commanded force acts against a
        viscous damping term (-c * v) and the EE speed is hard-clamped.
        Without damping the EE is an undamped double integrator -- PGS
        contact impulses launch it meters across the scene, which both
        wrecks the collected force -> motion statistics and makes the
        planning problem unstable.
        """
        self._apply_action_advance(action)
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
