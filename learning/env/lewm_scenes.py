"""StateLeWM scenes for two-room, push-T, and reacher.

Two-room and push-T are planar (top-down): walls / T live in x-y, no in-plane
gravity and no GroundDomain (that would be a side view). Push-T adds Coulomb
table drag with out-of-plane N = m * table_g. Reacher is a gravity two-link arm
with no circular EE: node 0 = link 1, node 1 = fixed base, node 2 = link 2.
Action is PD joint-angle targets (use_pd=2 → internal torque). Planning
and eval score pose_head xy of link 2 (box center), not the distal corner.
"""

from __future__ import annotations

import numpy as np

from flatworld import (
    BallRigid,
    BoxRigid,
    ExplicitLoop,
    FixedAll,
    Force,
    Gravity,
    RigidBodyDomain,
    RevoluteJoint,
    WeldJoint,
)
from rigidmanager import _patch_array

from learning.configs.default import Config
from learning.env.flatworld_wrapper import (
    OBJ_TYPE_BALL,
    OBJ_TYPE_BOX,
    OBJ_TYPE_EE,
    PushSceneEnv,
)
from learning.models.relations import REL_REVOLUTE, REL_WELD


def _rot(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _set_pose(rigid, xy, theta: float = 0.0):
    rigid.origin = np.asarray(xy, dtype=np.float32)
    rigid.angle = np.array([float(theta), 0.0], dtype=np.float32)


def _box_corners(xy, hw, hh, theta):
    c, s = float(np.cos(theta)), float(np.sin(theta))
    local = np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ R.T + np.asarray(xy, dtype=np.float64)


def _circle_obb_gap(ee_xy, ee_r, box_xy, hw, hh, theta):
    """Signed gap between a circle and a rotated box. Negative = penetration."""
    ee = np.asarray(ee_xy, dtype=np.float64).reshape(2)
    box = np.asarray(box_xy, dtype=np.float64).reshape(2)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    d = ee - box
    lx = c * d[0] + s * d[1]
    ly = -s * d[0] + c * d[1]
    if abs(lx) <= hw and abs(ly) <= hh:
        return float(-min(hw - abs(lx), hh - abs(ly)) - ee_r)
    qx = float(np.clip(lx, -hw, hw))
    qy = float(np.clip(ly, -hh, hh))
    return float(np.hypot(lx - qx, ly - qy) - ee_r)


def _aabb_sep(ee_xy, ee_r, box_xy, hw, hh):
    gx = abs(float(ee_xy[0]) - float(box_xy[0])) - (ee_r + hw)
    gy = abs(float(ee_xy[1]) - float(box_xy[1])) - (ee_r + hh)
    return float(max(gx, gy))


class LewmSceneEnv(PushSceneEnv):
    """PushSceneEnv with fixed bodies, joint skip-pairs, and no independent seating."""

    planar = False
    torque_control = False

    def __init__(self, cfg: Config = None, device: str = None):
        self.connected_pairs = set()
        self.joint_rel = {}
        self.fixed_slots = set()
        super().__init__(cfg, device)

    def rel_type_mat(self):
        from learning.models.relations import empty_rel, set_pair
        R = empty_rel(self.n_obj)
        for (i, j), t in self.joint_rel.items():
            set_pair(R, i, j, t)
        return R

    def _maybe_gravity(self, extra=None):
        bcs = list(extra or [])
        g = float(self.cfg.scene.gravity)
        if abs(g) > 1e-9:
            bcs.append(Gravity([0.0, -g]))
        return bcs

    def _register(self, joints=None):
        sc = self.cfg.scene
        domains = [self.ee_domain] + self.obj_domains
        damp = float(getattr(sc, "loop_damping", 0.0))
        use_pd = int(getattr(self, "use_pd", getattr(sc, "use_pd", 0)))
        self.use_pd = use_pd
        self.looper = ExplicitLoop(
            0.0, domains, joints or [], useAdapativeDT=True, damping=damp,
            considerRigidRigidContact=bool(getattr(self, "rigid_rigid_contact", True)),
            use_pd=use_pd)
        self.rm = self.looper.rigidManager
        self.ee_idx = self.ee_domain.ndOffset
        self.rigid_ids = [self.ee_idx] + [d.ndOffset for d in self.obj_domains]
        self.obj_types_np = np.asarray(self.obj_types, dtype=np.int64)
        self.n_obj = len(self.rigid_ids)
        self.obj_geom = np.zeros((self.n_obj, 2), dtype=np.float32)
        if int(self.obj_types[0]) == OBJ_TYPE_BOX:
            ext = np.asarray(self.ee_rigid.ext, dtype=np.float64)
            self.obj_geom[0] = (0.5 * ext[0], 0.5 * ext[1])
        else:
            self.obj_geom[0] = (sc.ee_radius, sc.ee_radius)
        for i, t in enumerate(self.obj_types):
            if i == 0:
                continue
            if t == OBJ_TYPE_BOX:
                ext = np.asarray(self.obj_rigids[i - 1].ext, dtype=np.float64)
                self.obj_geom[i] = (0.5 * ext[0], 0.5 * ext[1])
            else:
                r = float(self.obj_rigids[i - 1].radius)
                self.obj_geom[i] = (r, r)
        self.base_mass = [sc.ee_mass] + [r.mass for r in self.obj_rigids]
        self.obj_mass = np.asarray(self.base_mass, dtype=np.float32)
        self.obj_mu = np.full(self.n_obj, sc.friction, dtype=np.float32)
        self._noise_rng = np.random.default_rng(0)

    def attract_indices(self):
        return [i for i in range(1, self.n_obj) if i not in self.fixed_slots]

    def min_separating_gap(self) -> float:
        s = self.raw_states()
        g = self.obj_geom
        worst = 1.0e9
        n = len(s)
        skip = self.connected_pairs
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in skip or (j, i) in skip:
                    continue
                gx = abs(float(s[i, 0] - s[j, 0])) - float(g[i, 0] + g[j, 0])
                gy = abs(float(s[i, 1] - s[j, 1])) - float(g[i, 1] + g[j, 1])
                worst = min(worst, max(gx, gy))
        return float(worst)

    def _pin_fixed(self):
        for i in self.fixed_slots:
            gid = self.rigid_ids[i]
            _patch_array(self.rm.mass, gid, 1e10)
            _patch_array(self.rm.inertia, gid, 1e10)

    def _randomize_dynamics(self, rng: np.random.Generator):
        super()._randomize_dynamics(rng)
        self._pin_fixed()

    def _apply_base_dynamics(self):
        super()._apply_base_dynamics()
        self._pin_fixed()

    def _seat_on_ground(self):
        if self.planar or self.torque_control:
            return
        sc = self.cfg.scene
        ee_r = float(self.obj_geom[0, 0])
        y = max(float(self.ee_rigid.origin[1]), ee_r + sc.spawn_drop)
        self.ee_rigid.origin[1] = np.float32(y)
        _patch_array(self.rm.rigidParams, (self.ee_idx, 0), self.ee_rigid.origin)

    def _randomize_sizes(self, rng: np.random.Generator):
        if int(self.obj_types[0]) == OBJ_TYPE_BOX:
            return
        sc = self.cfg.scene
        lo, hi = getattr(sc, "dr_ee_r_scale", (1.0, 1.0))
        if sc.dr_size_enabled and hi > lo:
            r = float(sc.ee_radius) * float(rng.uniform(lo, hi))
        else:
            r = float(sc.ee_radius)
        self.ee_rigid.radius = r
        self.obj_geom[0] = (r, r)
        self.obj_half_heights[0] = r

    def _apply_joint_reset_quats(self):
        """Keep quat_initial at attach (0) so joint R = rot(current).

        ``looper.reset`` copies ``rigid.angle`` into both quat and
        quat_initial, which would freeze joints in the identity frame.
        """
        rigids = [self.ee_rigid] + self.obj_rigids
        for i, rigid in enumerate(rigids):
            gid = self.rigid_ids[i]
            ang = np.asarray(getattr(rigid, "angle", 0.0)).reshape(-1)
            theta = float(ang[0]) if ang.size else 0.0
            _patch_array(self.rm.quat, gid, theta)
            _patch_array(self.rm.quat_initial, gid, 0.0)
        self.rm.precompute_rigid_transforms()

    def _finish_reset(self, rng: np.random.Generator):
        sc = self.cfg.scene
        self._apply_geom_to_rm()
        self.looper.reset()
        self._apply_joint_reset_quats()
        self._seat_on_ground()
        self.rm.updateBBox()
        if sc.dr_enabled:
            self._randomize_dynamics(rng)
        else:
            self._apply_base_dynamics()
        self._cache_body_params()
        self._noise_rng = np.random.default_rng(int(rng.integers(1 << 31)))
        hold = self.hold_action()
        self.set_force(hold)


class TwoRoomEnv(LewmSceneEnv):
    """Planar two-room: two wall boxes stacked on y, gap in the middle.

    The walls share ``door_x``. Task is EE going left ↔ right through the hole.
    No payload, no ground.
    """

    planar = True
    WALL_BOT = 1
    WALL_TOP = 2

    def _build(self):
        sc = self.cfg.scene
        self.wall_hw = float(getattr(sc, "wall_hw", 0.04))
        self.door_gap = float(getattr(sc, "door_gap", 0.36))
        self.door_x = float(getattr(sc, "door_x", 1.10))
        self.door_y = float(getattr(sc, "door_y", 0.55))
        self.room_y0 = float(getattr(sc, "room_y0", -1.40))
        self.room_y1 = float(getattr(sc, "room_h", 2.40))
        bot, top = self._wall_ys()

        self.ee_rigid = BallRigid(2, [0.28, 0.55], sc.ee_radius, sc.ee_mass)
        self.ee_domain = RigidBodyDomain(
            self.ee_rigid, bcs=[Force([0], [0.0, 0.0])], friction=sc.friction
        )
        wall_bot = BoxRigid(
            2, [self.door_x, bot[0]], [2.0 * self.wall_hw, 2.0 * bot[1]],
            [0.0, 0.0], 20.0)
        wall_top = BoxRigid(
            2, [self.door_x, top[0]], [2.0 * self.wall_hw, 2.0 * top[1]],
            [0.0, 0.0], 20.0)
        self.obj_rigids = [wall_bot, wall_top]
        self.obj_types = [OBJ_TYPE_EE, OBJ_TYPE_BOX, OBJ_TYPE_BOX]
        self.obj_half_heights = [sc.ee_radius, bot[1], top[1]]
        self.obj_domains = [
            RigidBodyDomain(wall_bot, bcs=[FixedAll([0])], friction=sc.friction),
            RigidBodyDomain(wall_top, bcs=[FixedAll([0])], friction=sc.friction),
        ]
        self.fixed_slots = {self.WALL_BOT, self.WALL_TOP}
        self._register()

    def attract_indices(self):
        return []

    def _wall_ys(self, door_y=None, door_gap=None):
        """((bot_cy, bot_hh), (top_cy, top_hh)) for walls aligned at door_x."""
        dy = float(self.door_y if door_y is None else door_y)
        gap = float(self.door_gap if door_gap is None else door_gap)
        y0, y1 = float(self.room_y0), float(self.room_y1)
        gap_lo = dy - 0.5 * gap
        gap_hi = dy + 0.5 * gap
        bot_hh = max(0.04, 0.5 * (gap_lo - y0))
        bot_cy = y0 + bot_hh
        top_hh = max(0.04, 0.5 * (y1 - gap_hi))
        top_cy = gap_hi + top_hh
        return (bot_cy, bot_hh), (top_cy, top_hh)

    def _place_walls(self):
        bot, top = self._wall_ys()
        wb = self.obj_rigids[self.WALL_BOT - 1]
        wt = self.obj_rigids[self.WALL_TOP - 1]
        wb.ext = np.array([2.0 * self.wall_hw, 2.0 * bot[1]], dtype=np.float32)
        wt.ext = np.array([2.0 * self.wall_hw, 2.0 * top[1]], dtype=np.float32)
        _set_pose(wb, [self.door_x, bot[0]], 0.0)
        _set_pose(wt, [self.door_x, top[0]], 0.0)
        self.obj_geom[self.WALL_BOT] = (self.wall_hw, bot[1])
        self.obj_geom[self.WALL_TOP] = (self.wall_hw, top[1])
        self.obj_half_heights[self.WALL_BOT] = bot[1]
        self.obj_half_heights[self.WALL_TOP] = top[1]

    def _min_door_gap(self) -> float:
        """Opening must exceed EE diameter plus a small clearance."""
        ee_r = float(self.obj_geom[0, 0])
        return 2.0 * ee_r + 0.08

    def _ensure_passable_door(self):
        need = self._min_door_gap()
        if self.door_gap + 1e-6 < need:
            self.door_gap = need
            self._place_walls()
            self._apply_geom_to_rm()
            self.rm.updateBBox()

    def set_ee_radius(self, radius: float):
        super().set_ee_radius(radius)
        self._ensure_passable_door()

    def _reset_once(self, rng: np.random.Generator):
        self._randomize_sizes(rng)
        need = self._min_door_gap()
        lo = max(0.32, need)
        hi = max(lo + 0.06, 0.42)
        self.door_gap = float(rng.uniform(lo, hi))
        self.door_x = float(rng.uniform(1.02, 1.20))
        self.door_y = float(rng.uniform(0.42, 0.68))
        self._place_walls()
        self._ensure_passable_door()

        ee_r = float(self.obj_geom[0, 0])
        left = rng.random() < 0.5
        self.spawn_left = bool(left)
        margin = self.wall_hw + ee_r + 0.10

        def _unif(lo, hi, fallback):
            lo, hi = float(lo), float(hi)
            if hi > lo + 1e-4:
                return float(rng.uniform(lo, hi))
            return float(fallback)

        if left:
            ex = _unif(0.16, self.door_x - margin, 0.28)
        else:
            ex = _unif(self.door_x + margin, 2.05, 1.85)
        ey = _unif(0.12, 0.98, 0.55)
        _set_pose(self.ee_rigid, [ex, ey], 0.0)
        self._finish_reset(rng)


class PushTEnv(LewmSceneEnv):
    """Planar T (two boxes + weld); EE pushes it to a pose. No gravity."""

    planar = True
    STEM = 1
    BAR = 2

    def _build(self):
        sc = self.cfg.scene
        stem_full = np.asarray(getattr(sc, "stem_ext", (0.08, 0.28)), dtype=np.float64)
        bar_full = np.asarray(getattr(sc, "bar_ext", (0.32, 0.08)), dtype=np.float64)
        self.stem_hw, self.stem_hh = 0.5 * stem_full[0], 0.5 * stem_full[1]
        self.bar_hw, self.bar_hh = 0.5 * bar_full[0], 0.5 * bar_full[1]
        stem_xy, bar_xy, weld = self._t_parts(0.70, 0.40, 0.0)

        self.ee_rigid = BallRigid(2, [0.28, 0.40], sc.ee_radius, sc.ee_mass)
        self.ee_domain = RigidBodyDomain(
            self.ee_rigid, bcs=[Force([0], [0.0, 0.0])], friction=sc.friction
        )
        stem = BoxRigid(2, stem_xy, stem_full, [0.0, 0.0], float(getattr(sc, "stem_mass", 0.45)))
        bar = BoxRigid(2, bar_xy, bar_full, [0.0, 0.0], float(getattr(sc, "bar_mass", 0.40)))
        self.obj_rigids = [stem, bar]
        self.obj_types = [OBJ_TYPE_EE, OBJ_TYPE_BOX, OBJ_TYPE_BOX]
        self.obj_half_heights = [sc.ee_radius, self.stem_hh, self.bar_hh]
        self.obj_domains = [
            RigidBodyDomain(stem, bcs=self._maybe_gravity(), friction=sc.friction),
            RigidBodyDomain(bar, bcs=self._maybe_gravity(), friction=sc.friction),
        ]
        self.connected_pairs = {(self.STEM, self.BAR)}
        self.joint_rel = {(self.STEM, self.BAR): REL_WELD}
        joint = WeldJoint(1, 2, weld, bcs=[])
        self._register(joints=[joint])

    def _t_local(self):
        stem = np.array([0.0, self.stem_hh], dtype=np.float64)
        bar = np.array([0.0, 2.0 * self.stem_hh + self.bar_hh], dtype=np.float64)
        weld = np.array([0.0, 2.0 * self.stem_hh], dtype=np.float64)
        return stem, bar, weld

    def _t_parts(self, cx: float, cy: float, theta: float):
        stem_l, bar_l, weld_l = self._t_local()
        R = _rot(theta)
        origin = np.array([cx, cy], dtype=np.float64)
        stem = origin + R @ stem_l
        bar = origin + R @ bar_l
        weld = origin + R @ weld_l
        return stem, bar, weld

    def t_goal_parts(self, gx: float, gy: float, theta: float = 0.0):
        stem, bar, _ = self._t_parts(gx, gy, theta)
        return np.asarray(stem, dtype=np.float32), np.asarray(bar, dtype=np.float32)

    def _spawn_ee_xy(self, rng, stem_xy, bar_xy, theta, clearance: float = 0.10):
        """EE outside both T boxes by ``clearance`` (no initial penetration)."""
        ee_r = float(self.obj_geom[0, 0])
        stem_xy = np.asarray(stem_xy, dtype=np.float64)
        bar_xy = np.asarray(bar_xy, dtype=np.float64)
        bodies = (
            (stem_xy, self.stem_hw, self.stem_hh),
            (bar_xy, self.bar_hw, self.bar_hh),
        )
        for _ in range(64):
            box, hw, hh = bodies[int(rng.integers(0, 2))]
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            u = np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
            rad = ee_r + float(np.hypot(hw, hh)) + clearance
            ee = box + u * rad
            ee[0] = float(np.clip(ee[0], 0.18, 2.04))
            ee[1] = float(np.clip(ee[1], 0.14, 0.96))
            g_s = _circle_obb_gap(ee, ee_r, stem_xy, self.stem_hw, self.stem_hh, theta)
            g_b = _circle_obb_gap(ee, ee_r, bar_xy, self.bar_hw, self.bar_hh, theta)
            a_s = _aabb_sep(ee, ee_r, stem_xy, self.stem_hw, self.stem_hh)
            a_b = _aabb_sep(ee, ee_r, bar_xy, self.bar_hw, self.bar_hh)
            if min(g_s, g_b) >= clearance and min(a_s, a_b) >= 0.04:
                return ee
        left = np.array([max(0.18, float(stem_xy[0]) - 0.50),
                         float(np.clip(stem_xy[1], 0.20, 0.90))])
        return left

    def _obb_support(self, xy, hw, hh, th, nvec):
        """World-space support: center + n̂ · extent along an OBB."""
        n = np.asarray(nvec, dtype=np.float64)
        nn = float(np.linalg.norm(n))
        n = n / nn if nn > 1e-8 else np.array([-1.0, 0.0])
        c, s = float(np.cos(th)), float(np.sin(th))
        ext = abs(n[0] * c + n[1] * s) * hw + abs(-n[0] * s + n[1] * c) * hh
        center = np.asarray(xy, dtype=np.float64)
        return center + n * ext, ext, n

    def _spawn_ee_support(self, stem_xy, bar_xy, theta, gap: float = 0.015):
        """Place EE 1–2 cm behind the stem, opposite the task goal."""
        ee_r = float(self.obj_geom[0, 0])
        stem_xy = np.asarray(stem_xy, dtype=np.float64)
        tc = getattr(self.cfg, "task", None)
        gx = float(getattr(tc, "goal_x", 1.50))
        gy = float(getattr(tc, "goal_y", 0.52))
        c, s = float(np.cos(theta)), float(np.sin(theta))
        origin = stem_xy - np.array([-s * self.stem_hh, c * self.stem_hh])
        err = np.array([gx, gy], dtype=np.float64) - origin
        hx = -1.0 if err[0] >= 0.0 else 1.0
        nvec = np.array([hx, 0.0], dtype=np.float64)
        _, ext, n = self._obb_support(
            stem_xy, self.stem_hw, self.stem_hh, theta, nvec)
        stem_m = float(getattr(self.cfg.scene, "stem_mass", 0.45))
        bar_m = float(getattr(self.cfg.scene, "bar_mass", 0.40))
        com_y = (stem_m * stem_xy[1] + bar_m * float(bar_xy[1])) / (stem_m + bar_m)
        ee = stem_xy + n * (ee_r + ext + float(gap))
        ee[1] = float(com_y)
        for _ in range(10):
            ee[0] = float(np.clip(ee[0], 0.18, 2.04))
            ee[1] = float(np.clip(ee[1], 0.16, 0.92))
            g_s = _circle_obb_gap(ee, ee_r, stem_xy, self.stem_hw, self.stem_hh, theta)
            g_b = _circle_obb_gap(
                ee, ee_r, bar_xy, self.bar_hw, self.bar_hh, theta)
            need = min(g_s, g_b)
            if need >= 0.5 * gap:
                self._spawn_ee_gap = float(need)
                return ee
            ee = ee + n * 0.025
        self._spawn_ee_gap = float(min(
            _circle_obb_gap(ee, ee_r, stem_xy, self.stem_hw, self.stem_hh, theta),
            _circle_obb_gap(ee, ee_r, bar_xy, self.bar_hw, self.bar_hh, theta)))
        return ee

    def _reset_once(self, rng: np.random.Generator):
        sc = self.cfg.scene
        self._randomize_sizes(rng)
        cx = float(rng.uniform(0.45, 0.95))
        cy = float(rng.uniform(0.22, 0.85))
        theta = float(rng.uniform(-2.4, 2.4))
        stem_xy, bar_xy, _ = self._t_parts(cx, cy, theta)
        _set_pose(self.obj_rigids[0], stem_xy, theta)
        _set_pose(self.obj_rigids[1], bar_xy, theta)
        mode = str(getattr(self, "ee_spawn", None)
                   or getattr(self.cfg.task, "ee_spawn", "default"))
        if mode == "support":
            ee = self._spawn_ee_support(stem_xy, bar_xy, theta, gap=0.015)
        else:
            ee = self._spawn_ee_xy(rng, stem_xy, bar_xy, theta, clearance=0.10)
            ee_r = float(self.obj_geom[0, 0])
            self._spawn_ee_gap = float(min(
                _circle_obb_gap(ee, ee_r, stem_xy, self.stem_hw, self.stem_hh, theta),
                _circle_obb_gap(ee, ee_r, bar_xy, self.bar_hw, self.bar_hh, theta)))
        _set_pose(self.ee_rigid, ee, 0.0)
        self._finish_reset(rng)


class ReacherEnv(LewmSceneEnv):
    """Gravity two-link arm. No circular EE, no ground, no rigid-rigid contact.

    Domain 0 is link 1 (box). Revolute anchors are the shared box faces:
    shoulder at the base/link1 junction, elbow at the link1/link2 junction.
    Joints are attached at the stretched rest pose (θ=0) so local frames
    sit on those faces, not at the box centers.

    Action is (θ1, θ2) PD targets. ``use_pd=2`` converts them to joint
    torques internally: τ = kp·(target − angle) − kd·ω.
    """

    torque_control = True
    use_pd = 2
    rigid_rigid_contact = False
    LINK1 = 0
    BASE = 1
    LINK2 = 2

    def _build(self):
        sc = self.cfg.scene
        self.use_pd = int(getattr(sc, "use_pd", 2))
        self.l1 = float(getattr(sc, "link1_len", 0.28))
        self.l2 = float(getattr(sc, "link2_len", 0.24))
        thick = float(getattr(sc, "link_thick", 0.06))
        self.link_hh = 0.5 * thick
        self.pivot = np.array(getattr(sc, "pivot", (1.00, 0.52)), dtype=np.float64)
        base_hw, base_hh = 0.045, 0.045
        self.base_hw, self.base_hh = base_hw, base_hh
        mass = float(getattr(sc, "link_mass", 0.35))
        # Attach at θ=0 so joint_l1/l2 are body-local offsets to the faces.
        bxy, l1xy, l2xy, tip, a1, a2 = self._fk(0.0, 0.0)

        self.ee_rigid = BoxRigid(
            2, l1xy, [self.l1, thick], [0.0, 0.0], mass)
        self.ee_domain = RigidBodyDomain(
            self.ee_rigid,
            bcs=self._maybe_gravity(),
            friction=sc.friction,
        )
        base = BoxRigid(
            2, bxy, [2.0 * base_hw, 2.0 * base_hh], [0.0, 0.0], 8.0)
        link2 = BoxRigid(
            2, l2xy, [self.l2, thick], [0.0, 0.0], mass)
        self.obj_rigids = [base, link2]
        self.obj_types = [OBJ_TYPE_BOX, OBJ_TYPE_BOX, OBJ_TYPE_BOX]
        self.obj_half_heights = [self.link_hh, base_hh, self.link_hh]
        self.obj_domains = [
            RigidBodyDomain(base, bcs=[FixedAll([0])], friction=sc.friction),
            RigidBodyDomain(link2, bcs=self._maybe_gravity(),
                            friction=sc.friction),
        ]
        self.fixed_slots = {self.BASE}
        self.connected_pairs = {
            (self.LINK1, self.BASE), (self.LINK1, self.LINK2),
            (self.BASE, self.LINK2),
        }
        self.joint_rel = {
            (self.LINK1, self.BASE): REL_REVOLUTE,
            (self.LINK1, self.LINK2): REL_REVOLUTE,
        }
        # domain 0=link1, 1=base, 2=link2. stiff/damping become PD kp/kd.
        j1 = RevoluteJoint(1, 0, a1, axis=[0, 0], bcs=[], stiff=100.0, damping=3.0,
                           velocity_limit=8.0, effort_limit=4.0)
        j2 = RevoluteJoint(0, 2, a2, axis=[0, 0], bcs=[], stiff=100.0, damping=3.0,
                           velocity_limit=8.0, effort_limit=4.0)
        self._register(joints=[j1, j2])

    def min_separating_gap(self) -> float:
        return 1.0

    def _fk(self, th1: float, th2: float):
        """Centers, distal tip, and face anchors (pivot, elbow)."""
        pivot = self.pivot
        R1 = _rot(th1)
        R12 = _rot(th1 + th2)
        u1 = R1 @ np.array([self.l1, 0.0], dtype=np.float64)
        u2 = R12 @ np.array([self.l2, 0.0], dtype=np.float64)
        elbow = pivot + u1
        tip = elbow + u2
        l1c = pivot + 0.5 * u1
        l2c = elbow + 0.5 * u2
        base = pivot.copy()
        return base, l1c, l2c, tip, pivot.copy(), elbow.copy()

    def link_goals(self, th1: float, th2: float):
        _, l1c, l2c, tip, _, _ = self._fk(th1, th2)
        return (np.asarray(l1c, dtype=np.float32),
                np.asarray(l2c, dtype=np.float32),
                np.asarray(tip, dtype=np.float32))

    def tip_xy(self, th1: float, th2: float) -> np.ndarray:
        return self.link_goals(th1, th2)[2]

    def tip_from_states(self, st: np.ndarray) -> np.ndarray:
        c = np.asarray(st[self.LINK2, :2], dtype=np.float64)
        th = float(st[self.LINK2, 2])
        return (c + _rot(th) @ np.array([0.5 * self.l2, 0.0], dtype=np.float64)
                ).astype(np.float32)

    def goal_reachable(self, xy, margin: float = 0.01, scored: str = "center") -> bool:
        """Workspace of the scored point on link 2.

        ``center``: link2 box origin, reach |L1 − L2/2| … L1 + L2/2.
        ``tip``: distal corner, reach |L1 − L2| … L1 + L2.
        Mixing a tip-valid goal with a center metric leaves a half-link
        annulus that the box origin can never enter.
        """
        d = float(np.linalg.norm(np.asarray(xy, dtype=np.float64)[:2] - self.pivot))
        half = 0.5 * self.l2
        if scored == "tip":
            inner, outer = abs(self.l1 - self.l2), self.l1 + self.l2
        else:
            inner, outer = abs(self.l1 - half), self.l1 + half
        return inner + margin <= d <= outer - margin

    def set_force(self, f):
        """Write (θ1, θ2) into joint_control_target for PD (use_pd=2)."""
        a = np.asarray(f, dtype=np.float64).reshape(-1)
        t1 = float(a[0])
        t2 = float(a[1]) if a.size > 1 else 0.0
        if int(getattr(self, "use_pd", 0)) == 2:
            _patch_array(self.rm.joint_control_target, 0, t1)
            _patch_array(self.rm.joint_control_target, 1, t2)
            return
        _patch_array(self.rm.bcRValues, self.rigid_ids[self.LINK1], t1 - t2)
        _patch_array(self.rm.bcRValues, self.rigid_ids[self.LINK2], t2)

    def joint_angles(self) -> np.ndarray:
        """Relative revolute angles matching RigidManager PD (qb−q0b)−(qa−q0a)."""
        quat = self.rm.quat.numpy()
        q0 = self.rm.quat_initial.numpy()
        ja = self.rm.joint_id_a.numpy()
        jb = self.rm.joint_id_b.numpy()
        out = np.zeros(2, dtype=np.float32)
        n = min(2, int(self.rm.numAnchors))
        for j in range(n):
            a, b = int(ja[j]), int(jb[j])
            ang = (float(quat[b]) - float(q0[b])) - (float(quat[a]) - float(q0[a]))
            out[j] = np.arctan2(np.sin(ang), np.cos(ang))
        return out

    def hold_action(self) -> np.ndarray:
        return self.joint_angles()

    def attract_indices(self):
        return []

    def _reset_once(self, rng: np.random.Generator):
        th1 = float(rng.uniform(-2.4, 2.4))
        th2 = float(rng.uniform(-2.2, 2.2))
        bxy, l1xy, l2xy, _, _, _ = self._fk(th1, th2)
        _set_pose(self.ee_rigid, l1xy, th1)
        _set_pose(self.obj_rigids[0], bxy, 0.0)
        _set_pose(self.obj_rigids[1], l2xy, th1 + th2)
        self._angles0 = (th1, th2)
        self._finish_reset(rng)


def make_scene_env(cfg: Config, device: str = None):
    kind = getattr(cfg.scene, "kind", "push")
    if kind == "two_room":
        return TwoRoomEnv(cfg, device=device)
    if kind in ("push_t", "push-t"):
        return PushTEnv(cfg, device=device)
    if kind == "reacher":
        return ReacherEnv(cfg, device=device)
    return PushSceneEnv(cfg, device=device)
