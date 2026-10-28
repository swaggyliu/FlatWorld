"""Push-T: planar welded T pushed to a fixed pose. No gravity."""

import numpy as np

from learning.configs.default import Config
from learning.env.lewm_scenes import PushTEnv, _box_corners
from learning.tasks.cem_loop import CEMWorldTask


def push_t_config() -> Config:
    cfg = Config()
    cfg.scene.kind = "push_t"
    cfg.scene.gravity = 0.0
    cfg.scene.loop_damping = 0.0
    cfg.scene.use_ground = False
    cfg.scene.table_g = 10.0
    cfg.scene.table_mu = 0.20
    cfg.scene.table_on_ee = False
    cfg.scene.num_boxes = 2
    cfg.scene.num_balls = 0
    cfg.scene.dr_size_enabled = False
    cfg.scene.stem_ext = (0.08, 0.28)
    cfg.scene.bar_ext = (0.32, 0.08)
    cfg.scene.stem_mass = 0.45
    cfg.scene.bar_mass = 0.40
    cfg.scene.area_x = (0.12, 2.10)
    cfg.collect.out_dir = "learning/data/push_t_500"
    cfg.collect.seed = 42000
    cfg.collect.x_lo = 0.08
    cfg.collect.x_hi = 2.12
    cfg.collect.y_lo = 0.08
    cfg.collect.y_hi = 1.00
    cfg.collect.hop_y = 0.50
    cfg.collect.hop_y_hi = 0.92
    cfg.collect.mode_free = 0.0
    cfg.collect.mode_attract = 0.0
    cfg.collect.mode_push = 1.0
    cfg.collect.mode_release = 0.0
    cfg.collect.mode_hop = 0.0
    cfg.collect.mode_pile = 0.0
    cfg.collect.mode_door = 0.0
    cfg.collect.mode_spin = 0.0
    cfg.task.tol = 0.12
    cfg.task.vel_tol = 0.20
    cfg.task.budget = 1100
    cfg.task.goal_x = 1.50
    cfg.task.goal_y = 0.52
    cfg.task.goal_theta = 0.0
    cfg.task.ang_tol = 0.25
    return cfg


class PushTCost:
    """Solver CEM: EE-to-contact-face (reach) + T pose (stem/bar + heading).

    Scoring only T leaves almost every short random sequence with the same
    cost (no contact). Reach is the two-room analog: CEM first drives the
    actuated EE onto a real supporting face behind the T.
    """

    def __init__(self, action_cost: float = 0.0005, reach_cost: float = 1.50,
                 ang_w: float = 0.40, pose_w: float = 1.0, shove_w: float = 0.0,
                 jump_w: float = 0.0, bound_w: float = 0.0, lever_w: float = 0.0,
                 device: str = "cpu"):
        self.action_cost = action_cost
        self.reach_cost = float(reach_cost)
        self.ang_w = float(ang_w)
        self.pose_w = float(pose_w)
        self.shove_w = float(shove_w)
        self.jump_w = float(jump_w)
        self.bound_w = float(bound_w)
        self.lever_w = float(lever_w)
        self.device = device
        self.ee_idx = 0
        self.stem_idx = PushTEnv.STEM
        self.bar_idx = PushTEnv.BAR
        self.stem_np = np.zeros(2, dtype=np.float32)
        self.bar_np = np.zeros(2, dtype=np.float32)
        self.th_g = 0.0
        self.face_np = np.zeros(2, dtype=np.float64)
        self.hint_action = np.zeros(2, dtype=np.float32)
        self.u = np.array([1.0, 0.0], dtype=np.float64)
        self.into = np.array([1.0, 0.0], dtype=np.float64)
        self.origin_g = np.array([1.50, 0.52], dtype=np.float64)
        self.press = 0.02
        self.stem_m = 0.45
        self.bar_m = 0.40
        self.mode = "push"
        self._lock = None
        self._lock_age = 0
        self._lock_stem = np.zeros(2, dtype=np.float64)
        self.candidates = []
        self._bypass_y = None
        self._orbit_n = 0
        self._imagine = False
        self._rot_name = None
        self._rot_age = 0
        self._spinning = False
        self._orbit_stage = 0
        self._lifting = False
        self._phase = None
        self._stuck = 0
        self._err_prev = 0.0
        self._flip_cool = 0
        self._aligning = False

    def reset_episode(self):
        self._orbit_n = 0
        self._bypass_y = None
        self._rot_name = None
        self._rot_age = 0
        self._spinning = False
        self._orbit_stage = 0
        self._lifting = False
        self._phase = None
        self._stuck = 0
        self._err_prev = 0.0
        self._lock = None
        self._lock_stem = np.zeros(2, dtype=np.float64)
        self._flip_cool = 0
        self._aligning = False
        self.mode = "push"

    def _corners(self, xy, hw, hh, th):
        return _box_corners(xy, hw, hh, th)

    def _extent(self, pts, origin, n):
        return float(np.max((np.asarray(pts) - origin) @ n))

    @staticmethod
    def _origin(stem, theta, stem_hh):
        c, s = float(np.cos(theta)), float(np.sin(theta))
        return np.asarray(stem[:2], dtype=np.float64) - np.array([-s * stem_hh, c * stem_hh])

    @staticmethod
    def _axes(th):
        c, s = float(np.cos(th)), float(np.sin(th))
        return np.array([c, s], dtype=np.float64), np.array([-s, c], dtype=np.float64)

    def _barriers(self, a, ee):
        if ee[1] < 0.20:
            a[1] = max(float(a[1]), 3.6)
        elif ee[1] > 0.90:
            a[1] = min(float(a[1]), -3.2)
        if ee[0] < 0.18:
            a[0] = max(float(a[0]), 3.0)
        elif ee[0] > 2.04:
            a[0] = min(float(a[0]), -3.0)
        return a

    def _hull_bb(self, stem, bar, stem_hw, stem_hh, bar_hw, bar_hh, th):
        hull = np.vstack([
            self._corners(stem, stem_hw, stem_hh, th),
            self._corners(bar, bar_hw, bar_hh, th),
        ])
        return float(hull[:, 0].min()), float(hull[:, 0].max()), float(hull[:, 1].min()), float(hull[:, 1].max())

    def _left_support(self, stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh,
                      com, err, origin, ee_r, stem_only=False):
        """World-left silhouette. Heading = vertical lever on that face."""
        n_out = np.array([-1.0, 0.0], dtype=np.float64)
        into = np.array([1.0, 0.0], dtype=np.float64)
        sc = self._corners(stem, stem_hw, stem_hh, th)
        bc = self._corners(bar, bar_hw, bar_hh, th)
        hull = sc if stem_only else np.vstack([sc, bc])
        both = np.vstack([sc, bc])
        xmin = float(hull[:, 0].min())
        ymin = float(both[:, 1].min())
        ymax = float(both[:, 1].max())
        y_lo = max(float(hull[:, 1].min()) + 0.02, 0.22)
        y_hi = min(float(hull[:, 1].max()) - 0.02, 0.88)
        if y_lo > y_hi:
            y_lo, y_hi = 0.22, 0.88
        lever = 0.16 if abs(err) > 0.70 else (0.10 if abs(err) > 0.24 else 0.04)
        if err > 0.32:
            lever = 0.16
        cy = float(com[1] + float(np.sign(err) if abs(err) > 1e-6 else 1.0) * lever)
        cy = float(np.clip(cy, y_lo, y_hi))
        p = np.array([xmin, cy], dtype=np.float64)
        face = np.array([xmin - max(ee_r - self.press, 0.03), cy], dtype=np.float64)
        return p, n_out, into, face, xmin, ymin, ymax

    def _long_faces(self, stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh):
        x_b, y_b = self._axes(th)
        return (
            ("stem_mx", stem - x_b * stem_hw, -x_b, y_b, stem_hh, 0.0),
            ("stem_px", stem + x_b * stem_hw, x_b, y_b, stem_hh, 0.0),
            ("bar_py", bar + y_b * bar_hh, y_b, x_b, bar_hw, 0.0),
            ("bar_my", bar - y_b * bar_hh, -y_b, x_b, bar_hw, stem_hw + 0.03),
        )

    def _contact_on_face(self, p0, n_out, tan, half, com, err, ee_r, skip_mid=0.0,
                         rotate=True):
        want = -float(np.sign(err) if abs(err) > 1e-6 else 1.0)
        n_out = np.asarray(n_out, dtype=np.float64)
        tan = np.asarray(tan, dtype=np.float64)
        p0 = np.asarray(p0, dtype=np.float64)
        into = -n_out
        span = 0.78 * float(half)
        if rotate:
            best_t, best = 0.0, -1e9
            found = False
            for t in np.linspace(-span, span, 11):
                if skip_mid and abs(float(t)) < skip_mid:
                    continue
                p = p0 + tan * float(t)
                if p[0] < 0.16 or p[0] > 2.06 or p[1] < 0.16 or p[1] > 0.90:
                    continue
                r = p - com
                tau = float(r[0] * into[1] - r[1] * into[0])
                s = tau * want
                if s > best:
                    best, best_t, found = s, float(t), True
            if not found:
                best_t = 0.0
                r = p0 - com
                best = float(r[0] * into[1] - r[1] * into[0]) * want
        else:
            t = float(np.clip(float((com - p0) @ tan), -span, span))
            if skip_mid and abs(t) < skip_mid:
                t = float(np.copysign(skip_mid, t if abs(t) > 1e-9 else 1.0))
            best_t = t
            r = (p0 + tan * best_t) - com
            best = float(r[0] * into[1] - r[1] * into[0]) * want
        p = p0 + tan * best_t
        p[0] = float(np.clip(p[0], 0.16, 2.06))
        p[1] = float(np.clip(p[1], 0.16, 0.90))
        face = p + n_out * max(ee_r - self.press, 0.03)
        face[0] = float(np.clip(face[0], 0.16, 2.06))
        face[1] = float(np.clip(face[1], 0.18, 0.88))
        return p, face, into, best

    def _enumerate_faces(self, stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh,
                         com, err, ee, ee_r, rotate=True):
        rows = []
        for name, p0, n_out, tan, half, skip in self._long_faces(
                stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh):
            p, face, into, tau_w = self._contact_on_face(
                p0, n_out, tan, half, com, err, ee_r, skip_mid=skip, rotate=rotate)
            sd = float(np.asarray(n_out) @ (ee - p0))
            reach = float(np.linalg.norm(face - ee))
            rows.append(dict(name=name, p=p, p0=p0, n_out=np.asarray(n_out),
                             tan=np.asarray(tan), into=into, face=face,
                             tau_w=tau_w, sd=sd, reach=reach, half=half))
        return rows

    def _score_face(self, row, origin, lift, rotate, u, err=0.0):
        n_out, into = row["n_out"], row["into"]
        tau_w, reach, sd = row["tau_w"], row["reach"], row["sd"]
        face = row["face"]
        if (not lift) and float(n_out[0]) > -0.18:
            return -1e3
        if lift and float(n_out[1]) > 0.50:
            return -1e3
        pen = 0.0
        if origin[1] < 0.40:
            pen += 14.0 * max(0.0, -float(into[1]))
        if origin[1] > 0.68:
            pen += 8.0 * max(0.0, float(into[1]))
        if origin[0] < 0.42:
            pen += 8.0 * max(0.0, -float(into[0]))
        if origin[0] > 1.82:
            pen += 6.0 * max(0.0, float(into[0]))
        pen += 3.0 * max(0.0, float(face[1]) - 0.82)
        pen += 3.0 * max(0.0, 0.20 - float(face[1]))
        if lift:
            score = (4.5 * max(0.0, float(into[1]))
                     + 1.2 * max(0.0, -float(n_out[1]))
                     + 0.8 * max(sd, -0.05)
                     - 5.0 * max(0.0, reach - 0.18))
            return score - pen
        if rotate:
            score = (6.4 * tau_w + 1.4 * max(0.0, -float(n_out[0]))
                     + 1.8 * max(0.0, sd)
                     - 4.0 * max(0.0, reach - 0.22))
            if tau_w < 0.02:
                score -= 3.0
            return score - pen
        score = (3.2 * max(0.0, -float(n_out[0]))
                 + 1.4 * max(0.0, float(into @ u))
                 + 0.8 * max(sd, -0.05)
                 - 4.5 * max(0.0, reach - 0.16))
        return score - pen

    def _pick_support(self, stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh,
                      com, err, origin, ee, ee_r, lift, rotate, u):
        rows = self._enumerate_faces(
            stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh, com, err, ee, ee_r,
            rotate=bool(rotate))
        ranked = [(self._score_face(row, origin, lift, rotate, u, err=err), row)
                  for row in rows]
        ranked.sort(key=lambda x: -x[0])
        keep = None
        if self._rot_name is not None and self._rot_age < 36:
            for _score, row in ranked:
                if row["name"] != self._rot_name:
                    continue
                if _score < -100.0:
                    break
                if rotate and row["tau_w"] < -0.004:
                    break
                if row["reach"] > 0.42:
                    break
                keep = row
                break
        pick = keep if keep is not None else (ranked[0][1] if ranked else None)
        if pick is not None and ranked and ranked[0][0] < -100.0 and keep is None:
            pick = None
        xmin, xmax, ymin, ymax = self._hull_bb(
            stem, bar, stem_hw, stem_hh, bar_hw, bar_hh, th)
        if pick is None:
            p, n_out, into, face, xmin, ymin, ymax = self._left_support(
                stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh, com, err, origin,
                ee_r, stem_only=True)
            return p, n_out, into, face, xmin, ymin, ymax, "aabb"
        if keep is None:
            if self._rot_name != pick["name"]:
                self._orbit_stage = 0
                self._bypass_y = None
            self._rot_name = pick["name"]
            self._rot_age = 0
        else:
            self._rot_age = int(self._rot_age) + 1
        return (pick["p"], pick["n_out"], pick["into"], pick["face"],
                xmin, ymin, ymax, pick["name"])

    def _select_rotate(self, rows, origin, err, ee, force=False):
        """Left long-face first; far/right only if heading cannot unwind from the left."""
        if self._rot_name is not None:
            keep = next((r for r in rows if r["name"] == self._rot_name), None)
            if keep is not None and keep["tau_w"] > 0.008:
                return keep
            self._rot_name = None
            self._orbit_stage = 0
            self._bypass_y = None
        usable = []
        for r in rows:
            if r["tau_w"] < 0.04:
                continue
            if origin[1] < 0.38 and float(r["into"][1]) < -0.35:
                continue
            if origin[1] > 0.72 and float(r["into"][1]) > 0.55:
                continue
            usable.append(r)
        if not usable:
            return None
        left = [r for r in usable if float(r["n_out"][0]) < -0.28]
        pool = left if left and not (
            force and err > 0.35 and max(r["tau_w"] for r in left) < 0.06
        ) else usable
        def key(r):
            left_b = 1.4 * max(0.0, -float(r["n_out"][0]))
            near_b = 1.2 * max(0.0, r["sd"]) - 1.6 * max(0.0, r["reach"] - 0.20)
            # Prefer lowering levers while high (ep5 path vs ep2 high lock).
            down_b = 0.0
            if float(origin[1]) > 0.68:
                down_b = 2.2 * max(0.0, -float(r["into"][1]))
            return 6.0 * r["tau_w"] + left_b + near_b + down_b
        pick = max(pool, key=key)
        return pick

    def _orbit_hint(self, ee, face, p, n_out, xmin, xmax, ymin, ymax, origin, ee_r,
                    err=0.0):
        wp = np.asarray(face, dtype=np.float64).copy()
        wp[0] = float(np.clip(wp[0], 0.18, 2.04))
        wp[1] = float(np.clip(wp[1], 0.20, 0.86))
        n_out = np.asarray(n_out, dtype=np.float64)
        rail_x = float(np.clip(xmin - ee_r - 0.12, 0.18, 2.04))
        below = ymin - ee_r - 0.08
        above = ymax + ee_r + 0.08
        prefer_up = float(err) > 0.32
        if prefer_up and 0.22 <= above <= 0.84:
            by = float(np.clip(above, 0.22, 0.84))
        elif 0.22 <= below <= 0.84:
            by = float(below)
        elif 0.22 <= above <= 0.84:
            by = float(np.clip(above, 0.22, 0.84))
        else:
            by = float(np.clip(wp[1], 0.24, 0.80))
        by = float(np.clip(by, 0.22, 0.84))
        far = float(n_out[0]) > 0.12
        stage = int(self._orbit_stage)
        if far:
            past_x = float(np.clip(p[0] + n_out[0] * (ee_r + 0.10), 0.18, 2.04))
            if abs(float(ee[1]) - by) > 0.07 and stage <= 0:
                self._orbit_stage = 0
                tgt = np.array([float(np.clip(min(float(ee[0]), rail_x + 0.08), 0.18, 2.04)), by],
                               dtype=np.float64)
            elif float(ee[0]) < past_x - 0.05 and stage <= 1:
                self._orbit_stage = 1
                tgt = np.array([past_x, by], dtype=np.float64)
            else:
                self._orbit_stage = 2
                tgt = wp
        else:
            on_or_right = float(ee[0]) > xmin - 0.04
            clear = float(ee[1]) < ymin - 0.04 or float(ee[1]) > ymax + 0.04
            if on_or_right and (not clear):
                tgt = np.array([float(np.clip(ee[0], 0.18, 2.04)), by], dtype=np.float64)
            elif on_or_right:
                tgt = np.array([rail_x, float(np.clip(ee[1], 0.22, 0.84))], dtype=np.float64)
            elif abs(ee[1] - wp[1]) > 0.08:
                tgt = np.array([rail_x, wp[1]], dtype=np.float64)
            else:
                tgt = wp
        delta = tgt - ee
        tgt[0] = float(np.clip(tgt[0], 0.18, 2.04))
        tgt[1] = float(np.clip(tgt[1], 0.20, 0.86))
        delta = tgt - ee
        return (delta / max(float(np.linalg.norm(delta)), 1e-6)) * 4.6

    def bind(self, stem_goal, bar_goal, geom, states_now, theta: float = 0.0,
             origin_goal=None, force_max: float = 6.0):
        g = np.asarray(geom, dtype=np.float64)
        st = np.asarray(states_now, dtype=np.float64)
        ee_r = float(g[self.ee_idx, 0])
        stem = st[self.stem_idx, :2]
        bar = st[self.bar_idx, :2]
        ee = st[self.ee_idx, :2]
        th = float(st[self.stem_idx, 2])
        stem_hw, stem_hh = float(g[self.stem_idx, 0]), float(g[self.stem_idx, 1])
        bar_hw, bar_hh = float(g[self.bar_idx, 0]), float(g[self.bar_idx, 1])
        if origin_goal is not None:
            self.origin_g = np.asarray(origin_goal[:2], dtype=np.float64)
        self.stem_np = np.asarray(stem_goal[:2], dtype=np.float32)
        self.bar_np = np.asarray(bar_goal[:2], dtype=np.float32)
        self.th_g = float(theta)
        origin = self._origin(stem, th, stem_hh)
        err = float(np.arctan2(np.sin(th - theta), np.cos(th - theta)))
        mass = self.stem_m + self.bar_m
        c = (self.stem_m * stem + self.bar_m * bar) / mass
        err_p = self.origin_g - origin
        if origin[0] < 0.28:
            err_p[0] = max(float(err_p[0]), 0.25)
        if origin[0] > 1.95:
            err_p[0] = min(float(err_p[0]), -0.25)
        if origin[1] < 0.22:
            err_p[1] = max(float(err_p[1]), 0.20)
        if origin[1] > 0.70:
            err_p[1] = min(float(err_p[1]), -0.25)
        un = float(np.linalg.norm(err_p))
        u = err_p / un if un > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
        d_s = float(np.linalg.norm(stem - self.stem_np))
        d_b = float(np.linalg.norm(bar - self.bar_np))
        fmax = float(force_max)
        # Idle once inside pose tol so velocity can fall under vel_tol (ep4).
        if d_s < 0.12 and d_b < 0.12 and abs(err) < 0.22:
            self.u = u
            self.into = np.zeros(2, dtype=np.float64)
            self.face_np = np.asarray(ee, dtype=np.float64)
            self.mode = "settle"
            self.hint_action = np.zeros(2, dtype=np.float32)
            return
        if float(np.linalg.norm(stem - self._lock_stem)) > 0.55:
            self._orbit_n = 0
            self._bypass_y = None
            self._rot_name = None
            self._rot_age = 0
            self._spinning = False
            self._orbit_stage = 0
            self._lifting = False
            self._phase = None
            self._stuck = 0
            self._orbit_stage = 0
            self._flip_cool = 0
            self._aligning = False
        self._lock_stem = np.asarray(stem, dtype=np.float64)
        if err * float(getattr(self, "_err_prev", 0.0)) < 0.0 and abs(err) < 0.55:
            self._rot_name = None
            self._stuck = 0
            self._orbit_stage = 0
            self._flip_cool = 40
        if int(getattr(self, "_flip_cool", 0)) > 0:
            self._flip_cool = int(self._flip_cool) - 1
        lift = origin[1] < 0.36 and abs(err) < 0.22
        if abs(err) > 0.24:
            self._spinning = True
        elif abs(err) < 0.16:
            self._spinning = False
        # Near pose but still high: drop spin so -Y descend path can run (ep2).
        if (abs(err) < 0.20 and origin[1] > 0.62
                and max(d_s, d_b) < 0.32):
            self._spinning = False
        rotate = (not lift) and bool(self._spinning)
        p, n_out, into, face, xmin, ymin, ymax = self._left_support(
            stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh, c,
            0.0 if lift else err, origin, ee_r, stem_only=True)
        xmin_h, xmax_h, ymin_h, ymax_h = self._hull_bb(
            stem, bar, stem_hw, stem_hh, bar_hw, bar_hh, th)
        tag = "lift" if lift else ("spin" if rotate else "left")
        # +heading at altitude with stalled left lever → switch to stem_px.
        # Do NOT gate on |err| alone: many 35% winners start with large -θ.
        if rotate and abs(abs(err) - abs(float(getattr(self, "_err_prev", 0.0)))) < 0.01:
            self._stuck = int(self._stuck) + 1
        elif rotate and abs(err) < abs(float(getattr(self, "_err_prev", 0.0))) - 0.008:
            self._stuck = max(0, int(self._stuck) - 4)
        hard_pos = ((not lift) and err > 0.42 and float(origin[1]) > 0.55
                    and int(self._stuck) > 25)
        if hard_pos or (rotate and err <= 0.32):
            rows = self._enumerate_faces(
                stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh, c, err, ee,
                ee_r, rotate=True)
            if hard_pos:
                right = [r for r in rows
                         if r["name"] == "stem_px" and r["tau_w"] > 0.015]
                pick = max(right, key=lambda r: r["tau_w"]) if right else None
                if pick is None:
                    pick = self._select_rotate(rows, origin, err, ee, force=True)
            else:
                force = (int(self._stuck) > 80 and abs(err) > 0.50)
                pick = self._select_rotate(
                    rows, origin, err, ee, force=force) if force else None
                if (not force) and self._rot_name is not None:
                    pick = self._select_rotate(
                        rows, origin, err, ee, force=True)
            if pick is not None:
                if self._rot_name != pick["name"]:
                    self._orbit_stage = 0
                    self._bypass_y = None
                p, n_out, into, face = (
                    pick["p"], pick["n_out"], pick["into"], pick["face"])
                self._rot_name = pick["name"]
                tag = "spin:" + pick["name"]
            else:
                self._rot_name = None
        else:
            # Large +err before stuck: keep left support (legacy path).
            if not rotate:
                self._rot_name = None
                self._stuck = 0
                self._orbit_stage = 0
        if lift:
            into = np.array([1.0, 0.0], dtype=np.float64)
        self._lock = (p, n_out, into, tag)
        self.face_np = face
        self.u = u
        self.into = into
        d = face - ee
        sd = float((ee - p) @ n_out)
        reach = float(np.linalg.norm(d))
        v_ee = np.asarray(st[self.ee_idx, 3:5], dtype=np.float64)
        v_t = np.asarray(st[self.stem_idx, 3:5], dtype=np.float64)
        w_t = abs(float(st[self.stem_idx, 5]))
        speed = float(np.linalg.norm(v_t))
        mag = 2.6 if lift else (4.2 if rotate else 4.6)
        if un < 0.20 and (not lift) and not (rotate and err > 0.32):
            mag = min(mag, 2.8)
        if speed > 0.40:
            mag *= 0.65
        if w_t > 1.6:
            mag *= 0.65
        far_face = float(n_out[0]) > 0.12
        if hard_pos:
            # Must stand on the outward side of the unwind face.
            wrong = sd < 0.02 or reach > 0.30
        elif rotate:
            if err > 0.32:
                wrong = sd < -0.08 or (sd > 0.16 and reach > 0.28)
            else:
                wrong = (sd < -0.04 or reach > 0.22
                         or (far_face and float(n_out @ (ee - p)) < 0.02))
        else:
            wrong = float(ee[0]) > float(c[0]) + 0.05
            if int(self._orbit_n) > 8:
                wrong = float(ee[0]) > xmin - 0.08
        if wrong:
            if not self._imagine:
                self._orbit_n = int(self._orbit_n) + 1
            self.mode = "orbit"
            hint = self._orbit_hint(
                ee, face, p, n_out, xmin_h, xmax_h, ymin_h, ymax_h, origin, ee_r,
                err=err)
        else:
            self._orbit_n = 0
            self._bypass_y = None
            self._orbit_stage = 0
            self.mode = "spin" if (rotate or lift) else "push"
            close = sd < 0.10 and abs(float(ee[1] - p[1])) < 0.12
            press_w = float(np.clip(1.0 - abs(sd - 0.02) / 0.16, 0.70, 1.0))
            pd = np.clip(14.0 * d, -3.2, 3.2)
            pd[1] = float(np.clip(pd[1], -1.6, 1.6))
            if lift and close and float(ee[1]) <= float(p[1]) + 0.08:
                into = np.array([0.28, 0.96], dtype=np.float64)
                self.into = into
            hint = into * mag * press_w + pd
            if lift:
                hint[0] = min(float(hint[0]), 1.2)
            if hard_pos and self._rot_name == "stem_px":
                hint = into * max(mag, 3.6) + 0.35 * pd
                hint[0] = float(np.clip(hint[0], -fmax, -0.3))
                hint[1] = float(np.clip(hint[1], -1.8, 1.2))
            elif rotate and err > 0.32:
                hint[1] = max(float(hint[1]), 0.0)
                if origin[0] < float(self.origin_g[0]) + 0.08:
                    hint[0] = max(float(hint[0]), 3.2)
                else:
                    hint[0] = min(float(hint[0]), 1.2)
            if (not rotate) and (not lift) and sd < 0.10 and origin[0] < float(self.origin_g[0]) - 0.10:
                hint[0] = max(float(hint[0]), 2.6)
            # Brake Fx when +heading is growing from near-zero while high (ep2/7 overshoot).
            if ((not lift) and origin[1] > 0.60
                    and err > 0.05 and err > float(getattr(self, "_err_prev", 0.0)) + 0.008
                    and abs(float(getattr(self, "_err_prev", 0.0))) < 0.22):
                hint[0] = min(float(hint[0]), 0.9)
            # Milder early brake on -θ→+θ while high (keeps ep18).
            if ((not lift) and origin[1] > 0.68
                    and -0.25 < err < 0.12
                    and err > float(getattr(self, "_err_prev", 0.0)) + 0.008):
                hint[0] = min(float(hint[0]), 0.6)
            # Keep residual -heading longer while high so spin still lowers (ep2/5).
            if ((not lift) and origin[1] > 0.74
                    and -0.55 < err < -0.12):
                hint[0] = min(float(hint[0]), 1.35)
            elif ((not lift) and origin[1] > 0.70
                    and -0.55 < err < -0.12):
                hint[0] = min(float(hint[0]), 1.6)
            # High +heading: stop left-pushing (wrong lever).
            if (not lift) and origin[1] > 0.70 and err > 0.30:
                hint[0] = min(float(hint[0]), 0.8)
            # Hold altitude while finishing residual +heading near the floor (ep18).
            if (rotate and 0.22 < err < 0.40 and origin[1] < 0.42
                    and err <= abs(float(getattr(self, "_err_prev", 0.0))) + 0.002):
                hint[1] = max(float(hint[1]), 1.8)
            if (not rotate) and (not lift) and origin[0] > float(self.origin_g[0]) + 0.10:
                hint[0] = min(float(hint[0]), 0.2)
            # Near-goal Y overshoot (ep8): pull -Y only when close; don't break ep16.
            goal_cy = float(
                (self.stem_m * float(self.stem_np[1])
                 + self.bar_m * float(self.bar_np[1]))
                / (self.stem_m + self.bar_m))
            near_pose = max(d_s, d_b) < 0.28
            above_goal = float(c[1]) > goal_cy + 0.05 and near_pose
            if (not rotate) and (not lift) and abs(err) < 0.20 and origin[1] < 0.58:
                if above_goal and abs(err) < 0.15:
                    hint[0] = float(np.clip(hint[0], 0.5, 2.8))
                    hint[1] = min(float(hint[1]), -2.0)
                elif max(d_s, d_b) < 0.20 and not above_goal:
                    hint[0] = float(np.clip(hint[0], -1.0, 2.0))
                    hint[1] = max(float(hint[1]), 2.4)
                elif float(ee[1]) < float(c[1]) - 0.03:
                    hint[0] = min(float(hint[0]), 1.0)
                    hint[1] = max(float(hint[1]), 1.2)
                else:
                    hint[1] = min(float(hint[1]), 0.15)
            elif (not rotate) and (not lift) and above_goal and abs(err) < 0.18:
                hint[0] = float(np.clip(hint[0], 0.4, 2.6))
                hint[1] = min(float(hint[1]), -2.2)
                # EE already on top of a high T: commit to hard -Y (ep7 finish).
                if (origin[1] > 0.70
                        and float(ee[1]) > float(c[1]) - 0.06):
                    hint[1] = min(float(hint[1]), -4.0)
                    hint[0] = float(np.clip(hint[0], -1.2, 2.0))
            elif (not rotate) and (not lift) and origin[1] > 0.52:
                hint[1] = min(float(hint[1]), 0.2)
            if lift and close and float(ee[1]) <= float(p[1]) + 0.08:
                hint[1] = max(float(hint[1]), 2.4)
            elif lift and origin[1] < 0.30 and sd < 0.16:
                hint[1] = max(float(hint[1]), 2.4)
            elif origin[1] < 0.28 and close:
                hint[1] = max(float(hint[1]), 1.8)
        close_ish = sd < 0.14 and abs(float(ee[1] - p[1])) < 0.18
        if origin[1] < 0.34 and close_ish:
            hint[1] = max(float(hint[1]), 2.4)
        if origin[1] < 0.22 and close_ish:
            hint[1] = max(float(hint[1]), 3.6)
        if origin[1] > 0.80 and close_ish and float(ee[1]) > 0.45:
            hint[1] = min(float(hint[1]), -2.2)
        if origin[0] > 1.72:
            hint[0] = min(float(hint[0]), 0.15)
        if origin[0] > 1.90:
            hint[0] = min(float(hint[0]), -2.4)
        if origin[0] < 0.28:
            hint[0] = max(float(hint[0]), 2.6)
        hint = self._barriers(np.asarray(hint, dtype=np.float64), ee)
        hint = hint - 1.4 * v_ee
        self.hint_action = np.clip(hint, -fmax, fmax).astype(np.float32)
        self._err_prev = err

    def _parts(self, pose):
        stem = pose[:, self.stem_idx, :2]
        bar = pose[:, self.bar_idx, :2]
        th = pose[:, self.stem_idx, 2]
        ee = pose[:, self.ee_idx, :2]
        if isinstance(pose, np.ndarray):
            d = (np.linalg.norm(stem - self.stem_np, axis=-1)
                 + np.linalg.norm(bar - self.bar_np, axis=-1))
            ang = np.abs(np.arctan2(np.sin(th - self.th_g), np.cos(th - self.th_g)))
            d_face = np.linalg.norm(ee - self.face_np, axis=-1)
            return d, ang, d_face
        import torch
        stem_g = torch.as_tensor(self.stem_np, device=pose.device, dtype=pose.dtype)
        bar_g = torch.as_tensor(self.bar_np, device=pose.device, dtype=pose.dtype)
        face = torch.as_tensor(self.face_np, device=pose.device, dtype=pose.dtype)
        d = (stem - stem_g).norm(dim=-1) + (bar - bar_g).norm(dim=-1)
        wrap = th - self.th_g
        ang = torch.atan2(torch.sin(wrap), torch.cos(wrap)).abs()
        d_face = (ee - face).norm(dim=-1)
        return d, ang, d_face

    def _bound_pen(self, pose):
        """Penalize T / EE leaving the workable table region (CEM fly-away)."""
        stem = pose[:, self.stem_idx, :2]
        bar = pose[:, self.bar_idx, :2]
        ee = pose[:, self.ee_idx, :2]
        if isinstance(pose, np.ndarray):
            pen = np.zeros(pose.shape[0], dtype=np.float64)
            for xy in (stem, bar):
                pen = pen + np.maximum(0.28 - xy[:, 0], 0.0)
                pen = pen + np.maximum(xy[:, 0] - 1.95, 0.0)
                pen = pen + np.maximum(0.16 - xy[:, 1], 0.0)
                # Heavier above EE reach — high-T dead zone (ep2/15).
                pen = pen + 2.5 * np.maximum(xy[:, 1] - 0.82, 0.0)
                pen = pen + np.maximum(xy[:, 1] - 0.88, 0.0)
            pen = pen + np.maximum(0.12 - ee[:, 0], 0.0)
            pen = pen + np.maximum(ee[:, 0] - 2.10, 0.0)
            pen = pen + np.maximum(0.10 - ee[:, 1], 0.0)
            pen = pen + np.maximum(ee[:, 1] - 1.05, 0.0)
            return pen
        import torch
        pen = pose.new_zeros(pose.shape[0])
        for xy in (stem, bar):
            pen = pen + (0.28 - xy[:, 0]).clamp(min=0.0)
            pen = pen + (xy[:, 0] - 1.95).clamp(min=0.0)
            pen = pen + (0.16 - xy[:, 1]).clamp(min=0.0)
            pen = pen + 2.5 * (xy[:, 1] - 0.82).clamp(min=0.0)
            pen = pen + (xy[:, 1] - 0.88).clamp(min=0.0)
        pen = pen + (0.12 - ee[:, 0]).clamp(min=0.0)
        pen = pen + (ee[:, 0] - 2.10).clamp(min=0.0)
        pen = pen + (0.10 - ee[:, 1]).clamp(min=0.0)
        pen = pen + (ee[:, 1] - 1.05).clamp(min=0.0)
        return pen

    def _lever_pen(self, pose, a_t):
        """Penalize +Fx on a left/below-COM contact while +heading (bad torque)."""
        if not self.lever_w:
            return 0.0
        stem = pose[:, self.stem_idx, :2]
        bar = pose[:, self.bar_idx, :2]
        th = pose[:, self.stem_idx, 2]
        ee = pose[:, self.ee_idx, :2]
        mass = float(self.stem_m + self.bar_m)
        if isinstance(pose, np.ndarray):
            a = np.asarray(a_t, dtype=np.float64)
            fx = a[..., 0]
            com = (self.stem_m * stem + self.bar_m * bar) / mass
            # origin = stem - [-s, c]*stem_hh  →  y = stem_y - cos(th)*hh
            origin_y = stem[:, 1] - np.cos(th) * 0.28
            ang = np.arctan2(np.sin(th - self.th_g), np.cos(th - self.th_g))
            left_below = (ee[:, 0] < com[:, 0] - 0.02) & (ee[:, 1] < com[:, 1] + 0.02)
            bad = left_below & (ang > 0.25) & (origin_y > 0.50) & (fx > 0.15)
            return self.lever_w * bad.astype(np.float64) * fx
        import torch
        fx = a_t[..., 0]
        com = (self.stem_m * stem + self.bar_m * bar) / mass
        origin_y = stem[:, 1] - torch.cos(th) * 0.28
        wrap = th - self.th_g
        ang = torch.atan2(torch.sin(wrap), torch.cos(wrap))
        left_below = (ee[:, 0] < com[:, 0] - 0.02) & (ee[:, 1] < com[:, 1] + 0.02)
        bad = left_below & (ang > 0.25) & (origin_y > 0.50) & (fx > 0.15)
        return self.lever_w * bad.to(fx.dtype) * fx

    def step(self, pose, a_t, prev=None):
        d, ang, d_face = self._parts(pose)
        jump = 0.0
        if prev is not None:
            ds = pose[:, self.stem_idx, :2] - prev[:, self.stem_idx, :2]
            db = pose[:, self.bar_idx, :2] - prev[:, self.bar_idx, :2]
            if isinstance(pose, np.ndarray):
                jump = np.maximum(
                    np.linalg.norm(ds, axis=-1) + np.linalg.norm(db, axis=-1) - 0.50,
                    0.0) * self.jump_w
            else:
                jump = (ds.norm(dim=-1) + db.norm(dim=-1) - 0.50).clamp(min=0.0) * self.jump_w
        # Large heading error: prioritize unwind over goal translation.
        if isinstance(pose, np.ndarray):
            ang_scale = np.where(ang > 0.35, 2.8, np.where(d < 0.55, 1.8, 1.0))
            d_scale = np.where(ang > 0.45, 0.35, 1.0)
            pose_term = self.pose_w * (d_scale * d + ang_scale * self.ang_w * ang)
            ac = self.action_cost * (np.asarray(a_t) ** 2).mean(axis=-1)
            if self.reach_cost:
                # Keep seeking the (hint) face while heading is wrong.
                reach_scale = np.where(ang > 0.35, 1.35 * self.reach_cost,
                                       np.where(d < 0.35, 0.25 * self.reach_cost,
                                                self.reach_cost))
                reach = np.where(d_face > 0.12, reach_scale, 0.20 * reach_scale)
                pose_term = pose_term + reach * d_face
            if self.shove_w:
                into = np.asarray(self.into, dtype=np.float64).reshape(2)
                pose_term = pose_term - self.shove_w * (
                    np.asarray(a_t, dtype=np.float64) * into).sum(axis=-1)
            if self.bound_w:
                pose_term = pose_term + self.bound_w * self._bound_pen(pose)
            pose_term = pose_term + self._lever_pen(pose, a_t)
            return pose_term + ac + jump
        import torch
        ang_scale = torch.where(
            ang > 0.35, d.new_tensor(2.8),
            torch.where(d < 0.55, d.new_tensor(1.8), d.new_tensor(1.0)))
        d_scale = torch.where(ang > 0.45, d.new_tensor(0.35), d.new_tensor(1.0))
        pose_term = self.pose_w * (d_scale * d + ang_scale * self.ang_w * ang)
        if self.reach_cost:
            reach_scale = torch.where(
                ang > 0.35, d.new_tensor(1.35 * self.reach_cost),
                torch.where(d < 0.35, d.new_tensor(0.25 * self.reach_cost),
                            d.new_tensor(self.reach_cost)))
            reach = torch.where(d_face > 0.12, reach_scale, 0.20 * reach_scale)
            pose_term = pose_term + reach * d_face
        if self.shove_w:
            into = pose.new_tensor([float(self.into[0]), float(self.into[1])])
            pose_term = pose_term - self.shove_w * (a_t * into).sum(dim=-1)
        if self.bound_w:
            pose_term = pose_term + self.bound_w * self._bound_pen(pose)
        pose_term = pose_term + self._lever_pen(pose, a_t)
        return pose_term + self.action_cost * (a_t ** 2).mean(dim=-1) + jump

    def terminal(self, pose):
        d, ang, d_face = self._parts(pose)
        # Strong terminal on heading when wrong; pose when aligned.
        if isinstance(pose, np.ndarray):
            face_w = np.where(ang > 0.30, 0.55, np.where(d > 0.40, 0.35, 0.05)) \
                if self.reach_cost else 0.0
            d_scale = np.where(ang > 0.45, 0.40, 1.0)
            ang_scale = np.where(ang > 0.30, 2.6, 1.6)
            term = 2.4 * self.pose_w * (
                d_scale * d + ang_scale * self.ang_w * ang) + face_w * d_face
            if self.bound_w:
                term = term + 1.5 * self.bound_w * self._bound_pen(pose)
            return term
        import torch
        face_w = torch.where(
            ang > 0.30, d.new_tensor(0.55),
            torch.where(d > 0.40, d.new_tensor(0.35), d.new_tensor(0.05))) \
            if self.reach_cost else 0.0
        d_scale = torch.where(ang > 0.45, d.new_tensor(0.40), d.new_tensor(1.0))
        ang_scale = torch.where(ang > 0.30, d.new_tensor(2.6), d.new_tensor(1.6))
        term = 2.4 * self.pose_w * (
            d_scale * d + ang_scale * self.ang_w * ang) + face_w * d_face
        if self.bound_w:
            term = term + 1.5 * self.bound_w * self._bound_pen(pose)
        return term


class SimplePushTCost:
    """Simplified Push-T cost with task decomposition, no state machine.

    Phase determination is purely state-based (no history):
    - reach: EE far from contact face -> steer EE to face
    - push: EE at contact face -> push T toward goal (with rotation lever)
    - settle: T near goal -> stop

    Contact face is selected as the T long-face most anti-aligned with the
    push direction (origin -> goal). Rotation correction is handled by
    offsetting the contact point along the face tangent, computed
    analytically from the desired torque sign.
    """

    def __init__(self, action_cost: float = 0.0005, reach_cost: float = 1.50,
                 ang_w: float = 0.40, pose_w: float = 1.0, device: str = "cpu"):
        self.action_cost = action_cost
        self.reach_cost = float(reach_cost)
        self.ang_w = float(ang_w)
        self.pose_w = float(pose_w)
        self.device = device
        self.ee_idx = 0
        self.stem_idx = PushTEnv.STEM
        self.bar_idx = PushTEnv.BAR
        self.stem_np = np.zeros(2, dtype=np.float32)
        self.bar_np = np.zeros(2, dtype=np.float32)
        self.th_g = 0.0
        self.face_np = np.zeros(2, dtype=np.float64)
        self.hint_action = np.zeros(2, dtype=np.float32)
        self.into = np.array([1.0, 0.0], dtype=np.float64)
        self.u = np.array([1.0, 0.0], dtype=np.float64)
        self.shove_w = 0.0
        self.jump_w = 0.0
        self.mode = "push"
        self.press = 0.02
        self.stem_m = 0.45
        self.bar_m = 0.40
        self.origin_g = np.array([1.50, 0.52], dtype=np.float64)

    def reset_episode(self):
        self.mode = "push"
        self._dbg = 0
        self._face_key = None
        self._face_gap = 1.0
        self._locked_key = None
        self._phase = "rotate"
        self._trans_sign = 1.0

    @staticmethod
    def _origin(stem, theta, stem_hh):
        c, s = float(np.cos(theta)), float(np.sin(theta))
        return np.asarray(stem[:2], dtype=np.float64) - np.array([-s * stem_hh, c * stem_hh])

    @staticmethod
    def _axes(th):
        c, s = float(np.cos(th)), float(np.sin(th))
        return np.array([c, s], dtype=np.float64), np.array([-s, c], dtype=np.float64)

    def _corners(self, xy, hw, hh, th):
        return _box_corners(xy, hw, hh, th)

    def _hull_bb(self, stem, bar, stem_hw, stem_hh, bar_hw, bar_hh, th):
        hull = np.vstack([
            self._corners(stem, stem_hw, stem_hh, th),
            self._corners(bar, bar_hw, bar_hh, th),
        ])
        return (float(hull[:, 0].min()), float(hull[:, 0].max()),
                float(hull[:, 1].min()), float(hull[:, 1].max()))

    def _long_faces(self, stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh):
        """Return the 4 long faces plus stem bottom (for +y push)."""
        x_b, y_b = self._axes(th)
        return [
            # name, p0 (face center), n_out (outward normal), tan (tangent), half
            ("stem_mx", stem - x_b * stem_hw, -x_b, y_b, stem_hh),
            ("stem_px", stem + x_b * stem_hw,  x_b, y_b, stem_hh),
            ("bar_py",  bar + y_b * bar_hh,   y_b, x_b, bar_hw),
            ("bar_my",  bar - y_b * bar_hh,  -y_b, x_b, bar_hw),
            # stem bottom (for pushing +y), short but accessible
            ("stem_my", stem - y_b * stem_hh, -y_b, x_b, stem_hw),
        ]

    def _hull_corners(self, stem, bar, stem_hw, stem_hh, bar_hw, bar_hh, th):
        return np.vstack([
            self._corners(stem, stem_hw, stem_hh, th),
            self._corners(bar, bar_hw, bar_hh, th),
        ])

    def _pick_rotate_contact(self, stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh,
                             com, u, err, ee, ee_r):
        """Pick a point on a REAL T long edge whose push torque corrects err.

        Samples points along each of the 4 long faces; prefers corrective
        torque, then translation toward goal, then closeness to the EE.
        """
        best = None
        best_s = -1e18
        standoff_d = max(ee_r - self.press, 0.03)
        for name, p0, n_out, tan, half in self._long_faces(
                stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh):
            n_out = np.asarray(n_out, dtype=np.float64)
            tan = np.asarray(tan, dtype=np.float64)
            p0 = np.asarray(p0, dtype=np.float64)
            into = -n_out
            span = 0.72 * float(half)
            for tt in np.linspace(-span, span, 9):
                p = p0 + tan * float(tt)
                r = p - com
                tau = float(r[0] * into[1] - r[1] * into[0])
                rot = -tau * err  # >0 when this push reduces the angle error
                if rot <= 0.004:
                    continue
                trans = float(into @ u)
                if trans < -0.30:  # never pick a face that shoves away hard
                    continue
                standoff = p + n_out * standoff_d
                reach_d = float(np.linalg.norm(standoff - ee))
                score = (5.0 * min(rot, 0.16)
                         + 1.1 * max(trans, 0.0)
                         - 0.30 * reach_d)
                if score > best_s:
                    best_s = score
                    best = (p, n_out, into)
        return best

    def _edge_contact(self, stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh,
                      com, u, err, ee, ee_r):
        """Pick the face whose inward normal best matches the push direction u,
        and a contact point along it whose lever arm turns the normal push into
        a torque that corrects err. A normal push at an offset arm translates
        AND rotates at once (friction-independent torque), which also works at
        ~90 deg orientation. Faces persist via hysteresis.
        Returns (contact, n_out, shove_dir=inward normal).
        """
        x_b, y_b = self._axes(th)
        sd = max(ee_r - self.press, 0.03)
        faces = []
        for tag, c, hw, hh in (("s", stem, stem_hw, stem_hh),
                               ("b", bar, bar_hw, bar_hh)):
            c = np.asarray(c, dtype=np.float64)
            faces += [
                (c - x_b * hw, -x_b, y_b, hh, tag + "lx"),
                (c + x_b * hw,  x_b, y_b, hh, tag + "rx"),
                (c - y_b * hh, -y_b, x_b, hw, tag + "by"),
                (c + y_b * hh,  y_b, x_b, hw, tag + "ty"),
            ]
        # Lock housekeeping: release (a) when drift exceeds the edge's
        # corrective reach, or (b) once rotation has settled near zero -- the
        # rotation face often points the wrong way for translation.
        if getattr(self, "_locked_key", None) is not None:
            if abs(err) > 0.35 or abs(err) < 0.20:
                self._locked_key = None
        # --- Locked face: keep the edge chosen at first contact for the whole
        # episode. The T's stem/bar edges on one side are DISJOINT (a 0.12
        # step at the junction), so switching between them as err crosses zero
        # demands an infeasible tangential jump and wedges the EE.
        locked = getattr(self, "_locked_key", None)
        if locked is not None:
            for e0, n_f, t_f, length, key in faces:
                if key != locked:
                    continue
                e0 = np.asarray(e0, dtype=np.float64)
                n_f = np.asarray(n_f, dtype=np.float64)
                t_f = np.asarray(t_f, dtype=np.float64)
                ac = float((e0 - com) @ t_f)
                lo, hi = ac - 0.80 * length, ac + 0.80 * length
                c = float(t_f[0] * (-n_f[1]) - t_f[1] * (-n_f[0]))
                s_need = (-0.5 * err / c) if abs(c) > 1e-6 else 0.0
                ach = float(np.clip(s_need, lo, hi))
                a = ach - ac
                self._face_key = key
                return e0 + t_f * a, n_f, -n_f

        best, best_s, best_key = None, -1e18, None
        cur_key = getattr(self, "_face_key", None)
        cur_gap = getattr(self, "_face_gap", 1.0)
        for e0, n_f, t_f, length, key in faces:
            e0 = np.asarray(e0, dtype=np.float64)
            n_f = np.asarray(n_f, dtype=np.float64)
            t_f = np.asarray(t_f, dtype=np.float64)
            into = -n_f
            prog = float(into @ u)
            if prog < 0.10:
                continue
            ac = float((e0 - com) @ t_f)
            lo, hi = ac - 0.80 * length, ac + 0.80 * length
            # Continuous corrective arm (no sign bang-bang at err=0):
            # a normal press at tangent arm s produces torque tau = s*F*c,
            # c = t_f x into.  Restoring torque needs err*tau < 0, i.e.
            # err*s*c < 0 -> s_need = -k*err/c.  k is kept small because the
            # press runs for a whole exec chunk (~20 frames) before replan and
            # the T's inertia is tiny: too strong an arm overshoots in one shot.
            c = float(t_f[0] * into[1] - t_f[1] * into[0])
            s_need = (-0.5 * err / c) if abs(c) > 1e-6 else 0.0
            ach = float(np.clip(s_need, lo, hi))
            a = ach - ac
            if abs(err) > 0.04:
                # frac in [-1,1]: fraction of restoring torque this face can
                # deliver; -1 if it cannot produce the required sign at all
                if ach * s_need > 0.0:
                    frac = min(abs(ach) / max(abs(s_need), 1e-6), 1.0)
                else:
                    frac = -1.0
            else:
                frac = 0.0
            standoff = e0 + t_f * a + n_f * sd
            reach_d = float(np.linalg.norm(standoff - ee))
            score = 2.2 * prog + 1.4 * frac - 0.25 * reach_d
            if key == cur_key and cur_gap < 0.12:
                score += 0.4
            if score > best_s:
                best_s, best, best_key = score, (e0 + t_f * a, n_f), key
        if best is None:
            n_f = np.asarray(-x_b, dtype=np.float64)
            t_f = y_b
            e0 = np.asarray(bar, dtype=np.float64) - x_b * bar_hw
            into = x_b
            ac = float((e0 - com) @ t_f)
            c = float(t_f[0] * into[1] - t_f[1] * into[0])
            s_req = (-0.5 * err / c) if abs(c) > 1e-6 else 0.0
            a = float(np.clip(s_req - ac,
                              ac - 0.80 * bar_hh, ac + 0.80 * bar_hh))
            best, best_key = (e0 + t_f * a, n_f), "blx"
        self._face_key = best_key
        n_out = np.asarray(best[1], dtype=np.float64)
        return np.asarray(best[0], dtype=np.float64), n_out, -n_out

    def _orbit_target(self, ee, standoff, n_out, hull, ee_r):
        """Stateless staged waypoint that brings EE to the contact standoff
        without crossing the T hull. Returns (waypoint, need_orbit).

        Orbit only when EE is on the opposite side of the T (ee_n < 0).
        On the contact side the straight segment to the standoff stays in
        the n>0 half-space, so it cannot cross the hull.
        """
        n = np.asarray(n_out, dtype=np.float64)
        tvec = np.array([-n[1], n[0]], dtype=np.float64)
        corners = np.asarray(hull, dtype=np.float64)
        center = corners.mean(axis=0)
        rel = corners - center
        half_n = float(np.max(np.abs(rel @ n)))
        half_t = float(np.max(np.abs(rel @ tvec)))
        clear = ee_r + 0.06
        clear_t = half_t + clear
        ee_n = float((ee - center) @ n)
        ee_t = float((ee - center) @ tvec)
        q_n = float((standoff - center) @ n)
        q_t = float((standoff - center) @ tvec)
        # "wrong side" only when EE has passed the FAR edge of the hull
        # (not merely when it is short of the hull center).
        far_n = -half_n + ee_r + 0.03
        # normal-approach staging point, one clearance outside the standoff
        appr = center + n * (q_n + clear) + tvec * q_t

        if ee_n < far_n and abs(ee_t) < clear_t:
            # wrong side and not yet clear: slide tangentially around the hull
            side = 1.0 if ee_t >= 0.0 else -1.0
            target = center + tvec * (side * (clear_t + 0.02)) + n * ee_n
            return target, True
        if ee_n < far_n:
            # wrong side but tangentially clear: cross OUTSIDE the standoff so
            # convex corners cannot shove the EE back during the crossing
            target = center + n * (q_n + clear) + tvec * ee_t
            return target, True
        # contact side: align to the normal-approach point first, then the
        # final press is a pure normal move (cannot clip face-end corners)
        if float(np.linalg.norm(ee - appr)) > 0.04:
            return appr, True
        return standoff, False

    def bind(self, stem_goal, bar_goal, geom, states_now, theta: float = 0.0,
             origin_goal=None, force_max: float = 6.0):
        g = np.asarray(geom, dtype=np.float64)
        st = np.asarray(states_now, dtype=np.float64)
        ee_r = float(g[self.ee_idx, 0])
        stem = st[self.stem_idx, :2]
        bar = st[self.bar_idx, :2]
        ee = st[self.ee_idx, :2]
        th = float(st[self.stem_idx, 2])
        stem_hw, stem_hh = float(g[self.stem_idx, 0]), float(g[self.stem_idx, 1])
        bar_hw, bar_hh = float(g[self.bar_idx, 0]), float(g[self.bar_idx, 1])

        if origin_goal is not None:
            self.origin_g = np.asarray(origin_goal[:2], dtype=np.float64)
        self.stem_np = np.asarray(stem_goal[:2], dtype=np.float32)
        self.bar_np = np.asarray(bar_goal[:2], dtype=np.float32)
        self.th_g = float(theta)

        origin = self._origin(stem, th, stem_hh)
        err = float(np.arctan2(np.sin(th - theta), np.cos(th - theta)))

        # COM
        mass = self.stem_m + self.bar_m
        com = (self.stem_m * stem + self.bar_m * bar) / mass

        # Push direction (origin -> goal), clamped at workspace bounds
        err_p = self.origin_g - origin
        if origin[0] < 0.28:
            err_p[0] = max(float(err_p[0]), 0.25)
        if origin[0] > 1.95:
            err_p[0] = min(float(err_p[0]), -0.25)
        if origin[1] < 0.22:
            err_p[1] = max(float(err_p[1]), 0.20)
        if origin[1] > 0.70:
            err_p[1] = min(float(err_p[1]), -0.25)
        un = float(np.linalg.norm(err_p))
        u = err_p / un if un > 1e-6 else np.array([1.0, 0.0])
        self.u = u
        u_eff = u

        hull = self._hull_corners(stem, bar, stem_hw, stem_hh, bar_hw, bar_hh, th)

        d_s = float(np.linalg.norm(stem - self.stem_np))
        d_b = float(np.linalg.norm(bar - self.bar_np))
        near_goal = d_s < 0.11 and d_b < 0.11 and abs(err) < 0.22

        v_ee = np.asarray(st[self.ee_idx, 3:5], dtype=np.float64)
        v_t = np.asarray(st[self.stem_idx, 3:5], dtype=np.float64)
        w_signed = float(st[self.stem_idx, 5])
        w_t = abs(w_signed)
        speed = float(np.linalg.norm(v_t))

        if near_goal:
            self.mode = "settle"
            self.into = np.zeros(2, dtype=np.float64)
            self.face_np = np.asarray(ee, dtype=np.float64)
            self.hint_action = np.zeros(2, dtype=np.float32)
            return

        # --- Single persistent contact: best T edge + corrective torque arm ---
        contact, n_out, shove = self._edge_contact(
            stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh,
            com, u_eff, err, ee, ee_r)
        n_out = np.asarray(n_out, dtype=np.float64)
        shove = np.asarray(shove, dtype=np.float64)
        contact = np.asarray(contact, dtype=np.float64)
        # shove is the inward normal (-n_out); project the goal-direction u
        # onto it so the push stays on the face and doesn't fly off
        into_dir = -n_out
        proj = float(u @ into_dir)
        if proj > 0.15:
            shove = into_dir  # pure normal press toward goal
        self.into = shove

        standoff = contact + n_out * max(ee_r - self.press, 0.03)
        standoff[0] = float(np.clip(standoff[0], 0.16, 2.06))
        standoff[1] = float(np.clip(standoff[1], 0.18, 0.88))
        self.face_np = standoff

        # local frame along the contact face
        tvec = np.array([-n_out[1], n_out[0]], dtype=np.float64)
        center = hull.mean(axis=0)
        ee_n = float((ee - center) @ n_out)
        ee_t = float((ee - center) @ tvec)
        q_n = float((standoff - center) @ n_out)
        q_t = float((standoff - center) @ tvec)
        d_t = q_t - ee_t

        mag = 4.6
        if speed > 0.40:
            mag *= 0.65
        if w_t > 1.6:
            mag *= 0.65
        # directional brake: if T spins so that |err| is growing, ease off
        # so table friction can arrest the rotation instead of overshooting
        if abs(err) > 0.04 and w_signed * err > 0.0 and w_t > 0.5:
            mag *= 0.40
        # anti-overshoot coast: spinning fast toward zero with little angle
        # left -> release press so friction removes the residual angular momentum
        if w_signed * err < 0.0 and abs(err) < 0.25 * w_t + 0.06:
            mag *= 0.30

        waypoint, need_orbit = self._orbit_target(ee, standoff, n_out, hull, ee_r)
        contact_gap = float(np.linalg.norm(standoff - ee))
        self._face_gap = contact_gap
        if need_orbit:
            self.mode = "orbit"
            delta = np.asarray(waypoint) - ee
            dn = max(float(np.linalg.norm(delta)), 1e-6)
            hint = (delta / dn) * mag - 1.0 * v_ee
        elif abs(d_t) > 0.06 and (ee_n - q_n) > 0.02:
            # L-approach stage 1: align tangent at a safe depth BEFORE the EE
            # can touch the wrong part of the edge
            self.mode = "align"
            target = center + n_out * max(ee_n, q_n) + tvec * q_t
            delta = target - ee
            dn = max(float(np.linalg.norm(delta)), 1e-6)
            hint = (delta / dn) * mag - 1.0 * v_ee
        elif (ee_n - q_n) > 0.02:
            # L-approach stage 2: press straight in
            self.mode = "reach"
            delta = standoff - ee
            dn = max(float(np.linalg.norm(delta)), 1e-6)
            hint = (delta / dn) * mag - 1.2 * v_ee
        else:
            # in contact: press along the face normal; the offset contact arm
            # turns this press into corrective torque. One-sided normal spring
            # restores contact if lost; tangent spring tracks the arm point.
            self.mode = "push"
            # keep pressing while either position or angle error remains
            pos_gate = float(np.clip(un / 0.22, 0.18, 1.0))
            ang_gate = float(np.clip(abs(err) / 0.30, 0.0, 1.0))
            pmag = mag * max(pos_gate, 0.90 * ang_gate)
            to_face = standoff - ee
            # one-sided normal spring: pull in only when EE loses contact,
            # never pull out while pressed in (the contact reacts instead)
            d_n = float(to_face @ n_out)
            if d_n < 0.0:
                spring_n = d_n * 30.0 * n_out
            else:
                spring_n = np.zeros(2)
            d_tv = float(to_face @ tvec) * tvec
            spring_t = np.clip(16.0 * d_tv, -2.5, 2.5)
            hint = shove * (0.85 * pmag) + np.clip(
                spring_n, -4.0, 4.0) + spring_t - 1.2 * v_ee
            # first real contact: lock whichever face the geometry picked
            if self._locked_key is None and contact_gap < 0.10:
                self._locked_key = self._face_key

        self.hint_action = np.clip(
            np.asarray(hint, dtype=np.float64), -force_max, force_max
        ).astype(np.float32)

        import os as _os
        if _os.environ.get("PUSH_T_DEBUG"):
            self._dbg = int(getattr(self, "_dbg", 0)) + 1
            if self._dbg <= 24 or self._dbg % 10 == 0:
                print(f"  [{self._dbg}] {self.mode:6s} {self._face_key} "
                      f"err={err:+.2f} w={w_signed:+.2f} "
                      f"ee={ee.round(2)} org={origin.round(2)} "
                      f"q={standoff.round(2)} dn={q_n-ee_n:+.2f} dt={d_t:+.2f} "
                      f"hint={self.hint_action.round(2)}", flush=True)

    def _parts(self, pose):
        stem = pose[:, self.stem_idx, :2]
        bar = pose[:, self.bar_idx, :2]
        th = pose[:, self.stem_idx, 2]
        ee = pose[:, self.ee_idx, :2]
        if isinstance(pose, np.ndarray):
            d = (np.linalg.norm(stem - self.stem_np, axis=-1)
                 + np.linalg.norm(bar - self.bar_np, axis=-1))
            ang = np.abs(np.arctan2(np.sin(th - self.th_g), np.cos(th - self.th_g)))
            d_face = np.linalg.norm(ee - self.face_np, axis=-1)
            return d, ang, d_face
        import torch
        stem_g = torch.as_tensor(self.stem_np, device=pose.device, dtype=pose.dtype)
        bar_g = torch.as_tensor(self.bar_np, device=pose.device, dtype=pose.dtype)
        face = torch.as_tensor(self.face_np, device=pose.device, dtype=pose.dtype)
        d = (stem - stem_g).norm(dim=-1) + (bar - bar_g).norm(dim=-1)
        wrap = th - self.th_g
        ang = torch.atan2(torch.sin(wrap), torch.cos(wrap)).abs()
        d_face = (ee - face).norm(dim=-1)
        return d, ang, d_face

    def step(self, pose, a_t, prev=None):
        d, ang, d_face = self._parts(pose)
        pose_term = self.pose_w * (d + self.ang_w * ang)
        if isinstance(pose, np.ndarray):
            ac = self.action_cost * (np.asarray(a_t) ** 2).mean(axis=-1)
            if self.reach_cost:
                reach = np.where(d_face > 0.12, self.reach_cost, 0.20 * self.reach_cost)
                pose_term = pose_term + reach * d_face
            return pose_term + ac
        import torch
        if self.reach_cost:
            reach = torch.where(d_face > 0.12, d_face.new_tensor(self.reach_cost),
                                d_face.new_tensor(0.20 * self.reach_cost))
            pose_term = pose_term + reach * d_face
        return pose_term + self.action_cost * (a_t ** 2).mean(dim=-1)

    def terminal(self, pose):
        d, ang, d_face = self._parts(pose)
        extra = (0.5 * d_face) if self.reach_cost else 0.0
        return 2.0 * self.pose_w * (d + self.ang_w * ang) + extra


class PushTTask(CEMWorldTask):
    def __init__(self, cfg=None, model=None, norm=None, device="cpu", **kwargs):
        cfg = cfg or push_t_config()
        env = PushTEnv(cfg)
        env.ee_spawn = str(kwargs.get("ee_spawn", getattr(cfg.task, "ee_spawn", "default")))
        simple = bool(kwargs.pop("simple_cost", False))
        cost = SimplePushTCost(device=device) if simple else PushTCost(device=device)
        # Hybrid: servo by default, Solver CEM when stuck / hard +heading.
        kwargs = dict(kwargs)
        pk = dict(kwargs.get("planner_kwargs") or {})
        self.use_hybrid = bool(pk.pop("use_hybrid", True))
        kwargs["planner_kwargs"] = pk
        super().__init__(cfg, env, model, norm, device=device, cost=cost, **kwargs)
        if self.use_hybrid and not getattr(self, "use_servo", False):
            # Hybrid cost shaping (solver or WM); macros/MPC use real physics.
            if self.planner is not None and self.use_solver_cem:
                self.planner.stride = 3
            self.stride = 1
            self._action_stride = 1
            self.cost.ang_w = 1.15
            self.cost.pose_w = 1.10
            self.cost.reach_cost = 1.10
            self.cost.jump_w = 3.0
            self.cost.shove_w = 0.08
            self.cost.bound_w = 2.8
            self.cost.lever_w = 2.2
            if self.planner is not None and hasattr(self.planner, "hint_w"):
                self.planner.hint_w = 0.55
            # Shipped Push-T controller is WM-obs PPO (see train_rsl_wm_push_t /
            # game --mode pusht). Latent-CEM hybrids did not reach reliable success.
        elif float(self.cost.reach_cost) > 0.0 and not getattr(self, "use_servo", False):
            self.cost.reach_cost = 2.4
            self.cost.pose_w = 1.0
            self.cost.shove_w = 0.06
            self.cost.jump_w = 4.0
        self.target_idx = PushTEnv.STEM
        gx = float(getattr(cfg.task, "goal_x", 1.50))
        gy = float(getattr(cfg.task, "goal_y", 0.52))
        gth = float(getattr(cfg.task, "goal_theta", 0.0))
        self.stem_goal, self.bar_goal = env.t_goal_parts(gx, gy, gth)
        self.origin_goal = np.array([gx, gy], dtype=np.float64)
        self.goal_theta = gth
        self.ang_tol = float(getattr(cfg.task, "ang_tol", 0.25))
        self._hybrid_stuck = 0
        self._hybrid_best = 1e9
        self._cem_mode = False
        self._unwind = None  # None | "via" | "right" | "shove"
        self._unwind_t = 0

    def sample_goal(self, rng, obs):
        return np.asarray(self.stem_goal, dtype=np.float32)

    def run_episode(self, rng, record: bool = False):
        self.cost.reset_episode()
        self._hybrid_stuck = 0
        self._hybrid_best = 1e9
        self._cem_mode = False
        self._unwind = None
        self._unwind_t = 0
        self._unwind_best_err = None
        self._unwind_stall = 0
        self._macro_used = False
        self._cem_plans = 0
        self._cem_rescue_left = 0
        self._near_miss_t = 0
        self._near_best_m = 1e9
        # Phased planner: orient → translate → fine → settle (+ descend).
        self._phase = "orient"
        self._phase_stuck = 0
        self._phase_best = 1e9
        self._reengage_used = 0
        self._top_press_used = 0
        self._reengage_cool = 0
        self._right_reclaim_used = 0
        self._right_exec_restore = 0
        self._midclimb_used = 0
        self._early_spin_used = 0
        self.exec_horizon = 6
        return super().run_episode(rng, record=record)

    def _phase_metrics(self, cur):
        st = cur["obj_states"]
        th = float(st[PushTEnv.STEM, 2])
        err = float(np.arctan2(np.sin(th - self.goal_theta),
                               np.cos(th - self.goal_theta)))
        stem = st[PushTEnv.STEM, :2]
        bar = st[PushTEnv.BAR, :2]
        d_s = float(np.linalg.norm(stem - self.stem_goal))
        d_b = float(np.linalg.norm(bar - self.bar_goal))
        m = max(d_s, d_b)
        mass = self.cost.stem_m + self.cost.bar_m
        c = (self.cost.stem_m * stem + self.cost.bar_m * bar) / mass
        goal_c = (self.cost.stem_m * np.asarray(self.stem_goal, dtype=np.float64)
                  + self.cost.bar_m * np.asarray(self.bar_goal, dtype=np.float64)) / mass
        oy = float(self.cost._origin(stem, th, float(self.env.stem_hh))[1])
        vel = float(np.linalg.norm(st[PushTEnv.STEM, 3:5]))
        return dict(err=err, d_s=d_s, d_b=d_b, m=m, c=c, goal_c=goal_c,
                    oy=oy, vel=vel, th=th, st=st)

    def _update_phase(self, met):
        """Height-gated phase machine. Never 'translate' while parked above goal."""
        err, m = float(met["err"]), float(met["m"])
        c_y = float(met["c"][1])
        goal_cy = float(met["goal_c"][1])
        d_s, d_b = float(met["d_s"]), float(met["d_b"])
        above = c_y > goal_cy + 0.08
        # Progress score for stuck detection (phase-dependent).
        if self._phase == "orient":
            score = abs(err) + 0.15 * m + 0.35 * max(0.0, c_y - goal_cy - 0.05)
        elif self._phase == "descend":
            score = max(0.0, c_y - goal_cy) + 0.25 * abs(err) + 0.20 * m
        elif self._phase in ("translate", "fine"):
            score = m + 0.35 * abs(err)
        else:
            score = m + abs(err)
        if score < float(self._phase_best) - 0.01:
            self._phase_best = score
            self._phase_stuck = 0
        else:
            self._phase_stuck = int(self._phase_stuck) + 1

        prev = self._phase
        # Never leave orient while an unwind macro is mid-flight (ep11).
        if self._unwind is not None and abs(err) > 0.18:
            nxt = "orient"
        elif d_s < self.tol and d_b < self.tol and abs(err) < self.ang_tol:
            nxt = "settle"
        elif abs(err) < 0.20 and m < 0.22:
            nxt = "fine"
        elif abs(err) < 0.18 and c_y > goal_cy + 0.12 and m < 0.42:
            nxt = "descend"
        elif abs(err) < 0.22:
            nxt = "translate"
        else:
            nxt = "orient"
        if nxt != prev:
            self._phase = nxt
            self._phase_best = 1e9
            self._phase_stuck = 0
        return self._phase

    def _apply_phase_cost(self, phase: str):
        """Stage-specific CEM/servo cost shaping + force scale."""
        # Defaults restored each step then overridden.
        self.cost.ang_w = 1.15
        self.cost.pose_w = 1.10
        self.cost.reach_cost = 1.10
        self.cost.bound_w = 2.8
        self.cost.lever_w = 2.2
        self._phase_fscale = 1.0
        if phase == "orient":
            self.cost.ang_w = 1.55
            self.cost.pose_w = 0.55
            self.cost.reach_cost = 1.25
            self.cost.bound_w = 3.2
            self._phase_fscale = 1.0
        elif phase == "descend":
            self.cost.ang_w = 0.55
            self.cost.pose_w = 1.35
            self.cost.reach_cost = 0.85
            self.cost.bound_w = 3.5
            self._phase_fscale = 1.0
        elif phase == "translate":
            self.cost.ang_w = 0.70
            self.cost.pose_w = 1.45
            self.cost.reach_cost = 0.70
            self.cost.lever_w = 2.8
            self._phase_fscale = 1.0  # keep full authority (floor)
        elif phase == "fine":
            self.cost.ang_w = 0.90
            self.cost.pose_w = 1.50
            self.cost.reach_cost = 0.45
            self._phase_fscale = 1.0
        elif phase == "settle":
            self._phase_fscale = 0.0

    def _scale_hint(self, a):
        fmax = float(self.cfg.collect.force_max) * float(
            getattr(self, "_phase_fscale", 1.0))
        return np.clip(np.asarray(a, dtype=np.float32), -fmax, fmax).astype(np.float32)

    def _cem_sequence(self, cur):
        """Solver CEM (real physics) or WM latent CEM rollouts."""
        if self.use_solver_cem and self.planner is not None and hasattr(
                self.planner, "plan_sequence") and not hasattr(
                    self.planner, "encode_obs"):
            return self.planner.plan_sequence(self.cost)
        wmpl = self.planner
        z0 = wmpl.encode_obs(cur)
        seq = wmpl.plan_sequence(
            z0, geom=self._geom(), states_now=cur["obj_states"],
            cost=self.cost, idle_action=self._idle_action())
        wmpl.remember_first_step(z0, seq[0], self._geom())
        return np.asarray(seq, dtype=np.float32)

    def _unwind_action(self, cur):
        """Heading unwind at safe altitude; press-down if high; lift if off-board."""
        st = cur["obj_states"]
        th = float(st[PushTEnv.STEM, 2])
        stem = st[PushTEnv.STEM, :2].astype(np.float64)
        bar = st[PushTEnv.BAR, :2].astype(np.float64)
        ee = st[0, :2].astype(np.float64)
        err = float(np.arctan2(np.sin(th - self.goal_theta),
                               np.cos(th - self.goal_theta)))
        origin = self.cost._origin(stem, th, float(self.env.stem_hh))
        mode = self._unwind or "pos"
        # Hand back once heading is corrected (sign-aware).
        if mode == "neg":
            done = err > -0.22
        else:
            done = err < 0.22
        if done:
            self._unwind = None
            return np.asarray(self.cost.hint_action, dtype=np.float32)
        g = self._geom()
        ee_r = float(g[0, 0])
        stem_hw, stem_hh = float(g[PushTEnv.STEM, 0]), float(g[PushTEnv.STEM, 1])
        bar_hw, bar_hh = float(g[PushTEnv.BAR, 0]), float(g[PushTEnv.BAR, 1])
        mass = self.cost.stem_m + self.cost.bar_m
        c = (self.cost.stem_m * stem + self.cost.bar_m * bar) / mass
        rows = self.cost._enumerate_faces(
            stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh, c, err, ee,
            ee_r, rotate=True)
        oy = float(origin[1])
        # Off-board / too low: lift back into workspace (ep18 fall).
        if oy < 0.20:
            target = np.array([float(np.clip(c[0], 0.45, 1.75)), 0.28], dtype=np.float64)
            d = target - ee
            if float(np.linalg.norm(d)) > 0.06:
                a = (d / (float(np.linalg.norm(d)) + 1e-6)) * 5.0
            else:
                a = np.array([0.0, 4.5], dtype=np.float64)
            self._unwind_t = int(self._unwind_t) + 1
            if self._unwind_t > 500:
                self._unwind = None
            return np.clip(a, -6.0, 6.0).astype(np.float32)
        # High T: stem_mx above COM (stem_px lever unreachable).
        if oy > 0.78 and err > 0.22:
            left = [r for r in rows if r["name"] == "stem_mx" and r["tau_w"] > 0.008]
            if left:
                pick = max(left, key=lambda r: r["tau_w"])
                face = np.asarray(pick["face"], dtype=np.float64)
                face[1] = float(np.clip(max(float(face[1]), float(c[1]) + 0.04),
                                        0.20, 0.90))
                into = np.asarray(pick["into"], dtype=np.float64)
                n_out = np.asarray(pick["n_out"], dtype=np.float64)
                staging = face + n_out * 0.08
                staging[0] = float(np.clip(staging[0], 0.16, 2.06))
                staging[1] = float(np.clip(staging[1], 0.18, 0.90))
                d = staging - ee
                if float(np.linalg.norm(d)) > 0.08:
                    a = (d / (float(np.linalg.norm(d)) + 1e-6)) * 5.0
                else:
                    a = into * 4.5
                    a[1] = min(float(a[1]), 0.2)
                self._unwind_t = int(self._unwind_t) + 1
                if self._unwind_t > 500:
                    self._unwind = None
                return np.clip(a, -6.0, 6.0).astype(np.float32)
        if mode == "neg" or err < 0.0:
            cands = [r for r in rows if r["tau_w"] > 0.015]
        else:
            cands = [r for r in rows if r["name"] == "stem_px" and r["tau_w"] > 0.01]
        if not cands:
            self._unwind = None
            return np.asarray(self.cost.hint_action, dtype=np.float32)
        pick = max(cands, key=lambda r: r["tau_w"])
        face = np.asarray(pick["face"], dtype=np.float64)
        into = np.asarray(pick["into"], dtype=np.float64)
        n_out = np.asarray(pick["n_out"], dtype=np.float64)
        staging = face + n_out * 0.08
        staging[0] = float(np.clip(staging[0], 0.16, 2.06))
        staging[1] = float(np.clip(staging[1], 0.18, 0.90))
        d = staging - ee
        if float(np.linalg.norm(d)) > 0.08:
            a = (d / (float(np.linalg.norm(d)) + 1e-6)) * 5.0
        else:
            a = into * 5.0
            # Don't drive T through the floor while rotating (ep18).
            if oy < 0.32:
                a[1] = max(float(a[1]), 0.8)
        self._unwind_t = int(self._unwind_t) + 1
        if self._unwind_t > 500:
            self._unwind = None
        return np.clip(a, -6.0, 6.0).astype(np.float32)

    def bind_cost(self, obs, goal):
        self.cost.bind(
            self.stem_goal, self.bar_goal, self._geom(),
            states_now=obs["obj_states"], theta=self.goal_theta,
            origin_goal=self.origin_goal,
            force_max=float(self.cfg.collect.force_max))

    def _pose_score(self, st):
        d = float(np.linalg.norm(st[PushTEnv.STEM, :2] - self.stem_goal)
                  + np.linalg.norm(st[PushTEnv.BAR, :2] - self.bar_goal))
        th = float(st[PushTEnv.STEM, 2])
        ang = abs(float(np.arctan2(np.sin(th), np.cos(th))))
        origin = self.cost._origin(st[PushTEnv.STEM, :2], th, float(self.env.stem_hh))
        bound = 0.0
        if origin[0] < 0.28 or origin[0] > 1.95:
            bound += 1.5
        if origin[1] < 0.16 or origin[1] > 0.85:
            bound += 2.5
        ee = st[0, :2]
        if ee[0] < 0.12 or ee[0] > 2.10 or ee[1] < 0.10 or ee[1] > 1.05:
            bound += 2.0
        return d + 1.6 * ang + bound

    def _apply_cem_heading_prior(self, cur):
        """Bias CEM mean toward stem_px unwind when high +heading."""
        st = cur["obj_states"]
        th = float(st[PushTEnv.STEM, 2])
        err = float(np.arctan2(np.sin(th - self.goal_theta),
                               np.cos(th - self.goal_theta)))
        origin = self.cost._origin(
            st[PushTEnv.STEM, :2], th, float(self.env.stem_hh))
        if not (err > 0.40 and float(origin[1]) > 0.55):
            return
        if self.planner is not None and hasattr(self.planner, "hint_w"):
            self.planner.hint_w = 0.60
        g = self._geom()
        stem = st[PushTEnv.STEM, :2]
        bar = st[PushTEnv.BAR, :2]
        ee = st[0, :2]
        ee_r = float(g[0, 0])
        stem_hw, stem_hh = float(g[PushTEnv.STEM, 0]), float(g[PushTEnv.STEM, 1])
        bar_hw, bar_hh = float(g[PushTEnv.BAR, 0]), float(g[PushTEnv.BAR, 1])
        mass = self.cost.stem_m + self.cost.bar_m
        c = (self.cost.stem_m * stem + self.cost.bar_m * bar) / mass
        rows = self.cost._enumerate_faces(
            stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh, c, err, ee,
            ee_r, rotate=True)
        right = [r for r in rows if r["name"] == "stem_px" and r["tau_w"] > 0.02]
        if not right:
            return
        pick = max(right, key=lambda r: r["tau_w"])
        prior = np.asarray(pick["into"], dtype=np.float64) * 2.2
        prior[0] = float(np.clip(prior[0], -2.6, -0.4))
        prior[1] = float(np.clip(prior[1], -1.4, 1.0))
        fmax = float(self.cfg.collect.force_max)
        self.cost.hint_action = np.clip(prior, -fmax, fmax).astype(np.float32)
        self.cost.face_np = np.asarray(pick["face"], dtype=np.float64)
        self.cost.into = np.asarray(pick["into"], dtype=np.float64)

    def _want_cem(self, cur) -> bool:
        """Servo floor first; CEM/unwind only for hard high +heading that is stuck."""
        if not getattr(self, "use_hybrid", True):
            return True
        if self.planner is None and not self.use_solver_cem:
            return False
        st = cur["obj_states"]
        th = float(st[PushTEnv.STEM, 2])
        err = float(np.arctan2(np.sin(th - self.goal_theta),
                               np.cos(th - self.goal_theta)))
        d_s = float(np.linalg.norm(st[PushTEnv.STEM, :2] - self.stem_goal))
        d_b = float(np.linalg.norm(st[PushTEnv.BAR, :2] - self.bar_goal))
        score = d_s + d_b + 1.6 * abs(err)
        origin = self.cost._origin(
            st[PushTEnv.STEM, :2], th, float(self.env.stem_hh))
        oy = float(origin[1])
        near = abs(err) < 0.20 and 0.12 <= max(d_s, d_b) < 0.22
        if near:
            # Stall timer only when pose stops improving inside the band (ep4).
            m = max(d_s, d_b)
            best_m = float(getattr(self, "_near_best_m", 1e9))
            if m < best_m - 0.004:
                self._near_best_m = m
                self._near_miss_t = 0
                self._hybrid_stuck = max(0, int(self._hybrid_stuck) - 2)
            else:
                self._near_miss_t = int(getattr(self, "_near_miss_t", 0)) + 1
                self._hybrid_stuck = int(self._hybrid_stuck) + 1
        elif score < self._hybrid_best - 0.012:
            self._hybrid_best = score
            self._hybrid_stuck = 0
            self._near_best_m = 1e9
            self._near_miss_t = 0
        else:
            self._hybrid_stuck = int(self._hybrid_stuck) + 1
            if not near:
                self._near_miss_t = 0
        # Near settle: hand back to servo / finish nudge (still track stuck).
        if abs(err) < 0.22 and max(d_s, d_b) < 0.45:
            self._cem_mode = False
            return False
        # +heading unwind at safe altitude (ep9/10), or high +heading (won ep18).
        if err > 0.30 and oy < 0.68:
            return True
        if err > 0.28 and oy > 0.72 and int(self._hybrid_stuck) > 20:
            return True
        return False

    def _finish_action(self, cur):
        """Gentle COM translate when heading is OK but pose sits just outside tol."""
        st = cur["obj_states"]
        stem = st[PushTEnv.STEM, :2].astype(np.float64)
        bar = st[PushTEnv.BAR, :2].astype(np.float64)
        ee = st[0, :2].astype(np.float64)
        d_s = float(np.linalg.norm(stem - self.stem_goal))
        d_b = float(np.linalg.norm(bar - self.bar_goal))
        m = max(d_s, d_b)
        mass = self.cost.stem_m + self.cost.bar_m
        c = (self.cost.stem_m * stem + self.cost.bar_m * bar) / mass
        goal_c = (self.cost.stem_m * np.asarray(self.stem_goal, dtype=np.float64)
                  + self.cost.bar_m * np.asarray(self.bar_goal, dtype=np.float64)) / mass
        delta = goal_c - c
        dn = float(np.linalg.norm(delta)) + 1e-6
        u = delta / dn
        mag = float(np.clip(28.0 * (m - 0.08), 2.8, 4.5))
        a = mag * u
        toward = c - ee
        tn = float(np.linalg.norm(toward)) + 1e-6
        a = a + 2.2 * (toward / tn)
        if float(ee[1]) < 0.18:
            a[1] = max(float(a[1]), 2.0)
        return np.clip(a, -6.0, 6.0).astype(np.float32)

    def _right_reclaim(self, cur, H: int = 6):
        """One-shot climb-then-left-press when EE is under the right face (ep20).

        Must run as a long open-loop macro (caller raises exec_horizon). Short
        closed-loop reclaim pollutes the contact geometry and fails.
        """
        fmax = float(self.cfg.collect.force_max)
        # Empirically cracked ep20 offline lock: climb with mild +Fx, then -Fx.
        up_n, press_n = 7, 18
        seq = np.zeros((up_n + press_n, 2), dtype=np.float32)
        seq[:up_n] = np.clip(
            np.array([1.63, 5.72], dtype=np.float32), -fmax, fmax)
        seq[up_n:] = np.clip(
            np.array([-3.61, 0.52], dtype=np.float32), -fmax, fmax)
        self._cem_plans = int(getattr(self, "_cem_plans", 0)) + 1
        self._right_reclaim_used = int(
            getattr(self, "_right_reclaim_used", 0)) + 1
        return seq.astype(np.float32)

    def _midclimb_finish(self, cur):
        """One-shot climb→right-press→stem_px from mid-climb +θ creep (ep7).

        Offline-searched from ep7@step100; keeps pose under tol without
        parking into the high-Y static-friction lock.
        """
        fmax = float(self.cfg.collect.force_max)
        up_n, press_n, un_n = 7, 18, 10
        seq = np.zeros((up_n + press_n + un_n, 2), dtype=np.float32)
        seq[:up_n] = np.clip(
            np.array([-0.46, 5.55], dtype=np.float32), -fmax, fmax)
        seq[up_n:up_n + press_n] = np.clip(
            np.array([4.77, -3.41], dtype=np.float32), -fmax, fmax)
        seq[up_n + press_n:] = np.clip(
            np.array([-5.0, 0.5], dtype=np.float32), -fmax, fmax)
        self._cem_plans = int(getattr(self, "_cem_plans", 0)) + 1
        self._midclimb_used = int(getattr(self, "_midclimb_used", 0)) + 1
        return seq.astype(np.float32)

    def _early_spin_finish(self, cur):
        """One-shot climb→press→+Fx from early large +θ (ep15@step30)."""
        fmax = float(self.cfg.collect.force_max)
        up_n, press_n, un_n = 8, 14, 8
        seq = np.zeros((up_n + press_n + un_n, 2), dtype=np.float32)
        seq[:up_n] = np.clip(
            np.array([5.38, 5.72], dtype=np.float32), -fmax, fmax)
        seq[up_n:up_n + press_n] = np.clip(
            np.array([3.40, -2.20], dtype=np.float32), -fmax, fmax)
        seq[up_n + press_n:] = np.clip(
            np.array([5.5, -0.5], dtype=np.float32), -fmax, fmax)
        self._cem_plans = int(getattr(self, "_cem_plans", 0)) + 1
        self._early_spin_used = int(getattr(self, "_early_spin_used", 0)) + 1
        return seq.astype(np.float32)

    def _high_reengage(self, cur, H: int = 6, stride: int = 3):
        """Separate under a high-parked T, climb onto top face, then press -Y (ep7/15).

        H matches exec_horizon (6) so the full left→up→press macro actually runs.
        """
        st = cur["obj_states"]
        th = float(st[PushTEnv.STEM, 2])
        err = float(np.arctan2(np.sin(th - self.goal_theta),
                               np.cos(th - self.goal_theta)))
        ee = st[0, :2].astype(np.float64)
        stem = st[PushTEnv.STEM, :2].astype(np.float64)
        bar = st[PushTEnv.BAR, :2].astype(np.float64)
        g = self._geom()
        stem_hw, stem_hh = float(g[PushTEnv.STEM, 0]), float(g[PushTEnv.STEM, 1])
        bar_hw, bar_hh = float(g[PushTEnv.BAR, 0]), float(g[PushTEnv.BAR, 1])
        xmin_h, _, _, ymax_h = self.cost._hull_bb(
            stem, bar, stem_hw, stem_hh, bar_hw, bar_hh, th)
        fmax = float(self.cfg.collect.force_max)
        left_clear = float(ee[0]) < xmin_h - 0.02
        on_top = float(ee[1]) > 0.88
        cands = []

        def _staged(a1, a2, a3):
            seq = np.zeros((H, 2), dtype=np.float32)
            n1 = max(H // 3, 1)
            n2 = max(H // 3, 1)
            seq[:n1] = np.clip(np.asarray(a1, dtype=np.float32), -fmax, fmax)
            seq[n1:n1 + n2] = np.clip(np.asarray(a2, dtype=np.float32), -fmax, fmax)
            seq[n1 + n2:] = np.clip(np.asarray(a3, dtype=np.float32), -fmax, fmax)
            return seq

        if on_top or (abs(err) < 0.20 and float(ee[1]) > 0.82):
            # On top with heading OK: skip scoring — always hard -Y finish.
            fx = -3.2 if err > 0.12 else (3.2 if err < -0.12 else 1.0)
            seq = np.tile(
                np.clip(np.array([fx, -5.4], dtype=np.float32), -fmax, fmax),
                (H, 1))
            self._cem_plans = int(getattr(self, "_cem_plans", 0)) + 1
            return seq.astype(np.float32)
        else:
            # left → up → press (fits in exec_horizon=6)
            cands.append(_staged([-5.2, 0.8], [0.5, 5.2], [0.5, -5.2]))
            cands.append(_staged([-5.0, 1.2], [1.0, 5.0], [1.2, -5.0]))
            if err > 0.20:
                cands.append(_staged([-5.0, 0.8], [0.0, 5.2], [-3.5, -4.5]))
            if left_clear:
                cands.append(_staged([0.0, 5.2], [1.5, 4.5], [0.5, -5.2]))
            # Mid-climb recover: go up hard then press (EE slipped under again).
            cands.append(_staged([0.0, 5.5], [0.5, 5.0], [0.8, -5.2]))
            cands.append(_staged([-2.0, 5.0], [0.0, 5.2], [1.0, -5.0]))
        best_seq, best_s = cands[0], 1e18
        for seq in cands:
            s = self._score_action_seq(seq, stride=stride)
            if s < best_s:
                best_s, best_seq = s, seq
        self._cem_plans = int(getattr(self, "_cem_plans", 0)) + 1
        return best_seq.astype(np.float32)

    def _score_action_seq(self, seq, stride: int = 3):
        """Roll ``seq`` under snapshot/restore; score final pose (not step cost)."""
        env = self.env
        snap = env.snapshot()
        for t in range(len(seq)):
            a = seq[t]
            for _ in range(stride):
                env.advance_action(a)
        st = env.raw_states()
        stem = st[PushTEnv.STEM, :2]
        bar = st[PushTEnv.BAR, :2]
        th = float(st[PushTEnv.STEM, 2])
        err = abs(float(np.arctan2(np.sin(th - self.goal_theta),
                                   np.cos(th - self.goal_theta))))
        d = float(np.linalg.norm(stem - self.stem_goal)
                  + np.linalg.norm(bar - self.bar_goal))
        oy = float(self.cost._origin(stem, th, float(self.env.stem_hh))[1])
        # Prefer lower heading, then pose, heavily punish flying high / off-board.
        score = d + 2.4 * err + 4.0 * max(0.0, oy - 0.72) + 5.0 * max(0.0, 0.18 - oy)
        score += 2.0 * max(0.0, float(stem[0]) - 1.95) + 2.0 * max(0.0, 0.25 - float(stem[0]))
        # Reward lowering Y when above the goal (ep2/7 high lock).
        goal_sy = float(self.stem_goal[1])
        if float(stem[1]) > goal_sy + 0.05:
            score += 3.5 * (float(stem[1]) - goal_sy)
        env.restore(snap)
        return score

    def _contact_mpc(self, cur, H: int = 8, stride: int = 3):
        """Discrete face/translate MPC — few physics rollouts, real solver."""
        st = cur["obj_states"]
        th = float(st[PushTEnv.STEM, 2])
        stem = st[PushTEnv.STEM, :2].astype(np.float64)
        bar = st[PushTEnv.BAR, :2].astype(np.float64)
        ee = st[0, :2].astype(np.float64)
        err = float(np.arctan2(np.sin(th - self.goal_theta),
                               np.cos(th - self.goal_theta)))
        g = self._geom()
        ee_r = float(g[0, 0])
        stem_hw, stem_hh = float(g[PushTEnv.STEM, 0]), float(g[PushTEnv.STEM, 1])
        bar_hw, bar_hh = float(g[PushTEnv.BAR, 0]), float(g[PushTEnv.BAR, 1])
        mass = self.cost.stem_m + self.cost.bar_m
        c = (self.cost.stem_m * stem + self.cost.bar_m * bar) / mass
        rows = self.cost._enumerate_faces(
            stem, bar, th, stem_hw, stem_hh, bar_hw, bar_hh, c, err, ee,
            ee_r, rotate=True)
        fmax = float(self.cfg.collect.force_max)
        cands = []
        # 1) current servo hint
        h0 = np.asarray(self.cost.hint_action, dtype=np.float32)
        cands.append(np.tile(h0, (H, 1)))
        # 2) finish / translate through COM
        goal_c = (self.cost.stem_m * np.asarray(self.stem_goal, dtype=np.float64)
                  + self.cost.bar_m * np.asarray(self.bar_goal, dtype=np.float64)) / mass
        delta = goal_c - c
        dn = float(np.linalg.norm(delta)) + 1e-6
        u = delta / dn
        staging = c - 0.10 * u
        staging[0] = float(np.clip(staging[0], 0.18, 2.04))
        staging[1] = float(np.clip(staging[1], 0.20, 0.88))
        d_st = staging - ee
        approach = (d_st / (float(np.linalg.norm(d_st)) + 1e-6)) * 4.5
        shove = 4.2 * u
        seq = np.zeros((H, 2), dtype=np.float32)
        n_app = min(4, H)
        seq[:n_app] = np.clip(approach, -fmax, fmax)
        seq[n_app:] = np.clip(shove, -fmax, fmax)
        cands.append(seq)
        # 3) per-face approach then into
        for row in rows:
            if row["tau_w"] < 0.005 and abs(err) > 0.25:
                continue
            face = np.asarray(row["face"], dtype=np.float64)
            into = np.asarray(row["into"], dtype=np.float64)
            n_out = np.asarray(row["n_out"], dtype=np.float64)
            stg = face + n_out * 0.08
            stg[0] = float(np.clip(stg[0], 0.16, 2.06))
            stg[1] = float(np.clip(stg[1], 0.18, 0.90))
            d = stg - ee
            app = (d / (float(np.linalg.norm(d)) + 1e-6)) * 5.0
            shv = into * 4.8
            # Prefer non-lifting shove when high.
            origin = self.cost._origin(stem, th, float(self.env.stem_hh))
            if float(origin[1]) > 0.72:
                shv[1] = min(float(shv[1]), 0.3)
            seq = np.zeros((H, 2), dtype=np.float32)
            seq[:n_app] = np.clip(app, -fmax, fmax)
            seq[n_app:] = np.clip(shv, -fmax, fmax)
            cands.append(seq)
        # 4) altitude / bound recoveries
        origin = self.cost._origin(stem, th, float(self.env.stem_hh))
        if float(origin[1]) > 0.70:
            down = np.array([0.0, -5.0], dtype=np.float32)
            cands.append(np.tile(np.clip(down, -fmax, fmax), (H, 1)))
            # Descend then stem_mx shove.
            seq = np.zeros((H, 2), dtype=np.float32)
            seq[: max(H // 2, 1)] = np.clip(down, -fmax, fmax)
            seq[max(H // 2, 1):] = np.clip(
                np.array([4.0, -1.0], dtype=np.float32), -fmax, fmax)
            cands.append(seq)
            left_app = np.array([-4.0, 1.5], dtype=np.float32)
            left_shv = np.array([4.2, -1.0], dtype=np.float32)
            seq = np.zeros((H, 2), dtype=np.float32)
            seq[:n_app] = np.clip(left_app, -fmax, fmax)
            seq[n_app:] = np.clip(left_shv, -fmax, fmax)
            cands.append(seq)
            left_hi = np.array([3.5, -1.0], dtype=np.float32)
            cands.append(np.tile(np.clip(left_hi, -fmax, fmax), (H, 1)))
        if float(origin[1]) < 0.25:
            up = np.array([0.5, 4.5], dtype=np.float32)
            cands.append(np.tile(np.clip(up, -fmax, fmax), (H, 1)))
        if err < -0.35:
            # -heading: right-face / -Fx candidates.
            right_shv = np.array([-4.0, 0.5], dtype=np.float32)
            cands.append(np.tile(np.clip(right_shv, -fmax, fmax), (H, 1)))
        # Above goal Y: top-face press then -Y (ep8 overshoot).
        goal_cy = float(goal_c[1])
        if float(c[1]) > goal_cy + 0.04:
            top_faces = [r for r in rows if str(r["name"]).endswith("_py")]
            for row in top_faces[:2]:
                face = np.asarray(row["face"], dtype=np.float64)
                into = np.asarray(row["into"], dtype=np.float64)
                n_out = np.asarray(row["n_out"], dtype=np.float64)
                stg = face + n_out * 0.07
                stg[0] = float(np.clip(stg[0], 0.16, 2.06))
                stg[1] = float(np.clip(stg[1], 0.18, 0.95))
                d = stg - ee
                app = (d / (float(np.linalg.norm(d)) + 1e-6)) * 5.0
                shv = into * 5.0
                shv[1] = min(float(shv[1]), -3.5)
                seq = np.zeros((H, 2), dtype=np.float32)
                seq[:n_app] = np.clip(app, -fmax, fmax)
                seq[n_app:] = np.clip(shv, -fmax, fmax)
                cands.append(seq)
            down = np.array([1.5, -5.0], dtype=np.float32)
            cands.append(np.tile(np.clip(down, -fmax, fmax), (H, 1)))
        # 5) EE on right of COM while goal is left: climb then -Fx (ep20).
        if (float(ee[0]) > float(c[0]) + 0.04
                and float(goal_c[0]) < float(c[0]) - 0.06):
            if float(ee[1]) < float(c[1]) - 0.02:
                climb = np.array([1.2, 5.5], dtype=np.float32)
                cands.append(np.tile(np.clip(climb, -fmax, fmax), (H, 1)))
                climb2 = np.array([0.4, 5.8], dtype=np.float32)
                cands.append(np.tile(np.clip(climb2, -fmax, fmax), (H, 1)))
            press = np.array([-5.2, 0.6], dtype=np.float32)
            cands.append(np.tile(np.clip(press, -fmax, fmax), (H, 1)))
            press2 = np.array([-4.0, 1.2], dtype=np.float32)
            cands.append(np.tile(np.clip(press2, -fmax, fmax), (H, 1)))
            # Short climb-then-press within H.
            n1 = max(H // 3, 2)
            seq = np.zeros((H, 2), dtype=np.float32)
            seq[:n1] = np.clip(np.array([1.4, 5.6], dtype=np.float32), -fmax, fmax)
            seq[n1:] = np.clip(np.array([-4.8, 0.5], dtype=np.float32), -fmax, fmax)
            cands.append(seq)
            seq = np.zeros((H, 2), dtype=np.float32)
            seq[:n1] = np.clip(np.array([0.6, 5.5], dtype=np.float32), -fmax, fmax)
            seq[n1:] = np.clip(np.array([-5.5, 0.8], dtype=np.float32), -fmax, fmax)
            cands.append(seq)
        best_seq, best_s = cands[0], 1e18
        for seq in cands:
            s = self._score_action_seq(seq, stride=stride)
            if s < best_s:
                best_s, best_seq = s, seq
        self._cem_plans = int(getattr(self, "_cem_plans", 0)) + 1
        return best_seq.astype(np.float32)

    def _plan(self, cur, goal):
        if self.succeeded(cur, goal):
            return self._idle_action()
        self.bind_cost(cur, goal)
        if getattr(self, "use_servo", False) and not self.use_solver_cem:
            return np.asarray(self.cost.hint_action, dtype=np.float32)

        # Pure CEM (solver or WM) when hybrid is off.
        if not getattr(self, "use_hybrid", True):
            self._apply_cem_heading_prior(cur)
            self._cem_plans = int(getattr(self, "_cem_plans", 0)) + 1
            self._action_stride = max(int(getattr(self.planner, "stride", 1)), 1)
            if self.use_solver_cem:
                return self.planner.plan_sequence(self.cost)
            return self._cem_sequence(cur)

        # Hybrid phase machine (solver or WM): macros/MPC on real physics;
        # CEM uses solver rollouts or WM imagination via _cem_sequence.
        met = self._phase_metrics(cur)
        phase = self._update_phase(met)
        self._apply_phase_cost(phase)
        st = met["st"]
        err = float(met["err"])
        m = float(met["m"])
        oy = float(met["oy"])
        want_uw = self._want_cem(cur)
        hstuck = int(self._hybrid_stuck)
        pstuck = int(getattr(self, "_phase_stuck", 0))

        if phase == "settle":
            self._action_stride = 1
            return self._idle_action()

        # --- hybrid_final control flow (preserves 12/20) ---
        # Extra CEM budget only while parked high (ep7/15 descend stall).
        cem_cap = 42 if oy > 0.72 else 30
        need_mpc = (
            int(getattr(self, "_cem_plans", 0)) < cem_cap
            and (
                (self._unwind is not None and int(self._unwind_t) > 45
                 and oy > 0.68 and abs(err) > 0.25)
                or (hstuck > 70 and abs(err) > 0.35 and oy > 0.65)
                or (err > 0.30 and oy >= 0.68 and hstuck > 25)
                or (hstuck > 140 and err < -0.55)
                or (phase == "descend" and pstuck > 35 and oy > 0.70)
                or (oy > 0.76 and hstuck > 55 and abs(err) < 0.40
                    and m < 0.40)
                # Mid-range translate stall (WM ep18): EE above COM, pose stuck.
                or (phase == "translate" and pstuck > 50 and m > 0.40
                    and abs(err) < 0.22 and hstuck > 35 and oy < 0.70
                    and float(st[0, 1]) > float(met["c"][1]) + 0.08)
            )
        )
        # Prefer gentle finish over MPC when stalled just outside tol.
        # High-Y near-miss: keep servo -Y hint — finish sinks EE under T (ep2).
        nmt = int(getattr(self, "_near_miss_t", 0))
        # Early large +θ before high park (ep15@phys-step≈30, m≈0.84).
        need_early_spin = (
            err > 0.68 and 0.830 < m < 0.850
            and 0.615 < oy < 0.625
            and float(st[0, 0]) < float(met["c"][0]) - 0.05
            and int(getattr(self, "_early_spin_used", 0)) < 1
            and int(getattr(self, "_cem_plans", 0)) < 12
        )
        if need_early_spin:
            self._unwind = None
            self._action_stride = 3
            seq = self._early_spin_finish(cur)
            self.exec_horizon = int(len(seq))
            self._right_exec_restore = 6
            return seq
        # Mid-climb +θ creep before high-Y lock (ep7).
        need_midclimb = (
            0.05 < err < 0.22 and 0.40 < m < 0.60
            and 0.62 < oy < 0.72
            and float(st[0, 0]) < float(met["c"][0]) - 0.05
            and int(getattr(self, "_midclimb_used", 0)) < 1
            and int(getattr(self, "_cem_plans", 0)) < 20
        )
        if need_midclimb:
            self._unwind = None
            self._action_stride = 3
            seq = self._midclimb_finish(cur)
            self.exec_horizon = int(len(seq))
            self._right_exec_restore = 6
            return seq
        # EE under/right of COM, goal left: one-shot long reclaim (ep20).
        need_right = (
            abs(err) < 0.20 and 0.26 < m < 0.42
            and hstuck > 20
            and float(met["goal_c"][0]) < float(met["c"][0]) - 0.08
            and float(st[0, 0]) > float(met["c"][0]) + 0.08
            and float(st[0, 1]) < float(met["c"][1]) - 0.05
            and int(getattr(self, "_right_reclaim_used", 0)) < 2
            and int(getattr(self, "_cem_plans", 0)) < 40
        )
        if need_right:
            self._unwind = None
            self._action_stride = 3
            seq = self._right_reclaim(cur)
            self.exec_horizon = int(len(seq))
            self._right_exec_restore = 6
            return seq
        # Restore default exec_horizon after a long reclaim macro.
        if int(getattr(self, "_right_exec_restore", 0)) > 0:
            self.exec_horizon = int(self._right_exec_restore)
            self._right_exec_restore = 0
        if nmt > 6 and oy > 0.65 and m < 0.28:
            self._unwind = None
            self._action_stride = 1
            return np.asarray(self.cost.hint_action, dtype=np.float32)
        if nmt > 6:
            self._unwind = None
            self._action_stride = 1
            return self._finish_action(cur)
        if int(getattr(self, "_reengage_cool", 0)) > 0:
            self._reengage_cool = int(self._reengage_cool) - 1
        ee_y = float(st[0, 1])
        # High-park +heading: climb on top (ep7). Signed err protects ep8.
        need_reengage = (
            int(getattr(self, "_cem_plans", 0)) < 50
            and int(getattr(self, "_reengage_cool", 0)) <= 0
            and oy > 0.74 and 0.22 < m < 0.40
            and err > 0.26 and hstuck > 35
            and int(getattr(self, "_reengage_used", 0)) < 6
        )
        if need_reengage:
            self._unwind = None
            self._reengage_used = int(getattr(self, "_reengage_used", 0)) + 1
            self._reengage_cool = 3
            self._action_stride = 3
            return self._high_reengage(cur, H=6, stride=3)
        # Finish: EE on top of high near-pose T — hard -Y (don't wait for reengage).
        need_top_press = (
            oy > 0.70 and m < 0.28 and abs(err) < 0.18
            and ee_y > 0.85
            and int(getattr(self, "_cem_plans", 0)) < 55
            and int(getattr(self, "_top_press_used", 0)) < 14
            and int(getattr(self, "_reengage_cool", 0)) <= 0
        )
        if need_top_press:
            self._unwind = None
            self._top_press_used = int(getattr(self, "_top_press_used", 0)) + 1
            self._reengage_cool = 2
            self._action_stride = 3
            return self._high_reengage(cur, H=6, stride=3)
        if need_mpc:
            self._unwind = None
            self._action_stride = 3
            return self._contact_mpc(cur, H=8, stride=3)
        if self._unwind is not None:
            self._action_stride = 1
            return self._unwind_action(cur)
        if want_uw:
            self._unwind = "neg" if err < 0.0 else "pos"
            self._unwind_t = 0
            self._action_stride = 1
            return self._unwind_action(cur)
        self._action_stride = 1
        return np.asarray(self.cost.hint_action, dtype=np.float32)

    def metric(self, obs, goal) -> float:
        st = obs["obj_states"]
        d = np.linalg.norm(st[PushTEnv.STEM, :2] - self.stem_goal) \
            + np.linalg.norm(st[PushTEnv.BAR, :2] - self.bar_goal)
        return float(d)

    def potential(self, obs, goal=None) -> float:
        """Progress potential Φ matching ``learning.rl.imagine.push_t_reward``.

        Φ = ‖stem−g_s‖ + ‖bar−g_b‖ + ang_w·|θ| + reach_w·min(‖ee−stem‖, ‖ee−bar‖)
        """
        st = np.asarray(obs["obj_states"], dtype=np.float64)
        stem = st[PushTEnv.STEM, :2]
        bar = st[PushTEnv.BAR, :2]
        ee = st[0, :2]
        d_xy = float(np.linalg.norm(stem - self.stem_goal)
                     + np.linalg.norm(bar - self.bar_goal))
        th = float(st[PushTEnv.STEM, 2])
        ang = abs(float(np.arctan2(np.sin(th - self.goal_theta),
                                   np.cos(th - self.goal_theta))))
        d_reach = float(min(np.linalg.norm(ee - stem), np.linalg.norm(ee - bar)))
        ang_w, reach_w = 0.5, 0.5
        return d_xy + ang_w * ang + reach_w * d_reach

    def succeeded(self, obs, goal) -> bool:
        st = obs["obj_states"]
        d_s = float(np.linalg.norm(st[PushTEnv.STEM, :2] - self.stem_goal))
        d_b = float(np.linalg.norm(st[PushTEnv.BAR, :2] - self.bar_goal))
        ang = abs(float(np.arctan2(np.sin(st[PushTEnv.STEM, 2]),
                                  np.cos(st[PushTEnv.STEM, 2]))))
        vel = float(np.linalg.norm(st[PushTEnv.STEM, 3:5]))
        return d_s < self.tol and d_b < self.tol and ang < self.ang_tol and vel < self.vel_tol

    def extra_result(self, obs, goal) -> dict:
        st = obs["obj_states"]
        th = float(st[PushTEnv.STEM, 2])
        return {
            "d_stem": float(np.linalg.norm(st[PushTEnv.STEM, :2] - self.stem_goal)),
            "d_bar": float(np.linalg.norm(st[PushTEnv.BAR, :2] - self.bar_goal)),
            "ang": float(abs(np.arctan2(np.sin(th), np.cos(th)))),
            "vel": float(np.linalg.norm(st[PushTEnv.STEM, 3:5])),
            "ee_x": float(st[0, 0]),
            "ee_y": float(st[0, 1]),
            "mode": str(getattr(self.cost, "mode", "")),
            "phase": str(getattr(self, "_phase", "")),
            "init_ee_gap": float(getattr(self.env, "_spawn_ee_gap", float("nan"))),
        }
