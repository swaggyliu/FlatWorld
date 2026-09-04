"""Flat vector drawing for the push scene."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pyray as rl

from learning.env.flatworld_wrapper import OBJ_TYPE_BOX

from .camera import Camera
from .levels import CAMPAIGN_LEN

BG = rl.Color(22, 32, 48, 255)
GROUND = rl.Color(38, 62, 56, 255)
GROUND_LINE = rl.Color(120, 168, 150, 255)
GRID = rl.Color(40, 55, 72, 255)
EE = rl.Color(62, 207, 190, 255)
EE_CORE = rl.Color(200, 255, 245, 180)
BOX = rl.Color(186, 176, 160, 255)
BALL = rl.Color(176, 154, 206, 255)
TARGET = rl.Color(233, 196, 106, 255)
HAZARD = rl.Color(231, 111, 81, 255)
HAZARD_STRIPE = rl.Color(40, 20, 18, 180)
GOAL = rl.Color(230, 57, 70, 255)
STEER = rl.Color(120, 230, 255, 255)
SELECT = rl.Color(90, 220, 255, 255)
TRAIL = rl.Color(62, 207, 190, 90)
TEXT = rl.Color(230, 228, 220, 255)
MUTED = rl.Color(150, 160, 175, 255)
HP_OK = rl.Color(42, 157, 143, 255)
HP_BAD = rl.Color(231, 111, 81, 255)
PANEL = rl.Color(16, 24, 36, 210)
BRAIN = rl.Color(186, 140, 214, 255)
BRAIN_DEEP = rl.Color(92, 58, 128, 255)
BRAIN_GLOW = rl.Color(168, 120, 220, 50)
BTN = rl.Color(42, 157, 143, 255)
BTN_HOT = rl.Color(64, 190, 172, 255)
CARD = rl.Color(20, 30, 46, 230)
SPRING = rl.Color(214, 168, 86, 255)
SPRING_DK = rl.Color(122, 82, 36, 255)
ICE = rl.Color(210, 236, 244, 255)
ICE_DK = rl.Color(70, 130, 150, 255)
RUBBER = rl.Color(232, 118, 64, 255)
RUBBER_DK = rl.Color(110, 42, 22, 255)
SLIDER_BG = rl.Color(40, 48, 60, 255)
KNOB = rl.Color(240, 236, 228, 255)
SIZE = rl.Color(120, 196, 230, 255)

SCALE_LO, SCALE_HI = 0.5, 2.0

FONT = None
_FONT_SPACING = 0.6


def _font_candidates():
    """Bundled Roboto first so Windows and macOS look the same."""
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "Roboto-Bold.ttf")
    paths = [bundled]
    if sys.platform == "darwin":
        paths += [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    elif sys.platform.startswith("win"):
        fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        paths += [
            os.path.join(fonts, "arialbd.ttf"),
            os.path.join(fonts, "segoeuib.ttf"),
        ]
    else:
        paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    return paths


def _load_ttf(path: str, size: int = 64):
    """Load ASCII + Latin-1 so HUD punctuation (·, —) rasterizes."""
    cps = list(range(32, 256))
    try:
        arr = rl.ffi.new("int[]", cps)
        ptr = rl.ffi.cast("int *", arr)
        return rl.load_font_ex(path, size, ptr, len(cps))
    except (TypeError, AttributeError):
        try:
            return rl.load_font_ex(path, size, None, 0)
        except TypeError:
            return rl.load_font(path)


def load_ui_font():
    """Roboto Bold (bundled). Same TTF on Windows and macOS."""
    global FONT
    for path in _font_candidates():
        if not os.path.isfile(path):
            continue
        try:
            font = _load_ttf(path)
        except Exception:
            continue
        if font is None or int(getattr(font, "glyphCount", 0) or 0) <= 0:
            continue
        try:
            rl.set_texture_filter(font.texture, rl.TEXTURE_FILTER_BILINEAR)
        except Exception:
            pass
        FONT = font
        return font
    FONT = None
    return None


def unload_ui_font():
    global FONT
    if FONT is not None:
        try:
            rl.unload_font(FONT)
        except Exception:
            pass
        FONT = None


def text(s, x, y, size, color):
    s = str(s)
    if FONT is None:
        rl.draw_text(s, int(x), int(y), int(size), color)
        return
    rl.draw_text_ex(FONT, s, rl.Vector2(float(x), float(y)),
                    float(size), _FONT_SPACING, color)


def text_width(s, size) -> float:
    s = str(s)
    if FONT is None:
        return float(rl.measure_text(s, int(size)))
    return float(rl.measure_text_ex(FONT, s, float(size), _FONT_SPACING).x)


def scale_to_t(scale: float) -> float:
    return (float(scale) - SCALE_LO) / (SCALE_HI - SCALE_LO)


def t_to_scale(t: float) -> float:
    return SCALE_LO + float(t) * (SCALE_HI - SCALE_LO)


t_to_scale = t_to_scale
scale_to_t = scale_to_t


def _box(cam: Camera, x, y, th, hw, hh, fill, outline=None):
    sx = cam.to_screen(x, y)
    w, h = cam.px(2.0 * hw), cam.px(2.0 * hh)
    rec = rl.Rectangle(sx.x, sx.y, w, h)
    origin = rl.Vector2(w * 0.5, h * 0.5)
    deg = -math.degrees(th)
    rl.draw_rectangle_pro(rec, origin, deg, fill)
    if outline is not None:
        # slightly larger stroke via a second pass
        rec2 = rl.Rectangle(sx.x, sx.y, w + 3, h + 3)
        origin2 = rl.Vector2((w + 3) * 0.5, (h + 3) * 0.5)
        rl.draw_rectangle_pro(rec2, origin2, deg, outline)


def _tint(color, toward, t: float):
    return rl.color_lerp(color, toward, max(0.0, min(1.0, t)))


def _circle(cam: Camera, x, y, r, fill, th: float = 0.0, core=None):
    """Ball with rotating wedges so spin is visible."""
    c = cam.to_screen(x, y)
    pr = cam.px(r)
    deg = -math.degrees(th)
    dark = _tint(fill, rl.Color(28, 22, 32, 255), 0.28)
    pale = _tint(fill, rl.Color(255, 250, 240, 255), 0.22)
    segs = max(10, int(pr * 0.45))
    step = 60.0
    for i in range(6):
        col = pale if i % 2 == 0 else dark
        rl.draw_circle_sector(c, pr, deg + i * step, deg + (i + 1) * step,
                              segs, col)
    if core is not None:
        rl.draw_circle_v(c, pr * 0.28, core)


def draw_spring_along(ax, ay, bx, by, coils: int, amp: float,
                      color=SPRING, thick: float = 2.4):
    """Zigzag coil from (ax,ay) to (bx,by) in screen pixels."""
    dx, dy = bx - ax, by - ay
    ln = (dx * dx + dy * dy) ** 0.5
    if ln < 1.0:
        return
    ux, uy = dx / ln, dy / ln
    px, py = -uy, ux
    n = max(4, int(coils)) * 2
    prev = rl.Vector2(ax, ay)
    for i in range(1, n + 1):
        t = i / n
        side = amp if (i % 2) else -amp
        if i == n:
            side = 0.0
        cur = rl.Vector2(ax + ux * ln * t + px * side,
                         ay + uy * ln * t + py * side)
        rl.draw_line_ex(prev, cur, thick, color)
        prev = cur
    rl.draw_circle(int(ax), int(ay), 3, color)
    rl.draw_circle(int(bx), int(by), 3, color)


def draw_spring_icon(cx: float, cy: float, w: float, h: float, coils: int = 5):
    draw_spring_along(cx - w * 0.5, cy, cx + w * 0.5, cy, coils, h * 0.5,
                      SPRING, 2.6)
    rl.draw_circle(int(cx - w * 0.5), int(cy), 3, SPRING_DK)
    rl.draw_circle(int(cx + w * 0.5), int(cy), 3, SPRING_DK)


def _grip_color(mu: float):
    """Ice → rubber. Same map for the spirit rim and the grip slider."""
    t = max(0.0, min(1.0, float(mu)))
    return rl.color_lerp(ICE, RUBBER, t), rl.color_lerp(ICE_DK, RUBBER_DK, t)


def draw_grip_icon(cx: float, cy: float, mu: float, r: float = 11.0):
    """Mini body + ice/rubber rim, matching the spirit."""
    rim, _ = _grip_color(mu)
    rl.draw_circle(int(cx), int(cy), int(r + 3.5), rim)
    rl.draw_circle(int(cx), int(cy), int(r), rl.Color(72, 214, 196, 255))


def _draw_slider(track: rl.Rectangle, value: float, fill):
    rl.draw_rectangle_rounded(track, 0.55, 6, SLIDER_BG)
    v = max(0.0, min(1.0, float(value)))
    fw = max(track.height, track.width * v)
    filled = rl.Rectangle(track.x, track.y, fw, track.height)
    rl.draw_rectangle_rounded(filled, 0.55, 6, fill)
    kx = track.x + v * track.width
    ky = track.y + track.height * 0.5
    rl.draw_circle(int(kx), int(ky), int(track.height * 0.78), KNOB)
    rl.draw_circle(int(kx), int(ky), int(track.height * 0.40), fill)


def slider_value(track: rl.Rectangle, mx: float) -> float:
    if track.width <= 1.0:
        return 0.0
    return max(0.0, min(1.0, (mx - track.x) / track.width))


def draw_size_icon(cx: float, cy: float, scale: float):
    t = scale_to_t(scale)
    r = 5.0 + 9.0 * t
    rl.draw_circle(int(cx), int(cy), int(r + 2), SIZE)
    rl.draw_circle(int(cx), int(cy), int(max(3.0, r - 2)), rl.Color(210, 240, 250, 230))


def equipment_layout(width: int) -> dict:
    pw, ph = 232, 248
    px, py = width - pw - 16.0, 12.0
    panel = rl.Rectangle(px, py, pw, ph)
    track_w, track_h = 148.0, 20.0
    track_x = px + 66.0
    spring = rl.Rectangle(track_x, py + 48.0, track_w, track_h)
    grip = rl.Rectangle(track_x, py + 112.0, track_w, track_h)
    size = rl.Rectangle(track_x, py + 176.0, track_w, track_h)
    return {
        "panel": panel,
        "spring": spring,
        "grip": grip,
        "size": size,
        "icon_spring": (px + 34.0, py + 58.0),
        "icon_grip": (px + 34.0, py + 122.0),
        "icon_size": (px + 34.0, py + 186.0),
        "hud": rl.Rectangle(16, 12, 500, 168),
    }


def draw_equipment_panel(width: int, restitution: float, friction: float,
                         scale: float = 1.0) -> dict:
    lay = equipment_layout(width)
    rl.draw_rectangle_rounded(lay["panel"], 0.12, 6, PANEL)
    text("EQUIPMENT", lay["panel"].x + 16, lay["panel"].y + 10, 16, MUTED)
    sx, sy = lay["icon_spring"]
    draw_spring_icon(sx, sy, 30, 16, coils=5)
    gx, gy = lay["icon_grip"]
    draw_grip_icon(gx, gy, friction)
    zx, zy = lay["icon_size"]
    draw_size_icon(zx, zy, scale)
    grip_col = _grip_color(friction)[0]
    text("bounce", lay["spring"].x, lay["spring"].y - 16, 13, SPRING)
    text("grip", lay["grip"].x, lay["grip"].y - 16, 13, grip_col)
    text("size", lay["size"].x, lay["size"].y - 16, 13, SIZE)
    _draw_slider(lay["spring"], restitution, SPRING)
    _draw_slider(lay["grip"], friction, grip_col)
    _draw_slider(lay["size"], scale_to_t(scale), SIZE)
    text(f"{restitution:.2f}", lay["spring"].x + lay["spring"].width - 40,
         lay["spring"].y + 22, 14, SPRING)
    text(f"{friction:.2f}", lay["grip"].x + lay["grip"].width - 40,
         lay["grip"].y + 22, 14, grip_col)
    text(f"{scale:.2f}x", lay["size"].x + lay["size"].width - 48,
         lay["size"].y + 22, 14, SIZE)
    return lay


def _draw_body_spring(cam: Camera, x, y, r, dx, dy, rest: float):
    """One coil sticking out of the body along a unit direction."""
    s = 0.30 + 0.70 * rest
    ax = x + dx * r * 0.82
    ay = y + dy * r * 0.82
    length = r * (0.36 + 0.64 * s)
    bx = ax + dx * length
    by = ay + dy * length
    a = cam.to_screen(ax, ay)
    b = cam.to_screen(bx, by)
    amp = cam.px(r * (0.12 + 0.20 * rest))
    draw_spring_along(a.x, a.y, b.x, b.y, 3 + int(round(rest * 5)), amp,
                      SPRING, 2.2 + rest)


def _spirit_gear(cam: Camera, x, y, r, restitution):
    """Cardinal springs (restitution). Friction is the body rim color."""
    rest = max(0.0, min(1.0, float(restitution)))
    for dx, dy in ((-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        _draw_body_spring(cam, x, y, r, dx, dy, rest)


def draw_spirit(cam: Camera, x, y, r, look_x=1.0, look_y=0.0,
                hurt=0.0, happy=False, restitution: float = 0.15,
                friction: float = 0.5):
    """Chubby round spirit for the force-controlled EE."""
    ln = (look_x * look_x + look_y * look_y) ** 0.5
    if ln < 1e-6:
        lx, ly = 1.0, 0.0
    else:
        lx, ly = look_x / ln, look_y / ln

    pr = cam.px(r * 1.08)
    body_c = cam.to_screen(x, y)
    shadow = cam.to_screen(x, 0.012)
    rl.draw_ellipse(int(shadow.x), int(shadow.y), int(pr * 0.88),
                    max(4, int(pr * 0.26)), rl.Color(8, 12, 18, 80))

    body = rl.Color(72, 214, 196, 255)
    if hurt > 0:
        body = rl.color_lerp(body, HAZARD, min(1.0, hurt))
    rim, _ = _grip_color(friction)
    rl.draw_circle_v(body_c, pr + 5.0, rim)
    rl.draw_circle_v(body_c, pr, body)

    # dumpling ears
    ear_r = pr * 0.28
    ear_l = cam.to_screen(x - r * 0.55, y + r * 0.72)
    ear_rgt = cam.to_screen(x + r * 0.55, y + r * 0.72)
    rl.draw_circle_v(ear_l, ear_r + 2.5, rim)
    rl.draw_circle_v(ear_rgt, ear_r + 2.5, rim)
    rl.draw_circle_v(ear_l, ear_r, body)
    rl.draw_circle_v(ear_rgt, ear_r, body)

    # pale belly
    belly = cam.to_screen(x, y - r * 0.10)
    rl.draw_circle_v(belly, pr * 0.58, rl.Color(210, 250, 242, 210))

    # glossy highlight
    hi = cam.to_screen(x - r * 0.30, y + r * 0.30)
    rl.draw_circle_v(hi, pr * 0.20, rl.Color(255, 255, 255, 95))

    blush = rl.Color(255, 130, 150, 90 if not happy else 130)
    rl.draw_circle_v(cam.to_screen(x - r * 0.48, y - r * 0.06), pr * 0.16, blush)
    rl.draw_circle_v(cam.to_screen(x + r * 0.48, y - r * 0.06), pr * 0.16, blush)

    _spirit_gear(cam, x, y, r, restitution)

    eye_y = y + r * 0.10
    eye_dx = r * 0.30
    eye_pr = max(4.0, cam.px(r * 0.20))
    if happy:
        for s in (-1.0, 1.0):
            e = cam.to_screen(x + s * eye_dx, eye_y)
            rl.draw_line_ex(rl.Vector2(e.x - eye_pr * 0.7, e.y),
                            rl.Vector2(e.x, e.y - eye_pr * 0.55), 3.0,
                            rl.Color(30, 48, 58, 255))
            rl.draw_line_ex(rl.Vector2(e.x, e.y - eye_pr * 0.55),
                            rl.Vector2(e.x + eye_pr * 0.7, e.y), 3.0,
                            rl.Color(30, 48, 58, 255))
    elif hurt > 0.35:
        for s in (-1.0, 1.0):
            e = cam.to_screen(x + s * eye_dx, eye_y)
            d = eye_pr * 0.55
            rl.draw_line_ex(rl.Vector2(e.x - d, e.y - d),
                            rl.Vector2(e.x + d, e.y + d), 3.0,
                            rl.Color(40, 30, 36, 255))
            rl.draw_line_ex(rl.Vector2(e.x - d, e.y + d),
                            rl.Vector2(e.x + d, e.y - d), 3.0,
                            rl.Color(40, 30, 36, 255))
    else:
        look_px = eye_pr * 0.38
        for s in (-1.0, 1.0):
            e = cam.to_screen(x + s * eye_dx, eye_y)
            rl.draw_circle_v(e, eye_pr, rl.Color(250, 252, 255, 255))
            pupil = rl.Vector2(e.x + lx * look_px, e.y - ly * look_px)
            rl.draw_circle_v(pupil, eye_pr * 0.48, rl.Color(32, 46, 58, 255))
            rl.draw_circle_v(rl.Vector2(pupil.x - eye_pr * 0.15,
                                        pupil.y - eye_pr * 0.15),
                             eye_pr * 0.16, rl.Color(255, 255, 255, 230))

    # smile
    mouth = cam.to_screen(x, y - r * 0.28)
    mw = cam.px(r * 0.22)
    if happy:
        rl.draw_line_ex(rl.Vector2(mouth.x - mw, mouth.y),
                        rl.Vector2(mouth.x, mouth.y + mw * 0.55), 2.5,
                        rl.Color(40, 70, 78, 220))
        rl.draw_line_ex(rl.Vector2(mouth.x, mouth.y + mw * 0.55),
                        rl.Vector2(mouth.x + mw, mouth.y), 2.5,
                        rl.Color(40, 70, 78, 220))
    elif hurt <= 0.35:
        rl.draw_line_ex(rl.Vector2(mouth.x - mw * 0.7, mouth.y),
                        rl.Vector2(mouth.x + mw * 0.7, mouth.y), 2.2,
                        rl.Color(40, 70, 78, 200))


def draw_star(cam: Camera, x, y, r_m, color, rot_deg=0.0):
    c = cam.to_screen(x, y)
    rl.draw_poly(c, 5, cam.px(r_m), rot_deg, color)
    rl.draw_poly(c, 5, cam.px(r_m * 0.45), rot_deg + 36.0, rl.Color(255, 240, 200, 230))


def _select_ring(cam: Camera, x, y, hw, hh):
    c = cam.to_screen(x, y)
    rl.draw_ring(c, cam.px(max(hw, hh) + 0.012), cam.px(max(hw, hh) + 0.022),
                 0, 360, 24, SELECT)


def draw_steer(cam: Camera, x, y):
    c = cam.to_screen(x, y)
    s = cam.px(0.028)
    rl.draw_line_ex(rl.Vector2(c.x - s, c.y), rl.Vector2(c.x + s, c.y), 3.0, STEER)
    rl.draw_line_ex(rl.Vector2(c.x, c.y - s), rl.Vector2(c.x, c.y + s), 3.0, STEER)
    rl.draw_circle_lines(int(c.x), int(c.y), cam.px(0.022), STEER)


def _hazard_mark(cam: Camera, x, y, hw, hh):
    c = cam.to_screen(x, y)
    pulse = 0.55 + 0.45 * math.sin(rl.get_time() * 7.0)
    rad = cam.px(max(hw, hh) + 0.018)
    col = rl.color_alpha(HAZARD, pulse)
    rl.draw_ring(c, rad, rad + 5.0, 0, 360, 28, col)
    s = cam.px(max(hw, hh) * 0.42)
    rl.draw_line_ex(rl.Vector2(c.x - s, c.y - s), rl.Vector2(c.x + s, c.y + s),
                    3.5, HAZARD_STRIPE)
    rl.draw_line_ex(rl.Vector2(c.x - s, c.y + s), rl.Vector2(c.x + s, c.y - s),
                    3.5, HAZARD_STRIPE)


def draw_force_arrow(cam: Camera, x, y, r, fx, fy):
    mag = float(math.hypot(fx, fy))
    if mag < 0.12:
        return
    ux, uy = fx / mag, fy / mag
    length = 0.07 + 0.16 * min(1.0, mag / 6.0)
    ax, ay = x + ux * r * 0.35, y + uy * r * 0.35
    bx, by = ax + ux * length, ay + uy * length
    a = cam.to_screen(ax, ay)
    b = cam.to_screen(bx, by)
    rl.draw_line_ex(a, b, 3.4, STEER)
    px, py = -uy, ux
    head = 0.028
    p1 = cam.to_screen(bx - ux * head + px * head * 0.55,
                       by - uy * head + py * head * 0.55)
    p2 = cam.to_screen(bx - ux * head - px * head * 0.55,
                       by - uy * head - py * head * 0.55)
    rl.draw_triangle(b, p1, p2, STEER)


def draw_scene(cam: Camera, states: np.ndarray, types, geom: np.ndarray,
               target_idx: int, hazard_idx: int, goal, trail,
               hit_flash: float = 0.0, happy: bool = False,
               push_idx: int | None = None, steer_goal=None,
               restitution: float = 0.15, friction: float = 0.5,
               force=None):
    rl.draw_rectangle(0, int(cam.oy), cam.width, cam.height - int(cam.oy), GROUND)
    x0, x1 = cam.visible_x()
    for gx in np.arange(math.floor(x0 / 0.2) * 0.2, x1 + 0.2, 0.2):
        a = cam.to_screen(gx, 0.0)
        b = cam.to_screen(gx, 0.55)
        rl.draw_line_ex(a, b, 1.0, GRID)
    rl.draw_line_ex(rl.Vector2(0.0, cam.oy),
                    rl.Vector2(float(cam.width), cam.oy), 4.0, GROUND_LINE)

    if trail:
        for i in range(1, len(trail)):
            a = cam.to_screen(*trail[i - 1])
            b = cam.to_screen(*trail[i])
            rl.draw_line_ex(a, b, 3.0, TRAIL)

    n = len(states)
    focus = {0, target_idx, hazard_idx, push_idx}
    order = [i for i in range(n) if i not in focus]
    if hazard_idx not in (None, target_idx, 0):
        order.append(hazard_idx)
    if target_idx not in (None, 0):
        order.append(target_idx)
    if push_idx not in (None, 0, target_idx):
        order.append(push_idx)
    order.append(0)

    for i in order:
        x, y, th = float(states[i, 0]), float(states[i, 1]), float(states[i, 2])
        hw, hh = float(geom[i, 0]), float(geom[i, 1])
        t = int(types[i])
        if i == 0:
            if steer_goal is not None:
                look_x = float(steer_goal[0]) - x
                look_y = float(steer_goal[1]) - y
            elif push_idx not in (None, 0):
                look_x = float(states[push_idx, 0]) - x
                look_y = float(states[push_idx, 1]) - y
            elif target_idx not in (None, 0):
                look_x = float(states[target_idx, 0]) - x
                look_y = float(states[target_idx, 1]) - y
            else:
                look_x, look_y = 1.0, 0.0
            draw_spirit(cam, x, y, hw, look_x, look_y,
                        hurt=hit_flash, happy=happy,
                        restitution=restitution, friction=friction)
            if force is not None:
                draw_force_arrow(cam, x, y, hw, float(force[0]), float(force[1]))
            continue
        if t == OBJ_TYPE_BOX:
            fill = HAZARD if i == hazard_idx else (TARGET if i == target_idx else BOX)
            _box(cam, x, y, th, hw, hh, fill, outline=None)
            if i == hazard_idx:
                _hazard_mark(cam, x, y, hw, hh)
        else:
            fill = HAZARD if i == hazard_idx else (TARGET if i == target_idx else BALL)
            _circle(cam, x, y, hw, fill, th)
            if i == target_idx:
                rl.draw_poly(cam.to_screen(x, y), 4, cam.px(0.025),
                             -math.degrees(th) + 45.0,
                             rl.Color(40, 30, 10, 220))
            if i == hazard_idx:
                _hazard_mark(cam, x, y, hw, hh)
        if i == push_idx:
            _select_ring(cam, x, y, hw, hh)

    # Markers last so boxes / the spirit cannot cover the drop points.
    if goal is not None:
        draw_star(cam, float(goal[0]), float(goal[1]), 0.045, GOAL)
    if steer_goal is not None:
        draw_steer(cam, float(steer_goal[0]), float(steer_goal[1]))


def draw_hud(status: str, lines: list[str],
             width: int, level: int = 1,
             elapsed: float = 0.0, total_score: int = 0, pace: int = 0,
             par: float = 0.0, pushes: int = 0, push_cap: int = 3,
             best: float | None = None, stars: int = 0, hint: str = ""):
    panel = rl.Rectangle(16, 12, 500, 168)
    rl.draw_rectangle_rounded(panel, 0.12, 6, PANEL)
    if level <= CAMPAIGN_LEN:
        title = f"Spirit Push   ·   {level}/{CAMPAIGN_LEN}"
    else:
        title = f"Spirit Push   ·   {level}"
    text(title, 28, 20, 18, TEXT)
    text(f"TIME  {elapsed:5.1f}s", 28, 48, 16, TEXT)
    text(f"PAR {par:.0f}s", 188, 48, 16, MUTED)
    best_s = f"{best:.1f}s" if best is not None else "--"
    text(f"BEST {best_s}", 280, 48, 16, MUTED)
    text(f"+{pace}", 400, 48, 16, TARGET)
    text(f"PUSH {pushes}/{push_cap}", 28, 72, 16, TEXT)
    text(f"SCORE {total_score}", 188, 72, 16, TEXT)
    for i in range(3):
        col = TARGET if i < max(stars, 0) else rl.Color(70, 78, 90, 255)
        rl.draw_poly(rl.Vector2(400 + i * 22, 80), 5, 8, -90, col)
    col = rl.GOLD if "WIN" in status else (HAZARD if "LOSE" in status else TEXT)
    text(status, 28, 98, 16, col)
    if hint:
        text(hint, 28, 122, 14, MUTED)
    y = 192
    for line in lines:
        text(line, 20, y, 15, MUTED)
        y += 20
    return panel


def _hit(rec: rl.Rectangle, mx: float, my: float) -> bool:
    return rec.x <= mx <= rec.x + rec.width and rec.y <= my <= rec.y + rec.height


def draw_brain(cx: float, cy: float, scale: float = 1.0):
    """Stylized top-down brain icon (two lobes + sulci)."""
    s = 70.0 * scale
    rl.draw_circle(int(cx), int(cy - 4 * scale), int(s * 1.15), BRAIN_GLOW)
    rl.draw_ellipse(int(cx - 0.28 * s), int(cy), int(0.50 * s), int(0.40 * s), BRAIN_DEEP)
    rl.draw_ellipse(int(cx + 0.28 * s), int(cy), int(0.50 * s), int(0.40 * s), BRAIN_DEEP)
    rl.draw_ellipse(int(cx - 0.26 * s), int(cy - 2), int(0.44 * s), int(0.34 * s), BRAIN)
    rl.draw_ellipse(int(cx + 0.26 * s), int(cy - 2), int(0.44 * s), int(0.34 * s), BRAIN)
    wrinkle = rl.Color(72, 42, 102, 220)
    for dx, yoff, w in ((-0.22, -0.12, 0.28), (0.22, -0.10, 0.26),
                        (-0.18, 0.08, 0.24), (0.20, 0.10, 0.22),
                        (-0.08, -0.02, 0.16), (0.10, 0.02, 0.16)):
        a = rl.Vector2(cx + (dx - w * 0.5) * s, cy + yoff * s)
        b = rl.Vector2(cx + (dx + w * 0.5) * s, cy + (yoff + 0.06) * s)
        rl.draw_line_bezier(a, b, 2.4, wrinkle)
    stem = rl.Rectangle(cx - 7 * scale, cy + 0.28 * s, 14 * scale, 16 * scale)
    rl.draw_rectangle_rounded(stem, 0.6, 4, BRAIN_DEEP)
    rl.draw_circle(int(cx - 0.18 * s), int(cy - 0.16 * s), int(6 * scale),
                   rl.Color(230, 210, 255, 90))


def draw_start_menu(width: int, height: int, model_label: str, model_detail: str,
                    idx: int, n: int, error: str = "", loading: bool = False):
    """Title card. Returns clickable rects: prev, next, start."""
    mx = rl.get_mouse_x()
    my = rl.get_mouse_y()
    draw_brain(width * 0.5, 132, 1.15)
    title = "Spirit Push"
    tw = text_width(title, 42)
    text(title, (width - tw) * 0.5, 210, 42, TEXT)
    sub = "drive the spirit with a learned world model"
    sw = text_width(sub, 18)
    text(sub, (width - sw) * 0.5, 262, 18, MUTED)

    card_w, card_h = 520, 78
    card = rl.Rectangle((width - card_w) * 0.5, 320, card_w, card_h)
    rl.draw_rectangle_rounded(card, 0.14, 8, CARD)
    text("WORLD MODEL", card.x + 24, card.y + 10, 14, MUTED)
    text(model_label[:42], card.x + 24, card.y + 32, 22, TEXT)
    text(f"{model_detail}    {idx + 1}/{max(n, 1)}",
         card.x + 24, card.y + 56, 14, MUTED)

    prev = rl.Rectangle(card.x - 54, card.y + 18, 42, 42)
    nxt = rl.Rectangle(card.x + card_w + 12, card.y + 18, 42, 42)
    for rec, glyph in ((prev, "<"), (nxt, ">")):
        hot = _hit(rec, mx, my)
        rl.draw_rectangle_rounded(rec, 0.2, 6, BTN_HOT if hot else CARD)
        gw = text_width(glyph, 28)
        text(glyph, rec.x + (rec.width - gw) * 0.5, rec.y + 6, 28, TEXT)

    start = rl.Rectangle((width - 240) * 0.5, 430, 240, 58)
    hot = _hit(start, mx, my)
    rl.draw_rectangle_rounded(start, 0.18, 8, BTN_HOT if hot or loading else BTN)
    label = "LOADING..." if loading else "START"
    lw = text_width(label, 26)
    text(label, start.x + (start.width - lw) * 0.5, start.y + 16, 26, TEXT)

    hint = "drop your .pt into  learning/world_models     Esc = quit"
    hw = text_width(hint, 16)
    text(hint, (width - hw) * 0.5, height - 48, 16, MUTED)
    if error:
        ew = text_width(error, 16)
        text(error, (width - ew) * 0.5, 508, 16, HAZARD)
    return prev, nxt, start
