"""Hand-authored layouts that stay inside the trained scene (1 EE + 3 boxes + 2 balls).

Levels 1–5 are the campaign. From 6 onward, `spec_for` builds a seeded random
layout in the same N=6 scene so retry is stable but skip/win never loops.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class LevelSpec:
    name: str
    hint: str
    # World x for bodies 0..5: EE, box, box, box, ball, ball.
    xs: tuple
    target: int
    hazard: int
    goal_x: float
    par: float
    push_cap: int = 3
    bounce: float = 0.15
    grip: float = 0.50
    scale: float = 1.0


# Indices: 0 EE, 1-3 boxes, 4-5 balls. Ground is flat; variety is spacing + roles.
LEVELS = (
    LevelSpec(
        name="Push",
        hint="Push the gold box onto the star. Red hurts — one touch and you retry.",
        xs=(0.28, 0.62, 1.50, 1.82, 1.18, 2.05),
        target=1, hazard=5, goal_x=1.05,
        par=14.0, push_cap=3,
        bounce=0.15, grip=0.50, scale=1.0,
    ),
    LevelSpec(
        name="Squeeze",
        hint="The red wall leaves a tiny gap. Shrink the spirit (size 0.5) and slip through.",
        xs=(0.26, 1.38, 0.56, 0.96, 1.88, 2.10),
        target=1, hazard=2, goal_x=1.72,
        par=18.0, push_cap=4,
        bounce=0.10, grip=0.55, scale=0.5,
    ),
    LevelSpec(
        name="Bank",
        hint="Crank bounce and knock the gold ball onto the star.",
        xs=(0.30, 1.64, 1.88, 1.38, 0.66, 2.08),
        target=4, hazard=5, goal_x=1.12,
        par=16.0, push_cap=4,
        bounce=0.85, grip=0.25, scale=1.0,
    ),
    LevelSpec(
        name="Stop",
        hint="More grip. Park the gold ball on the star without sliding past.",
        xs=(0.30, 1.62, 1.86, 2.10, 0.68, 1.38),
        target=4, hazard=1, goal_x=1.05,
        par=16.0, push_cap=3,
        bounce=0.05, grip=0.90, scale=1.1,
    ),
    LevelSpec(
        name="Gauntlet",
        hint="Red sits in the lane. Shift = careful. Don't touch it.",
        xs=(0.26, 1.18, 1.62, 1.90, 0.68, 2.10),
        target=1, hazard=4, goal_x=1.48,
        par=20.0, push_cap=4,
        bounce=0.12, grip=0.60, scale=0.85,
    ),
)

CAMPAIGN_LEN = len(LEVELS)

# Match SceneConfig: EE radius 0.10, box half-w 0.12, ball radius 0.08.
_HW = (0.10, 0.12, 0.12, 0.12, 0.08, 0.08)
_AREA_LO, _AREA_HI = 0.16, 2.10
_MIN_GAP = 0.06  # scene.min_gap; x-AABBs must not overlap (gap >= this)

_FLAVORS = (
    dict(
        tag="Push", kind="box",
        bounce=(0.10, 0.22), grip=(0.40, 0.60), scale=(0.90, 1.15),
        hint="Push gold onto the star. Red still hurts.",
    ),
    dict(
        tag="Bank", kind="ball",
        bounce=(0.70, 0.95), grip=(0.15, 0.35), scale=(0.90, 1.15),
        hint="Bounce the gold ball onto the star.",
    ),
    dict(
        tag="Stop", kind="ball",
        bounce=(0.02, 0.12), grip=(0.78, 0.95), scale=(1.00, 1.20),
        hint="Park the gold ball. Don't slide past the star.",
    ),
    dict(
        tag="Squeeze", kind="box",
        bounce=(0.08, 0.16), grip=(0.45, 0.65), scale=(0.50, 0.62),
        hint="Shrink and slip past red.",
    ),
    dict(
        tag="Lane", kind="any",
        bounce=(0.10, 0.20), grip=(0.50, 0.70), scale=(0.75, 1.00),
        hint="Red sits in the lane. Shift = careful.",
    ),
)


def _boxes():
    return (1, 2, 3)


def _balls():
    return (4, 5)


def half_width(i: int, ee_scale: float = 1.0) -> float:
    """AABB half-extent along x. EE grows with the size slider."""
    i = int(i)
    if i == 0:
        return _HW[0] * float(ee_scale)
    return _HW[i]


def min_x_gap(xs, ee_scale: float = 1.0) -> float:
    """Smallest AABB gap along x. Negative means the x-intervals overlap."""
    n = len(xs)
    worst = 1.0e9
    for i in range(n):
        hi = half_width(i, ee_scale)
        for j in range(i + 1, n):
            hj = half_width(j, ee_scale)
            gap = abs(float(xs[i]) - float(xs[j])) - (hi + hj)
            if gap < worst:
                worst = gap
    return float(worst)


def ensure_x_clearance(xs, ee_scale: float = 1.0, min_gap: float = _MIN_GAP) -> tuple:
    """Separate x-AABBs. No-op when already clear (campaign layouts stay put)."""
    xs_in = tuple(float(x) for x in xs)
    if min_x_gap(xs_in, ee_scale) >= min_gap - 1e-9:
        return xs_in
    xs = list(xs_in)
    n = len(xs)
    hw = [half_width(i, ee_scale) for i in range(n)]
    xs[0] = max(xs[0], _AREA_LO + hw[0])
    order = [0] + sorted(range(1, n), key=lambda i: (xs[i], i))
    for a, b in zip(order, order[1:]):
        need = xs[a] + hw[a] + min_gap + hw[b]
        if xs[b] < need:
            xs[b] = need
    last = order[-1]
    overflow = (xs[last] + hw[last]) - _AREA_HI
    if overflow > 0:
        xs[0] = max(_AREA_LO + hw[0], xs[0] - overflow)
        cursor = xs[0] + hw[0]
        for i in order[1:]:
            xs[i] = cursor + min_gap + hw[i]
            cursor = xs[i] + hw[i]
    return tuple(round(x, 4) for x in xs)


def _spread_xs(rng: random.Random, ee_scale: float = 1.0) -> tuple:
    """EE on the left; objects packed left-to-right with extra slack, never overlapping."""
    hw = [half_width(i, ee_scale) for i in range(6)]
    order = [1, 2, 3, 4, 5]
    rng.shuffle(order)
    packed = sum(_MIN_GAP + 2.0 * hw[b] for b in order)
    ee_lo = _AREA_LO + hw[0]
    ee_hi = min(0.30, _AREA_HI - packed - hw[0])
    if ee_hi < ee_lo:
        ee_x = ee_lo
    else:
        ee_x = rng.uniform(ee_lo, ee_hi)
    left = ee_x + hw[0]
    slack = max(0.0, _AREA_HI - left - packed)
    weights = [rng.random() + 0.05 for _ in range(6)]
    wsum = sum(weights)
    extras = [slack * w / wsum for w in weights]
    xs = [0.0] * 6
    xs[0] = ee_x
    cursor = left
    for k, bid in enumerate(order):
        cursor += _MIN_GAP + extras[k] + hw[bid]
        xs[bid] = cursor
        cursor += hw[bid]
    return ensure_x_clearance(xs, ee_scale)


def _pick_roles(rng: random.Random, xs: tuple, flavor: dict) -> tuple[int, int]:
    kind = flavor["kind"]
    if kind == "box":
        pool = list(_boxes())
    elif kind == "ball":
        pool = list(_balls())
    else:
        pool = [1, 2, 3, 4, 5]
    target = rng.choice(pool)
    others = [i for i in range(1, 6) if i != target]
    # Lane / Squeeze: prefer a blocker between the spirit and the gold.
    ee_x, tx = xs[0], xs[target]
    between = [i for i in others if min(ee_x, tx) < xs[i] < max(ee_x, tx)]
    if flavor["tag"] in ("Lane", "Squeeze") and between:
        if flavor["tag"] == "Squeeze":
            boxes_between = [i for i in between if i in _boxes()]
            hazard = rng.choice(boxes_between or between)
        else:
            hazard = rng.choice(between)
    else:
        hazard = rng.choice(others)
    return int(target), int(hazard)


def _goal_x(rng: random.Random, xs: tuple, target: int, hazard: int) -> float:
    tx = xs[target]
    lo, hi = 0.55, 1.95
    candidates = []
    x = lo
    while x <= hi:
        if abs(x - tx) < 0.32:
            x += 0.08
            continue
        if abs(x - xs[0]) < 0.26 or abs(x - xs[hazard]) < 0.22:
            x += 0.08
            continue
        if any(abs(x - xs[i]) < 0.16 for i in range(1, 6) if i not in (target, hazard)):
            x += 0.08
            continue
        candidates.append(x)
        x += 0.08
    if not candidates:
        side = 0.45 if tx < 1.15 else -0.45
        pick = min(hi, max(lo, tx + side))
    else:
        ahead = [c for c in candidates if c >= tx + 0.32]
        pick = rng.choice(ahead or candidates)
    gx = float(pick)
    hx = xs[hazard]

    def _score(x: float) -> float:
        return min(abs(x - tx), abs(x - hx), abs(x - xs[0]))

    if abs(gx - hx) < 0.22 or abs(gx - tx) < 0.30:
        options = [gx, hx + 0.30, hx - 0.30, tx + 0.40, tx - 0.40]
        options = [min(hi, max(lo, o)) for o in options]
        gx = max(options, key=_score)
    return round(gx, 3)


def _random_spec(level: int, seed: int) -> LevelSpec:
    rng = random.Random(int(seed) ^ (int(level) * 1_000_003))
    flavor = _FLAVORS[(int(level) - 1) % len(_FLAVORS)]
    bounce = rng.uniform(*flavor["bounce"])
    grip = rng.uniform(*flavor["grip"])
    scale = rng.uniform(*flavor["scale"])
    xs = _spread_xs(rng, ee_scale=scale)
    xs = ensure_x_clearance(xs, scale)
    if min_x_gap(xs, scale) < _MIN_GAP - 1e-6:
        xs = ensure_x_clearance(xs, scale, min_gap=_MIN_GAP)
    target, hazard = _pick_roles(rng, xs, flavor)
    gx = _goal_x(rng, xs, target, hazard)
    dist = abs(gx - xs[target])
    par = 14.0 + 10.0 * dist
    push_cap = 3 if dist < 0.55 else 4
    return LevelSpec(
        name=f"{flavor['tag']} {level}",
        hint=flavor["hint"],
        xs=xs,
        target=target,
        hazard=hazard,
        goal_x=gx,
        par=round(par, 1),
        push_cap=push_cap,
        bounce=round(bounce, 3),
        grip=round(grip, 3),
        scale=round(scale, 3),
    )


def spec_for(level: int, seed: int | None = None) -> LevelSpec:
    """1-based level. Campaign for 1..5; seeded mix after that."""
    n = max(1, int(level))
    if n <= CAMPAIGN_LEN:
        return LEVELS[n - 1]
    if seed is None:
        seed = 1_000_003 * n
    return _random_spec(n, int(seed))
