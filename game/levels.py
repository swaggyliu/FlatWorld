"""Hand-authored layouts that stay inside the trained scene (1 EE + 3 boxes + 2 balls)."""

from __future__ import annotations

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
        xs=(0.30, 1.70, 1.96, 1.42, 0.66, 2.08),
        target=4, hazard=5, goal_x=1.12,
        par=16.0, push_cap=4,
        bounce=0.85, grip=0.25, scale=1.0,
    ),
    LevelSpec(
        name="Stop",
        hint="More grip. Park the gold ball on the star without sliding past.",
        xs=(0.30, 1.62, 1.90, 2.10, 0.68, 1.38),
        target=4, hazard=1, goal_x=1.05,
        par=16.0, push_cap=3,
        bounce=0.05, grip=0.90, scale=1.1,
    ),
    LevelSpec(
        name="Gauntlet",
        hint="Red sits in the lane. Shift = careful. Don't touch it.",
        xs=(0.26, 1.18, 1.62, 1.94, 0.68, 2.10),
        target=1, hazard=4, goal_x=1.48,
        par=20.0, push_cap=4,
        bounce=0.12, grip=0.60, scale=0.85,
    ),
)


def spec_for(level: int) -> LevelSpec:
    """1-based level number, wraps after the last authored stage."""
    n = max(1, int(level))
    return LEVELS[(n - 1) % len(LEVELS)]
