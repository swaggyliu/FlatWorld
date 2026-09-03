"""Rolling disks must dissipate energy on frictional ground.

Coulomb friction alone cannot stop a disk that is already rolling without
slip: the contact point is instantaneously at rest, so the tangent impulse
does no work. RigidManager.rolling_resistance adds a PGS torque row so an
unforced ball comes to rest.
"""
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import (
    BallRigid,
    ExplicitLoop,
    GroundDomain,
    Gravity,
    InitialAngVel,
    InitialVel,
    RigidBodyDomain,
)
from test_utils import init_sim


def _coast_ball(rolling_resistance, v0=0.6, end_time=3.0, mu=0.5,
                radius=0.08, mass=0.3, gravity=10.0):
    init_sim()
    ball = BallRigid(2, [0.5, radius], radius, mass)
    # Pure rolling to +x: ω = -v / R (clockwise).
    bcs = [Gravity([0.0, -gravity])]
    initials = [
        InitialVel([0], [v0, 0.0]),
        InitialAngVel([0], [-v0 / radius]),
    ]
    domain = RigidBodyDomain(ball, bcs, friction=mu, initials=initials)
    ground = GroundDomain(2, [0.0, 0.0], [0.0, 1.0])
    looper = ExplicitLoop(0.0, [ground, domain], useAdapativeDT=True)
    looper.rigidManager.rolling_resistance = float(rolling_resistance)
    looper.stableTime = 1.0 / 600.0

    x0 = float(domain.getCurrentRefPoint()[0])
    frame_dt = 1.0 / 60.0
    t = 0.0
    while t < end_time:
        looper.advanceWithTime(frame_dt)
        t += frame_dt
    pos = domain.getCurrentRefPoint()
    gid = domain.ndOffset
    v = looper.rigidManager.V.numpy()[gid]
    speed = float(np.hypot(v[0], v[1]))
    return float(pos[0] - x0), speed


def test_rolling_ball_coasts_without_rolling_resistance():
    dist, speed = _coast_ball(0.0)
    assert dist > 0.20, f"Coulomb-only disk should coast; travelled {dist:.3f} m"
    # Numerical dissipation may bleed a little speed, but it must not park.
    assert speed > 0.02 or dist > 0.25, (
        f"Coulomb-only disk stopped too soon: dist={dist:.3f} m speed={speed:.3f}"
    )


def test_rolling_ball_stops_with_rolling_resistance():
    dist, speed = _coast_ball(0.25)
    assert dist < 0.16, f"rolling resistance should stop the disk inside 16 cm, got {dist:.3f} m"
    assert speed < 0.02, f"disk should be at rest, leftover speed={speed:.3f} m/s"


def test_rolling_resistance_beats_coulomb_only():
    d0, _ = _coast_ball(0.0)
    d1, s1 = _coast_ball(0.25)
    assert d1 < 0.5 * d0, f"RR travel {d1:.3f} m should be << Coulomb-only {d0:.3f} m"
    assert s1 < 0.02


if __name__ == "__main__":
    for c in (0.0, 0.25):
        dist, speed = _coast_ball(c)
        print(f"C_rr={c:.2f}: distance={dist:.4f} m  final_speed={speed:.4f} m/s")
    test_rolling_ball_coasts_without_rolling_resistance()
    test_rolling_ball_stops_with_rolling_resistance()
    test_rolling_resistance_beats_coulomb_only()
    print("ok")
