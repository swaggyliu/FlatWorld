"""Rigid vs HeightField / Voxel: correctness and batched-kernel speedup.

PYTEST_DONT_REWRITE
"""

import os
import sys
import time

import numpy as np
import warp as wp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import (  # noqa: E402
    BallRigid,
    ExplicitLoop,
    Gravity,
    HeightFieldDomain,
    RigidBodyDomain,
    VoxelGridDomain,
)
from test_utils import init_sim  # noqa: E402


def _bowl_heights(nx=257):
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    h = np.empty(nx, dtype=np.float32)
    r, cx, zc = 0.5, 0.5, 0.5
    for i, x in enumerate(xs):
        inside = r * r - (x - cx) ** 2
        h[i] = zc - np.sqrt(inside) if inside >= 0.0 else zc
    return h


def _voxel_pit(nx=80, ny=50):
    occ = np.zeros((nx, ny), dtype=np.int32)
    cx = nx * 0.5
    r = nx * 0.45
    max_h = int(ny * 0.55)
    for i in range(nx):
        inside = r * r - (i - cx) ** 2
        if inside > 0:
            depth = inside / (r * r)
            hh = int(max_h * (1.3 - depth))
        else:
            hh = max_h
        if hh > 0:
            occ[i, :hh] = 1
    return VoxelGridDomain(d=2, nx=nx, ny=ny, lb=[0.0, 0.0], ub=[1.0, 1.0], occupancy_np=occ, considerContact=True)


def _ball_grid(n, y0=0.72, radius=0.025):
    domains = []
    side = int(np.ceil(np.sqrt(n)))
    k = 0
    for i in range(side):
        for j in range(side):
            if k >= n:
                break
            x = 0.35 + 0.3 * (i + 0.5) / max(side, 1)
            y = y0 + 0.06 * j
            rigid = BallRigid(2, [float(x), float(y)], radius, 1.0)
            domains.append(
                RigidBodyDomain(rigid, [Gravity([0.0, -9.8])], considerContact=True, restitution=0.0)
            )
            k += 1
    return domains


def _make_loop(ground, n_balls):
    domains = _ball_grid(n_balls) + [ground]
    return ExplicitLoop(0.0, domains, useAdapativeDT=True)


def _min_ball_y(loop, n_balls):
    return float(loop.rigidManager.rigidParams.numpy()[:n_balls, 0, 1].min())


def _time_detect(loop, iters):
    mgr = loop.rigidManager
    mgr._generate_ground_pairs_direct_kernel()
    for _ in range(3):
        mgr.reset_contact_caches_kernel()
        mgr.detectRigidGroundContact()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mgr.reset_contact_caches_kernel()
        mgr.detectRigidGroundContact()
    wp.synchronize()
    return time.perf_counter() - t0


def _time_sim(loop, frames, frame_dt=1.0 / 60.0):
    for _ in range(2):
        loop.advanceWithTime(frame_dt)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(frames):
        loop.advanceWithTime(frame_dt)
    wp.synchronize()
    return time.perf_counter() - t0


def test_rigid_balls_rest_on_heightfield():
    init_sim()
    hf = HeightFieldDomain(2, _bowl_heights(), lb=[0.0, 0.0], ub=[1.0, 1.0], considerContact=True)
    loop = _make_loop(hf, n_balls=4)
    for _ in range(90):
        loop.advanceWithTime(1.0 / 60.0)
    ys = loop.rigidManager.rigidParams.numpy()[:4, 0]
    assert float(ys[:, 1].min()) > 0.0, "balls fell through heightfield"
    assert float(ys[:, 1].max()) < 0.55, "balls did not settle onto heightfield"


def test_rigid_balls_rest_on_voxel():
    init_sim()
    loop = _make_loop(_voxel_pit(), n_balls=4)
    for _ in range(90):
        loop.advanceWithTime(1.0 / 60.0)
    assert _min_ball_y(loop, 4) > 0.02, "balls fell through voxel pit"


def test_batched_hf_voxel_kernel_runs(capsys):
    init_sim()
    n_balls = 64
    detect_iters = 40
    sim_frames = 20

    hf = _make_loop(
        HeightFieldDomain(2, _bowl_heights(), lb=[0.0, 0.0], ub=[1.0, 1.0], considerContact=True),
        n_balls,
    )
    vox = _make_loop(_voxel_pit(), n_balls)

    t_hf = _time_detect(hf, detect_iters)
    t_vx = _time_detect(vox, detect_iters)
    s_hf = _time_sim(hf, sim_frames)
    s_vx = _time_sim(vox, sim_frames)

    print("\n=== HeightField / Voxel batched kernel ===")
    print(f"scene: {n_balls} balls")
    print(f"HF  detect x{detect_iters}: {t_hf*1e3:.1f} ms")
    print(f"Vox detect x{detect_iters}: {t_vx*1e3:.1f} ms")
    print(f"HF  sim {sim_frames} frames: {s_hf:.3f} s")
    print(f"Vox sim {sim_frames} frames: {s_vx:.3f} s")
    print(f"HF min y={_min_ball_y(hf, n_balls):.3f}")
    print(f"Vox min y={_min_ball_y(vox, n_balls):.3f}")

    assert _min_ball_y(hf, n_balls) > 0.0
    assert _min_ball_y(vox, n_balls) > 0.02
