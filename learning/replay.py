"""Replay PushToGoal eval rollouts in the game renderer and dump MP4s.

Existing eval JSONs only store per-episode scores. This loads the matching
``task_trajectories_{tag}.npz`` when present, otherwise re-runs those
eval seeds with the listed checkpoint so you can watch the physics.

Usage (repo root):
    python -m learning.replay
    python -m learning.replay --eval learning/results/task_eval_9kft_H32_leftmost.json --mp4
    python -m learning.replay --eval learning/results/task_eval_9kft_H32_*.json --only-fail
    python -m learning.replay --eval ... --play
    python -m learning.replay --eval ... --indices 0,12,33 --play

Window keys: Space pause, Left/Right step, N/P episode, S save MP4, Esc quit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from learning.configs.default import Config

RESULTS = "learning/results"
DEFAULT_EVALS = (
    "learning/results/task_eval_9kft_H32_leftmost.json",
    "learning/results/task_eval_9kft_H32_rightmost.json",
)


@dataclass
class Episode:
    states: np.ndarray
    geom: np.ndarray
    types: np.ndarray
    goal: np.ndarray
    actions: np.ndarray
    masks: np.ndarray | None
    success: bool
    target_idx: int
    episode_idx: int
    final_dist: float
    settle_frame: int
    tag: str = ""


@dataclass
class Pack:
    tag: str
    eval_path: str
    traj_path: str
    summary: dict
    episodes: list[Episode] = field(default_factory=list)


def _as_list(arr) -> list[np.ndarray]:
    arr = np.asarray(arr)
    if arr.dtype == object:
        return [np.asarray(x) for x in arr.tolist()]
    return [np.asarray(arr[i]) for i in range(len(arr))]


def _default_types(n: int) -> np.ndarray:
    sc = Config().scene
    types = [0] + [1] * int(sc.num_boxes) + [2] * int(sc.num_balls)
    if len(types) != n:
        types = [0] + [1] * (n - 1)
    return np.asarray(types, dtype=np.int64)


def traj_path_for(eval_path: str, tag: str = "") -> str:
    d = os.path.dirname(os.path.abspath(eval_path)) or RESULTS
    name = os.path.basename(eval_path)
    if name.startswith("task_eval") and name.endswith(".json"):
        return os.path.join(d, "task_trajectories" + name[len("task_eval"):-5] + ".npz")
    suffix = f"_{tag}" if tag else ""
    return os.path.join(d, f"task_trajectories{suffix}.npz")


def load_eval_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_traj_npz(path: str, tag: str = "", summary: dict | None = None) -> list[Episode]:
    d = np.load(path, allow_pickle=True)
    states = _as_list(d["states"])
    geoms = _as_list(d["geom"]) if "geom" in d.files else [None] * len(states)
    acts = _as_list(d["actions"]) if "actions" in d.files else [np.zeros((0, 2), np.float32)] * len(states)
    masks = _as_list(d["masks"]) if "masks" in d.files else [None] * len(states)
    goals = np.asarray(d["goal"], dtype=np.float32)
    succ = np.asarray(d["success"]).astype(bool)
    tgt = np.asarray(d["target_idx"] if "target_idx" in d.files else np.ones(len(states)))
    if "types" in d.files:
        types = _as_list(d["types"])
    else:
        types = [_default_types(s.shape[1]) for s in states]
    idx = np.asarray(d["episode_idx"], dtype=int) if "episode_idx" in d.files else np.arange(len(states))
    dist = np.asarray(d["final_dist"], dtype=np.float32) if "final_dist" in d.files else np.full(len(states), np.nan)
    settle = np.asarray(d["settle_frame"], dtype=int) if "settle_frame" in d.files else np.zeros(len(states), dtype=int)
    out = []
    for i, st in enumerate(states):
        st = np.asarray(st, dtype=np.float32)
        g = np.asarray(geoms[i], dtype=np.float32) if geoms[i] is not None else None
        if g is None:
            raise ValueError(f"{path} episode {i} missing geom")
        fd = float(dist[i])
        if not np.isfinite(fd):
            ti = int(tgt[i])
            fd = float(np.linalg.norm(st[-1, ti, :2] - goals[i, :2]))
        out.append(Episode(
            states=st, geom=g, types=np.asarray(types[i], dtype=np.int64),
            goal=np.asarray(goals[i], dtype=np.float32),
            actions=np.asarray(acts[i], dtype=np.float32),
            masks=None if masks[i] is None else np.asarray(masks[i]),
            success=bool(succ[i]), target_idx=int(tgt[i]),
            episode_idx=int(idx[i]), final_dist=fd,
            settle_frame=int(settle[i]), tag=tag,
        ))
    return out


def save_traj_npz(path: str, episodes: list[Episode]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        states=np.array([e.states for e in episodes], dtype=object),
        masks=np.array([e.masks if e.masks is not None else np.zeros((len(e.states), 0))
                        for e in episodes], dtype=object),
        actions=np.array([e.actions for e in episodes], dtype=object),
        goal=np.stack([e.goal for e in episodes]).astype(np.float32),
        success=np.asarray([e.success for e in episodes]),
        target_idx=np.asarray([e.target_idx for e in episodes], dtype=np.int32),
        geom=np.stack([e.geom for e in episodes]).astype(np.float32),
        types=np.stack([e.types for e in episodes]).astype(np.int64),
        episode_idx=np.asarray([e.episode_idx for e in episodes], dtype=np.int32),
        final_dist=np.asarray([e.final_dist for e in episodes], dtype=np.float32),
        settle_frame=np.asarray([e.settle_frame for e in episodes], dtype=np.int32),
    )


def _task_from_summary(summary: dict, device: str):
    from learning.tasks.push_to_goal import PushToGoalTask, load_ensemble

    cfg = Config()
    if summary.get("planner") == "solver_cem":
        pk = dict(
            use_solver_cem=True,
            horizon=int(summary.get("horizon") or 16),
            population=int(summary.get("population") or 12),
            iterations=int(summary.get("iterations") or 2),
            seed=int(summary.get("seed") or 1000),
            exec_horizon=int(summary.get("exec_horizon") or 4),
        )
        return PushToGoalTask(
            cfg, None, None, device="cpu",
            budget=summary.get("budget"),
            stride=int(summary.get("stride") or 5),
            target_mode=summary.get("mode") or summary.get("target_mode") or "leftmost",
            planner_kwargs=pk)

    ckpt = summary.get("checkpoint") or "learning/checkpoints"
    model, norm, stride = load_ensemble(ckpt, device)
    cem = summary.get("cem") or {}
    pk = dict(horizon=int(cem.get("horizon") or 32),
              population=int(cem.get("population") or 96),
              iterations=int(cem.get("iterations") or 5),
              seed=int(summary.get("seed") or 1000))
    if cem.get("uncert_cost") is not None:
        pk["uncert_cost"] = float(cem["uncert_cost"])
    pk["tactile_mode"] = summary.get("tactile") or "gate"
    ab = summary.get("cost_ablate") or {}
    if ab.get("reach") is False:
        pk["reach_cost"] = 0.0
    return PushToGoalTask(
        cfg, model, norm, device=device,
        tol=summary.get("tol"), vel_tol=summary.get("vel_tol"),
        budget=summary.get("budget"), stride=stride,
        target_mode=summary.get("target_mode") or "leftmost",
        ee_scale=float(summary.get("ee_scale") or 1.0),
        planner_kwargs=pk)


def collect_episodes(summary: dict, indices: list[int],
                     device: str | None = None) -> list[Episode]:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(summary.get("seed") or 1000)
    tag = summary.get("tag") or ""
    task = _task_from_summary(summary, device)
    out = []
    n = len(indices)
    t0 = time.time()
    for k, i in enumerate(indices):
        rng = np.random.default_rng(seed + int(i))
        r = task.run_episode(rng, record=True)
        states, masks, actions = r["frames"]
        ep = Episode(
            states=np.asarray(states, dtype=np.float32),
            geom=task._geom().copy(),
            types=np.asarray(task.env.obj_types_np, dtype=np.int64),
            goal=np.asarray(r["goal"], dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            masks=np.asarray(masks),
            success=bool(r["success"]),
            target_idx=int(r["target_idx"]),
            episode_idx=int(i),
            final_dist=float(r["final_dist"]),
            settle_frame=int(r["settle_frame"]),
            tag=tag,
        )
        out.append(ep)
        mark = "OK" if ep.success else "FAIL"
        print(f"[collect {k + 1}/{n}] ep {i} {mark}  "
              f"d={ep.final_dist:.3f}  T={len(ep.states)}  "
              f"({time.time() - t0:.0f}s)")
    return out


def load_pack(eval_path: str, collect_missing: bool = True,
              indices: list[int] | None = None) -> Pack:
    data = load_eval_json(eval_path)
    summary = data.get("summary") or {}
    tag = summary.get("tag") or ""
    recs = data.get("episodes") or []
    tpath = traj_path_for(eval_path, tag)
    wanted = set(indices) if indices is not None else None
    episodes: list[Episode] = []
    have: set[int] = set()
    if os.path.isfile(tpath):
        episodes = load_traj_npz(tpath, tag=tag, summary=summary)
        have = {e.episode_idx for e in episodes}
        print(f"loaded {len(episodes)} trajectories from {tpath}")
        if recs:
            by_json = {int(r.get("episode_idx", i)): r for i, r in enumerate(recs)}
            for e in episodes:
                meta = by_json.get(e.episode_idx)
                if meta is None:
                    continue
                e.success = bool(meta.get("success", e.success))
                e.final_dist = float(meta.get("final_dist", e.final_dist))
                e.settle_frame = int(meta.get("settle_frame", e.settle_frame))
                e.target_idx = int(meta.get("target_idx", e.target_idx))
    need = []
    if recs:
        for i, r in enumerate(recs):
            idx = int(r.get("episode_idx", i))
            if wanted is not None and idx not in wanted:
                continue
            if idx not in have:
                need.append(idx)
    elif wanted:
        need = [i for i in sorted(wanted) if i not in have]
    if need:
        if not collect_missing:
            raise FileNotFoundError(
                f"{tpath} missing episodes {need[:8]}{'...' if len(need) > 8 else ''}")
        print(f"collecting {len(need)} missing episode(s) from "
              f"{summary.get('checkpoint')}")
        extra = collect_episodes(summary, need)
        episodes.extend(extra)
        episodes.sort(key=lambda e: e.episode_idx)
        if indices is None and recs and len(episodes) >= len(recs):
            save_traj_npz(tpath, episodes)
            print(f"wrote {tpath}")
    if wanted is not None:
        episodes = [e for e in episodes if e.episode_idx in wanted]
    return Pack(tag=tag, eval_path=eval_path, traj_path=tpath,
                summary=summary, episodes=episodes)


def _physics_notes(states: np.ndarray, types: np.ndarray, geom: np.ndarray) -> list[str]:
    notes = []
    xy = states[:, :2]
    vel = np.linalg.norm(states[:, 3:5], axis=1)
    if float(vel.max()) > 3.5:
        notes.append(f"fast {vel.max():.1f}m/s")
    for i, st in enumerate(states):
        hh = float(geom[i, 1])
        if float(st[1]) - hh < -0.04:
            notes.append(f"obj{i} through floor")
    n = len(states)
    for i in range(n):
        ri = float(max(geom[i, 0], geom[i, 1]))
        for j in range(i + 1, n):
            rj = float(max(geom[j, 0], geom[j, 1]))
            d = float(np.linalg.norm(xy[i] - xy[j]))
            pen = ri + rj - d
            if pen > 0.03:
                notes.append(f"overlap {i}/{j} {pen:.3f}m")
                if len(notes) >= 3:
                    return notes
    return notes[:3]


def _action_at(ep: Episode, t: int) -> np.ndarray | None:
    a = ep.actions
    if a is None or len(a) == 0:
        return None
    if t <= 0:
        return a[0]
    return a[min(t - 1, len(a) - 1)]


# ---------------------------------------------------------------------------
# raylib view / encode
# ---------------------------------------------------------------------------

def _parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _image_to_rgb(rl, img) -> np.ndarray:
    w, h = int(img.width), int(img.height)
    fmt = int(img.format)
    rgba = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    rgb = getattr(rl, "PIXELFORMAT_UNCOMPRESSED_R8G8B8", None)
    if rgb is not None and fmt == rgb:
        nbytes, ch = w * h * 3, 3
    else:
        if fmt != rgba:
            rl.image_format(img, rgba)
        nbytes, ch = w * h * 4, 4
    buf = rl.ffi.buffer(img.data, nbytes)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, ch)
    return np.ascontiguousarray(arr[:, :, :3])


class ReplayView:
    def __init__(self, width: int, height: int, hidden: bool = False, fps: int = 0):
        import pyray as rl
        from game.camera import Camera
        from game.render import load_ui_font

        self.rl = rl
        flags = rl.FLAG_MSAA_4X_HINT
        if hidden and hasattr(rl, "FLAG_WINDOW_HIDDEN"):
            flags |= rl.FLAG_WINDOW_HIDDEN
        rl.set_config_flags(flags)
        if hasattr(rl, "set_trace_log_level"):
            rl.set_trace_log_level(rl.LOG_WARNING)
        rl.init_window(width, height, "FlatWorld replay")
        if fps > 0:
            rl.set_target_fps(fps)
        load_ui_font()
        self.cam = Camera(width, height)
        self.width = width
        self.height = height
        self.rt = rl.load_render_texture(width, height)

    def close(self):
        from game.render import unload_ui_font
        rl = self.rl
        try:
            rl.unload_render_texture(self.rt)
        except Exception:
            pass
        unload_ui_font()
        if rl.is_window_ready():
            rl.close_window()

    def draw(self, ep: Episode, t: int, *, hud: bool = True,
             vel_ticks: bool = True, extra: str = "") -> None:
        rl = self.rl
        from game.render import (
            BG, GOAL, HAZARD, MUTED, PANEL, TARGET, TEXT, draw_scene, text,
        )
        t = int(np.clip(t, 0, len(ep.states) - 1))
        st = ep.states[t]
        trail = [tuple(p) for p in ep.states[max(0, t - 40):t + 1, 0, :2]]
        force = _action_at(ep, t)
        rl.begin_texture_mode(self.rt)
        rl.clear_background(BG)
        draw_scene(
            self.cam, st, ep.types, ep.geom, ep.target_idx, -1, ep.goal,
            trail, force=force)
        if vel_ticks:
            _draw_vel(rl, self.cam, st)
        if hud:
            _draw_hud(rl, self.height, ep, t, extra=extra,
                      notes=_physics_notes(st, ep.types, ep.geom),
                      PANEL=PANEL, TEXT=TEXT, MUTED=MUTED,
                      TARGET=TARGET, HAZARD=HAZARD, text=text)
        rl.end_texture_mode()
        src = rl.Rectangle(0, 0, float(self.width), -float(self.height))
        rl.begin_drawing()
        rl.clear_background(BG)
        rl.draw_texture_rec(self.rt.texture, src, rl.Vector2(0, 0), rl.WHITE)
        rl.end_drawing()

    def grab(self) -> np.ndarray:
        rl = self.rl
        img = rl.load_image_from_texture(self.rt.texture)
        rl.image_flip_vertical(img)
        rgb = _image_to_rgb(rl, img)
        rl.unload_image(img)
        return rgb


def _draw_vel(rl, cam, states, scale: float = 0.12):
    col = rl.Color(250, 250, 250, 150)
    for st in states:
        vx, vy = float(st[3]), float(st[4])
        if vx * vx + vy * vy < 4e-4:
            continue
        a = cam.to_screen(float(st[0]), float(st[1]))
        b = cam.to_screen(float(st[0]) + vx * scale, float(st[1]) + vy * scale)
        rl.draw_line_ex(a, b, 2.0, col)


def _draw_hud(rl, height, ep: Episode, t: int, extra: str, notes: list[str],
              PANEL, TEXT, MUTED, TARGET, HAZARD, text):
    mark = "OK" if ep.success else "FAIL"
    col = TARGET if ep.success else HAZARD
    panel = rl.Rectangle(16, 12, 560, 92 if not notes else 114)
    rl.draw_rectangle_rounded(panel, 0.12, 6, PANEL)
    tag = ep.tag or "eval"
    text(f"{tag}   ep {ep.episode_idx}   {mark}", 28, 20, 18, col)
    tgt = ep.states[t, ep.target_idx]
    dist = float(np.linalg.norm(tgt[:2] - ep.goal[:2]))
    spd = float(np.linalg.norm(tgt[3:5]))
    text(f"t {t}/{len(ep.states) - 1}   d={dist:.3f}   "
         f"v={spd:.2f}   settle {ep.settle_frame}", 28, 46, 16, TEXT)
    text(f"target {ep.target_idx}   final d={ep.final_dist:.3f}   "
         f"goal ({ep.goal[0]:.2f},{ep.goal[1]:.2f})", 28, 68, 14, MUTED)
    if notes:
        text("  |  ".join(notes), 28, 90, 14, HAZARD)
    if extra:
        text(extra, 28, height - 36, 14, MUTED)


def _hold_tail(ep: Episode, n: int) -> np.ndarray:
    idx = list(range(len(ep.states)))
    if n > 0 and idx:
        idx.extend([idx[-1]] * n)
    return np.asarray(idx, dtype=int)


def _open_writer(path: str, fps: int):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    import imageio.v2 as imageio
    return imageio.get_writer(
        path, fps=int(fps), format="FFMPEG", mode="I",
        codec="libx264", pixelformat="yuv420p", quality=8,
        macro_block_size=1)


def render_mp4(view: ReplayView, ep: Episode, path: str, *,
               fps: int = 30, skip: int = 2, hold: int = 20,
               vel_ticks: bool = True) -> str:
    frames = _hold_tail(ep, hold)
    if skip > 1:
        frames = np.concatenate([frames[::skip], frames[-1:]])
    writer = _open_writer(path, fps)
    try:
        for t in frames:
            if view.rl.window_should_close():
                break
            view.draw(ep, int(t), vel_ticks=vel_ticks)
            writer.append_data(view.grab())
    finally:
        writer.close()
    return path


def play_window(packs: list[Pack], *, size: str, vel_ticks: bool,
                skip: int, hold: int, fps: int) -> None:
    eps = [e for p in packs for e in p.episodes]
    if not eps:
        print("nothing to play")
        return
    w, h = _parse_size(size)
    view = ReplayView(w, h, hidden=False, fps=max(int(fps), 1))
    rl = view.rl
    k = 0
    t = 0
    paused = False
    extra = "Space pause   arrows step   N/P episode   S save mp4   Esc quit"
    try:
        while not rl.window_should_close():
            ep = eps[k]
            T = len(ep.states)
            if rl.is_key_pressed(rl.KEY_SPACE):
                paused = not paused
            if rl.is_key_pressed(rl.KEY_RIGHT) or rl.is_key_pressed(rl.KEY_PERIOD):
                paused = True
                t = min(t + max(skip, 1), T - 1)
            if rl.is_key_pressed(rl.KEY_LEFT) or rl.is_key_pressed(rl.KEY_COMMA):
                paused = True
                t = max(t - max(skip, 1), 0)
            if rl.is_key_pressed(rl.KEY_N):
                k = (k + 1) % len(eps)
                t = 0
                paused = False
            if rl.is_key_pressed(rl.KEY_P):
                k = (k - 1) % len(eps)
                t = 0
                paused = False
            if rl.is_key_pressed(rl.KEY_S):
                mark = "OK" if ep.success else "FAIL"
                out = os.path.join(
                    RESULTS, "replays", ep.tag or "eval",
                    f"ep{ep.episode_idx:03d}_{mark}.mp4")
                print(f"saving {out}")
                render_mp4(view, ep, out, fps=fps, skip=skip, hold=hold,
                           vel_ticks=vel_ticks)
                print(f"wrote {out}")
            if rl.is_key_pressed(rl.KEY_ESCAPE):
                break
            if not paused:
                t += max(skip, 1)
                if t >= T:
                    t = T - 1
                    paused = True
            view.draw(ep, t, vel_ticks=vel_ticks,
                      extra=("PAUSED  " if paused else "") + extra)
    finally:
        view.close()


def dump_mp4s(packs: list[Pack], *, out_dir: str, size: str, fps: int,
              skip: int, hold: int, vel_ticks: bool) -> list[str]:
    eps = [e for p in packs for e in p.episodes]
    if not eps:
        print("nothing to render")
        return []
    w, h = _parse_size(size)
    view = ReplayView(w, h, hidden=True, fps=0)
    written = []
    t0 = time.time()
    try:
        for i, ep in enumerate(eps):
            mark = "OK" if ep.success else "FAIL"
            sub = os.path.join(out_dir, ep.tag or "eval")
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, f"ep{ep.episode_idx:03d}_{mark}.mp4")
            render_mp4(view, ep, path, fps=fps, skip=skip, hold=hold,
                       vel_ticks=vel_ticks)
            written.append(path)
            print(f"[mp4 {i + 1}/{len(eps)}] {path}  "
                  f"T={len(ep.states)}  ({time.time() - t0:.0f}s)")
    finally:
        view.close()
    man = os.path.join(out_dir, "index.json")
    rows = []
    if os.path.isfile(man):
        try:
            rows = json.load(open(man, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            rows = []
    keep_tags = {e.tag for e in eps}
    rows = [r for r in rows if r.get("tag") not in keep_tags]
    for e in eps:
        mark = "OK" if e.success else "FAIL"
        rows.append({
            "tag": e.tag, "episode_idx": e.episode_idx, "success": e.success,
            "final_dist": e.final_dist, "settle_frame": e.settle_frame,
            "n_frames": int(len(e.states)),
            "mp4": os.path.join(e.tag or "eval", f"ep{e.episode_idx:03d}_{mark}.mp4"),
        })
    with open(man, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {len(written)} mp4s + {man}")
    return written


def dump_eval_fails(eval_path: str, out_dir: str = None) -> list[str]:
    """Render FAIL episodes from a just-written eval JSON + trajectory npz."""
    pack = load_pack(eval_path, collect_missing=False)
    fails = [e for e in pack.episodes if not e.success]
    pack.episodes = fails
    if not fails:
        print(f"{eval_path}: no failures to replay")
        return []
    out_dir = out_dir or os.path.join(RESULTS, "replays")
    print(f"{eval_path}: {len(fails)} FAIL replay(s)")
    return dump_mp4s([pack], out_dir=out_dir, size="1280x720", fps=30,
                     skip=2, hold=20, vel_ticks=True)


def _expand_evals(patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            out.extend(hits)
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"skip missing {p}", file=sys.stderr)
    seen, uniq = set(), []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def _parse_indices(s: str) -> list[int] | None:
    if not s.strip():
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser(description="Replay eval episodes to a window or MP4s")
    p.add_argument("--eval", nargs="+", default=list(DEFAULT_EVALS),
                   help="eval JSON path(s); globs ok. default: current 9kft evals")
    p.add_argument("--out", type=str, default=os.path.join(RESULTS, "replays"),
                   help="directory for batch MP4s")
    p.add_argument("--mp4", action="store_true", default=False,
                   help="write one MP4 per episode (default if --play is off)")
    p.add_argument("--play", action="store_true",
                   help="interactive raylib window")
    p.add_argument("--only-fail", action="store_true")
    p.add_argument("--only-ok", action="store_true")
    p.add_argument("--indices", type=str, default="",
                   help="comma-separated episode_idx filter")
    p.add_argument("--no-collect", action="store_true",
                   help="do not re-run missing trajectories")
    p.add_argument("--size", type=str, default="1280x720")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--skip", type=int, default=2,
                   help="keep every Nth sim frame (2 ≈ realtime at 30fps)")
    p.add_argument("--hold", type=int, default=20,
                   help="repeat last frame this many sim steps")
    p.add_argument("--no-vel", action="store_true",
                   help="hide velocity ticks")
    return p


def main():
    args = build_parser().parse_args()
    paths = _expand_evals(args.eval)
    if not paths:
        print("no eval JSON files", file=sys.stderr)
        sys.exit(2)
    idx = _parse_indices(args.indices)
    packs = []
    for path in paths:
        pack = load_pack(path, collect_missing=not args.no_collect, indices=idx)
        eps = pack.episodes
        if args.only_fail:
            eps = [e for e in eps if not e.success]
        if args.only_ok:
            eps = [e for e in eps if e.success]
        pack.episodes = eps
        print(f"{path}: {len(eps)} episode(s) "
              f"({sum(e.success for e in eps)} ok / {sum(not e.success for e in eps)} fail)")
        packs.append(pack)
    do_mp4 = args.mp4 or not args.play
    if do_mp4:
        dump_mp4s(packs, out_dir=args.out, size=args.size, fps=args.fps,
                  skip=args.skip, hold=args.hold, vel_ticks=not args.no_vel)
    if args.play:
        play_window(packs, size=args.size, vel_ticks=not args.no_vel,
                    skip=args.skip, hold=args.hold, fps=args.fps)


if __name__ == "__main__":
    main()
