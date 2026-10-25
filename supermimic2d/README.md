# SuperMimic-2D

A **2D recreation of [SuperMimic](https://supermimic.github.io/)** on [FlatWorld](../).

FlatWorld’s StateLeWM learns a differentiable dynamics model and plans through
imagined rollouts. SuperMimic uses the same idea for **motion tracking**: fit one
world model of shared rigid-body dynamics, then improve a goal-conditioned policy
by **backpropagation-through-time** through short imagined rollouts (instead of
model-free PPO in the simulator).

```
(1) Collect     πθ + noise  →  FlatWorld sim  →  replay B
(2) Fit WM      k-step windows from B  →  L_WM
(3) Improve π   H-step imagination through frozen fω  →  ∇θ Ĵ (BPTT + critic)
```

## Layout

```
supermimic2d/
├── character.py       # 10-body 2D humanoid, 8 PD joints
├── motion.py          # LAFAN-like clips + ground contact + baked loaders
├── bake_refs.py       # PD-track drafts in FlatWorld → physics-baked refs
├── models.py          # world model / residual policy / critic
├── buffer.py          # replay buffer
├── train.py           # SuperMimic training loop
├── render_video.py    # single-clip ref / policy / overlay MP4
├── render_library.py  # all clips → *_ref.mp4 / *_overlay.mp4
├── render_showcase.py # ~30s multi-skill reel
├── render_promo.py    # branded FlatWorld × SuperMimic promo
├── diagnose_wm.py     # WM vs FlatWorld rollout MSE
└── run_demo.py        # one-shot dance train + render
```

## Quick start

```bash
# from repo root
python -m supermimic2d.run_demo

# multi-skill (recommended)
python -m supermimic2d.bake_refs          # optional: physics-bake refs first
python -m supermimic2d.train --out supermimic2d/runs/lafan2d --library lafan2d --iters 80
python -m supermimic2d.render_promo --ckpt supermimic2d/runs/lafan2d/best.pt
```

Main deliverables under `runs/lafan2d/`:

| File | Description |
|------|-------------|
| `promo_supermimic.mp4` | Branded ~33s promo (title → skills → end card) |
| `showcase_30s.mp4` | Multi-skill overlay reel |
| `best.pt` | Best checkpoint |
| `baked_refs/*.npz` | Physics-baked reference motions |
| `*_ref.mp4` / `*_overlay.mp4` | Per-skill reference / policy+ghost |

## Character

Side-view humanoid: torso + head + 2 arms + 2 legs (8 actuated revolutes).
Actions are absolute joint-angle PD targets (`use_pd=2`), same pattern as
`learning`’s `ReacherEnv`. Reward is BeyondMimic-style pose / velocity / root
tracking, plus a mild root assist toward the reference.

## Reference motions (LAFAN-like 2D)

Clips (no BVH download):  
`walk` · `run` · `sprint` · `dance` · `jump` · `crouch` · `fight` · `fall_getup`

Real LAFAN1 is mocap + retarget. Here the pipeline is:

1. **Procedural draft** — analytic joint/root curves, temporally smoothed  
2. **Penetration-only lift** — raise `root_y` only when feet/support go below ground (never pull airborne poses down)  
3. **Physics bake** (optional but recommended) — PD-track the draft once in FlatWorld; save the simulated `(q, root)` as the training reference so contact/friction reduce hover and foot-skate  

```bash
# bake refs (no trained policy needed)
python -m supermimic2d.bake_refs --out supermimic2d/runs/lafan2d/baked_refs --root-kp 65

# train (loads baked_refs/*.npz when present)
python -m supermimic2d.train --out supermimic2d/runs/lafan2d --library lafan2d --iters 80

# resume
python -m supermimic2d.train --out supermimic2d/runs/lafan2d --library lafan2d --resume-best --iters 40
```

`make_lafan2d_library(use_baked=True)` (default) prefers baked npz; pass
`use_baked=False` for pure procedural drafts.

## Rendering

```bash
# every skill: kinematic/baked ref + optional policy overlay
python -m supermimic2d.render_library --out supermimic2d/runs/lafan2d \
  --overlay --ckpt supermimic2d/runs/lafan2d/best.pt

# website showcase (~30s)
python -m supermimic2d.render_showcase --ckpt supermimic2d/runs/lafan2d/best.pt \
  --out supermimic2d/runs/lafan2d/showcase_30s.mp4

# branded promo
python -m supermimic2d.render_promo --ckpt supermimic2d/runs/lafan2d/best.pt \
  --out supermimic2d/runs/lafan2d/promo_supermimic.mp4

# single clip
python -m supermimic2d.render_video --clip jump --reference --out .../jump_ref.mp4
python -m supermimic2d.render_video --clip jump --overlay --ckpt .../best.pt
```

Overlay ghosts use the **full planted clip pose** (share live torso *x* only) so
ghost feet match the ground-safe reference.

## Diagnostics

```bash
python -m supermimic2d.diagnose_wm --ckpt supermimic2d/runs/lafan2d/best.pt --all
```

Early tracking error often mixes cold-start from rest and world-model mismatch
while the buffer is still thin — the same failure mode SuperMimic avoids by
fitting the WM on logged transitions before relying on BPTT.
