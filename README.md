# Flat World

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/swaggyliu/FlatWorld/actions/workflows/ci.yml/badge.svg)](https://github.com/swaggyliu/FlatWorld/actions/workflows/ci.yml)

**English** | [中文](README.zh.md)

Warp-accelerated **2D** physics engine with **explicit FEM**, **impulse-based rigid bodies**, and **mixed-domain contact** (analytical ground, height fields, voxel maps).

## Features

- **FEM / soft bodies** — explicit dynamics, linear elastic / Neo-Hookean / J2 plasticity
- **Rigid bodies** — ball, box, capsule, mesh; 2D SAT; PGS impulses
- **Ground types** — `GroundDomain` (plane), `HeightFieldDomain`, `VoxelGridDomain`; rigid and FEM contacts run as batched Warp kernels
- **Joints** — revolute, weld, prismatic, spherical
- **GPU batching** — unified FEM / spring manager + `MixedContact` penalty kernels



## Quick start

Once published to PyPI (Trusted Publisher / OIDC):

```bash
pip install flatworld
```

From source:

```bash
git clone https://github.com/swaggyliu/FlatWorld.git
cd FlatWorld
pip install -r requirements.txt
pip install -e .
```

```python
import numpy as np
from flatworld import Mesh, FemDomain, SolidProp, Elastic, Gravity, ExplicitLoop, GroundDomain
from flatworld.wp_init import ensure_warp

ensure_warp()

conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)
mesh = Mesh(2, conn, coords)
domain = FemDomain(mesh, SolidProp(Elastic(E=2e4, nu=0.2, rho=40.0)), bcs=[Gravity([0, -1.0])])
ground = GroundDomain(2, [0, 0.0], [0, 1])

looper = ExplicitLoop(0.0, [domain, ground], useAdapativeDT=True)
for _ in range(60):
    looper.advanceWithTime(1.0 / 60.0)
```

`.msh` / `.vtu` meshes go through `FEMesher` and need **meshio** (`pip install meshio`). The Spirit Push game needs **raylib**.

## Tests

```bash
# Headless (recommended for CI)
set HEADLESS=1          # Windows
export HEADLESS=1       # Linux / macOS
pytest test2D -q

# With GUI when a display is available
pytest test2D/test_2Dfem_elastic.py
```

51 pytest modules under `test2D/` cover FEM, rigid contact, joints, friction, and all ground types.

## Learning stacks (StateLeWM)

One **state + tactile latent world model** (~320K params), three controllers.
Full architecture / losses: **[learning/README.md](learning/README.md)**.
Paradigm comparison + roadmap: **[docs/PARADIGMS.md](docs/PARADIGMS.md)**.
Site: [GitHub Pages](https://swaggyliu.github.io/FlatWorld/)
(enable once: **Settings → Pages → Source = GitHub Actions**) ·
Play: `python -m game --mode pusht` (**You vs AI**) ·
Manuscript: arXiv forthcoming (**cs.RO**, no PDF in-repo).


| Paradigm      | Tasks                           | Result               | Weights                                              |
| ------------- | ------------------------------- | -------------------- | ---------------------------------------------------- |
| **WM + CEM**  | PushToGoal · Two-room · Reacher | 96–98% (50 ep)       | `learning/checkpoints/{push_game,two_room,reacher}`  |
| **WM + BPTT** | SuperMimic-2D skills            | multi-skill tracking | `supermimic2d/runs/lafan2d`                          |
| **WM + PPO**  | Push-T                          | **36/40 = 90%**      | `learning/checkpoints/push_t_wm_rsl` (+ WM `push_t`) |


<p align="center">
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#pusht"><strong>▶ Push-T / You vs AI</strong></a>
  ·
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#supermimic"><strong>SuperMimic promo</strong></a>
  ·
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#dance"><strong>dance overlay</strong></a>
</p>

<p align="center">
  <video src="https://swaggyliu.github.io/FlatWorld/media/pusht_promo.mp4" width="680" controls playsinline preload="metadata"></video>
</p>
<p align="center">
  <video src="https://swaggyliu.github.io/FlatWorld/media/promo_supermimic.mp4" width="680" controls playsinline preload="metadata"></video>
</p>

<p align="center">
  <img src="docs/statelewm_worldmodel.png" width="720" alt="StateLeWM one-step world model" />
</p>

<p align="center">
  <img src="docs/statelewm_inference.png" width="720" alt="StateLeWM inference loop" />
</p>

### 1. WM + CEM — Push · Two-room · Reacher

Goal-free interaction → StateLeWM → **online CEM** on latent rollouts.
No policy network. Same encoder/predictor/losses across scenes; only data and
typed readouts change.


| scene                                      | success (50 ep, seed 1000) | tol / vel         | weights                          |
| ------------------------------------------ | -------------------------- | ----------------- | -------------------------------- |
| PushToGoal (leftmost / rightmost / random) | 98 / 96 / 96 %             | 0.08 m / 0.08 m/s | `learning/checkpoints/push_game` |
| Two-room                                   | **49/50 = 98%**            | 0.10 m / 0.15 m/s | `learning/checkpoints/two_room`  |
| Reacher                                    | **48/50 = 96%**            | 0.06 m / 0.18 m/s | `learning/checkpoints/reacher`   |



| method                                         | Two-Room | Reacher |
| ---------------------------------------------- | -------- | ------- |
| **StateLeWM** (this repo — state + EE tactile) | **98**   | **96**  |
| LeWM                                           | 87       | 86      |
| Fast-LeWM                                      | 98       | 88      |
| DINO-WM                                        | 100      | 79      |
| PLDM                                           | 97       | 78      |


Success %. StateLeWM is measured here; LeWM-family rows are published numbers on
*their* Two-Room / Reacher (pixels, own protocols — LeWM: Maes et al., 2026;
Fast-LeWM: Gao & Xu, 2026; DINO-WM: Zhou et al., 2025; PLDM: Sobal et al., 2025).
Size anchor, not a controlled head-to-head.

### 2. WM + BPTT — SuperMimic-2D

A **2D recreation of [SuperMimic](https://supermimic.github.io/)**: fit a shared
rigid-body WM, then improve a goal-conditioned policy by **BPTT through short
imagined rollouts** (walk / run / sprint / dance / jump / crouch / fight /
fall-getup). Write-up: **[supermimic2d/README.md](supermimic2d/README.md)**.

<p align="center">
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#supermimic"><strong>▶ Watch the promo (~33s)</strong></a>
  ·
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#dance">dance overlay</a>
</p>
<p align="center">
  <video src="https://swaggyliu.github.io/FlatWorld/media/promo_supermimic.mp4" width="640" controls playsinline preload="metadata"></video>
  &nbsp;
  <video src="https://swaggyliu.github.io/FlatWorld/media/dance_overlay.mp4" width="640" controls playsinline preload="metadata"></video>
</p>

```bash
python -m supermimic2d.bake_refs
python -m supermimic2d.train --out supermimic2d/runs/lafan2d --library lafan2d --iters 80
python -m supermimic2d.render_promo --ckpt supermimic2d/runs/lafan2d/best.pt
```



### 3. WM + PPO — Push-T

Underactuated T-block alignment where CEM alone is weak. Freeze Push-T StateLeWM
(`push_t`), train **rsl-rl PPO** with policy obs = WM latents ‖ task features,
**rewards / dynamics on real FlatWorld**, vectorized envs. Sim eval **36/40 = 90%**.

<p align="center">
  <img src="docs/push_t_success.png" width="680" alt="Push-T WM-obs PPO success summary 90%" />
</p>

```bash
python -m learning.train_rsl_wm_push_t \
  --wm learning/checkpoints/push_t \
  --out learning/checkpoints/push_t_wm_rsl \
  --num-envs 128 --num-workers 6 --target-rate 0.9
python -m learning.eval_rl_push_t_report \
  --policy learning/checkpoints/push_t_wm_rsl/best.pt \
  --wm learning/checkpoints/push_t \
  --out learning/results/wm_rl --tag wm_rsl
```



## Spirit Push

A raylib prototype that uses the world model as a **spirit**: you click
what to push and where; CEM plans the force. Five campaign stages stay
inside the trained scene (1 circular EE + 3 boxes + 2 balls). After
Gauntlet, layouts **randomize** (seeded — retry is the same mix).

```bash
python -m game
# or: python -m game --checkpoint learning/checkpoints/push_game
# Push-T: You vs AI (WASD left, WM+PPO right)
python -m game --mode pusht
```

Gold is the target, red is a one-hit hazard, the star is the goal.
Sliders set bounce / grip / size.

<p align="center">
  <img src="docs/game/menu.png" width="640" alt="Spirit Push start menu" />
</p>
<p align="center">
  <img src="docs/game/push.png" width="260" alt="Level 1 Push" />
  <img src="docs/game/bank.png" width="260" alt="Level 3 Bank" />
  <img src="docs/game/gauntlet.png" width="260" alt="Level 5 Gauntlet" />
</p>


| shot     | stage                                                                |
| -------- | -------------------------------------------------------------------- |
| menu     | pick `best.pt` (or drop your own `.pt` into `learning/world_models`) |
| Push     | shove the gold box onto the star                                     |
| Bank     | high bounce, knock the gold ball                                     |
| Gauntlet | red sits in the lane — don't touch it                                |


How the planner talks to the engine: [learning/README.md](learning/README.md).
Regenerate shots: `python scripts/capture_game.py`.

## Gallery

Live-rendered screenshots from `test2D` demos:

<p align="center">
  <img src="docs/gallery/domino.png" width="180" alt="Domino chain" />
  <img src="docs/gallery/fem-rigid-mixed.png" width="180" alt="FEM + rigid" />
  <img src="docs/gallery/robot.png" width="180" alt="2D robot arm" />
  <img src="docs/gallery/rigid-25-boxes.png" width="180" alt="25 boxes" />
  <img src="docs/gallery/fem-heightfield.png" width="180" alt="FEM height field" />
  <img src="docs/gallery/pendulum.png" width="180" alt="Pendulum" />
</p>

**35 scenes** — full catalog: [docs/gallery/README.md](docs/gallery/README.md)

Regenerate locally: `python scripts/capture_gallery.py`

## Documentation

- [Learning world models](learning/README.md)
- [Control paradigms (CEM / BPTT / PPO)](docs/PARADIGMS.md)
- paradigms write-up in-repo: [docs/PARADIGMS.md](docs/PARADIGMS.md)
- [Parallel envs / env_id](docs/ENGINE_PARALLEL.md)
- [Citation](CITATION.cff)
- [SuperMimic-2D](supermimic2d/README.md)
- [Theory & implementation](docs/THEORY_AND_IMPLEMENTATION.md) · [中文](docs/THEORY_AND_IMPLEMENTATION.zh.md)
- [Contributing](CONTRIBUTING.md)



## Project layout

```
FlatWorld/
├── flatworld/          # Core engine
│   ├── explicitloop.py # Main simulation loop
│   ├── mixedcontact.py # Batched FEM / spring / ground contact
│   ├── femspringmanager.py
│   ├── rigidmanager.py # Rigid SAT / PGS / batched ground contact
│   └── sat.py          # 2D box–box SAT
├── learning/           # StateLeWM: CEM tasks + Push-T PPO
├── supermimic2d/       # SuperMimic-style 2D motion imitation (WM+BPTT)
├── game/               # raylib prototype (python -m game)
├── test2D/             # 2D examples & pytest suite
└── docs/               # gallery, diagrams, paradigms
```



## Third-party dependencies


| Package                                                                   | License    | Used for                       |
| ------------------------------------------------------------------------- | ---------- | ------------------------------ |
| [NVIDIA Warp](https://github.com/NVIDIA/warp)                             | Apache-2.0 | GPU/CPU kernels                |
| numpy                                                                     | BSD        | Arrays                         |
| meshio                                                                    | MIT        | `.msh` / `.vtu` via `FEMesher` |
| matplotlib                                                                | PSF        | Viewer / plots                 |
| [raylib](https://github.com/electronstudio/raylib-python-cffi) (optional) | zlib       | `python -m game`               |




## License

Copyright 2025-2026 Dongyu Liu

Licensed under the [Apache License, Version 2.0](LICENSE).
