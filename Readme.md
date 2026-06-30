# Flat World

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**English** | [中文](#flat-world-中文)

Taichi-accelerated 2D/3D physics engine with **explicit FEM**, **impulse-based rigid bodies**, and **batched mixed-domain contact** (analytical ground, height fields, voxel maps).

## Features

- **FEM / soft bodies** — explicit dynamics, linear elastic / Neo-Hookean / J2 plasticity
- **Rigid bodies** — ball, box, capsule, mesh; SAT / GJK; PGS constraints
- **Ground types** — `GroundDomain` (plane), `HeightFieldDomain`, `VoxelGridDomain`
- **Joints** — revolute, weld, prismatic, spherical
- **GPU batching** — unified FEM manager + `MixedContact` batched kernels

## Quick start

```bash
git clone https://github.com/swaggyliu/FlatWorld.git
cd FlatWorld
pip install -r requirements.txt
pip install -e .
```

```python
import taichi as ti
import numpy as np
from flatworld import Mesh, FemDomain, SolidProp, Elastic, Gravity, ExplicitLoop, GroundDomain

ti.init(arch=ti.cpu)

conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)
mesh = Mesh(2, conn, coords)
domain = FemDomain(mesh, SolidProp(Elastic(E=2e4, nu=0.2, rho=40.0)), bcs=[Gravity([0, -1.0])])
ground = GroundDomain(2, [0, 0.0], [0, 1])

looper = ExplicitLoop(0.0, [domain, ground], useAdapativeDT=True)
for _ in range(60):
    looper.advanceWithTime(1.0 / 60.0)
```

## Tests

```bash
# Headless (recommended for CI)
HEADLESS=1 pytest test2D -q

# With GUI when a display is available
pytest test2D/test_2Dfem_elastic.py
```

51 test files under `test2D/` cover FEM, rigid contact, joints, friction, and all ground types.

## Documentation

- [Theory & implementation (中文)](docs/THEORY_AND_IMPLEMENTATION.md)
- [Contributing](CONTRIBUTING.md)

## Project layout

```
FlatWorld/
├── flatworld/          # Core engine
│   ├── explicitloop.py # Main simulation loop
│   ├── mixedcontact.py # Batched FEM/rigid/ground contact
│   ├── femspringmanager.py
│   └── rigidmanager.py
├── test2D/             # 2D examples & pytest suite
└── docs/
```

## Third-party dependencies

| Package | License |
|---------|---------|
| [Taichi](https://github.com/taichi-dev/taichi) | Apache-2.0 |
| numpy, scipy | BSD |
| meshio | MIT |
| tetgen, pymeshlab, usd-core | See respective projects |

## License

Copyright 2025-2026 Dongyu Liu

Licensed under the [Apache License, Version 2.0](LICENSE).

---

# Flat World (中文)

基于 **Taichi** 的实时物理仿真引擎，支持刚体、有限元（FEM）、弹簧-质量系统，以及三种地面表示（解析平面、高度场、体素网格）。

## 功能概览

| 模块 | 说明 |
|------|------|
| 有限元 | 显式动力学，线性弹性 / Neo-Hookean / J2 塑性 |
| 刚体 | 球、盒、胶囊、网格；PGS 冲量求解 |
| 地面 | `GroundDomain`、`HeightFieldDomain`、`VoxelGridDomain` |
| 接触 | `mixedcontact.py` 批处理惩罚接触内核 |

## 安装与运行

```bash
pip install -r requirements.txt
pip install -e .
pytest test2D
```

无显示器环境（CI）请设置：

```bash
set HEADLESS=1          # Windows
export HEADLESS=1       # Linux / macOS
pytest test2D -q
```

## 文档

详见 [docs/THEORY_AND_IMPLEMENTATION.md](docs/THEORY_AND_IMPLEMENTATION.md)。

## 开源协议

本项目采用 [Apache License 2.0](LICENSE)。
