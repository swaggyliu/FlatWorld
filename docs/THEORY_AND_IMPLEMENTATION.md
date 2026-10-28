# Flat World Physics Engine — Theory and Implementation

**English** | [中文](THEORY_AND_IMPLEMENTATION.zh.md)

> **Flat World** (FlatWorld) is a real-time multiphysics simulation platform built on [NVIDIA Warp](https://nvidia.github.io/warp/). It supports **rigid-body dynamics**, the **finite element method (FEM)**, **spring–mass systems**, and three static ground representations: **analytical plane (Ground)**, **height field (HeightField)**, and **voxel grid (Voxel)**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Overall Architecture](#2-overall-architecture)
3. [Finite Element Method (FEM)](#3-finite-element-method-fem)
4. [Rigid-Body Dynamics](#4-rigid-body-dynamics)
5. [Ground Types](#5-ground-types)
6. [Contact and Collision](#6-contact-and-collision)
7. [Time Integration and Adaptive Step Size](#7-time-integration-and-adaptive-step-size)
8. [test2D Test Cases](#8-test2d-test-cases)
9. [Quick-Start Examples](#9-quick-start-examples)

---

## 1. Project Overview

### 1.1 Tech Stack

| Component | Description |
|-----------|-------------|
| Language | Python 3 |
| Compute backend | NVIDIA Warp (GPU/CPU parallel kernels) |
| Dependencies | warp-lang, numpy, scipy, meshio, pytest |

### 1.2 Directory Structure

```
FlatWorld/
├── flatworld/                  # Core engine
│   ├── explicitloop.py         # Main simulation loop (ExplicitLoop)
│   ├── femspringmanager.py     # Unified FEM / spring data management and integration
│   ├── rigidmanager.py         # Rigid-body state, collision detection, PGS constraint solver
│   ├── femdomain.py            # FEM domain wrapper
│   ├── rigiddomain.py          # Rigid-body domain wrapper
│   ├── grounddomain.py         # Analytical plane ground (GroundDomain)
│   ├── heightfielddomain.py    # Height-field ground
│   ├── voxeldomain.py          # Voxel-grid ground
│   ├── femcontact.py           # Soft-body penalty contact
│   ├── mesh.py                 # Mesh, shape functions, B matrix
│   ├── contact_detection.py    # Point–edge, point–face contact detection
│   ├── sat.py / gjk.py         # 2D SAT / GJK-EPA collision
│   ├── bvh.py                  # Broad-phase collision detection
│   ├── joints.py               # Joint definitions
│   ├── bcs.py                  # Boundary conditions
│   └── materials/              # Constitutive models
└── test2D/                     # 2D tests and demos (51 files)
```

### 1.3 Domain Types (DomainType)

| Type | Constant | Typical class |
|------|----------|---------------|
| Analytical plane | `ANALYTICAL` | `GroundDomain` |
| Finite element | `FEM` | `FemDomain` |
| Rigid body | `RIGID` | `RigidBodyDomain` |
| Spring–mass | `SPRINGMASS` | `SpringMassDomain` |
| Height field | `HEIGHTFIELD` | `HeightFieldDomain` |
| Voxel map | `VOXELMAP` | `VoxelGridDomain` |

---

## 2. Overall Architecture

### 2.1 Simulation Pipeline

`ExplicitLoop` is the simulation entry point and coordinates all subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│                      ExplicitLoop                           │
│  advanceWithTime(frame_dt)                                  │
│    └─ Adaptive substeps by stableTime                       │
│         └─ advance()                                        │
│              1. BVH broad phase → candidate collision pairs │
│              2. Batched contact forces (FEM ↔ ground/rigid) │
│              3. FemSpringManager.substep() → FEM integration│
│              4. RigidManager.substep() → rigid + PGS        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Dual-Manager Design

| Manager | Domains | Contact method |
|---------|---------|----------------|
| `FemSpringManager` | FEM, SpringMass | **Penalty forces** (force level, added to `Fext`) |
| `RigidManager` | Rigid, Ground, HeightField, Voxel | **Impulses + PGS** (velocity-level constraints) |

### 2.3 Contact-Type Matrix

| Contact pair | Status | Method |
|--------------|--------|--------|
| FEM ↔ analytical plane | ✅ | Penalty contact |
| FEM ↔ height field | ✅ | Penalty contact |
| FEM ↔ voxel | ✅ | Penalty contact |
| FEM ↔ FEM | ✅ | Penalty contact |
| FEM ↔ rigid | ✅ | Penalty contact |
| Rigid ↔ analytical plane | ✅ | Impulse + PGS |
| Rigid ↔ height field | ✅ | Impulse + PGS |
| Rigid ↔ voxel | ✅ | Impulse + PGS |
| Rigid ↔ rigid | ✅ | SAT/GJK + PGS |

---

## 3. Finite Element Method (FEM)

### 3.1 Theoretical Background

The engine uses a hybrid of **explicit dynamics** and a **Total Lagrangian** formulation:

#### 3.1.1 Elements and Shape Functions

- **2D**: Linear triangular elements (3 nodes)
- **3D**: Linear tetrahedral elements (4 nodes)
- Shape functions (2D triangle, barycentric coordinates):

\[
N_1 = \psi,\quad N_2 = \eta,\quad N_3 = 1 - \psi - \eta
\]

- The Jacobian and B matrix are evaluated at the element centroid (\(\psi = \eta = 0\)) (constant strain / constant stress)

#### 3.1.2 Deformation Gradient

The reference-configuration Jacobian \(J_{\text{ref}}\) is stored at initialization. In the current configuration:

\[
J = \frac{\partial x}{\partial \xi},\quad F = J \cdot J_{\text{ref}}^{-1}
\]

#### 3.1.3 Strain and Stress

**Green–Lagrange strain** (Voigt form):

\[
\varepsilon = \tfrac{1}{2}(F^T F - I)
\]

**Constitutive relation** (PK2 stress → stored as Cauchy stress):

\[
\sigma_{\text{Cauchy}} = \frac{1}{J} F S F^T
\]

#### 3.1.4 Internal Force Assembly

\[
f_{\text{int}} = B^T \sigma \cdot V_{\text{elem}}
\]

where \(B\) is the strain–displacement matrix and \(V_{\text{elem}}\) is the element volume (area in 2D).

#### 3.1.5 Time Integration

Explicit central-difference style:

\[
a = \frac{F_{\text{ext}} - c \cdot v}{m},\quad v \leftarrow v + a \cdot \Delta t,\quad u \leftarrow u + v \cdot \Delta t
\]

Implementation: `femspringmanager.py` → `femStep()` kernel.

#### 3.1.6 Stable Time Step

\[
\Delta t_{\text{stable}} \approx 0.4 \cdot \frac{L_{\text{char}}}{\sqrt{E/\rho}}
\]

where \(L_{\text{char}}\) is the mesh minimum edge length, \(E\) is Young’s modulus, and \(\rho\) is density.

Implementation: `femdomain.py`.

### 3.2 Material Models

| Type | Enum | Description |
|------|------|-------------|
| Linear elastic | `MaterialType.ELASTIC` | Lamé parameters \(\lambda, \mu\), Hooke’s law |
| Neo-Hookean | `MaterialType.NEOHOOKEAN` | Compressible hyperelasticity |
| J2 von Mises | `MaterialType.MISES` | Radial-return plasticity, isotropic hardening |

**Linear elastic Lamé parameters**:

\[
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)},\quad \mu = \frac{E}{2(1+\nu)}
\]

**J2 plasticity**: Uses radial return mapping. When the von Mises equivalent stress exceeds the yield surface, the deviatoric stress is corrected and the plastic strain history is updated.

Implementation: `materials/materialfunctions.py`.

### 3.3 Implementation Notes

| Module | File | Responsibility |
|--------|------|----------------|
| Domain wrapper | `femdomain.py` | Mesh, material, BCs, stable step size |
| Unified manager | `femspringmanager.py` | Global node/element arrays, batched `femStep` |
| Mesh utilities | `mesh.py` | Shape functions, Jacobian, B matrix, boundary-edge extraction |
| Mesh generation | `femesher.py` | Circle meshes, Gmsh/VTU import |
| Boundary conditions | `bcs.py` | Gravity, forces, fixed, velocity/acceleration constraints |

**Boundary-edge extraction**: Traverse all element edges; edges that appear only once are boundary edges, used for contact detection.

---

## 4. Rigid-Body Dynamics

### 4.1 Theoretical Background

Rigid bodies use **semi-implicit Euler integration** + **impulse-based contact** + a **projected Gauss–Seidel (PGS)** constraint solver.

#### 4.1.1 State Variables (2D)

| Variable | Meaning |
|----------|---------|
| `U` | Reference-point position |
| `V` | Linear velocity |
| `quat` | Scalar rotation angle (2D about the z-axis) |
| `RotV` | Angular velocity |

#### 4.1.2 Moment of Inertia

| Shape | 2D moment of inertia |
|-------|----------------------|
| Ball | \(I = \tfrac{1}{2} m r^2\) |
| Box | \(I = \tfrac{1}{12} m (w^2 + h^2)\) |

#### 4.1.3 Contact Response

- **Normal**: Baumgarte ERP (\(\text{ERP} = 0.2\)) corrects penetration
- **Tangential**: Coulomb friction cone
- **Restitution**: Configurable restitution coefficient

Constraints are solved in velocity space; PGS iterations update impulses.

#### 4.1.4 Joint Constraints

Supported joint types (`JointType`):

| Joint | Constrained DOFs |
|-------|------------------|
| Revolute | Keeps 1 rotational DOF |
| Weld | Fully fixes relative pose |
| Prismatic | Keeps 1 translational DOF |
| Spherical | Keeps 3 rotational DOFs |

### 4.2 Rigid-Body Shapes

| Type | Class | Collision detection |
|------|-------|---------------------|
| Ball | `BallRigid` | Analytical SDF |
| Box | `BoxRigid` | 2D SAT |
| Capsule | `CapsuleRigid` | Segment–point / ball |
| Mesh | `MeshRigid` | Boundary edges + GJK-EPA |

Implementation: `rigid.py`, `rigiddomain.py`, `rigidmanager.py`.

### 4.3 Rigid Contact Pipeline

Each substep of `RigidManager.substep()` performs:

1. Velocity integration (including damping)
2. **Broad phase**: BVH or spatial hashing
3. **Narrow phase**: Dispatch by pair type (ball–ball, box–box, mesh–mesh, etc.)
4. Assemble contact rows + joint rows → PGS matrix
5. Iterative solve via `solve_pgs(iterations)`
6. Position integration + AABB update

### 4.4 Collision Filtering

Uses ODE-style category/collide bitmasks:

```
Pair (a, b) is valid ⟺ (collide_bits[a] & category_bits[b]) ≠ 0
                    and (collide_bits[b] & category_bits[a]) ≠ 0
```

Predefined categories: `COLLISION_CATEGORY_GROUND`, `COLLISION_CATEGORY_FEM`, `COLLISION_CATEGORY_ORDINARY_RIGID`, etc.

---

## 5. Ground Types

All three grounds are **static obstacles** with `category_bits = COLLISION_CATEGORY_GROUND`, but differ in geometric representation and contact detection.

### 5.1 GroundDomain — Analytical Plane

**File**: `grounddomain.py`  
**Type**: `DomainType.ANALYTICAL`

#### Theory

An infinite plane is defined by a reference point \(\mathbf{p}\) and a unit normal \(\mathbf{n}\):

\[
(\mathbf{x} - \mathbf{p}) \cdot \mathbf{n} = 0
\]

In 2D the tangent is \([ -n_y, n_x ]\). The plane can translate/rotate via BCs bound through `RigidManager`.

#### Contact Detection

- **Rigid**: Signed distance from point to plane → impulse applied at the contact point
- **FEM**: Boundary-node penetration check → penalty normal force

#### Usage Example

```python
from flatworld import GroundDomain

# Horizontal ground: y=0.1, normal upward
ground = GroundDomain(2, point=[0, 0.1], normal=[0, 1])
```

---

### 5.2 HeightFieldDomain — Height Field

**File**: `heightfielddomain.py`  
**Type**: `DomainType.HEIGHTFIELD`

#### Theory

- **2D**: Curve \(z = h(x)\), stored as a 1D array
- **3D**: Surface \(z = h(x, y)\), stored as a 2D array

Sampling uses **linear interpolation**; normals are computed from the height gradient:

\[
\mathbf{n} = \text{normalize}([-dh/dx,\ 1]) \quad \text{(2D)}
\]

#### Core Methods

| Method | Function |
|--------|----------|
| `sample_height_2d(x)` | Linearly interpolate height |
| `sample_normal_2d(x)` | Compute normal from \(dh/dx\) |
| `nearest_on_curve_2d(x, z)` | Nearest point, normal, and signed distance |

#### Typical Use Cases

Smooth terrain: bowl-shaped grooves, sine waves, slopes, etc.

#### Usage Example

```python
from flatworld import HeightFieldDomain
import numpy as np

xs = np.linspace(0, 1, 100)
heights = 0.2 + 0.1 * np.sin(2 * np.pi * xs)
hf = HeightFieldDomain(2, heights, lb=[0.0, 0.0], ub=[1.0, 1.0])
```

---

### 5.3 VoxelGridDomain — Voxel Grid

**File**: `voxeldomain.py`  
**Type**: `DomainType.VOXELMAP`

#### Theory

An axis-aligned **occupancy grid** with `occ[i,j] = 1` for solid cells. At initialization, **boundary edges** (interfaces between solid and empty cells) are precomputed.

#### Signed Distance

- **Interior** (inside an occupied cell): Distance to the nearest face (negative for penetration)
- **Exterior**: Traverse precomputed boundary edges to find the nearest edge and normal

#### Typical Use Cases

Concave static geometry: pits, steps, walls, complex obstacles.

#### Usage Example

```python
from flatworld import VoxelGridDomain
import numpy as np

occ = np.zeros((10, 10), dtype=np.int32)
occ[:3, :] = 1  # Fill bottom 3 rows
vox = VoxelGridDomain(2, nx=10, ny=10, lb=[0, 0], ub=[1, 1], occupancy_np=occ)
```

---

### 5.4 Comparison of the Three Ground Types

| Property | Ground | HeightField | Voxel |
|----------|--------|-------------|-------|
| Geometry | Infinite plane | Parametric curve/surface | Occupancy grid |
| Concave support | ❌ | Partial (single-valued function) | ✅ |
| Memory | Very low | O(nx) or O(nx·ny) | O(nx·ny) |
| FEM contact | Penalty force | Penalty force | Penalty force |
| Rigid contact | PGS impulse | PGS impulse | PGS impulse |
| Movable | ✅ (BC-driven) | ❌ Static | ❌ Static |

---

## 6. Contact and Collision

### 6.1 Soft-Body Penalty Contact (FEM / Spring)

#### Penalty Stiffness (LS-DYNA style)

\[
k = \text{slsfac} \cdot K \cdot \frac{A_{\text{seg}}^2}{V_{\text{elem}}}
\]

where the bulk modulus is \(K = E / [3(1-2\nu)]\).

2D triangle simplification: \(k = 2 \cdot \text{slsfac} \cdot K\).

#### Stable Contact Step Size

\[
\Delta t_{\text{contact}} = 0.9 \sqrt{\frac{m_{\text{node}}}{k}}
\]

#### Contact Force

When penetration depth \(d > 0\):

\[
F_n = k \cdot d \cdot \mathbf{n}
\]

Optional tangential damping: \(F_t = -\mu k v_n \Delta t\).

Implementation: `femcontact.py`, batched kernels in `explicitloop.py`.

### 6.2 Rigid Impulse Contact

Velocity-level constraints solved iteratively with PGS:

\[
J \lambda = b
\]

where \(J\) is the constraint Jacobian, \(\lambda\) is the impulse, and \(b\) includes Baumgarte correction terms.

### 6.3 Broad Phase / Narrow Phase

| Stage | Method | File |
|-------|--------|------|
| Broad phase | BVH (LBVH) | `bvh.py` |
| Broad phase (alt.) | Spatial hashing | `spatialmanager.py` |
| 2D convex | SAT | `sat.py` |
| 3D convex mesh | GJK-EPA | `gjk.py` |
| Point–edge | `pointToEdgeContact` | `contact_detection.py` |

---

## 7. Time Integration and Adaptive Step Size

### 7.1 Fixed-Frame-Rate Substepping

`ExplicitLoop.advanceWithTime(frame_dt)` splits each frame (default 1/60 s) into multiple substeps:

```python
while remaining > 0:
    sub_dt = min(stableTime, remaining)
    advance()          # single substep
    remaining -= sub_dt
```

### 7.2 Sources of the Stable Step Size

Global `stableTime` is the minimum across domains:

\[
\Delta t_{\text{global}} = \min(\Delta t_{\text{FEM}},\ \Delta t_{\text{contact}},\ \Delta t_{\text{rigid}})
\]

### 7.3 Optional Mass Scaling

When `mass_scaling_dt > 0`, LS-DYNA-style selective mass scaling is enabled so that small elements meet the target time step.

---

## 8. test2D Test Cases

Run all tests:

```bash
pytest test2D
```

### 8.1 FEM / Soft Body

| Test file | What it verifies |
|-----------|------------------|
| `test_2Dfem_elastic.py` | 2-triangle FEM block freefall, adaptive dt |
| `test_2Dfem_elastic10000.py` | 100×100 mesh (~20k elements) performance and stability |
| `test_2Dfem_bcs.py` | FEM boundary conditions |
| `test_2Dfem_force.py` | Applied force BCs |
| `test_2Dfem_ics.py` | Initial velocities / conditions |
| `test_2Dfem_contact.py` | FEM self-contact |
| `test_2Dfem2domain_elastic.py` | Two coexisting FEM domains |
| `test_2Dfemanalytic_contact.py` | FEM ↔ analytical plane contact |
| `test_2Dfem_heightfield_contact.py` | FEM disk falling into a bowl height field |
| `test_2Dfem_voxel_contact.py` | FEM disk falling into a concave voxel pit |
| `test_2dfemrigid_contact.py` | FEM contact with ball/box/mesh rigid bodies |
| `test_2Dfemcontact_types.py` | Contact types and penalty-stiffness unit tests |
| `test_2Dmaterial_functions.py` | Constitutive-model unit tests |

### 8.2 Rigid Body

| Test file | What it verifies |
|-----------|------------------|
| `test_2Drigid_contact.py` | Two balls colliding head-on |
| `test_2Drigid_contact100.py` | 25-ball grid + random gravity |
| `test_2Drigid_contact100_boxes.py` | 100-box contact stress test |
| `test_2Drigid_contact100_capsules.py` | 100-capsule contact |
| `test_2Drigid_contact100_mesh.py` | 100 mesh-rigid contact |
| `test_2Drigid_bcs.py` | Rigid BCs (fixed, velocity, acceleration, rotation) |
| `test_2Drigid_rot.py` | Rigid rotational dynamics |
| `test_2Drigid_rotation_benchmark.py` | Rotation BC benchmark |
| `test_2Drigidanalytic_contact.py` | Rigid ↔ analytical ground |
| `test_2Dbox_collision_response.py` | Box–box collision impulse response |
| `test_2Dbox_friction.py` | Coulomb friction (static/dynamic) |
| `test_2Dball_rolling_friction.py` | Disk rolling resistance (ground-contact dissipation) |
| `test_2Dmeshrigid.py` | Mesh rigid-body contact |
| `test_2Dcapsule_bbox.py` | Capsule AABB correctness |
| `test_2Ddomino.py` | Domino chain reaction |

### 8.3 Ground Types

| Test file | What it verifies |
|-----------|------------------|
| `test_2Danalyticaldomain.py` | Ground / HeightField / Voxel creation and ball landing |
| `test_2Danalytical_bcs.py` | Analytical ground + BCs |
| `test_2Danalytical_rotation_benchmark.py` | Rotating analytical plane |

### 8.4 Joints / Mechanisms

| Test file | What it verifies |
|-----------|------------------|
| `test_2Drevolute_joint.py` | Revolute joint |
| `test_2Djoint_bcs.py` | Joint + rotation BC |
| `test_2Djoint_bcs_pdvel.py` | PD velocity control |
| `test_2Djoint_bcs_pdforce.py` | PD torque control |
| `test_2Djoint_kernels_unit.py` | Joint kernel unit tests |
| `test_2Dpendulum_simple.py` | Multibody pendulum chain |
| `test_2Dlinkage.py` | Complex linkage mechanism |
| `test_2Drobot.py` | Multibody robot |
| `test_2Dsolar_system.py` | Nested revolute joints (Earth–Moon system) |
| `test_2D_wheel_rolling.py` | Wheel + friction |
| `test_2Dwheel_rolling_initials.py` | Wheel initial-condition variants |

### 8.5 Spring–Mass

| Test file | What it verifies |
|-----------|------------------|
| `test_2Dspring_bcs.py` | Spring-network BCs |
| `test_2Dspring_10000.py` | 100×100 spring lattice performance |

### 8.6 Collision / Geometry Unit Tests

| Test file | What it verifies |
|-----------|------------------|
| `test_2Dcontact_detection.py` | Point–plane, point–edge primitives |
| `test_2Dsat_collision.py` | 2D SAT |
| `test_2Dgjk_collision.py` | GJK placeholder test |
| `test_2Dsegment_segment.py` | Segment–segment nearest point |
| `test_2Dbvh.py` | BVH broad phase |

### 8.7 Representative Case Notes

#### Case A: FEM Freefall (`test_2Dfem_elastic.py`)

- Small square made of 2 triangles, gravity \(g = -1\)
- Verifies fall out of view after ~1 s (\(y < 0\))
- Demonstrates `ExplicitLoop(useAdapativeDT=True)` adaptive step size

#### Case B: FEM into Bowl Height Field (`test_2Dfem_heightfield_contact.py`)

- Disk FEM body generated with `FEMesher.createCircle`
- Height field is a semicircle bowl: \(h(x) = z_c - \sqrt{r^2 - (x-c_x)^2}\)
- Verifies nodes do not penetrate the bowl bottom (\(y_{\min} > 0.05\))

#### Case C: FEM into Voxel Pit (`test_2Dfem_voxel_contact.py`)

- Voxel occupancy forms a bowl-shaped pit
- FEM disk falls from above
- Verifies no penetration of the pit bottom (\(y_{\min} > 0.02\))

#### Case D: FEM–Rigid Multi-Type Contact (`test_2dfemrigid_contact.py`)

- Same simulation includes: FEM disk + BallRigid + BoxRigid + MeshRigid + Ground
- Demonstrates multiphysics coupling

#### Case E: Three Ground Types Compared (`test_2Danalyticaldomain.py`)

- Separately tests GroundDomain, HeightFieldDomain, VoxelGridDomain
- Ball freefall; verifies no ground penetration

---

## 9. Quick-Start Examples

### 9.1 FEM Freefall

```python
import numpy as np
from flatworld import Mesh, FemDomain, SolidProp, Elastic, Gravity, ExplicitLoop
from flatworld.wp_init import ensure_warp

ensure_warp()

conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)
mesh = Mesh(2, conn, coords)

mat = Elastic(E=2e4, nu=0.2, rho=40.0)
domain = FemDomain(mesh, SolidProp(mat), bcs=[Gravity([0, -1.0])])

looper = ExplicitLoop(0.0, [domain], useAdapativeDT=True)
for _ in range(60):
    looper.advanceWithTime(1.0 / 60.0)
```

### 9.2 Rigid Body + Analytical Ground

```python
from flatworld import BallRigid, RigidBodyDomain, GroundDomain, Gravity, ExplicitLoop

rigid = BallRigid(2, [0.5, 0.5], 0.02, 1.0)
domain = RigidBodyDomain(rigid, [Gravity([0, -9.8])])
ground = GroundDomain(2, [0, 0.1], [0, 1])

looper = ExplicitLoop(0.0, [domain, ground])
for _ in range(120):
    looper.advanceWithTime(1.0 / 60.0)
```

### 9.3 FEM + Height-Field Ground

```python
from flatworld import FEMesher, FemDomain, HeightFieldDomain, Elastic, SolidProp, Gravity, ExplicitLoop
import numpy as np

mesh = FEMesher(2).createCircle([0.5, 0.65], 0.1)
fem = FemDomain(mesh, SolidProp(Elastic(E=2e4, nu=0.2, rho=40.0)), [Gravity([0, -10])])

heights = np.ones(100, dtype=np.float32) * 0.2
hf = HeightFieldDomain(2, heights, lb=[0, 0], ub=[1, 1])

looper = ExplicitLoop(0.0, [hf, fem], useAdapativeDT=True)
for _ in range(60):
    looper.advanceWithTime(1.0 / 60.0)
```

---

## Appendix: Key Source Index

| Topic | File | Key symbols |
|-------|------|-------------|
| Main loop | `explicitloop.py` | `ExplicitLoop`, `advanceWithTime`, `advance` |
| FEM integration | `femspringmanager.py` | `femStep`, `FemSpringManager` |
| Rigid solver | `rigidmanager.py` | `substep`, `solve_pgs` |
| Shape functions / B matrix | `mesh.py` | `getJacobian2D`, `getBMatrix2D` |
| Constitutive | `materials/materialfunctions.py` | `getStress2D`, `misesReturnMap2D` |
| Penalty contact | `femcontact.py` | `ContactFlexAnalytical`, `_segment_penalty` |
| Analytical ground | `grounddomain.py` | `GroundDomain` |
| Height field | `heightfielddomain.py` | `sample_height_2d`, `nearest_on_curve_2d` |
| Voxel | `voxeldomain.py` | `signed_distance_to_edges_2d`, `build_edges` |
| Contact detection | `contact_detection.py` | `pointToEdgeContact` |

---

*This document is auto-generated from the current FlatWorld codebase and reflects functionality verified in test2D.*
