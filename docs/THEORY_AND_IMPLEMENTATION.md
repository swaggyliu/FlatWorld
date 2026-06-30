# Flat World 物理引擎 — 理论与实现文档

> **Flat World**（FlatWorld）是基于 [Taichi](https://www.taichi-lang.org/) 的实时多物理场仿真平台，支持 **刚体动力学**、**有限元法（FEM）**、**弹簧-质量系统**，以及三种静态地面表示：**解析平面（Ground）**、**高度场（HeightField）**、**体素网格（Voxel）**。

---

## 目录

1. [项目概览](#1-项目概览)
2. [整体架构](#2-整体架构)
3. [有限元法（FEM）](#3-有限元法fem)
4. [刚体动力学](#4-刚体动力学)
5. [地面类型](#5-地面类型)
6. [接触与碰撞](#6-接触与碰撞)
7. [时间积分与自适应步长](#7-时间积分与自适应步长)
8. [test2D 测试案例](#8-test2d-测试案例)
9. [快速入门示例](#9-快速入门示例)

---

## 1. 项目概览

### 1.1 技术栈

| 组件 | 说明 |
|------|------|
| 语言 | Python 3 |
| 计算后端 | Taichi（GPU/CPU 并行内核） |
| 依赖 | numpy, scipy, meshio, tetgen, pymeshlab, usd-core, pytest |

### 1.2 目录结构

```
FlatWorld/
├── flatworld/                  # 核心引擎
│   ├── explicitloop.py         # 主仿真循环（ExplicitLoop）
│   ├── femspringmanager.py     # FEM / 弹簧统一数据管理与积分
│   ├── rigidmanager.py         # 刚体状态、碰撞检测、PGS 约束求解
│   ├── femdomain.py            # FEM 域封装
│   ├── rigiddomain.py          # 刚体域封装
│   ├── grounddomain.py         # 解析平面地面（GroundDomain）
│   ├── heightfielddomain.py    # 高度场地面
│   ├── voxeldomain.py          # 体素网格地面
│   ├── femcontact.py           # 软体惩罚接触
│   ├── mesh.py                 # 网格、形函数、B 矩阵
│   ├── contact_detection.py    # 点-边、点-面接触检测
│   ├── sat.py / gjk.py         # 2D SAT / GJK-EPA 碰撞
│   ├── bvh.py                  # 宽相碰撞检测
│   ├── joints.py               # 关节定义
│   ├── bcs.py                  # 边界条件
│   └── materials/              # 本构模型
└── test2D/                     # 2D 测试与演示（51 个文件）
```

### 1.3 域类型（DomainType）

| 类型 | 常量 | 典型类 |
|------|------|--------|
| 解析平面 | `ANALYTICAL` | `GroundDomain` |
| 有限元 | `FEM` | `FemDomain` |
| 刚体 | `RIGID` | `RigidBodyDomain` |
| 弹簧-质量 | `SPRINGMASS` | `SpringMassDomain` |
| 高度场 | `HEIGHTFIELD` | `HeightFieldDomain` |
| 体素地图 | `VOXELMAP` | `VoxelGridDomain` |

---

## 2. 整体架构

### 2.1 仿真流程

`ExplicitLoop` 是仿真入口，负责协调各子系统：

```
┌─────────────────────────────────────────────────────────────┐
│                      ExplicitLoop                           │
│  advanceWithTime(frame_dt)                                  │
│    └─ 按 stableTime 自适应子步进                             │
│         └─ advance()                                        │
│              1. BVH 宽相 → 候选碰撞对                        │
│              2. 批处理接触力（FEM ↔ 地面/刚体）              │
│              3. FemSpringManager.substep() → FEM 积分        │
│              4. RigidManager.substep() → 刚体积分 + PGS      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 双管理器设计

| 管理器 | 负责域 | 接触方式 |
|--------|--------|----------|
| `FemSpringManager` | FEM、SpringMass | **惩罚力**（力级，加入 `Fext`） |
| `RigidManager` | Rigid、Ground、HeightField、Voxel | **冲量 + PGS**（速度级约束） |

### 2.3 接触类型矩阵

| 接触对 | 状态 | 方法 |
|--------|------|------|
| FEM ↔ 解析平面 | ✅ | 惩罚接触 |
| FEM ↔ 高度场 | ✅ | 惩罚接触 |
| FEM ↔ 体素 | ✅ | 惩罚接触 |
| FEM ↔ FEM | ✅ | 惩罚接触 |
| FEM ↔ 刚体 | ✅ | 惩罚接触 |
| 刚体 ↔ 解析平面 | ✅ | 冲量 + PGS |
| 刚体 ↔ 高度场 | ✅ | 冲量 + PGS |
| 刚体 ↔ 体素 | ✅ | 冲量 + PGS |
| 刚体 ↔ 刚体 | ✅ | SAT/GJK + PGS |

---

## 3. 有限元法（FEM）

### 3.1 理论基础

本引擎采用 **显式动力学** 与 **Total Lagrangian** 混合方案：

#### 3.1.1 单元与形函数

- **2D**：线性三角形单元（3 节点）
- **3D**：线性四面体单元（4 节点）
- 形函数（2D 三角形，重心坐标）：

\[
N_1 = \psi,\quad N_2 = \eta,\quad N_3 = 1 - \psi - \eta
\]

- 在单元形心（\(\psi = \eta = 0\)）处计算 Jacobian 与 B 矩阵（常应变/常应力）

#### 3.1.2 变形梯度

参考构形 Jacobian \(J_{\text{ref}}\) 在初始化时存储，当前构形：

\[
J = \frac{\partial x}{\partial \xi},\quad F = J \cdot J_{\text{ref}}^{-1}
\]

#### 3.1.3 应变与应力

**Green-Lagrange 应变**（Voigt 形式）：

\[
\varepsilon = \tfrac{1}{2}(F^T F - I)
\]

**本构关系**（PK2 应力 → Cauchy 应力存储）：

\[
\sigma_{\text{Cauchy}} = \frac{1}{J} F S F^T
\]

#### 3.1.4 内力组装

\[
f_{\text{int}} = B^T \sigma \cdot V_{\text{elem}}
\]

其中 \(B\) 为应变-位移矩阵，\(V_{\text{elem}}\) 为单元体积（2D 为面积）。

#### 3.1.5 时间积分

显式中心差分风格：

\[
a = \frac{F_{\text{ext}} - c \cdot v}{m},\quad v \leftarrow v + a \cdot \Delta t,\quad u \leftarrow u + v \cdot \Delta t
\]

实现位置：`femspringmanager.py` → `femStep()` 内核。

#### 3.1.6 稳定时间步

\[
\Delta t_{\text{stable}} \approx 0.4 \cdot \frac{L_{\text{char}}}{\sqrt{E/\rho}}
\]

其中 \(L_{\text{char}}\) 为网格最小边长，\(E\) 为杨氏模量，\(\rho\) 为密度。

实现位置：`femdomain.py`。

### 3.2 材料模型

| 类型 | 枚举 | 说明 |
|------|------|------|
| 线性弹性 | `MaterialType.ELASTIC` | Lamé 参数 \(\lambda, \mu\)，Hooke 定律 |
| Neo-Hookean | `MaterialType.NEOHOOKEAN` | 可压缩超弹性 |
| J2 von Mises | `MaterialType.MISES` | 径向回归塑性，各向同性硬化 |

**线性弹性 Lamé 参数**：

\[
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)},\quad \mu = \frac{E}{2(1+\nu)}
\]

**J2 塑性**：采用径向回归映射（radial return mapping），当 von Mises 等效应力超过屈服面时修正偏应力并更新塑性应变历史。

实现位置：`materials/materialfunctions.py`。

### 3.3 实现要点

| 模块 | 文件 | 职责 |
|------|------|------|
| 域封装 | `femdomain.py` | 网格、材料、BC、稳定步长 |
| 统一管理 | `femspringmanager.py` | 全局节点/单元数组，批处理 `femStep` |
| 网格工具 | `mesh.py` | 形函数、Jacobian、B 矩阵、边界边提取 |
| 网格生成 | `femesher.py` | 圆形网格、Gmsh/VTU 导入 |
| 边界条件 | `bcs.py` | 重力、力、固定、速度/加速度约束 |

**边界边提取**：遍历所有单元边，仅出现一次的边为边界边，用于接触检测。

---

## 4. 刚体动力学

### 4.1 理论基础

刚体采用 **半隐式 Euler 积分** + **冲量法接触** + **投影 Gauss-Seidel（PGS）** 约束求解。

#### 4.1.1 状态变量（2D）

| 变量 | 含义 |
|------|------|
| `U` | 参考点位置 |
| `V` | 线速度 |
| `quat` | 标量转角（2D 绕 z 轴） |
| `RotV` | 角速度 |

#### 4.1.2 转动惯量

| 形状 | 2D 转动惯量 |
|------|-------------|
| 球（Ball） | \(I = \tfrac{1}{2} m r^2\) |
| 矩形（Box） | \(I = \tfrac{1}{12} m (w^2 + h^2)\) |

#### 4.1.3 接触响应

- **法向**：Baumgarte ERP（\( \text{ERP} = 0.2 \)）修正穿透
- **切向**：Coulomb 摩擦锥
- **恢复系数**：可配置 restitution

约束在速度空间求解，PGS 迭代更新冲量。

#### 4.1.4 关节约束

支持的关节类型（`JointType`）：

| 关节 | 约束自由度 |
|------|-----------|
| Revolute（转动） | 保留 1 个转动 DOF |
| Weld（焊接） | 完全固定相对位姿 |
| Prismatic（滑动） | 保留 1 个平移 DOF |
| Spherical（球铰） | 保留 3 个转动 DOF |

### 4.2 刚体形状

| 类型 | 类 | 碰撞检测 |
|------|-----|----------|
| 球 | `BallRigid` | 解析 SDF |
| 矩形 | `BoxRigid` | 2D SAT |
| 胶囊 | `CapsuleRigid` | 线段-点/球 |
| 网格 | `MeshRigid` | 边界边 + GJK-EPA |

实现位置：`rigid.py`、`rigiddomain.py`、`rigidmanager.py`。

### 4.3 刚体接触流水线

`RigidManager.substep()` 每子步执行：

1. 速度积分（含阻尼）
2. **宽相**：BVH 或空间哈希
3. **窄相**：按对类型分发（ball-ball、box-box、mesh-mesh 等）
4. 组装接触行 + 关节行 → PGS 矩阵
5. `solve_pgs(iterations)` 迭代求解
6. 位置积分 + AABB 更新

### 4.4 碰撞过滤

采用 ODE 风格的 category/collide 位掩码：

```
配对 (a, b) 有效 ⟺ (collide_bits[a] & category_bits[b]) ≠ 0
              且 (collide_bits[b] & category_bits[a]) ≠ 0
```

预定义类别：`COLLISION_CATEGORY_GROUND`、`COLLISION_CATEGORY_FEM`、`COLLISION_CATEGORY_ORDINARY_RIGID` 等。

---

## 5. 地面类型

三种地面均为 **静态障碍物**，`category_bits = COLLISION_CATEGORY_GROUND`，但几何表示与接触检测方式不同。

### 5.1 GroundDomain — 解析平面

**文件**：`grounddomain.py`  
**类型**：`DomainType.ANALYTICAL`

#### 理论

无限平面由参考点 \(\mathbf{p}\) 和单位法向 \(\mathbf{n}\) 定义：

\[
(\mathbf{x} - \mathbf{p}) \cdot \mathbf{n} = 0
\]

2D 中切向量为 \([ -n_y, n_x ]\)。平面可通过 `RigidManager` 绑定 BC 实现平移/旋转。

#### 接触检测

- **刚体**：点到平面有符号距离 → 冲量施加于接触点
- **FEM**：边界节点穿透检测 → 惩罚法向力

#### 使用示例

```python
from flatworld import GroundDomain

# 水平地面：y=0.1，法向朝上
ground = GroundDomain(2, point=[0, 0.1], normal=[0, 1])
```

---

### 5.2 HeightFieldDomain — 高度场

**文件**：`heightfielddomain.py`  
**类型**：`DomainType.HEIGHTFIELD`

#### 理论

- **2D**：曲线 \(z = h(x)\)，存储为 1D 数组
- **3D**：曲面 \(z = h(x, y)\)，存储为 2D 数组

采样采用 **线性插值**，法向由高度梯度计算：

\[
\mathbf{n} = \text{normalize}([-dh/dx,\ 1]) \quad \text{(2D)}
\]

#### 核心方法

| 方法 | 功能 |
|------|------|
| `sample_height_2d(x)` | 线性插值采样高度 |
| `sample_normal_2d(x)` | 由 \(dh/dx\) 计算法向 |
| `nearest_on_curve_2d(x, z)` | 最近点、法向、有符号距离 |

#### 适用场景

平滑地形：碗形凹槽、正弦波、斜坡等。

#### 使用示例

```python
from flatworld import HeightFieldDomain
import numpy as np

xs = np.linspace(0, 1, 100)
heights = 0.2 + 0.1 * np.sin(2 * np.pi * xs)
hf = HeightFieldDomain(2, heights, lb=[0.0, 0.0], ub=[1.0, 1.0])
```

---

### 5.3 VoxelGridDomain — 体素网格

**文件**：`voxeldomain.py`  
**类型**：`DomainType.VOXELMAP`

#### 理论

轴对齐 **占据网格** `occ[i,j] = 1` 表示固体单元。初始化时预计算 **边界边**（固体与空单元交界处）。

#### 有符号距离

- **内部**（占据单元内）：到最近面的距离（负值表示穿透）
- **外部**：遍历预计算边界边，求最近边及法向

#### 适用场景

凹形静态几何：坑洞、台阶、墙体、复杂障碍。

#### 使用示例

```python
from flatworld import VoxelGridDomain
import numpy as np

occ = np.zeros((10, 10), dtype=np.int32)
occ[:3, :] = 1  # 底部 3 行填充
vox = VoxelGridDomain(2, nx=10, ny=10, lb=[0, 0], ub=[1, 1], occupancy_np=occ)
```

---

### 5.4 三种地面对比

| 特性 | Ground | HeightField | Voxel |
|------|--------|-------------|-------|
| 几何 | 无限平面 | 参数曲线/曲面 | 占据网格 |
| 凹形支持 | ❌ | 部分（单值函数） | ✅ |
| 内存 | 极低 | O(nx) 或 O(nx·ny) | O(nx·ny) |
| FEM 接触 | 惩罚力 | 惩罚力 | 惩罚力 |
| 刚体接触 | PGS 冲量 | PGS 冲量 | PGS 冲量 |
| 可运动 | ✅（BC 驱动） | ❌ 静态 | ❌ 静态 |

---

## 6. 接触与碰撞

### 6.1 软体惩罚接触（FEM / Spring）

#### 惩罚刚度（LS-DYNA 风格）

\[
k = \text{slsfac} \cdot K \cdot \frac{A_{\text{seg}}^2}{V_{\text{elem}}}
\]

其中体积模量 \(K = E / [3(1-2\nu)]\)。

2D 三角形简化：\(k = 2 \cdot \text{slsfac} \cdot K\)。

#### 稳定接触步长

\[
\Delta t_{\text{contact}} = 0.9 \sqrt{\frac{m_{\text{node}}}{k}}
\]

#### 接触力

穿透深度 \(d > 0\) 时：

\[
F_n = k \cdot d \cdot \mathbf{n}
\]

可选切向阻尼：\(F_t = -\mu k v_n \Delta t\)。

实现位置：`femcontact.py`、`explicitloop.py` 批处理内核。

### 6.2 刚体冲量接触

速度级约束，PGS 迭代求解：

\[
J \lambda = b
\]

其中 \(J\) 为约束 Jacobian，\(\lambda\) 为冲量，\(b\) 含 Baumgarte 修正项。

### 6.3 宽相 / 窄相

| 阶段 | 方法 | 文件 |
|------|------|------|
| 宽相 | BVH（LBVH） | `bvh.py` |
| 宽相（备选） | 空间哈希 | `spatialmanager.py` |
| 2D 凸体 | SAT | `sat.py` |
| 3D 凸网格 | GJK-EPA | `gjk.py` |
| 点-边 | `pointToEdgeContact` | `contact_detection.py` |

---

## 7. 时间积分与自适应步长

### 7.1 固定帧率子步进

`ExplicitLoop.advanceWithTime(frame_dt)` 将每帧（默认 1/60 s）拆分为多个子步：

```python
while remaining > 0:
    sub_dt = min(stableTime, remaining)
    advance()          # 单个子步
    remaining -= sub_dt
```

### 7.2 稳定步长来源

全局 `stableTime` 取各域最小值：

\[
\Delta t_{\text{global}} = \min(\Delta t_{\text{FEM}},\ \Delta t_{\text{contact}},\ \Delta t_{\text{rigid}})
\]

### 7.3 可选质量缩放

`mass_scaling_dt > 0` 时启用 LS-DYNA 风格选择性质量缩放，使小单元满足目标时间步。

---

## 8. test2D 测试案例

运行全部测试：

```bash
pytest test2D
```

### 8.1 FEM / 软体

| 测试文件 | 验证内容 |
|----------|----------|
| `test_2Dfem_elastic.py` | 2 三角形 FEM 块自由落体，自适应 dt |
| `test_2Dfem_elastic10000.py` | 100×100 网格（~2 万元）性能与稳定性 |
| `test_2Dfem_bcs.py` | FEM 边界条件 |
| `test_2Dfem_force.py` | 施加力 BC |
| `test_2Dfem_ics.py` | 初始速度/条件 |
| `test_2Dfem_contact.py` | FEM 自接触 |
| `test_2Dfem2domain_elastic.py` | 双 FEM 域共存 |
| `test_2Dfemanalytic_contact.py` | FEM ↔ 解析平面接触 |
| `test_2Dfem_heightfield_contact.py` | FEM 圆盘落入碗形高度场 |
| `test_2Dfem_voxel_contact.py` | FEM 圆盘落入凹形体素坑 |
| `test_2dfemrigid_contact.py` | FEM 与球/盒/网格刚体接触 |
| `test_2Dfemcontact_types.py` | 接触类型与惩罚刚度单元测试 |
| `test_2Dmaterial_functions.py` | 本构模型单元测试 |

### 8.2 刚体

| 测试文件 | 验证内容 |
|----------|----------|
| `test_2Drigid_contact.py` | 两球相向碰撞 |
| `test_2Drigid_contact100.py` | 25 球网格 + 随机重力 |
| `test_2Drigid_contact100_boxes.py` | 100 盒体接触压力测试 |
| `test_2Drigid_contact100_capsules.py` | 100 胶囊接触 |
| `test_2Drigid_contact100_mesh.py` | 100 网格刚体接触 |
| `test_2Drigid_bcs.py` | 刚体 BC（固定、速度、加速度、旋转） |
| `test_2Drigid_rot.py` | 刚体旋转动力学 |
| `test_2Drigid_rotation_benchmark.py` | 旋转 BC 基准 |
| `test_2Drigidanalytic_contact.py` | 刚体 ↔ 解析地面 |
| `test_2Dbox_collision_response.py` | 盒-盒碰撞冲量响应 |
| `test_2Dbox_friction.py` | Coulomb 摩擦（静/动摩擦） |
| `test_2Dmeshrigid.py` | 网格刚体接触 |
| `test_2Dcapsule_bbox.py` | 胶囊 AABB 正确性 |
| `test_2Ddomino.py` | 多米诺骨牌连锁 |

### 8.3 地面类型

| 测试文件 | 验证内容 |
|----------|----------|
| `test_2Danalyticaldomain.py` | Ground / HeightField / Voxel 创建与球体落地 |
| `test_2Danalytical_bcs.py` | 解析地面 + BC |
| `test_2Danalytical_rotation_benchmark.py` | 旋转解析平面 |

### 8.4 关节 / 机构

| 测试文件 | 验证内容 |
|----------|----------|
| `test_2Drevolute_joint.py` | 转动关节 |
| `test_2Djoint_bcs.py` | 关节 + 旋转 BC |
| `test_2Djoint_bcs_pdvel.py` | PD 速度控制 |
| `test_2Djoint_bcs_pdforce.py` | PD 力矩控制 |
| `test_2Djoint_kernels_unit.py` | 关节内核单元测试 |
| `test_2Dpendulum_simple.py` | 多体摆链 |
| `test_2Dlinkage.py` | 复杂连杆机构 |
| `test_2Drobot.py` | 多体机器人 |
| `test_2Dsolar_system.py` | 嵌套转动关节（地月系统） |
| `test_2D_wheel_rolling.py` | 滚轮 + 摩擦 |
| `test_2Dwheel_rolling_initials.py` | 滚轮初始条件变体 |

### 8.5 弹簧-质量

| 测试文件 | 验证内容 |
|----------|----------|
| `test_2Dspring_bcs.py` | 弹簧网络 BC |
| `test_2Dspring_10000.py` | 100×100 弹簧格点性能 |

### 8.6 碰撞 / 几何单元测试

| 测试文件 | 验证内容 |
|----------|----------|
| `test_2Dcontact_detection.py` | 点-平面、点-边原语 |
| `test_2Dsat_collision.py` | 2D SAT |
| `test_2Dgjk_collision.py` | GJK 占位测试 |
| `test_2Dsegment_segment.py` | 线段-线段最近点 |
| `test_2Dbvh.py` | BVH 宽相 |

### 8.7 代表性案例说明

#### 案例 A：FEM 自由落体（`test_2Dfem_elastic.py`）

- 2 个三角形组成的小方块，重力 \(g = -1\)
- 验证约 1 s 后落出视野（\(y < 0\)）
- 演示 `ExplicitLoop(useAdapativeDT=True)` 自适应步长

#### 案例 B：FEM 落入碗形高度场（`test_2Dfem_heightfield_contact.py`）

- 用 `FEMesher.createCircle` 生成圆盘 FEM 体
- 高度场为半圆碗：\(h(x) = z_c - \sqrt{r^2 - (x-c_x)^2}\)
- 验证节点未穿透碗底（\(y_{\min} > 0.05\)）

#### 案例 C：FEM 落入体素坑（`test_2Dfem_voxel_contact.py`）

- 体素占据场形成碗状凹坑
- FEM 圆盘从上方落下
- 验证未穿透坑底（\(y_{\min} > 0.02\)）

#### 案例 D：FEM-刚体多类型接触（`test_2dfemrigid_contact.py`）

- 同一仿真中包含：FEM 圆盘 + BallRigid + BoxRigid + MeshRigid + Ground
- 演示多物理场耦合

#### 案例 E：三种地面对比（`test_2Danalyticaldomain.py`）

- 分别测试 GroundDomain、HeightFieldDomain、VoxelGridDomain
- 球体自由落体，验证未穿透地面

---

## 9. 快速入门示例

### 9.1 FEM 自由落体

```python
import taichi as ti
import numpy as np
from flatworld import Mesh, FemDomain, SolidProp, Elastic, Gravity, ExplicitLoop

ti.init(arch=ti.gpu)

conn = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
coords = np.array([[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [0.7, 0.7]], dtype=np.float32)
mesh = Mesh(2, conn, coords)

mat = Elastic(E=2e4, nu=0.2, rho=40.0)
domain = FemDomain(mesh, SolidProp(mat), bcs=[Gravity([0, -1.0])])

looper = ExplicitLoop(0.0, [domain], useAdapativeDT=True)
for _ in range(60):
    looper.advanceWithTime(1.0 / 60.0)
```

### 9.2 刚体 + 解析地面

```python
from flatworld import BallRigid, RigidBodyDomain, GroundDomain, Gravity, ExplicitLoop

rigid = BallRigid(2, [0.5, 0.5], 0.02, 1.0)
domain = RigidBodyDomain(rigid, [Gravity([0, -9.8])])
ground = GroundDomain(2, [0, 0.1], [0, 1])

looper = ExplicitLoop(0.0, [domain, ground])
for _ in range(120):
    looper.advanceWithTime(1.0 / 60.0)
```

### 9.3 FEM + 高度场地面

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

## 附录：关键源码索引

| 主题 | 文件 | 关键符号 |
|------|------|----------|
| 主循环 | `explicitloop.py` | `ExplicitLoop`, `advanceWithTime`, `advance` |
| FEM 积分 | `femspringmanager.py` | `femStep`, `FemSpringManager` |
| 刚体求解 | `rigidmanager.py` | `substep`, `solve_pgs` |
| 形函数/B 矩阵 | `mesh.py` | `getJacobian2D`, `getBMatrix2D` |
| 本构 | `materials/materialfunctions.py` | `getStress2D`, `misesReturnMap2D` |
| 惩罚接触 | `femcontact.py` | `ContactFlexAnalytical`, `_segment_penalty` |
| 解析地面 | `grounddomain.py` | `GroundDomain` |
| 高度场 | `heightfielddomain.py` | `sample_height_2d`, `nearest_on_curve_2d` |
| 体素 | `voxeldomain.py` | `signed_distance_to_edges_2d`, `build_edges` |
| 接触检测 | `contact_detection.py` | `pointToEdgeContact` |

---

*文档基于 FlatWorld 当前代码库自动生成，反映 test2D 中已验证的功能。*
