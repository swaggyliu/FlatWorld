# Flat World

[English](README.md) | **中文**

基于 **NVIDIA Warp** 的 **2D** 实时物理仿真引擎，支持刚体、有限元（FEM）、弹簧-质量系统，以及三种地面表示（解析平面、高度场、体素网格）。

## 功能概览


| 模块  | 说明                                                                                    |
| --- | ------------------------------------------------------------------------------------- |
| 有限元 | 显式动力学，线性弹性 / Neo-Hookean / J2 塑性                                                      |
| 刚体  | 球、盒、胶囊、网格；2D SAT；PGS 冲量求解                                                             |
| 地面  | `GroundDomain`、`HeightFieldDomain`、`VoxelGridDomain`；刚体与 FEM 接触均为 batched Warp kernel |
| 关节  | 转动、焊接、滑动、球铰                                                                           |
| 接触  | `mixedcontact.py` FEM/弹簧惩罚接触；`rigidmanager.py` 刚体 PGS                                 |




## 安装与运行

```bash
pip install -r requirements.txt
pip install -e .
```

读 `.msh` / `.vtu` 需要 `meshio`；Spirit Push 游戏需要 `raylib`。

无显示器环境（CI）请设置：

```bash
set HEADLESS=1          # Windows
export HEADLESS=1       # Linux / macOS
pytest test2D -q
```

`test2D/` 下共 51 个 pytest 模块。

## 图库 Gallery

`test2D` 可渲染案例的实时截图（共 35 张）：

![多米诺](docs/gallery/domino.png)![FEM 刚体混合](docs/gallery/fem-rigid-mixed.png)![机械臂](docs/gallery/robot.png)![盒子碰撞](docs/gallery/rigid-25-boxes.png)![高度场](docs/gallery/fem-heightfield.png)![摆](docs/gallery/pendulum.png)

完整列表：[docs/gallery/README.md](docs/gallery/README.md) · 重新生成：`python scripts/capture_gallery.py`

## 学习栈（StateLeWM）

一套 **状态 + 触觉潜变量世界模型**（约 32 万参数），三种控制器。
架构与损失：**[learning/README.md](learning/README.md)**；范式对比与路线图：
**[docs/PARADIGMS.md](docs/PARADIGMS.md)**。


| 范式            | 任务                              | 成绩              | 权重                                                  |
| ------------- | ------------------------------- | --------------- | --------------------------------------------------- |
| **WM + CEM**  | PushToGoal · Two-room · Reacher | 96–98%（50 局）    | `learning/checkpoints/{push_game,two_room,reacher}` |
| **WM + BPTT** | SuperMimic-2D 技能                | 多技能跟踪           | `supermimic2d/runs/lafan2d`                         |
| **WM + PPO**  | Push-T                          | **36/40 = 90%** | `learning/checkpoints/push_t_wm_rsl`（WM：`push_t`）   |


<p align="center">
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#pusht"><strong>▶ Push-T / You vs AI</strong></a>
  ·
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#supermimic"><strong>SuperMimic 宣传片</strong></a>
</p>

<p align="center">
  <img src="docs/statelewm_worldmodel.png" width="720" alt="StateLeWM 一步预测" />
</p>

<p align="center">
  <img src="docs/statelewm_inference.png" width="720" alt="StateLeWM 推理流程" />
</p>

### 1. WM + CEM — 推箱 · 穿门 · 机械臂

无目标交互学 StateLeWM，再在潜空间做 **在线 CEM**。无策略网络；场景之间只换数据与 typed 读出。


| 场景                       | 成功率（50 局，seed 1000） | tol / vel         | 权重                               |
| ------------------------ | ------------------- | ----------------- | -------------------------------- |
| PushToGoal（最左 / 最右 / 随机） | 98 / 96 / 96 %      | 0.08 m / 0.08 m/s | `learning/checkpoints/push_game` |
| Two-room                 | **49/50 = 98%**     | 0.10 m / 0.15 m/s | `learning/checkpoints/two_room`  |
| Reacher                  | **48/50 = 96%**     | 0.06 m / 0.18 m/s | `learning/checkpoints/reacher`   |



| 方法                            | Two-Room | Reacher |
| ----------------------------- | -------- | ------- |
| **StateLeWM**（本仓库，状态 + EE 触觉） | **98**   | **96**  |
| LeWM                          | 87       | 86      |
| Fast-LeWM                     | 98       | 88      |
| DINO-WM                       | 100      | 79      |
| PLDM                          | 97       | 78      |


量级参照，非受控对比（像素系数字来自各自论文实现）。

### 2. WM + BPTT — SuperMimic-2D

FlatWorld 上的 **2D [SuperMimic](https://supermimic.github.io/)**：拟合共享刚体世界模型，再在短想象轨迹上 **BPTT** 改进目标条件策略。说明：**[supermimic2d/README.md](supermimic2d/README.md)**。

<p align="center">
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#supermimic"><strong>▶ 观看宣传片（约 33 秒）</strong></a>
  ·
  <a href="https://swaggyliu.github.io/FlatWorld/media.html#dance">dance 对比</a>
</p>

```bash
python -m supermimic2d.bake_refs
python -m supermimic2d.train --out supermimic2d/runs/lafan2d --library lafan2d --iters 80
python -m supermimic2d.render_promo --ckpt supermimic2d/runs/lafan2d/best.pt
```



### 3. WM + PPO — Push-T

欠驱动 T 块对齐（CEM 单独不够用）。冻结 Push-T StateLeWM（`push_t`），用
**rsl-rl PPO**：策略观测 = WM 潜变量 ‖ 任务特征，**奖励与转移在真实 FlatWorld**，
向量化并行环境。仿真评测 **36/40 = 90%**。

<p align="center">
  <img src="docs/push_t_success.png" width="680" alt="Push-T WM-obs PPO 成功率 90%" />
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



## Spirit Push 游戏

用世界模型当「精灵」的 raylib 原型：点要推的物体和落点，CEM 规划力。前五关是固定战役（1 个圆形末端执行器 + 3 箱 + 2 球）；第 6 关起按种子**随机出关**（重试同一关不变，跳过/过关是新布局）。

```bash
python -m game
python -m game --mode pusht   # Push-T：You vs AI split-screen (WASD vs PPO)
```

金色是目标，红色碰一次就重来，星星是终点。右侧滑条改弹性 / 摩擦力 / 尺寸。

<p align="center">
  <img src="docs/game/menu.png" width="640" alt="Spirit Push 开始菜单" />
</p>
<p align="center">
  <img src="docs/game/push.png" width="260" alt="第 1 关 Push" />
  <img src="docs/game/bank.png" width="260" alt="第 3 关 Bank" />
  <img src="docs/game/gauntlet.png" width="260" alt="第 5 关 Gauntlet" />
</p>

截图再生成：`python scripts/capture_game.py`。

## 文档

- [学习世界模型](learning/README.md)
- [控制范式（CEM / BPTT / PPO）](docs/PARADIGMS.md)
- 手稿将上传 arXiv（**cs.RO**）；仓库内范式说明：[docs/PARADIGMS.md](docs/PARADIGMS.md)
- [并行环境 / env_id](docs/ENGINE_PARALLEL.md)
- [引用](CITATION.cff)
- [SuperMimic-2D](supermimic2d/README.md)
- [理论与实现](docs/THEORY_AND_IMPLEMENTATION.zh.md) · [English](docs/THEORY_AND_IMPLEMENTATION.md)
- [贡献指南](CONTRIBUTING.md)



## 开源协议

本项目采用 [Apache License 2.0](LICENSE)。
