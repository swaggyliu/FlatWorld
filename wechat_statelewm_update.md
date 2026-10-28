# 一个世界模型，三种玩法：CEM / BPTT / PPO

> StateLeWM 更新：Push-T 用 WM+PPO 跑到 90%；仓库里三条控制范式终于对齐

---

先一句话：FlatWorld 里的 StateLeWM **不学像素**。输入是物理状态（位姿、速度、形状）+ 末端触觉，输出是「下一帧物体在哪」。整张网大约 **32 万参数**。真正变的是**怎么用这个脑子** —— 仓库现在明确成三条栈：

| 范式 | 在干什么 | 代表任务 | 成绩 |
|------|----------|----------|------|
| **WM + CEM** | 在线想象规划，没有策略网 | 推箱 / 穿门 / 二连杆 | 96–98% |
| **WM + BPTT** | 短想象轨迹反传改进策略 | SuperMimic-2D 技能 | 多技能跟踪 |
| **WM + PPO** | 冻结 WM 当观测，真仿真里强化学习 | **Push-T** | **90%** |

![StateLeWM 一步世界模型](docs/statelewm_worldmodel.png)

![StateLeWM 推理流程](docs/statelewm_inference.png)

更完整的对比与路线图：[docs/PARADIGMS.md](docs/PARADIGMS.md)

---

## 1. WM + CEM：还是那套「无目标交互 → 规划」

PushToGoal / Two-room / Reacher **不是三个架构**，是同一套配方换场景：

- 推物堆（最左 / 最右 / 随机）：**98 / 96 / 96%**
- Two-room 穿门：**49/50 = 98%**
- Reacher 重力二连杆：**48/50 = 96%**

适合：目标在 `(x,y)` 上好写、短时域想象够准、需要**实时重规划**（比如 Spirit Push 游戏）。

和像素系 LeWM 同名任务并排（量级参照，不是同一操场）：StateLeWM Two-Room **98** / Reacher **96**；LeWM 87/86；Fast-LeWM 98/88；DINO-WM 100/79；PLDM 97/78。

---

## 2. WM + BPTT：SuperMimic-2D

有参考动作时，不靠模型无关 PPO 硬刷物理，而是：**拟合共享刚体世界模型 → 在短想象轨迹上 BPTT** 改进目标条件策略（走跑跳蹲打架起身……）。

适合：模仿、动画、高自由度跟踪；前提是**有 demo / 稠密跟踪奖励**。

宣传片：`supermimic2d/runs/lafan2d/promo_supermimic.mp4`

---

## 3. WM + PPO：Push-T 为什么换栈

Push-T 是欠驱动的 T 块对齐：末端用力推，还要转到位姿。短视界 CEM 在非凸接触上很吃力；纯「在世界模型里想象着做 RL」早期也只有个位数成功率 —— 想象动力学漂了。

现在的打法：

1. 重训 Push-T 世界模型 `push_t`
2. 冻结 WM，策略只看 `z ‖ 任务特征`
3. **奖励和转移留在真实 FlatWorld**，`rsl-rl` PPO，128 并行环境
4. 进度奖励 `Φ_prev − Φ`，仿真评测 **36/40 = 90%**

![Push-T 成功率](docs/push_t_success.png)

![Push-T 训练曲线](docs/push_t_train_curve.png)

适合：接触丰富、欠驱动、CEM 代价景观太拧巴、需要探索的任务（插销-lite、多接触装配、软接触推齐……）。

---

## 怎么选

- 要交互、目标好写、在线重规划 → **CEM**
- 有动作参考、高 DoF 跟踪 → **BPTT**
- 接触欠驱动、规划搜不动 → **PPO（真仿真 + WM 观测）**

一个潜变量前端，三种控制器 —— 这是这次想讲清楚的故事。

---

## 已经落地的路线条目

1. **统一 Task 接口** — `learning/tasks/base.py`（obs / goal / Φ / success）  
5. **产品向** — `python -m game --mode pusht`；`python -m learning.render_pusht_promo`；范式文 [`docs/PARADIGMS.md`](docs/PARADIGMS.md)；全文稿本地 arXiv（cs.RO）另行上传  
6. **引擎并行** — `env_id` 隔离文档 [`docs/ENGINE_PARALLEL.md`](docs/ENGINE_PARALLEL.md)；`updateMeshCoords` / plane–mesh 已 kernel 化，其余见 `improvement.md` 

还开着的：Push-T 变体、新 CEM 场景、SuperMimic 加宽。

代码与权重：<https://github.com/swaggyliu/FlatWorld>
