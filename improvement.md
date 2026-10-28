# 引擎后续优化（先不改）

刚体 vs HeightField / Voxel 已改成 batched Warp kernel，host Python 检测已删。下面这些还在热路径里，**明确要求前不要动**。记录于 2026-09-11；并行 env 段落更新于 2026-10-28。

多环境隔离说明见 [`docs/ENGINE_PARALLEL.md`](docs/ENGINE_PARALLEL.md)。

## 已经做完的

- 2D SAT 盒–盒（不再抬到 3D GJK）
- 原始刚体–刚体、平面 vs 原始刚体：Warp kernel
- PGS early-exit（`dim=1`，最多 200 次）
- FEM / 弹簧 vs 平面、HeightField、Voxel：`mixedcontact.py` 里已是 batched kernel
- 刚体 vs HeightField / Voxel：`rigidmanager.py` 里 batched kernel；旧 Python 已删（`detectHeightField2Rigids_`、`detectHeightField2MeshContacts_`、`detectVoxel2Rigids_`、`detectVoxel2MeshContacts_`、`detectAnalaytical2Rigid`、host `get_box_vertex`）
- **多 env 接触隔离**：`RigidBodyDomain.env_id` + BVH / pair filter（同 id 才碰；`<0` 为全局）。向量化 Push-T PPO 依赖此约定。测试：`test2D/test_env_id_isolation.py`。

RTX 5060 Ti、64 球、相对旧 host 循环（删除前测的）：HeightField detect ~200×，Voxel detect ~660×；整段 20 帧仿真 ~14× / ~61×。

## 影响最大（mesh 刚体场景，每个 substep）

名字带 `*_kernel`，实际仍是 Python + `.numpy()`。

1. **`detect_mesh_mesh_contacts_kernel`**（`rigidmanager.py`）  
   仍为 host 窄相；已每次 batch 只 hoist 一次 `.numpy()`。  
   下一步：一次 launch，按 pair 或 (pair, node)。

2. **`detect_mixed_contacts_kernel`**  
   Mesh vs 球 / 盒 / 胶囊：host + 同批 cache。  
   下一步：node vs primitive、采样点 vs mesh 批量化。

3. **`detect_analyticalmesh_contacts_kernel`** — **done (Warp)**  
   平面 vs mesh：pair 并行 + deepest/±t1 representative contacts（`_detect_plane_mesh_contacts_wp`）。  
   Host `detectAnalytical2MeshPair` 保留作参考，热路径已走 kernel。

4. **`updateMeshCoords`** — **done (Warp)**  
   `_update_mesh_coords_wp`：一 mesh 一线程，世界系边界点 + AABB。

5. **`update_mesh_element_aabbs`**  
   FEM–mesh 接触：逐元素 AABB，`_patch_array` 写回。  
   下一步：一次 launch 扫 mesh 元素。

6. **`populate_spatial_hash_filtered`** / **`SpatialHashManager.addElement`**  
   Mesh hash 重建时在 Python 里插元素。  
   下一步：device insert，或更少重建。

## 次一级（仍每个 substep 或每次 rebuild）

7. **`cacheContact` / `cacheGroundContact`**  
   上面 mesh / 平面–mesh 路径的 host 写入。Kernel 已用 `_cache_*_func`。那几条 kernel 化后可以一起删。

8. **`get_max_linear_velocity`**  
   Spatial hash 重建时 `.numpy()` 扫全部刚体速度。刚体不多或重建不勤时很轻。

9. **BVH `detectInnerCollision`**  
   碰撞核是 Warp，但每次仍 `node_count.numpy()` / `pair_count.numpy()`。  
   下一步：计数留在 device 上，和原始刚体 pair 计数一样。

## 已经是 kernel，只是 `dim=1` 串行（不是 Python）

mesh host 路径做完后再考虑，不要当成「把 Python 搬进 kernel」。

- PGS（`solve_pgs`，前后扫）
- 原始刚体–刚体（`_detect_primitive_contacts_wp`）
- 平面 vs 原始刚体（`_detect_analytical_prim_contacts_wp`）
- 地面 pair 的 classify / generate

要再加速是改成按 pair / constraint 并行 launch。

## 不在积分热循环里

- `HeightFieldDomain.sample_*` / `draw`，`VoxelGridDomain.build_edges` / `signed_distance_to_edges_2d`：初始化、测试、可视化
- FEM vs HeightField / Voxel：已经 batched
- 同一 FEM 场景里多个不同 HeightField 或 Voxel 仍会在 build 时进 `_unbatched_contacts`（少见）

下一步若只改一块，收益最大的是 **mesh–mesh / mesh–primitive / 平面–mesh** 这三条 Python 窄相。

## Diagnostics

Scratch `learning/debug_*.py` probes were removed from the tree.
Use `learning.eval_task` / `learning.eval_rl_push_t_report` and
`docs/PARADIGMS.md` for shipped evals.
