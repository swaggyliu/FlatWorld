# Parallel environments & contact isolation

FlatWorld supports **many independent rigid scenes in one process** for
vectorized RL (Push-T PPO uses up to 128 envs). Isolation is enforced at
the BVH / pair-filter layer via `env_id`.

## Contract

| field | meaning |
|-------|---------|
| `RigidBodyDomain.env_id` | integer tag; default `-1` |
| `env_id >= 0` | collides only with bodies that share the same id |
| `env_id < 0` | **global** — may collide with every body (legacy single-scene) |

`ExplicitLoop` passes each domain’s `env_id` into `BVH.addObjects_batch`.
Narrow-phase pair generation skips cross-env pairs even if AABBs overlap.

## Compact managers

Training packs several Push-T copies into one `RigidManager` / loop with
disjoint body index ranges and matching `env_id`s (see
`learning/rl/push_t_vec_env.py`). Choose `--num-workers` / envs-per-worker
from **memory and Warp stability** on the machine you have (a typical
Push-T PPO run uses 128 envs); if process spawn fails, reduce workers and
increase envs packed per process.

## Checklist before adding a new multi-env scene

1. Set `env_id=k` on every rigid of copy `k` (including EE).
2. Prefer one shared ground / analytical plane as `env_id=-1` **or**
   duplicate grounds per env — do not mix conflicting globals carelessly.
3. Joints / anchors must only reference bodies inside the same env.
4. Smoke-test with `pytest test2D/test_env_id_isolation.py`.

## Mesh contact status

| path | status |
|------|--------|
| `updateMeshCoords` | Warp (one thread per mesh) |
| plane–mesh | Warp (pair-parallel, representative contacts) |
| mesh–mesh / mesh–primitive | host narrowphase; `.numpy()` hoisted once per batch |
| HF / voxel / prim–prim | batched Warp |

Tracking list: repo-root [`improvement.md`](../improvement.md).
Do not regress env_id filtering when remaining mesh kernels land.

## RL packing & WM encode (Push-T PPO)

Recommended defaults (tune to **memory / Warp stability**, not OS):

| knob | typical |
|------|---------|
| `--num-envs` | 128 |
| `--num-workers` | 4–8; if spawn fails, fewer workers and more envs packed per process |
| CEM game horizon | keep short (Spirit Push / interactive) |

`WMObsVecEnv` (`learning/train_rsl_wm_push_t.py`) encodes every step:
raw states → frozen WM `encode_batch` → `[z ‖ task_feat]`. Cost is dominated
by the GNN encode (batch of N obs dicts), not the VecEnv pipe. Profile tips:

1. Time `base.step` vs `_encode_td` separately (torch profiler or simple timers).
2. Prefer one encode per PPO step; avoid re-encoding in logging/eval loops.
3. Keep tactile dropped (`use_tactile=False`) to match the frozen Push-T WM.
