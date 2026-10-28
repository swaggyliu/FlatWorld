# StateLeWM — Learning World Models over Physics States in FlatWorld

A lightweight **state + tactile conditioned latent world model** on top of
the FlatWorld physics engine. No vision: the model consumes structured
physical states and the end-effector's variable-length contact set, and
predicts **per-object** latent dynamics for model-predictive control.

<p align="center">
  <img src="../docs/statelewm_worldmodel.png" width="920" alt="StateLeWM one-step world model: current frame to predicted next frame" />
</p>

<p align="center">
  <img src="../docs/statelewm_inference.png" width="920" alt="StateLeWM inference: observe, encode, latent, CEM imagination, act in FlatWorld" />
</p>

Actions are **not** produced by the world model itself. Controllers
plug in on top of the same latent dynamics:

| Paradigm | Role of the WM | Flagship | Doc |
|----------|----------------|----------|-----|
| **WM + CEM** | imagine rollouts for online planning | Push / Two-room / Reacher | this file |
| **WM + BPTT** | differentiable imagination for π | SuperMimic-2D | [`../supermimic2d/README.md`](../supermimic2d/README.md) |
| **WM + PPO** | frozen encoder → policy obs `z‖feat` | Push-T (≥90%) | below + [`../docs/PARADIGMS.md`](../docs/PARADIGMS.md) |

## Overview

Default scene still has **N = 6** bodies: 1 circular EE + 3 boxes + 2 balls
(counts are not randomized). Tensor index 0 is always the EE; along *x* the
EE sits at a **random rank** and the five objects are shuffled each reset.
Weights are **shared across bodies** (like a CNN over pixels): one encoder /
predictor / contact+xy readout, applied to every object. `N` is not baked into
parameter shapes. The shipped checkpoints are **320K parameters** (0.32M):
encoder 89,872, predictor 201,877, pair head 24,705, ground head 4,161. There is **no
full-state decoder**; CEM scores `xy_head`.

The network is a **one-step predictor**: encode the current frame, apply a
force on the EE, read out the next `xy`. Repeat that step to unroll a horizon.

```
 per frame
   obj_types / obj_states / obj_geom  ──►  state GNN  ──►  h_i
   EE tactile (K×7 + summary)         ──►  residual Δ on EE only
        ▲                                    (off unless last xy miss)
        └── gated at CEM: state is the default encode

                    z_t  (N, 128)
                      │
         pair_prob(z) → object–object edges (who touches whom)
         ground_prob(z) → per-node GRU input (on floor / in air)
         a_t on EE only
                      │
              Interaction Net + GRU  ──►  z_{t+1}
                      │
              xy_head:   z_i → (x, y)     ← CEM cost
              vel_head:  z_i → (vx, vy)   (edge rel_v)
```

Downstream: receding-horizon **CEM / MPC**. Dynamics roll in latent space;
the cost is scored on residual-corrected `xy_head` coordinates. The first
action is executed in the **simulator**. A Python + raylib prototype lives
in `game/` (`python -m game`).

## Directory layout

```
learning/
├── configs/default.py          # dataclass config (scene / tactile / collect)
├── env/flatworld_wrapper.py    # PushSceneEnv
│                               #   6-d state per body, EE-only tactile,
│                               #   mass / friction / size DR + sensor noise
├── data/
│   ├── collect.py              # random-policy rollouts → npz
│   ├── dataset.py              # time-window dataset (optional frame stride)
│   └── normalizer.py           # per-dim standardization
├── models/
│   ├── encoder.py              # per-object sparse graph + EE Deep Sets
│   ├── predictor.py            # Interaction Network + per-node GRU
│   └── lewm.py                 # assembled model + losses
├── planner.py                  # generic latent CEM (sample / roll / elite)
├── tasks/base.py               # FlatWorldTask protocol (obs / Φ / success)
├── tasks/adapters.py           # CEM / PPO / Mimic helpers
├── tasks/push_to_goal.py       # PushToGoal env loop + xy_head cost
├── tasks/push_t.py             # Push-T task + progress reward Φ
├── rl/                         # vec env, WM encode obs, PPO helpers
├── train.py                    # WM training + long-horizon open-loop eval
├── train_rsl_wm_push_t.py      # WM-obs rsl-rl PPO (Push-T)
├── eval_task.py                # CEM task success-rate evaluation
├── eval_rl_push_t_report.py    # Push-T PPO sim eval + figures
├── render_pusht_promo.py       # short WM+PPO promo mp4
├── replay.py                   # dump failed-eval MP4s
└── make_report.py              # curves / success stats / scene renders
```

### Unified task API

Every scene task exposes `observe` / `sample_goal` / `potential` (Φ) /
`metric` / `succeeded` (`learning/tasks/base.py`). CEM plans with geometric
costs; PPO uses `progress_reward` / `adapters.ppo_step_reward`
(`r = Φ_prev − Φ`). SuperMimic uses the parallel `MimicRefTask` surface.


## Observation / npz format

Each rollout stores `T+1` state frames and `T` actions (`(s_t, a_t) → s_{t+1}`).

| key               | shape           | description |
|-------------------|-----------------|-------------|
| `obj_types`       | `(N,)`          | 0 = EE, 1 = box, 2 = ball (not z-scored) |
| `obj_states`      | `(T+1, N, 6)`   | `x, y, θ, vx, vy, ω` (`θ` wrapped to `[−π, π]`) |
| `obj_geom`        | `(N, 2)`        | half-extents; EE/ball `(r, r)`, box `(hw, hh)` |
| `actions`         | `(T, 2)`        | force `(Fx, Fy)` on the EE |
| `contact_feat`    | `(T+1, K, 7)`   | EE contacts: rel_pos(2) + normal(2) + force_on_ee(2) + pen(1) |
| `contact_mask`    | `(T+1, K)`      | 1 = real contact, 0 = padding (not z-scored) |
| `tactile_summary` | `(T+1, 4)`      | `ΣFx, ΣFy, Στ, num_contacts` |
| `pair_contact`    | `(T+1, N, N)`   | solver rigid–rigid contact, 0/1 (not z-scored) |
| `ground_contact`  | `(T+1, N)`      | solver body–plane contact, 0/1 (not z-scored) |

`K = 16`. Contacts involving only objects (box–box, box–ground) are **not**
written into `contact_feat`; those effects appear through the next frame's
`obj_states` and through `pair_contact` / `ground_contact`. Deep Sets + mask
make the EE tactile branch permutation- and count-invariant.

**Normalization.** Train time z-scores `obj_states`, `actions`, `contact_feat`
(padding rows re-zeroed after), `tactile_summary`, and `obj_geom`. Types and
0/1 flags stay raw so they match embeddings and BCE targets. `knn` / signed
gap use the same standardized `xy` and `geom` as the node MLP (`gap_cut=0.12`
in that space). CEM applies the checkpoint's `normalizer.json` the same way.

At collection, domain randomization covers object/EE mass, friction, and
per-object size. Gaussian noise is added to positions, velocities, and
contact forces. The EE is inserted into the lineup at a random rank
(`ee_rank_range`), so it is not always left of the pile.

## Encoder / predictor

**Encoder (`StateTactileEncoder`)** — one forward pass per frame.

1. Node: type embedding (32) + state MLP (6→64) + geom MLP (2→32),
   projected to 64-d `h_i`.
2. Tactile: each of `K` contacts through an MLP, mask-sum (Deep Sets),
   concatenated with the 4-d summary, **added only to the EE node as a
   residual**. Training drops this residual with probability
   `--tactile-drop-prob` (default 0.5) so the state backbone stands alone.
   At CEM the default encode is state-only; tactile turns on when the last
   imagined `xy` missed the real scene (slip / unexpected contact), matching
   a vision-baseline + touch-residual pattern. Zero tactile (no contact)
   already makes the residual ≈ 0.
3. `n_mp` rounds of sparse message passing (train / shipped default **3**).
   An edge is kept if it is among the `k=2` nearest neighbours
   **or** the signed gap `dist − r_i − r_j` is below `0.12`. Edges also
   carry relative velocity ``rel_v = (Δvx, Δvy)`` and a contact flag;
   messages **sum**. Do not project onto the CoM–CoM line (that is not
   the contact normal for boxes). Nodes update with a shared `GRUCell`.
   Output `z ∈ (B, N, 128)`.

**Predictor (`LatentPredictor`)** — Interaction Network in latent space.
Neighbourhoods are rebuilt each step from a cheap `xy` readout + geom.
The 2-d action is concatenated **only on the EE node**. A per-node
`GRUCell` carries short contact memory across steps. A `vel_head` rebuilds
`rel_v` on imagined edges, including the previous-step contact flag
(persistence).

**Contact heads** (on `StateLeWM`): privileged **pair** (N×N rigid–rigid)
and **ground** (per-body vs the floor plane). Imagined CEM rollouts feed
`pair_prob(z)` onto interaction edges and `ground_prob(z)` into the
predictor GRU. There is no EE-contact focal head; EE–object contact is
the first row of pair. Coordinates for CEM come from `xy_head`.

## Losses

```
Total = MSE(z_pred, sg(z_{t+1}))                         # latent dynamics (JEPA-style)
      + w_var      * SIGReg                              # hinge: max(0, 1 − std(z_d))
      + w_xy       * xy_head vs true xy
      + w_pair     * focal BCE(solver pair contact)
      + w_ground   * focal BCE(solver ground contact)
      + w_drift    * predicted object motion at near-zero EE force
      + w_vel      * vel_head vs true (vx, vy)
```

Pair already contains EE–object flags; ground is a separate body–plane
channel and is used in imagination.

`z_target` is stop-grad so the predictor cannot collapse the encoder.
CEM scores `xy_head`. SIGReg is a variance floor, not a maximize-variance
term.

## Data collection

Default output: `learning/data/rollouts_500` (`num_rollouts=500`,
`episode_len=200`, seed 30000). Ground contact uses rolling resistance
(`C_rr = 0.25`) so disks come to rest. Six interleaved modes
(`CollectConfig` shares):

| mode      | share | behaviour |
|-----------|-------|-----------|
| `free`    | 15%   | hold a random-direction force for a stretch + light OU. Teaches force → EE motion, and "objects stay still without contact". |
| `attract` | 15%   | OU force + capped attraction to a random object (horizontal-only near the ground). Contact / tactile. |
| `push`    | 15%   | timed push / withdraw toward a random object. Contact making and breaking. |
| `release` | 10%   | after a shove, zero/small force so objects come to rest. |
| `hop`     | 10%   | lift on +Fy, cruise, then dive into a random object. Aerial hits (Bank-style). |
| `pile`    | 35%   | shove the body in front of a random target (the target may sit behind other objects). Chain-contact / Random-task coverage. |

OU (Ornstein–Uhlenbeck) is temporally correlated force noise,
`a_t ≈ 0.85 a_{t−1} + 1.5 ε_t`, so trajectories are smooth rather than
white-noise jitter. Soft workspace barriers keep the EE in-bounds; the
barrier force is stored as part of the action.

## Training

```bash
python -m learning.train --data learning/data/rollouts_500 --epochs 80 --ensemble 1
```

- Window of 24 model steps; **stride 5**: one model step covers five sim
  frames, and the stored action is the mean force over the chunk (makes
  the force → motion signal large enough that dynamics cannot ignore it).
- `--ensemble K` trains `K` independently seeded copies of the **same**
  architecture (not one checkpoint per object). Used at plan time as
  mean cost + disagreement penalty. `K=1` writes `best.pt`; `K>1` writes
  `ens_0.pt` … `ens_{K-1}.pt` plus a copy of member 0 as `best.pt`.
- Train default `--n-mp 3`, `--epochs 80`, `--stride 5`,
  `--tactile-drop-prob 0.5`.
- Checkpoints store `n_obj`, `latent_dim`, `stride`, `n_mp`,
  `tactile_residual`, and point at `normalizer.json`.

## PushToGoal + CEM

The CEM loop is task-agnostic (`learning/planner.py`): encode the true
observation (state-only, plus tactile residual if the last step missed) →
sample force sequences → roll the latent predictor with `pair_prob` on
edges and `ground_prob` on nodes → score residual-corrected `xy_head` →
keep elites → execute the first action and replan.
PushToGoal supplies the cost (`PushToGoalCost` in
`learning/tasks/push_to_goal.py`); swapping tasks should swap the cost,
not the sampler.

Protocol (`eval_task.py`):

- `leftmost` / `rightmost` / `random`: which object is the target.
- Goal along ±x, clipped to the free-run gap, 10–30 cm (floor 10 cm >
  tolerance 8 cm). Occupied slots are rejected.
- Success: target–goal distance `< 0.08 m` **and** speed `< 0.08 m/s`
  within 400 sim frames.
- Force limit 6 N. Eval CEM: `horizon=32`, `population=96`,
  `iterations=5`. The game uses a shorter CEM (`H=6`, `P=48`, `iters=2`)
  to hold 60 Hz.

### World-model vs solver rollouts

CEM imagines `population × iterations × horizon` trajectories. The world
model does that **batched on GPU**; the engine must step each candidate
serially (`stride=5` visual frames per model step). Timed on this box
(RTX 5060 Ti, Warp CPU scene, N=6):

| CEM size | physics frames / plan | world model | solver CEM | ratio |
|----------|----------------------:|------------:|-----------:|------:|
| eval `H=32 P=96 I=5` | 76 800 | **0.24 s** | ~303 s | **~1 300×** |
| game `H=6 P=48 I=2` | 2 880 | **27 ms** | ~11 s | **~400×** |

The game replans every `stride=5` frames at 60 Hz (**83 ms** budget).
World-model CEM fits (27 ms). The same CEM on the solver would be ~11 s
per plan — about **140×** over budget, so the prototype would drop to
well under 1 Hz. Eval-sized solver CEM is ~5 min **per replan**, not per
episode.

Cost on residual-corrected `xy_head` (unchanged by the planner split):

1. `|target − goal|` each step, plus `2 ×` that distance at the horizon.
2. Reach: EE toward the contact face behind the target (from the current
   pose vs goal). When the target rolls fast toward the goal, the reach
   target is a bumper at the goal. Inside the 8 cm band the executed
   force is zero.
3. Action L2. Ensemble: mean cost + `uncert_cost ×` std (default 0.1).

Imagined `xy` is residual-corrected against a **zero-action baseline**
so resting bodies do not hallucinate drift. The **shipped controller is
CEM-only**: every executed `(Fx, Fy)` comes from CEM over the world
model. Geometric terms shape the cost; they do not emit the force.

## Current results

Numbers below are PushToGoal with **ensemble=1**, CEM H=32, 50 episodes,
seed 1000. The shipped model is trained on `learning/data/rollouts_500`
(500 npz, seed 30000). Weights: `learning/checkpoints/push_game/best.pt`.

| protocol  | success | mean final dist | mean settle |
|-----------|---------|-----------------|-------------|
| leftmost  | 49/50 = **98%** | 0.037 m | 153 |
| rightmost | 48/50 = **96%** | 0.037 m | 167 |
| random    | 48/50 = **96%** | 0.056 m | 153 |

JSON: `learning/results/task_eval_500v2_H32_{leftmost,rightmost,random}.json`.
Failed episodes write MP4s under `learning/results/replays/500v2_H32_{mode}/`.

### Other scenes (same architecture, per-scene checkpoint)

| scene    | success | mean final dist | mean settle | tol / vel | weights |
|----------|---------|-----------------|-------------|-----------|---------|
| two-room | 49/50 = **98%** | 0.080 m | 139 | 0.10 m / 0.15 m/s | `learning/checkpoints/two_room` |
| reacher  | 48/50 = **96%** | 0.043 m | 107 | 0.06 m / 0.18 m/s | `learning/checkpoints/reacher` |

Both are 50 episodes, seed 1000, CEM H=32 (stride 5), ensemble=1. The
reacher eval uses an **in-band hold**: once the link-2 centre is within
`tol`, the executed action is the idle hold (current joint angles) and the
PD controller dissipates the remaining velocity — the same idea as the
push task's in-band zero force. JSON: `learning/results/reacher/task_eval.json`
(reacher), `learning/results/task_eval_random.json` (two-room).

These are the same two benchmark tasks (`Two-Room`, `Reacher`) used by the
pixel-based latent world models, so success rates line up across methods:

| method | Two-Room | Reacher | observation |
|--------|----------|---------|-------------|
| **StateLeWM** (this repo) | **98** | **96** | 6-d state + EE tactile |
| LeWM | 87 | 86 | pixels |
| Fast-LeWM | 98 | 88 | pixels |
| DINO-WM | 100 | 79 | pixels |
| PLDM | 97 | 78 | pixels |

Success %. StateLeWM is measured here (50 episodes); the
LeWM-family rows are published numbers on *their own* Two-Room / Reacher (LeWM:
Maes et al., 2026; Fast-LeWM: Gao & Xu, 2026; DINO-WM: Zhou et al., 2025; PLDM:
Sobal et al., 2025) with different episode counts, success thresholds, and pixel
observations — a size anchor, not a controlled head-to-head.

```bash
python -m learning.eval_task --task two_room \
    --checkpoint learning/checkpoints/two_room --episodes 50
python -m learning.eval_task --task reacher \
    --checkpoint learning/checkpoints/reacher --episodes 50
```

## Push-T — WM + PPO

Push-T is an underactuated T-block alignment task: force EE must contact and
rotate/translate the T onto a pose goal. Short-horizon CEM on the same WM
family is a weak fit (non-convex contact, long horizon). The working stack:

1. Collect goal-free Push-T + contact-rich rollouts; train StateLeWM
   (`learning/checkpoints/push_t_v4`).
2. Freeze the WM. Policy obs = flatten(`z`) ‖ task features (pose / EE–T).
3. PPO (`rsl-rl`) in **real** FlatWorld with progress reward
   `Φ_prev − Φ`, vectorized envs (128 envs / ≤6 Windows workers).
4. Eval in sim until ≥90% success.

| run | success | notes |
|-----|---------|-------|
| WM-obs rsl-rl (`push_t_wm_rsl`) | **36/40 = 90%** | seed 1000, hold=5, iter 350 hit target |
| raw-state rsl-rl ablation (`push_t_rsl`) | 90% train-eval | confirms task is learnable without WM |

```bash
python -m learning.train_rsl_wm_push_t \
    --wm learning/checkpoints/push_t_v4 \
    --out learning/checkpoints/push_t_wm_rsl \
    --num-envs 128 --num-workers 6 --target-rate 0.9
python -m learning.eval_rl_push_t_report \
    --policy learning/checkpoints/push_t_wm_rsl/best.pt \
    --wm learning/checkpoints/push_t_v4 \
    --out learning/results/wm_rl --tag wm_rsl
```

Figures: `learning/results/wm_rl/` and `docs/push_t_*.png`.
Playable duel: `python -m game --mode pusht` (WASD vs this policy).
Why CEM vs BPTT vs PPO: **[docs/PARADIGMS.md](../docs/PARADIGMS.md)**.

Early pure-imagination PPO on Push-T drifted (~8%); that path was removed.
Shipped stack keeps **physics in the RL loop**.

## Pipeline

```bash
# 1. collect rollouts
python -m learning.data.collect --num-rollouts 500 --episode-len 200 \
    --seed 30000 --out learning/data/rollouts_500

# 2. train
python -m learning.train --data learning/data/rollouts_500 --epochs 80 \
    --ensemble 1 --out learning/checkpoints/push_game --results learning/results

# 3. evaluate
python -m learning.eval_task --checkpoint learning/checkpoints/push_game \
    --episodes 50 --target-mode leftmost --tag 500v2_H32_leftmost
python -m learning.eval_task --checkpoint learning/checkpoints/push_game \
    --episodes 50 --target-mode rightmost --tag 500v2_H32_rightmost
python -m learning.eval_task --checkpoint learning/checkpoints/push_game \
    --episodes 50 --target-mode random --tag 500v2_H32_random

# 4. report figures
python -m learning.make_report --current-best

# 5. playable prototype
python -m game --checkpoint learning/checkpoints/push_game
```

Outputs land in `learning/checkpoints/<scene>/` (push defaults to
`learning/checkpoints/push_game`; or pass `--out`) and
`learning/results/` (`train_log.csv`, `task_eval_<tag>.json`, figures
under `current_best/` after `--current-best`).

## Validation criteria

1. Tactile non-zero on EE-contact frames (checked at collection).
2. Dynamics and pair/ground losses fall together; latent per-dim std stays
   at or above 1 (SIGReg is a floor, typical mean `z_std` is 1.2–1.4).
   Watch **val pair** — it has been overfitting.
3. Long-horizon open-loop (10 / 25 / 50 model steps) stays bounded.
4. PushToGoal success on both `leftmost` and `random` (the latter
   requires chain contact, not just a clear-corridor side push).

## Further work

See **[docs/PARADIGMS.md](../docs/PARADIGMS.md)** (roadmap). Short list:

1. Shared `Task` protocol for CEM / BPTT / PPO.
2. Variable-N / new geometry for CEM levels (not more PushToGoal contact labels).
3. Push-T variants + optional CEM residual on the PPO policy.
4. Longer-horizon WM calibration if revisiting pure imagination RL.
