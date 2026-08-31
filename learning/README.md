# StateLeWM — Learning World Models over Physics States in FlatWorld

A lightweight **state + tactile conditioned latent world model** on top of
the FlatWorld physics engine. No vision: the model consumes structured
physical states and the end-effector's variable-length contact set, and
predicts **per-object** latent dynamics for model-predictive control.

Actions are **not** produced by the network. The world model answers
"what happens if I apply this force"; a CEM planner (or the collection
policy) supplies `(Fx, Fy)` on the end-effector.

## Overview

Default scene still has **N = 6** bodies: 1 circular EE + 3 boxes + 2 balls
(counts are not randomized). Tensor index 0 is always the EE; along *x* the
EE sits at a **random rank** and the five objects are shuffled each reset.
Weights are **shared across bodies** (like a CNN over pixels): one encoder /
predictor / decoder, applied to every object. `N` is not baked into the
parameter shapes.

```
 per frame
   obj_types (N,)          type embedding
   obj_states (N, 6)  ──►  state MLP  ─┐
   obj_geom (N, 2)         geom MLP   ─┴─ node h_i (64)
   contact[K,7] + mask     Deep Sets  ──► inject into EE node only
   tactile_summary (4)     MLP        ──┘
                              │
                    sparse graph MP (k-NN + signed gap, n_mp rounds)
                              │
                    z_t  (N, 128)   one latent per body
                              │
              z_t + a_t ──► Interaction Net + per-node GRU ──► z_{t+1}
                              │
              decoder (shared):  z_i → state_i (6)
                                 z_EE → tactile summary (4)
                                 z_EE → P(EE in contact)
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
│   ├── decoder.py              # z_i → state_i; z_EE → tactile summary
│   └── lewm.py                 # assembled model + losses
├── planner.py                  # generic latent CEM (sample / roll / elite)
├── tasks/push_to_goal.py       # PushToGoal env loop + decoded-xy cost
├── train.py                    # training + long-horizon open-loop eval
├── eval_task.py                # task success-rate evaluation
└── make_report.py              # curves / success stats / scene renders
```

## Observation / npz format

Each rollout stores `T+1` state frames and `T` actions (`(s_t, a_t) → s_{t+1}`).

| key               | shape           | description |
|-------------------|-----------------|-------------|
| `obj_types`       | `(N,)`          | 0 = EE, 1 = box, 2 = ball |
| `obj_states`      | `(T+1, N, 6)`   | `x, y, θ, vx, vy, ω` (`θ` wrapped to `[−π, π]`) |
| `obj_geom`        | `(N, 2)`        | half-extents; EE/ball `(r, r)`, box `(hw, hh)` |
| `actions`         | `(T, 2)`        | force `(Fx, Fy)` on the EE |
| `contact_feat`    | `(T+1, K, 7)`   | EE contacts: rel_pos(2) + normal(2) + force_on_ee(2) + pen(1) |
| `contact_mask`    | `(T+1, K)`      | 1 = real contact, 0 = padding |
| `tactile_summary` | `(T+1, 4)`      | `ΣFx, ΣFy, Στ, num_contacts` |

`K = 16`. Contacts involving only objects (box–box, box–ground) are **not**
written into `contact_feat`; those effects appear only through the next
frame's `obj_states`. Deep Sets + mask make the EE tactile branch
permutation- and count-invariant.

At collection, domain randomization covers object/EE mass, friction, and
per-object size. Gaussian noise is added to positions, velocities, and
contact forces. The EE is inserted into the lineup at a random rank
(`ee_rank_range`), so it is not always left of the pile.

## Encoder / decoder / predictor

**Encoder (`StateTactileEncoder`)** — one forward pass per frame.

1. Node: type embedding (32) + state MLP (6→64) + geom MLP (2→32),
   projected to 64-d `h_i`.
2. Tactile: each of `K` contacts through an MLP, mask-sum (Deep Sets),
   concatenated with the 4-d summary, **added only to the EE node**.
3. `n_mp` rounds of sparse message passing (default 2; shipped pair19
   uses 3). An edge is kept if it is among the `k=2` nearest neighbours
   **or** the signed gap `dist − r_i − r_j` is below `0.12`. With
   `rich_edges=True` (pair14+ default) edges also carry relative velocity
   (normal / tangent) and a contact flag; messages **sum**. Nodes update
   with a shared `GRUCell`. Output `z ∈ (B, N, 128)`.

**Decoder (`StateTactileDecoder`)** — shared heads, not one network per body.

- `trunk`: `128 → 128 → 128` (GELU), applied to every `z_i`
- `state_head`: linear `128 → 6` → reconstructed rigid state
- `summary_head`: `128 → 64 → 4` on **`z_EE` only**

**Predictor (`LatentPredictor`)** — Interaction Network in latent space.
Neighbourhoods are rebuilt each step from a cheap `xy` readout + geom.
The 2-d action is concatenated **only on the EE node**. A per-node
`GRUCell` carries short contact memory across steps. `rich_edges=True`
adds a `vel_head` so imagined rollouts can rebuild velocity-aware
edges, including the previous-step contact flag (persistence).

**Contact head** (on `StateLeWM`): `z_EE →` logit for "EE in contact this
frame", trained with BCE on both encoded `z` and predicted `z`.

## Losses

```
Total = MSE(z_pred, sg(z_{t+1}))                         # latent dynamics (JEPA-style)
      + w_rec      * Recon(z_enc)                        # states + tactile summary
      + w_pred_rec * Recon(z_pred)
      + w_var      * SIGReg                              # hinge: max(0, 1 − std(z_d))
      + w_c        * [BCE(contact | z) + BCE(contact | z_pred)]
```

`z_target` is stop-grad so the predictor cannot collapse the encoder.
Reconstruction on **both** encoded and predicted latents keeps `z`
physically readable for CEM (which scores decoded coordinates, not latent
distance). SIGReg is a variance floor, not a maximize-variance term.

## Data collection

Default output: `learning/data/rollouts_lr` (`num_rollouts=100`,
`episode_len=200`). Four interleaved modes:

| mode      | share | behaviour |
|-----------|-------|-----------|
| `free`    | 30%   | hold a random-direction force for a stretch + light OU. Teaches force → EE motion with no +x bias, and "objects stay still without contact". |
| `attract` | 25%   | OU force + capped attraction to a random object (horizontal-only near the ground). Contact / tactile. |
| `push`    | 15%   | timed push / withdraw toward a random object. Contact making and breaking. |
| `pile`    | 30%   | shove the body in front of a random (possibly buried) target. Chain-contact / Random-task coverage. |

OU (Ornstein–Uhlenbeck) is temporally correlated force noise,
`a_t ≈ 0.85 a_{t−1} + 1.5 ε_t`, so trajectories are smooth rather than
white-noise jitter. Soft workspace barriers keep the EE in-bounds; the
barrier force is stored as part of the action.

## Training

```bash
python -m learning.train --data learning/data/rollouts_lr --epochs 60 --ensemble 5
```

- Window of 24 model steps; **stride 5**: one model step covers five sim
  frames, and the stored action is the mean force over the chunk (makes
  the force → motion signal large enough that dynamics cannot ignore it).
- `--ensemble K` trains `K` independently seeded copies of the **same**
  architecture (not one checkpoint per object). Used at plan time as
  mean cost + disagreement penalty. Default `--ensemble 1` writes
  `best.pt`; `K>1` writes `ens_0.pt` … `ens_{K-1}.pt`.
- Checkpoints also store `n_obj`, `latent_dim`, `stride`, `n_mp`,
  `rich_edges`, and point at `normalizer.json`.

Shipped weights: `learning/checkpoints/best.pt` (pair19 member 0, `n_mp=3`,
`rich_edges=True`, stride 5). Extra `ens_*.pt` files are local-only; if
they are present, game and eval load the ensemble.

## PushToGoal + CEM

The CEM loop is task-agnostic (`learning/planner.py`): encode the true
observation → sample force sequences → roll the latent predictor →
score a cost on residual-corrected `xy_head` trajectories → keep elites
→ execute the first action and replan (warm-start by shifting the mean).
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

Cost on decoded `xy` (unchanged by the planner split):

1. `|target − goal|` each step, plus `2 ×` that distance at the horizon.
2. Reach: EE toward the contact face of the target, faded as it nears
   the goal.
3. Coast: near-goal penalty if EE stays closer than standoff + 1 cm.
4. Brake: near-goal penalty on target speed (linear + quadratic).
5. Action L2. Ensemble: mean cost + `uncert_cost ×` std (default 0.1).

Imagined `xy` is residual-corrected against a **zero-action baseline**
so resting bodies do not hallucinate drift. The **shipped controller is
CEM-only**: every executed `(Fx, Fy)` comes from CEM over the world
model. Geometric terms shape the cost; they do not emit the force.

## Current results

`learning/checkpoints` (pair19, 2 members, CEM H=32, 50 episodes):

| protocol  | success | mean final dist | mean settle |
|-----------|---------|-----------------|-------------|
| leftmost  | 43/50 = **86%** | 0.061 m | 176 |
| rightmost | 43/50 = **86%** | 0.133 m | 199 |
| random    | 39/50 = **78%** | 0.231 m | 178 |

Figures: `results/current_best/`.

Later contact-label and chain-Δv fine-tunes (pair20 / pair21) did **not**
beat this baseline. pair19 stays shipped.

## Pipeline

```bash
# 1. collect rollouts
python -m learning.data.collect --num-rollouts 100 --episode-len 200 \
    --out learning/data/rollouts_lr

# 2. train (optional ensemble)
python -m learning.train --data learning/data/rollouts_lr --epochs 60 --ensemble 5

# 3. evaluate
python -m learning.eval_task --checkpoint learning/checkpoints \
    --episodes 50 --target-mode leftmost
python -m learning.eval_task --checkpoint learning/checkpoints \
    --episodes 50 --target-mode rightmost --tag rightmost
python -m learning.eval_task --checkpoint learning/checkpoints \
    --episodes 50 --target-mode random --tag random

# 4. report figures
python -m learning.make_report --current-best

# 5. playable prototype
python -m game --checkpoint learning/checkpoints
```

Outputs land in `learning/checkpoints/` (or `--out`) and
`learning/results/` (`train_log.csv`, `task_eval.json`, `*.png`).

## Diagnostics

| script | purpose |
|--------|---------|
| `learning.debug_force`   | ground-truth EE force response (F=ma sanity) |
| `learning.debug_data`    | collected-data action balance / physics checks |
| `learning.debug_cost`    | planner cost landscape for constant actions |
| `learning.debug_contact` | in-contact imagination vs. ground truth |
| `learning.debug_plan`    | closed-loop EE / target trajectory dump |
| `learning.debug_imagined`| open-loop decoded rollout vs. simulator |

## Validation criteria

1. Tactile non-zero on EE-contact frames (checked at collection).
2. Dynamics and reconstruction fall together; latent per-dim std stays
   near 1 (SIGReg). Watch **val contact** — it has been overfitting.
3. Long-horizon open-loop (10 / 25 / 50 model steps) stays bounded.
4. PushToGoal success on both `leftmost` and `random` (the latter
   requires chain contact, not just a clear-corridor side push).

## Further work

Tracked in [`improvement.md`](improvement.md). Short version: the
per-object GNN + `rich_edges` + ensemble **training path** are in place;
the remaining leverage is **variable-N / new geometry for levels**, not
more contact-label or chain-Δv training on this PushToGoal split.
