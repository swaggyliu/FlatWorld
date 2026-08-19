# StateLeWM — Learning World Models over Physics States in FlatWorld

A lightweight **state + tactile conditioned latent world model** built on top of
the FlatWorld physics engine. No vision: the model consumes structured
physical states and variable-length tactile contact sets, and predicts
latent dynamics that can be rolled out fast for model-predictive control.

## Overview

```
                    ┌────────────────────────────────────────────┐
 obj_types  ──►Embed│                                            │
 obj_states ──►MLP  │        StateTactileEncoder                 │
 contact[K,7]──►DeepSets ──► fusion ──► z_t (latent, 128-d)     │
 summary(4) ──►MLP  │                                            │
                    └────────────────────────────────────────────┘
                                        │
                     z_t + a_t ──►  GRU LatentPredictor ──► z_{t+1}
                                        │
                    ┌────────────────────────────────────────────┐
                    │ StateTactileDecoder (anti-collapse)        │
                    │  z ──► obj_states (N,6), tactile summary   │
                    └────────────────────────────────────────────┘
```

Downstream use: **CEM / MPC planning** in latent space for a PushToGoal
task; actions are executed in the simulator, success is measured in the
simulator (the world model only plans).

## Directory layout

```
learning/
├── configs/default.py          # dataclass configuration (scene / tactile / collect)
├── env/flatworld_wrapper.py    # PushSceneEnv: FlatWorld push scene with
│                               #   - 6-dof-per-object state readout (pos, theta, vel, omega)
│                               #   - variable-length tactile extraction (Deep Sets input)
│                               #   - domain randomization (mass / friction) + sensor noise
├── data/
│   ├── collect.py              # random-policy rollouts -> npz
│   ├── dataset.py              # time-window dataset for training
│   └── normalizer.py           # per-dim standardization stats
├── models/
│   ├── encoder.py              # StateTactileEncoder (type emb + state MLP + tactile DeepSets)
│   ├── predictor.py            # GRU latent dynamics z_t + a_t -> z_{t+1}
│   ├── decoder.py              # latent -> (states, tactile summary)
│   └── lewm.py                 # assembled model + Phase 2 losses
├── tasks/push_to_goal.py       # PushToGoal task + CEM latent planner
├── train.py                    # training loop + long-horizon open-loop eval
├── eval_task.py                # task success-rate evaluation
└── make_report.py              # training curves / success stats / scene renders
```

## Data format (npz per rollout)

| key              | shape          | description                                    |
|------------------|----------------|------------------------------------------------|
| `obj_types`      | `(N,)`         | 0 = end-effector, 1 = box, 2 = ball            |
| `obj_states`     | `(T+1, N, 6)`  | pos(2) + theta(1) + vel(2) + omega(1)          |
| `actions`        | `(T, 2)`       | force (Fx, Fy) on the end-effector             |
| `contact_feat`   | `(T+1, K, 7)`  | rel_pos(2) + normal(2) + force(2) + pen(1)     |
| `contact_mask`   | `(T+1, K)`     | 1 = real contact, 0 = padding                  |
| `tactile_summary`| `(T+1, 4)`     | sum_Fx, sum_Fy, sum_tau, num_contacts          |

Variable-length tactile: contact sets are padded to `K = 16` and handled by
a mask-aware Deep Sets encoder (permutation- and count-invariant).

## Losses (Phase 2)

```
Total = MSE(z_pred, z_target.detach())                  # latent dynamics
      + w_rec  * [Recon(z_enc) + Recon(z_pred)]          # states + tactile
      + w_var  * SIGReg                                  # hinge on per-dim std(z)
```

- reconstruction is applied to **both** encoded and predicted latents so the
  latent transition stays physically meaningful
- SIGReg (hinge form of `-var(z)`) prevents latent dimension collapse
- domain randomization (object mass ×[0.5, 2], friction [0.3, 0.8]) and
  sensor noise are applied at collection time for generalization

## Pipeline

```bash
# 1. collect rollouts (balanced 3-mode mix, 120 x 200 frames)
python -m learning.data.collect --num-rollouts 120 --episode-len 200 \
    --out learning/data/rollouts_v2

# 2. train the world model
python -m learning.train --data learning/data/rollouts_v2 --epochs 80

# 3. evaluate PushToGoal success rate with CEM planning
python -m learning.eval_task --checkpoint learning/checkpoints/best.pt --episodes 50

# 4. generate report figures
python -m learning.make_report
```

Outputs land in `learning/checkpoints/` and `learning/results/`
(`train_log.csv`, `task_eval.json`, `*.png`).

## Data collection policy (v2)

Three interleaved modes per rollout keep the action distribution balanced —
critical for a controllable world model:

| mode    | share | behaviour                                                       |
|---------|-------|-----------------------------------------------------------------|
| `free`  | 30%   | symmetric OU force noise in free space (soft workspace barriers). Teaches the true force -> EE-motion map and the "objects stay still without contact" gating. |
| `attract` | 50% | OU noise + weak attraction to the nearest object. Rich contact / tactile data. |
| `push`  | 20%   | timed push-toward / withdraw-from the nearest object. Teaches contact making/breaking. |

The v1 dataset (pure attraction) biased all forces toward +x and produced a
model whose imagined EE response was sign-flipped for negative forces — the
planner then reliably fled the target. The balanced mix removes the bias.

## PushToGoal task protocol

- target = **leftmost object** (the EE spawns left of the pile, so the
  approach corridor is clear by construction)
- goal = target + `clip(free_run - 0.02, 0.10, 0.30)` to the right; the
  floor of 0.10 m > tolerance 0.08 m guarantees a real push is required
- success = final target-goal distance < 0.08 m **and** target speed < 0.08
  m/s within a 300-frame budget
- force limit 6 N (covers the domain-randomized friction range)

## CEM planner details

The planner rolls out candidate force sequences through the **learned**
latent dynamics and decodes the EE trajectory. Two robustness measures
against learned-model hallucinations:

- **baseline subtraction**: a zero-action latent rollout is subtracted from
  the imagined states, keeping only the action-dependent displacement on
  top of the true current state (cancels hallucinated drift of resting
  objects);
- **monotone push prior**: the decoded target motion is not trusted
  (telekinesis hallucination); instead the target is assumed pushed to
  `ee_x + standoff` whenever the imagined EE reaches contact, never
  backwards. The learned EE dynamics (which generalizes across randomized
  mass/friction) remain the core of the plan.

## Diagnostics

| script | purpose |
|--------|---------|
| `learning.debug_force`   | ground-truth EE force response (F=ma sanity) |
| `learning.debug_data`    | collected-data action balance / physics checks |
| `learning.debug_cost`    | planner cost landscape for constant actions |
| `learning.debug_contact` | in-contact imagination vs. ground truth |
| `learning.debug_plan`    | closed-loop EE / target trajectory dump |

## Validation criteria

1. tactile signal non-zero on contact frames (verified at collection)
2. dynamics + reconstruction losses decrease jointly; latent per-dim std
   stays healthy (no collapse)
3. long-horizon open-loop rollouts (10 / 25 / 50 steps) stay bounded
4. PushToGoal success rate (see `results/task_success_summary.png`)

## Future work

Ideas ranked by expected value for a fast, physically consistent 2D
state-conditioned world model are tracked in the project plan; highlights:
energy/momentum-consistency penalties, contact-event auxiliary heads,
probabilistic (ensemble) latents, graph-structured object encoders,
half-step symplectic prediction, and online fine-tuning during deployment.
