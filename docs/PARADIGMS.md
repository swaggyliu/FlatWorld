# Control paradigms on StateLeWM

FlatWorld uses **one** state + tactile latent world model (StateLeWM), then
plugs in three different **controllers**. Same engine, same latent dynamics
idea; different when you *plan online*, *imitate through imagination*, or
*explore with RL*.

| # | Paradigm | Flagship tasks | Decision time | Needs demos? |
|---|----------|----------------|---------------|--------------|
| 1 | **WM + CEM** | PushToGoal, Two-room, Reacher | online planning | no |
| 2 | **WM + BPTT** | SuperMimic-2D (LAFAN-like skills) | offline improve π | yes (refs) |
| 3 | **WM + PPO** | Push-T | offline train π | no |

```
                 ┌── CEM / MPC ──► act in FlatWorld     (1)
 observe ─► WM ──┤
                 ├── BPTT through imag. ─► policy π     (2)
                 └── encode z ─► PPO in sim ─► policy π (3)
```

---

## 1. WM + CEM (reacher · push · two-room)

**Idea.** Freeze a goal-free world model. At every replan, sample force /
joint-target sequences in latent space, score a task cost on `xy_head`
(and typed readouts), execute the first action, repeat.

**Traits**

- No policy network; **zero RL** once the WM is trained.
- Cost is cheap geometry in latent readout space (target–goal, reach face).
- Imagination is **batched on GPU** (~400–1300× vs serial solver CEM).
- Works when short-horizon latent rollouts stay calibrated and the cost
  landscape is not too multimodal from contact underactuation.

**When to use**

- Sparse goal reaching with a clear metric in `(x, y)` / tip pose.
- Navigation-through-contact (doorway) where the WM learns walls for free.
- Interactive tools (Spirit Push game) that need **replan every few frames**.

**Generalizes to**

- Multi-object rearrange with typed edges.
- Tool-use / bank shots if contact data covers them.
- Soft-body or height-field goals if the same state schema exists.

**Weak when**

- Long-horizon underactuated contact (Push-T-style T alignment).
- Cost needs a skilled prior; pure CEM explores poorly in non-convex contact.

**Entry points**

```bash
python -m learning.data.collect --out learning/data/rollouts_500 ...
python -m learning.train --data learning/data/rollouts_500 --out learning/checkpoints/push_game
python -m learning.eval_task --checkpoint learning/checkpoints/push_game --target-mode random
python -m game --checkpoint learning/checkpoints/push_game
```

Weights: `learning/checkpoints/{push_game,two_room,reacher}`.

---

## 2. WM + BPTT (SuperMimic-2D)

**Idea.** Fit a shared rigid-body WM from on-policy (or mixed) rollouts.
Improve a goal-conditioned policy by **backprop through short imagined
horizons** with a mimic reward / critic — SuperMimic-style, not model-free
PPO in the physics loop for the improve step.

**Traits**

- Dense tracking reward from reference motions.
- Gradients flow through the frozen (or slowly fit) WM.
- High-DoF continuous control (8 PD joints, 10-body humanoid).
- Skills share one dynamics model; policy is goal / skill conditioned.

**When to use**

- Motion imitation, animation, loco-manipulation with mocap / baked refs.
- When demos exist and you want sample-efficient policy improvement.
- When pure CEM cannot express stylistic tracking objectives.

**Generalizes to**

- More LAFAN / custom clips, multi-character.
- Dexterous or multi-limb tracking once the state schema is fixed.
- Residual policies on top of PD / IK.

**Weak when**

- No reference / no dense reward — BPTT needs a shaped objective.
- WM bias can lock the policy into imagined dynamics that diverge from sim
  (diagnose with `supermimic2d.diagnose_wm`).

**Entry points**

```bash
python -m supermimic2d.bake_refs
python -m supermimic2d.train --out supermimic2d/runs/lafan2d --library lafan2d --iters 80
python -m supermimic2d.render_promo --ckpt supermimic2d/runs/lafan2d/best.pt
```

---

## 3. WM + PPO (Push-T)

**Idea.** Train a frozen StateLeWM on goal-free Push-T contact data.
Policy observations are **`z ‖ task_features`** (WM encode + pose features);
**rewards and transitions stay on real FlatWorld physics**. PPO via
`rsl-rl` on a vectorized env (multiprocess workers).

**Traits**

- Progress reward `Φ_prev − Φ` (pose + EE–T coupling); no fail-dones that
  encourage suicide resets.
- Parallel envs (e.g. 128) with compact RigidManagers + env_id isolation.
- Hits **≥90%** sim success where pure latent-imagination RL and CEM struggled.
- Ablation: raw-state PPO also reaches 90% — WM-obs matches that bar while
  keeping a transferable latent front-end.

**When to use**

- Contact-rich, underactuated pushing / alignment.
- When CEM cost landscapes are non-convex and short imagination drifts.
- When you need a reactive policy, not a multi-second replan.

**Generalizes to**

- Peg-in-slot / keyhole-lite in 2D, multi-contact fixture assembly.
- Soft or frictional Push-T variants with the same progress Φ.
- Hybrid: CEM warm-start or residual on a PPO policy.

**Weak when**

- Pure imagination PPO (roll dynamics in WM only) — Push-T WM drift made
  that path ~8% in early ablations; keep physics in the loop for RL.
- Tasks that only need a sparse xy cost — CEM is simpler.

**Entry points**

```bash
# WM
python -m learning.data.collect ... --out learning/data/push_t_500
python -m learning.train --data learning/data/push_t_500 learning/data/push_t_contact500 \
    --out learning/checkpoints/push_t_v4 ...

# WM-obs PPO (tune --num-workers to machine memory / Warp stability)
python -m learning.train_rsl_wm_push_t \
    --wm learning/checkpoints/push_t_v4 \
    --out learning/checkpoints/push_t_wm_rsl \
    --num-envs 128 --num-workers 6 --target-rate 0.9

# report eval
python -m learning.eval_rl_push_t_report \
    --policy learning/checkpoints/push_t_wm_rsl/best.pt \
    --wm learning/checkpoints/push_t_v4 \
    --out learning/results/wm_rl --tag wm_rsl
```

Result (40 ep, seed 1000): **36/40 = 90%**, mean settle ~127 frames.
Weights: `learning/checkpoints/push_t_v4`, `learning/checkpoints/push_t_wm_rsl`.

---

## Choosing a stack

| Question | Prefer |
|----------|--------|
| Interactive / replan online, clear xy cost? | **WM + CEM** |
| Have mocap / style refs, high DoF? | **WM + BPTT** |
| Underactuated contact, CEM fails, need exploration? | **WM + PPO** |
| Want one latent front-end for several controllers? | Train StateLeWM once; swap (1)–(3) |

---

## Roadmap

1. **Unify task API** — **done.** `learning/tasks/base.py` (`FlatWorldTask`:
   observe / sample_goal / potential Φ / metric / succeeded) +
   `adapters.py` (CEM / PPO / Mimic helpers). CEM tasks via `CEMWorldTask` /
   `PushToGoalTask`; Push-T overrides `potential` to match RL progress reward.
2. **Push-T follow-ups** — multi-goal T, friction / soft T, CEM residual on PPO; revisit imagination RL only with calibrated long-horizon WM losses.
3. **New CEM scenes** — variable-N rearrange, height-field / soft push with the same recipe.
4. **SuperMimic breadth** — more clips, contact-rich fight / getup stress tests, WM–sim gap dashboards.
5. **Productization** — **done.** Push-T **You vs AI** split-screen
   (`python -m game --mode pusht`: WASD left, WM+PPO right, 3s countdown,
   first to seat the T wins). Promo helper:
   `python -m learning.render_pusht_promo`. Paradigm write-up = this file;
   full manuscript is local-only (arXiv cs.RO by author).
6. **Engine** — **done (isolation)** / **open (mesh).** Parallel env +
   `env_id` documented in [`ENGINE_PARALLEL.md`](ENGINE_PARALLEL.md) with
   `test2D/test_env_id_isolation.py`. Mesh contact batching remains on the
   backlog in [`improvement.md`](../improvement.md).

**Cleanup note.** Failed imagination-PPO experiment entrypoints were removed
from `learning/`; the takeaway stays here: pure imagination RL on Push-T
drifted (~8%), so shipped Push-T uses **physics-loop PPO** with a frozen WM
encoder.

## Content backlog (P5)

| track | next demos |
|-------|------------|
| Push-T variants | multi-goal T pose, friction / soft-T, CEM residual on PPO |
| New CEM scenes | variable-N rearrange; height-field / soft push with the same StateLeWM recipe |
| SuperMimic | more clips; contact-rich fight / getup stress; WM–sim gap dashboards |

These do not block shipping; track progress in this file.
