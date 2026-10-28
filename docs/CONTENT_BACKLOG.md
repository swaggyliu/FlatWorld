# Content backlog (after ship)

Short list for **P5** demos. Do not block releases on these.

## Push-T variants

- Multi-goal T poses (randomized stem/bar goals)
- Friction / soft-T material variants
- Optional CEM residual on top of frozen WM-obs PPO

Entry points today: `learning/tasks/push_t.py`, `learning/train_rsl_wm_push_t.py`,
`python -m game --mode pusht`.

## New CEM scenes

- Variable-N rearrange (same StateLeWM recipe as PushToGoal)
- Height-field / soft-body push with typed readouts

Scaffold: `learning/tasks/base.py` + `CEMWorldTask` / `adapters.py`.

## SuperMimic breadth

- More LAFAN-style clips
- Contact-rich fight / getup stress tests
- WM–sim gap dashboards next to `supermimic2d/runs/`

See also [`PARADIGMS.md`](PARADIGMS.md) roadmap.
