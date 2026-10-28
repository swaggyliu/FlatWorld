# Spirit Push / Push-T duel

```bash
# WM+CEM click-to-push campaign
python -m game

# You vs AI on Push-T (WASD left, WM-obs PPO right)
python -m game --mode pusht \
  --wm learning/checkpoints/push_t \
  --policy learning/checkpoints/push_t_wm_rsl/best.pt
```

Push-T duel: 3s countdown, first to seat the T wins, Space/R rematch.
No ground horizon — top-down table feel. Bounce on the spirit is a jelly
squash/stretch aura (not a coil spring).
