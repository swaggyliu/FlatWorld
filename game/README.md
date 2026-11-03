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
Both players use the saved evaluation's 0.12 m position tolerance for each T
part; the existing angle and settling-speed requirements still apply.
Web rounds default to 30 playable seconds (countdown and pauses excluded).
Settings → Round length accepts 10–180 seconds, saved on this device and applied
on the next rematch or connection. Both sides always share the same round clock.
Rendering reads noiseless solver poses, buffers 60 ms of snapshots for smooth
motion, and draws the welded T as one rigid shape; policy observations keep
their original sensor noise.
The T uses the original centered crossbar and dimensions: stem 0.08 × 0.28,
bar 0.32 × 0.08. Its goal origin is fixed at (1.50, 0.52), with zero rotation.
Starting positions and orientations use the original random reset on every
round. Both players receive identical starting poses and dynamics. `--seed`
reproduces the sequence of openings (default: 1000).
The web difficulty buttons are Easy / Normal / Hard. They change AI **decision
rate** (2 / 4 / 20 Hz, capped by the server tick) and **force scale**
(~0.32 / 0.50 / 1.0) on the same PPO checkpoint — Easy/Normal stay winnable;
Hard is full strength. Physics keeps advancing with the held action between
replans. The web server defaults to 20 Hz with one physics step per tick.
No ground horizon — top-down table feel. Bounce on the spirit is a jelly
squash/stretch aura (not a coil spring).
