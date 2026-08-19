"""Check how often the EE grinds on the ground in the collected data."""

import glob

import numpy as np

fs = sorted(glob.glob("learning/data/rollouts_v2/*.npz"))
ys, fys = [], []
for f in fs:
    d = np.load(f)
    ys.append(d["obj_states"][:, 0, 1])
    fys.append(d["actions"][:, 1])
ys = np.concatenate(ys)
fys = np.concatenate(fys)
print(f"frames {len(ys)}")
print(f"EE y < 0.105 (ground contact): {(ys < 0.105).mean() * 100:.1f}%")
print(f"EE y < 0.115 (near ground):    {(ys < 0.115).mean() * 100:.1f}%")
print(f"Fy < -1: {(fys < -1).mean() * 100:.1f}%  Fy > +1: {(fys > 1).mean() * 100:.1f}%")
print(f"EE y percentiles:",
      {p: round(float(np.percentile(ys, p)), 3) for p in (5, 25, 50, 75, 95)})
