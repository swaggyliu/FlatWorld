"""Quick data inspection: box shape (square vs flat) and contact stats."""
import glob
import numpy as np

files = sorted(glob.glob("learning/data/rollouts/*.npz"))
print("n_rollouts:", len(files))

d = np.load(files[0])
print("keys:", list(d.keys()))
s = d["obj_states"]
print("obj_states shape:", s.shape)
# obj order: boxes first, then balls; y position reveals half-extent
print("y at t=0 :", s[0, :, 1].round(3).tolist())
print("y at end :", s[-1, :, 1].round(3).tolist())
print("theta end:", s[-1, :, 2].round(2).tolist())

# Tipping check across all rollouts: box |theta| exceeding ~35 deg
n_tip = 0
for f in files:
    d = np.load(f)
    th = d["obj_states"][:, :3, 2]  # first 3 objects are boxes
    if np.abs(th).max() > 0.6:
        n_tip += 1
print("rollouts with box |theta|>0.6 rad:", n_tip, "/", len(files))

# config stored inside the rollout
import json
cfg = json.loads(str(d["config_json"]))
scene = cfg.get("scene", cfg)
for k in ("box_ext", "restitution", "friction", "area_x", "first_gap",
          "dr_friction_range", "dr_mass_range", "gravity"):
    print(f"cfg {k}: {scene.get(k, '?')}")
