"""Summarize task_eval.json episodes (success / failure breakdown)."""

import json

d = json.load(open("learning/results/task_eval.json"))
for e in d["episodes"]:
    tag = "OK  " if e["success"] else "FAIL"
    print(f"{tag} ep tgt={e['target_idx']} init_d={e['init_dist']:.3f} "
          f"final_d={e['final_dist']:.3f} settle={e['settle_frame']:3d} "
          f"goal={tuple(round(g, 3) for g in e['goal'])}")
