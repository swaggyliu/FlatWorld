"""Print a compact summary of learning/results/task_eval.json."""
import json

d = json.load(open("learning/results/task_eval.json"))
s = d["summary"]
print(f"rate {s['success_rate']:.3f}  n {s['n_success']}/{s['episodes']}  "
      f"mean_dist {s['mean_final_dist']:.3f}  mean_settle {s['mean_settle_frame']:.0f}")
for i, e in enumerate(d["episodes"]):
    tag = "ok  " if e["success"] else "FAIL"
    print(f"ep{i:02d} {tag} tgt {e['target_idx']} dist {e['final_dist']:.3f} "
          f"settle {e['settle_frame']:3d} goal_y {e['goal'][1]:.3f}")
