"""Quick sanity check of collected rollout data (v2 balance probe)."""

import glob

import numpy as np

fs = sorted(glob.glob("learning/data/rollouts_v2/*.npz"))
A = np.concatenate([np.load(f)["actions"] for f in fs])
S = np.concatenate([np.load(f)["obj_states"] for f in fs])
print("files", len(fs))
print("action mean", A.mean(0).round(3), "std", A.std(0).round(3))
print("Fx>1 frac", float((A[:, 0] > 1).mean().__round__(3)),
      "Fx<-1 frac", float((A[:, 0] < -1).mean().__round__(3)))

# EE displacement per 5-frame chunk vs mean action in the chunk (raw physics
# sanity: positive Fx should produce positive EE dx on average)
stride = 5
n_chunk = len(A) // stride
ax = A[: n_chunk * stride].reshape(n_chunk, stride, 2).mean(axis=1)[:, 0]
starts = S[::stride][:n_chunk]
ends = S[stride::stride][:n_chunk]
dx = ends[:, 0, 0] - starts[:, 0, 0]
m = (ax > 2) | (ax < -2)
if m.sum() > 10:
    c = np.corrcoef(ax[m], dx[m])[0, 1]
    print(f"chunk-level corr(Fx, EE dx) = {c:.3f}  (n={int(m.sum())})")
    print("mean EE dx | Fx>+2:", float(dx[m & (ax > 2)].mean().__round__(4)),
          " | Fx<-2:", float(dx[m & (ax < -2)].mean().__round__(4)))

    # remove the velocity carry-over: the model sees v, so what it must
    # learn beyond v*dt is the force-driven part
    dt = 5.0 / 60.0
    v0 = starts[:, 0, 3]
    resid = dx - v0 * dt
    contact = np.concatenate([np.load(f)["contact_mask"].sum(axis=1) > 0
                              for f in fs])
    contact_chunks = contact[::stride][:n_chunk]
    for tag, msk in [("free", ~contact_chunks & m), ("contact", contact_chunks & m)]:
        if msk.sum() > 10:
            cc = np.corrcoef(ax[msk], resid[msk])[0, 1]
            print(f"[{tag:7s}] n={int(msk.sum()):4d} "
                  f"corr(Fx, dx - v*dt) = {cc:+.3f} "
                  f"mean resid Fx>+2: {resid[msk & (ax > 2)].mean():+.4f} "
                  f"Fx<-2: {resid[msk & (ax < -2)].mean():+.4f}")
