"""Quick sanity check of collected rollout data (v3 balance probe)."""

import glob

import numpy as np

fs = sorted(glob.glob("learning/data/rollouts/*.npz"))
A = np.concatenate([np.load(f)["actions"] for f in fs])
S = np.concatenate([np.load(f)["obj_states"] for f in fs])
print("files", len(fs))
print("action mean", A.mean(0).round(3), "std", A.std(0).round(3))
print("EE y range", float(S[:, 0, 1].min().__round__(3)), float(S[:, 0, 1].max().__round__(3)),
      "y>0.5 frac", float((S[:, 0, 1] > 0.5).mean().__round__(3)))

contact = np.concatenate([np.load(f)["contact_mask"].sum(axis=1) > 0
                          for f in fs])
print("contact frame frac", float(contact.mean().__round__(3)))

# EE displacement per 5-frame chunk vs mean action in the chunk (raw physics
# sanity: positive F should produce positive EE displacement on average)
stride = 5
n_chunk = len(A) // stride
a_chunk = A[: n_chunk * stride].reshape(n_chunk, stride, 2).mean(axis=1)
starts = S[::stride][:n_chunk]
ends = S[stride::stride][:n_chunk]
contact_chunks = contact[::stride][:n_chunk]
dt = 5.0 / 60.0

for k, name in [(0, "x"), (1, "y")]:
    a_k = a_chunk[:, k]
    d = ends[:, 0, k] - starts[:, 0, k]
    v0 = starts[:, 0, 3 + k]
    resid = d - v0 * dt
    m = (a_k > 2) | (a_k < -2)
    for tag, msk in [("free", ~contact_chunks & m), ("contact", contact_chunks & m)]:
        if msk.sum() > 10:
            cc = np.corrcoef(a_k[msk], resid[msk])[0, 1]
            print(f"[{name} {tag:7s}] n={int(msk.sum()):4d} "
                  f"corr(F{name}, d{name} - v*dt) = {cc:+.3f} "
                  f"mean resid F>+2: {resid[msk & (a_k > 2)].mean():+.4f} "
                  f"F<-2: {resid[msk & (a_k < -2)].mean():+.4f}")

# expected free-space residual: 0.5 * a * dt^2 * stride = 0.5*F/m*(1/12)^2*5
print("expected free resid for |F|=2 (m=1):", 0.5 * 2.0 * (dt / 5 * 5) ** 2 * 5)
