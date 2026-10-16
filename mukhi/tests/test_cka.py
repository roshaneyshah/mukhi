"""Ground-truth tests for the CKA estimators. These must all pass before any
number produced by this pipeline is trusted."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from cka import linear_cka, unbiased_linear_cka, cka_with_baseline, _center_columns

RTOL = 1e-8
results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

rng = np.random.default_rng(0)
n, p = 200, 64
X = rng.normal(size=(n, p))
Y = rng.normal(size=(n, p))

# --- 1. Self-similarity is exactly 1 -----------------------------------------
check("linear_cka(X,X) == 1", abs(linear_cka(X, X) - 1.0) < 1e-10,
      f"got {linear_cka(X,X):.12f}")
check("unbiased_cka(X,X) == 1", abs(unbiased_linear_cka(X, X) - 1.0) < 1e-10,
      f"got {unbiased_linear_cka(X,X):.12f}")

# --- 2. Symmetry --------------------------------------------------------------
check("linear_cka symmetric", abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-12)
check("unbiased_cka symmetric",
      abs(unbiased_linear_cka(X, Y) - unbiased_linear_cka(Y, X)) < 1e-12)

# --- 3. Invariance to orthogonal transform of either argument -----------------
Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
check("linear_cka invariant to orthogonal Q",
      abs(linear_cka(X, Y) - linear_cka(X, Y @ Q)) < 1e-10,
      f"{linear_cka(X,Y):.10f} vs {linear_cka(X, Y@Q):.10f}")
check("unbiased_cka invariant to orthogonal Q",
      abs(unbiased_linear_cka(X, Y) - unbiased_linear_cka(X, Y @ Q)) < 1e-10)

# --- 4. Invariance to isotropic scaling ---------------------------------------
check("linear_cka invariant to isotropic scale",
      abs(linear_cka(X, Y) - linear_cka(X, 7.3 * Y)) < 1e-10)
check("unbiased_cka invariant to isotropic scale",
      abs(unbiased_linear_cka(X, Y) - unbiased_linear_cka(X, 7.3 * Y)) < 1e-10)

# --- 5. Invariance to translation (this is what centering buys) ---------------
check("linear_cka invariant to translation",
      abs(linear_cka(X, Y) - linear_cka(X, Y + 5.0)) < 1e-10)

# --- 6. NOT invariant to arbitrary invertible transform (sanity: it's CKA,
#        not CCA). If this "passes" invariance, centering or the norm is wrong.
A = rng.normal(size=(p, p))
different = abs(linear_cka(X, Y) - linear_cka(X, Y @ A)) > 1e-3
check("linear_cka NOT invariant to arbitrary invertible A (correct behaviour)",
      different)

# --- 7. Known-signal recovery: Y = X rotated + noise should score high --------
Xs = rng.normal(size=(n, p))
Ys = Xs @ Q + 0.05 * rng.normal(size=(n, p))
hi = linear_cka(Xs, Ys)
check("rotated+low-noise copy scores > 0.95", hi > 0.95, f"got {hi:.4f}")

# --- 8. Independent matrices: unbiased estimator near 0, biased inflated ------
b = linear_cka(X, Y)
u = unbiased_linear_cka(X, Y)
check("biased estimator inflated on independent data (>0.1)", b > 0.1, f"biased={b:.4f}")
check("unbiased estimator near 0 on independent data (|u|<0.05)", abs(u) < 0.05,
      f"unbiased={u:.4f}")

# --- 9. The inflation grows with p/n -- the reason we default to unbiased -----
rows = []
for pp in [32, 128, 512, 2048]:
    Xa = rng.normal(size=(100, pp)); Ya = rng.normal(size=(100, pp))
    rows.append((pp, linear_cka(Xa, Ya), unbiased_linear_cka(Xa, Ya)))
mono = all(rows[i][1] <= rows[i+1][1] + 1e-9 for i in range(len(rows)-1))
check("biased CKA inflates monotonically with p at fixed n", mono,
      "  ".join(f"p={p_}:{bb:.3f}" for p_, bb, _ in rows))
check("unbiased CKA stays near 0 across all p",
      all(abs(uu) < 0.06 for _, _, uu in rows),
      "  ".join(f"p={p_}:{uu:+.3f}" for p_, _, uu in rows))

# --- 10. Permutation baseline machinery --------------------------------------
res_sig = cka_with_baseline(Xs, Ys, estimator="unbiased", n_permutations=30, seed=1)
check("baseline: real signal gives large positive delta", res_sig["delta"] > 0.5,
      f"delta={res_sig['delta']:.4f} z={res_sig['z']:.1f}")
res_null = cka_with_baseline(X, Y, estimator="unbiased", n_permutations=30, seed=1)
check("baseline: null data gives small delta", abs(res_null["delta"]) < 0.05,
      f"delta={res_null['delta']:.4f}")

# --- 11. Centering actually happens ------------------------------------------
Xc = _center_columns(X)
check("_center_columns zeroes the column means", np.allclose(Xc.mean(axis=0), 0, atol=1e-12))

# --- 12. Different feature widths are accepted (p1 != p2) --------------------
Y_narrow = rng.normal(size=(n, 16))
try:
    v = linear_cka(X, Y_narrow)
    ok = np.isfinite(v)
except Exception as e:
    ok = False; v = str(e)
check("handles p1 != p2", ok, f"got {v}")

print()
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)
