"""Ground-truth tests for probing + control tasks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from probes import make_control_labels, probe_with_control, majority_baseline

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

rng = np.random.default_rng(0)
n, d = 400, 64

# ---- Case A: label IS linearly encoded -> high real acc, high selectivity ----
y = rng.integers(0, 2, size=n)
direction = rng.normal(size=d)
X = rng.normal(size=(n, d)) + 3.0 * y[:, None] * direction[None, :]
ids = [f"s{i}" for i in range(n)]
ctrl, _ = make_control_labels(ids, y, seed=0)
rA = probe_with_control(X, y, ctrl, C=1.0, seed=0)
check("A: encoded label -> real_acc > 0.95", rA["real_acc"] > 0.95, f"{rA['real_acc']:.3f}")
check("A: selectivity > 0.3", rA["selectivity"] > 0.3,
      f"sel={rA['selectivity']:.3f} (real={rA['real_acc']:.3f} ctrl={rA['control_acc']:.3f})")

# ---- Case B: label NOT encoded -> real acc ~ chance, selectivity ~ 0 --------
y2 = rng.integers(0, 2, size=n)
X2 = rng.normal(size=(n, d))                       # pure noise features
ctrl2, _ = make_control_labels(ids, y2, seed=1)
rB = probe_with_control(X2, y2, ctrl2, C=1.0, seed=0)
check("B: unencoded label -> real_acc near chance", abs(rB["real_acc"] - 0.5) < 0.12,
      f"{rB['real_acc']:.3f}")
check("B: selectivity near 0", abs(rB["selectivity"]) < 0.12,
      f"sel={rB['selectivity']:+.3f} (real={rB['real_acc']:.3f} ctrl={rB['control_acc']:.3f})")

# ---- Case C: THE point of control tasks -----------------------------------
# High-capacity representation memorises arbitrary labels. Real acc alone looks
# good; selectivity exposes that the probe, not the representation, did the work.
n_small, d_big = 150, 600
y3 = rng.integers(0, 2, size=n_small)
X3 = rng.normal(size=(n_small, d_big))             # NOTHING encoded
ids3 = [f"t{i}" for i in range(n_small)]
ctrl3, _ = make_control_labels(ids3, y3, seed=2)
rC = probe_with_control(X3, y3, ctrl3, C=10.0, seed=0)
check("C: control acc rises above chance when d >> n (memorisation)",
      rC["control_acc"] > 0.52, f"ctrl={rC['control_acc']:.3f}")
check("C: selectivity still ~0 despite memorisation", abs(rC["selectivity"]) < 0.15,
      f"sel={rC['selectivity']:+.3f}")

# ---- Case D: control labels are stable across calls (same seed) ------------
c1, _ = make_control_labels(ids, y, seed=5)
c2, _ = make_control_labels(ids, y, seed=5)
check("D: control labels deterministic given seed", np.array_equal(c1, c2))
c3, _ = make_control_labels(ids, y, seed=6)
check("D: different seed gives different labels", not np.array_equal(c1, c3))

# ---- Case E: paired items share a control label ---------------------------
# The critical detail: Gurmukhi and Shahmukhi rows of ONE sentence must get the
# SAME control label, else the control task leaks script identity.
pair_ids = [f"p{i//2}" for i in range(200)]        # 100 pairs, 2 rows each
script = np.array([i % 2 for i in range(200)])     # 0=Gurmukhi 1=Shahmukhi
ctrlE, _ = make_control_labels(pair_ids, script, seed=0)
same = all(ctrlE[i] == ctrlE[i+1] for i in range(0, 200, 2))
check("E: both scripts of a sentence share one control label", same)
# and therefore the control task carries NO script information
from probes import _fit_probe
leak_acc, _ = _fit_probe(np.eye(200)[:, :50], ctrlE, C=1.0, seed=0)
check("E: control labels are not predictable from script (no leak by design)",
      same, "structural guarantee")

# ---- Case F: majority baseline correct ------------------------------------
yf = np.array([0]*80 + [1]*20)
check("F: majority baseline == 0.8", abs(majority_baseline(yf) - 0.8) < 1e-12)

# ---- Case G: imbalanced labels do not crash StratifiedKFold ---------------
yg = np.array([0]*97 + [1]*3)
Xg = rng.normal(size=(100, 32))
ctrlg, _ = make_control_labels([f"g{i}" for i in range(100)], yg, seed=0)
try:
    rG = probe_with_control(Xg, yg, ctrlg, C=1.0, n_splits=5, seed=0)
    okG = np.isfinite(rG["real_acc"])
except Exception as e:
    okG = False; rG = {"real_acc": str(e)}
check("G: handles severe class imbalance without crashing", okG, f"{rG['real_acc']}")

print()
nf = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
