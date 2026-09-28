import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from extract import pool_hidden

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

# batch=2, seq=4, dim=3. Row 0 has 2 real tokens, row 1 has 4.
h = np.array([
    [[1.,1.,1.], [3.,3.,3.], [99.,99.,99.], [99.,99.,99.]],
    [[2.,0.,0.], [2.,0.,0.], [2.,0.,0.],    [6.,0.,0.]],
])
m = np.array([[1,1,0,0],
              [1,1,1,1]])

mean = pool_hidden(h, m, "mean")
check("mean ignores padding (row0 == 2.0 not 50.5)", np.allclose(mean[0], [2.,2.,2.]),
      f"got {mean[0]}")
check("mean correct on fully-real row", np.allclose(mean[1], [3.,0.,0.]), f"got {mean[1]}")

last = pool_hidden(h, m, "last")
check("last picks final REAL token (row0 -> idx1)", np.allclose(last[0], [3.,3.,3.]),
      f"got {last[0]}")
check("last picks final token (row1 -> idx3)", np.allclose(last[1], [6.,0.,0.]),
      f"got {last[1]}")

cls = pool_hidden(h, m, "cls")
check("cls picks position 0", np.allclose(cls[0], [1.,1.,1.]) and np.allclose(cls[1], [2.,0.,0.]))

# Left padding: row 0 has 2 real tokens at the END.
h_left = np.array([
    [[99.,99.,99.], [99.,99.,99.], [1.,1.,1.], [3.,3.,3.]],
    [[2.,0.,0.],    [2.,0.,0.],    [2.,0.,0.], [6.,0.,0.]],
])
m_left = np.array([[0,0,1,1],[1,1,1,1]])
lastL = pool_hidden(h_left, m_left, "last")
check("last correct under LEFT padding", np.allclose(lastL[0], [3.,3.,3.]), f"got {lastL[0]}")
meanL = pool_hidden(h_left, m_left, "mean")
check("mean correct under LEFT padding", np.allclose(meanL[0], [2.,2.,2.]), f"got {meanL[0]}")
old_idx = m_left.sum(axis=1) - 1
check("old sum-1 index would have been WRONG under left padding (bug guarded)",
      not np.allclose(h_left[0, old_idx[0]], [3.,3.,3.]))

# The failure this guards against: naive mean over ALL positions
naive = h.mean(axis=1)
check("naive mean differs from masked mean (bug this prevents)",
      not np.allclose(naive[0], mean[0]), f"naive={naive[0]} masked={mean[0]}")

# Shape and error handling
check("output shape is (batch, dim)", mean.shape == (2,3), str(mean.shape))
for bad, why in [((np.zeros((2,4)), m), "2-D hidden"),
                 ((h, np.zeros((3,4))), "mask/hidden mismatch"),
                 ((h, np.zeros((2,4))), "all-zero mask")]:
    try:
        pool_hidden(*bad, "mean"); ok = False
    except ValueError:
        ok = True
    check(f"raises on {why}", ok)

try:
    pool_hidden(h, m, "nonsense"); ok = False
except ValueError:
    ok = True
check("raises on unknown strategy", ok)

print()
nf = sum(1 for _,ok,_ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
