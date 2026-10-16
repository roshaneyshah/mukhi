"""Planted-structure tests for run_contrast, apply_bh and the
information / script-shape decomposition."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from analysis import run_contrast, apply_bh, decompose_punjabi_gap

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))


def planted(info_w, shape_w, n=250, d=48, L=4, seed=0):
    """G -> Gdev differs by an 'information' direction; Gdev -> S by a
    'shape' direction. Both constant across layers."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    v = rng.normal(size=d); v /= np.linalg.norm(v)
    sem = rng.normal(size=(n, d)) * 2
    G, Gd, S = (np.zeros((L, n, d)) for _ in range(3))
    for l in range(L):
        G[l] = sem + rng.normal(size=(n, d)) * 0.5
        Gd[l] = sem - info_w * u + rng.normal(size=(n, d)) * 0.5
        S[l] = sem - info_w * u + shape_w * v + rng.normal(size=(n, d)) * 0.5
    return G, Gd, S


def run_all(G, Gd, S, n):
    ids = [f"i{k}" for k in range(n)]
    kw = dict(n_perm_cka=60, n_perm_pd=60, seed=0)
    return {"pan_Guru|pan_Shah": run_contrast(G, S, ids, **kw),
            "pan_Guru|pan_Guru_devowel": run_contrast(G, Gd, ids, **kw),
            "pan_Guru_devowel|pan_Shah": run_contrast(Gd, S, ids, **kw)}

n = 250
# ---- gap is ALL information ---------------------------------------------
R1 = run_all(*planted(info_w=4.0, shape_w=0.0), n)
dec1 = decompose_punjabi_gap(R1)
info1 = np.mean([r["info"] for r in dec1]); shape1 = np.mean([r["shape"] for r in dec1])
check("info-only world: info displacement large", info1 > 0.3, f"{info1:.3f}")
check("info-only world: shape displacement ~0", abs(shape1) < 0.05, f"{shape1:.3f}")
check("info-only world: shape_share ~0",
      abs(np.nanmean([r['shape_share'] for r in dec1])) < 0.15,
      f"{np.nanmean([r['shape_share'] for r in dec1]):.3f}")

# ---- gap is ALL script shape --------------------------------------------
R2 = run_all(*planted(info_w=0.0, shape_w=4.0, seed=1), n)
dec2 = decompose_punjabi_gap(R2)
info2 = np.mean([r["info"] for r in dec2]); shape2 = np.mean([r["shape"] for r in dec2])
check("shape-only world: shape displacement large", shape2 > 0.3, f"{shape2:.3f}")
check("shape-only world: info displacement ~0", abs(info2) < 0.05, f"{info2:.3f}")
check("shape-only world: shape_share ~1",
      abs(np.nanmean([r['shape_share'] for r in dec2]) - 1) < 0.2,
      f"{np.nanmean([r['shape_share'] for r in dec2]):.3f}")

# ---- BH over the whole family --------------------------------------------
m = apply_bh(R1)
check("BH applied to every layer x contrast x statistic", m == 3 * 2 * 4, str(m))
nulls = [r for r in R1["pan_Guru_devowel|pan_Shah"]["paired"]]
check("BH: planted-null contrast not significant", not any(r["significant_bh"] for r in nulls),
      str([round(r["q_bh"], 3) for r in nulls]))
reals = [r for r in R1["pan_Guru|pan_Guru_devowel"]["paired"]]
check("BH: planted-real contrast significant at every layer",
      all(r["significant_bh"] for r in reals), str([round(r["q_bh"], 3) for r in reals]))

# ---- probe inside run_contrast -------------------------------------------
# Claim under test: the probe detects a real contrast (above chance AND above
# its control) at every layer. Its selectivity is modest (~0.18) because shared
# content variance dilutes it -- FINDING_02 again; the paired statistic reads
# the same contrast at ~0.56.
pr = R1["pan_Guru|pan_Guru_devowel"]["probe"]
check("probe detects a real contrast at every layer",
      all(r["real_acc"] > 0.7 and r["selectivity"] > 0.1 for r in pr),
      " ".join(f"L{r['layer']}:{r['real_acc']:.2f}/{r['selectivity']:.2f}" for r in pr))

# ---- guards ----------------------------------------------------------------
G, Gd, S = planted(1.0, 1.0)
try:
    run_contrast(G, S[:, :10], [f"i{k}" for k in range(n)]); ok = False
except ValueError:
    ok = True
check("run_contrast rejects misaligned conditions", ok)
check("decompose returns None when a contrast is missing",
      decompose_punjabi_gap({"pan_Guru|pan_Shah": R1["pan_Guru|pan_Shah"]}) is None)

print()
nf = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
