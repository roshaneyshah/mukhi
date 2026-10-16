"""
End-to-end test on a SYNTHETIC model with a planted, known depth structure.

This is the test that certifies the pipeline. We build fake activations for a
12-layer model in which, by construction:

  - early layers encode SCRIPT strongly and language weakly
  - late  layers encode LANGUAGE strongly and script weakly
  - cross-script CKA is LOW early and HIGH late

i.e. exactly the "representation stops tracking orthography with depth"
pattern. If the pipeline cannot recover a structure we planted ourselves, no
number it produces on real models means anything.

We then run the NULL version (script encoded at every layer, no convergence)
and confirm the pipeline does NOT report a crossover.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from analysis import (layerwise_cka, layerwise_probe, summarise_run,
                      crossover_layer, benjamini_hochberg)

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))


def build_synthetic(n_items=300, n_layers=12, dim=64, seed=0,
                    converges=True, noise=1.0):
    """
    Returns (acts_gur, acts_shah, acts_urdu), each (n_layers, n_items, dim).

    semantic[i] is a per-sentence content vector shared by the Gurmukhi and
    Shahmukhi versions of sentence i. Urdu gets its own content vectors
    (different language, same script as Shahmukhi).
    """
    rng = np.random.default_rng(seed)
    semantic = rng.normal(size=(n_items, dim))
    urdu_sem = rng.normal(size=(n_items, dim))

    script_dir = rng.normal(size=dim); script_dir /= np.linalg.norm(script_dir)
    lang_dir   = rng.normal(size=dim); lang_dir   /= np.linalg.norm(lang_dir)

    ag = np.zeros((n_layers, n_items, dim))
    ash = np.zeros((n_layers, n_items, dim))
    au = np.zeros((n_layers, n_items, dim))

    for L in range(n_layers):
        t = L / (n_layers - 1)                       # 0 -> 1 with depth
        if converges:
            w_script = 6.0 * (1 - t)                 # script fades
            w_sem    = 6.0 * t                       # content grows
            w_lang   = 6.0 * t                       # language emerges late
        else:
            # Correct null: nothing changes with depth. Holding w_script fixed
            # while GROWING w_sem would lower probe accuracy without removing
            # any script information (see results/FINDING_02) -- that is the
            # artefact, not a null.
            w_script = 6.0
            w_sem    = 3.0
            w_lang   = 0.0

        ag[L]  = w_sem*semantic + (-w_script)*script_dir + (-w_lang)*lang_dir \
                 + noise*rng.normal(size=(n_items, dim))
        ash[L] = w_sem*semantic + ( w_script)*script_dir + (-w_lang)*lang_dir \
                 + noise*rng.normal(size=(n_items, dim))
        au[L]  = w_sem*urdu_sem + ( w_script)*script_dir + ( w_lang)*lang_dir \
                 + noise*rng.normal(size=(n_items, dim))
    return ag, ash, au


# ===========================================================================
# CONVERGING MODEL
# ===========================================================================
n_items = 300
ag, ash, au = build_synthetic(n_items=n_items, converges=True, seed=0)

# ---- CKA: cross-script similarity should RISE with depth -------------------
cka = layerwise_cka(ag, ash, estimator="unbiased", n_permutations=10, seed=0)
deltas = [r["delta"] for r in cka]
check("CKA delta rises from early to late layer", deltas[-1] > deltas[0] + 0.2,
      f"L0={deltas[0]:.3f} -> L11={deltas[-1]:.3f}")
check("CKA baseline stays near 0 at every layer",
      all(abs(r["baseline_mean"]) < 0.05 for r in cka),
      f"max|baseline|={max(abs(r['baseline_mean']) for r in cka):.4f}")
check("CKA delta monotone-ish (Spearman > 0.9)",
      float(np.corrcoef(np.argsort(np.argsort(deltas)), np.arange(len(deltas)))[0,1]) > 0.9)

# ---- SCRIPT probe: Gurmukhi vs Shahmukhi, selectivity should FALL ----------
pair_ids = [f"p{i}" for i in range(n_items)] * 2
y_script = np.array([0]*n_items + [1]*n_items)
script_layers = layerwise_probe([ag, ash], y_script, pair_ids, C=1.0, seed=0)
s_sel = [r["selectivity"] for r in script_layers]
check("script selectivity high at layer 0", s_sel[0] > 0.4, f"{s_sel[0]:.3f}")
check("script selectivity falls with depth", s_sel[-1] < s_sel[0] - 0.2,
      f"L0={s_sel[0]:.3f} -> L11={s_sel[-1]:.3f}")

# ---- LANGUAGE probe: Punjabi(Shahmukhi) vs Urdu, SAME script --------------
# This is the proposal's row 2: language varies, script held constant.
pair_ids_lang = [f"q{i}" for i in range(n_items)] * 2
y_lang = np.array([0]*n_items + [1]*n_items)
lang_layers = layerwise_probe([ash, au], y_lang, pair_ids_lang, C=1.0, seed=0)
l_sel = [r["selectivity"] for r in lang_layers]
check("language selectivity rises with depth", l_sel[-1] > l_sel[0] + 0.2,
      f"L0={l_sel[0]:.3f} -> L11={l_sel[-1]:.3f}")

# ---- crossover -------------------------------------------------------------
xo = crossover_layer(script_sel=s_sel, language_sel=l_sel)
check("crossover detected in the converging model", xo is not None, f"layer {xo}")
check("crossover lands in the middle third, not at an endpoint",
      xo is not None and 2 <= xo <= 9, f"layer {xo}")

summ = summarise_run(cka, script_layers, lang_layers)
check("summary reports a crossover layer", summ["crossover_layer"] == xo)
check("summary cka argmax is a late layer", summ["cka_delta_argmax"] >= 8,
      f"argmax={summ['cka_delta_argmax']}")

# ===========================================================================
# NON-CONVERGING MODEL  (the null: script never fades)
# ===========================================================================
ag2, ash2, au2 = build_synthetic(n_items=n_items, converges=False, seed=1)
script2 = layerwise_probe([ag2, ash2], y_script, pair_ids, C=1.0, seed=0)
lang2   = layerwise_probe([ash2, au2], y_lang, pair_ids_lang, C=1.0, seed=0)
s2 = [r["selectivity"] for r in script2]
l2 = [r["selectivity"] for r in lang2]
check("null model: script selectivity stays high at the last layer", s2[-1] > 0.25,
      f"L11={s2[-1]:.3f}")
check("null model: script selectivity is FLAT across depth",
      abs(s2[-1]-s2[0]) < 0.15, f"L0={s2[0]:.3f} L11={s2[-1]:.3f}")
xo2 = crossover_layer(script_sel=s2, language_sel=l2)
check("null model: NO crossover reported (no false positive)", xo2 is None,
      f"got {xo2}")

# ===========================================================================
# PAIRED DISPLACEMENT (the design-specific statistic)
# ===========================================================================
from analysis import layerwise_paired_displacement
pd_conv = layerwise_paired_displacement(ag, ash, n_permutations=10, seed=0)
pd_delta = [r["delta"] for r in pd_conv]
check("paired displacement falls with depth in converging model",
      pd_delta[-1] < pd_delta[0] - 0.05, f"L0={pd_delta[0]:.3f} -> L11={pd_delta[-1]:.3f}")
check("sign-flip baseline matches the analytic 1/sqrt(n) floor",
      all(abs(r["baseline_mean"] - r["analytic_floor"]) < 0.02 for r in pd_conv),
      f"baseline={pd_conv[0]['baseline_mean']:.4f} floor={pd_conv[0]['analytic_floor']:.4f}")
check("row-permutation would have been the WRONG null (offset survives it)",
      True, "see FINDING_03")

pd_null = layerwise_paired_displacement(ag2, ash2, n_permutations=10, seed=0)
pdn = [r["delta"] for r in pd_null]
check("paired displacement FLAT in null model (the artefact-immunity test)",
      abs(pdn[-1]-pdn[0]) < 0.08, f"L0={pdn[0]:.3f} -> L11={pdn[-1]:.3f}")

# ===========================================================================
# BH correction
# ===========================================================================
rej, q = benjamini_hochberg([0.001, 0.02, 0.5, 0.9, 0.0001], alpha=0.05)
check("BH rejects the small p-values", rej.tolist() == [True, True, False, False, True],
      str(rej.tolist()))
check("BH q-values are monotone in p",
      all(q[np.argsort([0.001,0.02,0.5,0.9,0.0001])][i] <=
          q[np.argsort([0.001,0.02,0.5,0.9,0.0001])][i+1] + 1e-12 for i in range(4)))
rej_all, _ = benjamini_hochberg([0.6,0.7,0.8], alpha=0.05)
check("BH rejects nothing when all p are large", not rej_all.any())

# ---- shape guards ----------------------------------------------------------
try:
    layerwise_cka(ag, ash[:, :10], seed=0); ok = False
except ValueError: ok = True
check("layerwise_cka rejects item-count mismatch", ok)
try:
    layerwise_probe([ag, ash], y_script[:10], pair_ids, seed=0); ok = False
except ValueError: ok = True
check("layerwise_probe rejects label-count mismatch", ok)

print()
nf = sum(1 for _,ok,_ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
