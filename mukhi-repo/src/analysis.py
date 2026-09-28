"""
Layerwise orchestration: CKA curves and probe/selectivity curves.

Multiple comparisons matter here and are easy to forget. A single model with
33 layers, four contrasts and two probe targets is already ~264 tests. With
seven models it is four figures' worth. `benjamini_hochberg` is provided and
the runner applies it across the whole layer sweep, not per layer.
"""
import numpy as np

from cka import cka_with_baseline
from probes import (make_control_labels, probe_with_control,
                    paired_script_displacement, layer_norm_stats)


def benjamini_hochberg(pvals, alpha=0.05):
    """Returns (rejected_mask, qvalues). Standard BH step-up."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], bool), np.array([])

    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]     # enforce monotonicity
    q = np.clip(q, 0, 1)

    out_q = np.empty(n, float)
    out_q[order] = q
    return out_q <= alpha, out_q


def layerwise_cka(acts_a, acts_b, estimator="unbiased", n_permutations=200, seed=0):
    """
    acts_a, acts_b: (n_layers, n_items, dim), rows aligned across the two.
    Returns list of per-layer dicts from cka_with_baseline.
    """
    if acts_a.shape[0] != acts_b.shape[0]:
        raise ValueError(f"layer count mismatch: {acts_a.shape[0]} vs {acts_b.shape[0]}")
    if acts_a.shape[1] != acts_b.shape[1]:
        raise ValueError(f"item count mismatch: {acts_a.shape[1]} vs {acts_b.shape[1]}")

    out = []
    for li in range(acts_a.shape[0]):
        r = cka_with_baseline(np.asarray(acts_a[li], np.float64),
                              np.asarray(acts_b[li], np.float64), estimator=estimator,
                              n_permutations=n_permutations, seed=seed + li)
        r["layer"] = li
        out.append(r)
    return out


def layerwise_probe(acts_by_condition, labels, pair_ids, C=1.0, n_splits=5,
                    seed=0, control_seed=0):
    """
    acts_by_condition: list of (n_layers, n_items, dim) arrays, one per
        condition, all sharing the same item order.
    labels: length len(conditions)*n_items, the real target (script or language)
    pair_ids: same length; items that are the SAME SENTENCE in different
        conditions must share a pair_id so the control task does not leak.

    Control labels are drawn ONCE here and reused for every layer.
    """
    n_layers = acts_by_condition[0].shape[0]
    for a in acts_by_condition:
        if a.shape[0] != n_layers:
            raise ValueError("conditions disagree on layer count")

    labels = np.asarray(labels)
    control, _ = make_control_labels(pair_ids, labels, seed=control_seed)

    out = []
    for li in range(n_layers):
        X = np.concatenate([np.asarray(a[li], np.float64) for a in acts_by_condition], axis=0)
        if X.shape[0] != len(labels):
            raise ValueError(f"layer {li}: {X.shape[0]} rows vs {len(labels)} labels")
        r = probe_with_control(X, labels, control, C=C, n_splits=n_splits, seed=seed)
        r["layer"] = li
        out.append(r)
    return out


def crossover_layer(cka_curve=None, script_sel=None, language_sel=None):
    """
    The proposal's headline question: at what depth does representation stop
    tracking orthography and start tracking language?

    Operationalised as the first layer where language selectivity exceeds
    script selectivity. Returns None if it never crosses -- which for a
    low-resource script is itself the result, and is the finding the proposal
    anticipates ("and for lower-resource languages, does it ever?").
    """
    if script_sel is None or language_sel is None:
        return None
    s = np.asarray(script_sel, float)
    l = np.asarray(language_sel, float)
    n = min(len(s), len(l))
    for i in range(n):
        if np.isfinite(s[i]) and np.isfinite(l[i]) and l[i] > s[i]:
            return i
    return None


def summarise_run(cka_layers, script_layers, language_layers):
    """Collapse the three curves into the numbers that go in the abstract."""
    cka_delta = [r["delta"] for r in cka_layers]
    s_sel = [r["selectivity"] for r in script_layers]
    l_sel = [r["selectivity"] for r in language_layers]

    return {
        "n_layers": len(cka_delta),
        "cka_delta_max": float(np.nanmax(cka_delta)),
        "cka_delta_argmax": int(np.nanargmax(cka_delta)),
        "cka_delta_final": float(cka_delta[-1]),
        "script_sel_max": float(np.nanmax(s_sel)),
        "script_sel_argmax": int(np.nanargmax(s_sel)),
        "script_sel_final": float(s_sel[-1]),
        "language_sel_max": float(np.nanmax(l_sel)),
        "language_sel_argmax": int(np.nanargmax(l_sel)),
        "crossover_layer": crossover_layer(script_sel=s_sel, language_sel=l_sel),
    }


def layerwise_paired_displacement(acts_a, acts_b, n_permutations=200, seed=0,
                                  normalise=True):
    """
    Paired script-displacement consistency per layer, with a SIGN-FLIP null.

    Choice of null matters and the obvious choice is wrong. Row-permuting B --
    the null used for CKA -- does NOT work here: a consistent script
    displacement is a constant offset, and a constant offset survives row
    permutation intact, so the "baseline" tracks the signal and delta collapses
    to zero. Verified in-container.

    The correct null flips the sign of each delta_i independently. That
    destroys any shared direction while preserving every magnitude, and it
    agrees with the analytic floor for isotropic noise, E[consistency] =
    1/sqrt(n), which is reported as `analytic_floor` as a cross-check.
    """
    if acts_a.shape != acts_b.shape:
        raise ValueError(f"shape mismatch: {acts_a.shape} vs {acts_b.shape}")

    rng = np.random.default_rng(seed)
    n_items = acts_a.shape[1]
    out = []

    for li in range(acts_a.shape[0]):
        A = np.asarray(acts_a[li], np.float64)
        B = np.asarray(acts_b[li], np.float64)
        real = paired_script_displacement(A, B, normalise=normalise)

        if normalise:
            na = np.linalg.norm(A, axis=1, keepdims=True); na[na == 0] = 1.0
            nb = np.linalg.norm(B, axis=1, keepdims=True); nb[nb == 0] = 1.0
            delta = B / nb - A / na
        else:
            delta = B - A

        rms = np.sqrt((delta ** 2).sum(axis=1).mean())
        base = []
        for _ in range(n_permutations):
            signs = rng.choice([-1.0, 1.0], size=(delta.shape[0], 1))
            md = (delta * signs).mean(axis=0)
            base.append(np.linalg.norm(md) / rms if rms > 0 else np.nan)

        base = np.array(base, float)
        bm = float(np.nanmean(base))
        bs = float(np.nanstd(base, ddof=1)) if len(base) > 1 else 0.0

        out.append({
            "layer": li,
            "consistency": real["consistency"],
            "pc1_var": real["pc1_var"],
            "baseline_mean": bm,
            "baseline_std": bs,
            "analytic_floor": float(1.0 / np.sqrt(n_items)),
            "delta": real["consistency"] - bm,
            "z": float((real["consistency"] - bm) / bs) if bs > 0 else np.nan,
            "p_perm": float((1 + np.sum(base >= real["consistency"])) / (1 + len(base))),
            "mean_norm_a": layer_norm_stats(A)["mean_norm"],
            "mean_norm_b": layer_norm_stats(B)["mean_norm"],
        })
    return out


def run_contrast(acts_a, acts_b, item_ids, C=1.0, n_perm_cka=200, n_perm_pd=200,
                 seed=0, sweep_C=False):
    """
    Everything for one contrast between two aligned conditions:
    layerwise CKA, a probe discriminating the two conditions (with a control
    task keyed to item_ids), and paired displacement.

    The probe label is simply "which condition", so the same function serves a
    script contrast, a language contrast and every control.
    """
    n = acts_a.shape[1]
    if acts_b.shape[1] != n or len(item_ids) != n:
        raise ValueError("conditions and item_ids must be aligned")
    y = np.array([0] * n + [1] * n)
    pid = list(item_ids) * 2

    out = {
        "cka": layerwise_cka(acts_a, acts_b, n_permutations=n_perm_cka, seed=seed),
        "probe": layerwise_probe([acts_a, acts_b], y, pid, C=C, seed=seed),
        "paired": layerwise_paired_displacement(acts_a, acts_b,
                                                n_permutations=n_perm_pd, seed=seed),
    }
    if sweep_C:
        from probes import make_control_labels, sweep_regularisation
        ctrl, _ = make_control_labels(pid, y, seed=0)
        out["C_sweep"] = []
        for li in range(acts_a.shape[0]):
            X = np.concatenate([np.asarray(acts_a[li], np.float64),
                                np.asarray(acts_b[li], np.float64)])
            out["C_sweep"].append({"layer": li,
                                   "results": sweep_regularisation(X, y, ctrl, seed=seed)})
    return out


def apply_bh(contrast_results, alpha=0.05):
    """
    Benjamini-Hochberg across EVERY layer x contrast x statistic in one
    family, in place. Adds q and significant fields to each layer record.
    Correcting per layer or per contrast would understate the family size.
    """
    refs = []
    for cname, res in contrast_results.items():
        for stat in ("cka", "paired"):
            for rec in res.get(stat, []):
                if np.isfinite(rec.get("p_perm", np.nan)):
                    refs.append(rec)
    if not refs:
        return 0
    rej, q = benjamini_hochberg([r["p_perm"] for r in refs], alpha=alpha)
    for r, rj, qq in zip(refs, rej, q):
        r["q_bh"] = float(qq)
        r["significant_bh"] = bool(rj)
    return len(refs)


def decompose_punjabi_gap(results, stat="paired"):
    """
    The information / script-shape decomposition that FINDING_04 enables.
    Per layer, reports the paired-displacement delta for
        total  : pan_Guru vs pan_Shah
        info   : pan_Guru vs pan_Guru_devowel
        shape  : pan_Guru_devowel vs pan_Shah
    Displacements are vectors, so info + shape need not equal total; the
    function reports all three and the ratio shape/total, which is the
    quantity of interest: how much of the script gap survives once
    information is matched.
    """
    keys = {"total": "pan_Guru|pan_Shah", "info": "pan_Guru|pan_Guru_devowel",
            "shape": "pan_Guru_devowel|pan_Shah"}
    if not all(k in results for k in keys.values()):
        return None
    L = len(results[keys["total"]][stat])
    rows = []
    for li in range(L):
        t = results[keys["total"]][stat][li]["delta"]
        i = results[keys["info"]][stat][li]["delta"]
        sh = results[keys["shape"]][stat][li]["delta"]
        rows.append({"layer": li, "total": t, "info": i, "shape": sh,
                     "shape_share": sh / t if t and np.isfinite(t) and abs(t) > 1e-9 else np.nan})
    return rows
