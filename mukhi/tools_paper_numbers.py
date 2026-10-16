"""Recompute every headline number the paper quotes, from the result files.

The paper's §5.2 and §5.4 figures were originally computed by hand, so there
was nothing to re-run when the NFC repair (Finding 9) changed the underlying
activations. This script is that missing stage. It writes
results/paper_numbers.json, and tests/test_paper_numbers.py asserts that the
draft's prose agrees with it.

Convention, fixed here once so it cannot drift again:
  - the reported quantity is trained consistency minus random-init
    consistency, at the same layer and contrast;
  - where several random-init seeds exist they are averaged, and the spread
    is reported alongside;
  - the trough is the minimum of that difference over layers.

Usage:
    python tools_paper_numbers.py
"""
import glob
import json
import os
import statistics as st
from datetime import date

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Paper short name -> result file stem.
MODELS = {
    "xlm-roberta-base": "xlm-roberta-base",
    "xlm-roberta-large": "xlm-roberta-large",
    "muril-base-cased": "muril-base-cased",
}
CONTRASTS = {
    "punjabi": "pan_Guru|pan_Shah",
    "serbian": "srp_Cyrl|srp_Latn",
    "script_form": "pan_Guru_devowel|pan_Shah",
    "information": "pan_Guru|pan_Guru_devowel",
    "no_script": "pan_Shah_rom|urd_rom",
}


def _consistency(path, contrast):
    d = json.load(open(path))
    block = d["contrasts"].get(contrast)
    if block is None:
        return None
    return {x["layer"]: x["consistency"] for x in block["paired"]}


def _curve(stem, contrast):
    """Trained minus mean random-init, per layer, plus the seed spread."""
    trained = _consistency(os.path.join(RESULTS, "main_flores_%s.json" % stem), contrast)
    if trained is None:
        return None
    seeds = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "main_flores_%s_randinit*.json" % stem))):
        c = _consistency(f, contrast)
        if c is not None:
            seeds.append(c)
    if not seeds:
        return None
    layers = sorted(trained)
    diff = {}
    spread = {}
    for L in layers:
        per_seed = [trained[L] - s[L] for s in seeds if L in s]
        diff[L] = st.mean(per_seed)
        spread[L] = (max(per_seed) - min(per_seed)) if len(per_seed) > 1 else 0.0
    trough = min(diff, key=diff.get)
    return {
        "n_random_seeds": len(seeds),
        "embedding": diff[layers[0]],
        "trough": diff[trough],
        "trough_layer": trough,
        "trough_seed_spread": spread[trough],
        "final": diff[layers[-1]],
        "per_layer": {str(L): diff[L] for L in layers},
    }


def main():
    out = {
        "computed": str(date.today()),
        "script": "tools_paper_numbers.py",
        "statistic": "trained consistency minus mean random-init consistency",
        "models": {},
    }
    for name, stem in MODELS.items():
        out["models"][name] = {}
        for cname, contrast in CONTRASTS.items():
            c = _curve(stem, contrast)
            if c is not None:
                out["models"][name][cname] = c

    # Script-form share: how much of the Punjabi gap survives when
    # information is matched. Formed from trained-minus-random on both legs
    # (Finding 7), per layer, then averaged over layers.
    for name, stem in MODELS.items():
        m = out["models"][name]
        if "punjabi" not in m or "script_form" not in m:
            continue
        pj = m["punjabi"]["per_layer"]
        sf = m["script_form"]["per_layer"]
        keys = sorted(pj, key=int)
        ratios = [sf[k] / pj[k] for k in keys]
        m["script_form_ratio"] = {
            "mean": st.mean(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "embedding": ratios[0],
            "final": ratios[-1],
            "n_layers": len(ratios),
        }
        # The same ratio on raw displacements, with no baseline subtracted:
        # what a pipeline without random-init controls would report.
        raw_t = _consistency(os.path.join(RESULTS, "main_flores_%s.json" % stem),
                             CONTRASTS["punjabi"])
        raw_s = _consistency(os.path.join(RESULTS, "main_flores_%s.json" % stem),
                             CONTRASTS["script_form"])
        raw = [raw_s[L] / raw_t[L] for L in sorted(raw_t)]
        m["script_form_ratio_raw"] = {"mean": st.mean(raw), "min": min(raw), "max": max(raw)}
        # And on the random-init runs alone, to show the ratio is near 1
        # there regardless of training.
        rr = []
        for f in sorted(glob.glob(os.path.join(RESULTS, "main_flores_%s_randinit*.json" % stem))):
            a = _consistency(f, CONTRASTS["punjabi"])
            b = _consistency(f, CONTRASTS["script_form"])
            if a and b:
                rr.extend(b[L] / a[L] for L in sorted(a))
        if rr:
            m["script_form_ratio_randinit"] = {"min": min(rr), "max": max(rr)}

    # Tokeniser premium, Shahmukhi over Gurmukhi, tokens per character.
    fert = json.load(open(os.path.join(RESULTS, "fertility_flores.json")))["per_model"]
    out["fertility"] = {}
    for m, d in fert.items():
        pc = d.get("per_condition", {})
        if "pan_Guru" in pc and "pan_Shah" in pc:
            g = pc["pan_Guru"]["tokens_per_char_mean"]
            s = pc["pan_Shah"]["tokens_per_char_mean"]
            out["fertility"][m] = {
                "guru_tokens_per_char": g,
                "shah_tokens_per_char": s,
                "premium_pct": (s / g - 1.0) * 100.0,
                "ratio_guru_over_shah": g / s,
            }

    # Intervention, straight from the probe results.
    iv = json.load(open(os.path.join(RESULTS, "intervention_sib.json")))
    out["intervention"] = {
        m: {
            "best_layer": d["best_layer"],
            "acc": d["summary"]["acc"],
            "gap": d["summary"]["gap_guru_minus_shah"],
            "gain": d["summary"]["gain_restored_minus_shah"],
            "gain_ci95": d["summary"]["gain_ci95"],
            "gap_closed_frac": d["summary"]["gap_closed_frac"],
            "p_guru_vs_shah": d["summary"]["mcnemar_guru_vs_shah"][2],
            "p_shah_vs_restored": d["summary"]["mcnemar_shah_vs_restored"][2],
            "p_restored_vs_devowel": d["summary"]["mcnemar_restored_vs_devowel"][2],
        }
        for m, d in iv.items()
    }

    path = os.path.join(RESULTS, "paper_numbers.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # Human readable digest.
    for name, m in out["models"].items():
        for cname in ("punjabi", "serbian", "script_form", "no_script"):
            if cname in m:
                c = m[cname]
                print("%-18s %-12s emb=%.4f  trough=%.4f@L%-2d (spread %.4f)  final=%.4f  [%d seeds]"
                      % (name, cname, c["embedding"], c["trough"], c["trough_layer"],
                         c["trough_seed_spread"], c["final"], c["n_random_seeds"]))
        if "information" in m:
            c = m["information"]
            print("%-18s %-12s emb=%.4f  final=%.4f"
                  % (name, "information", c["embedding"], c["final"]))
        if "script_form_ratio" in m:
            r = m["script_form_ratio"]
            q = m["script_form_ratio_raw"]
            z = m.get("script_form_ratio_randinit", {})
            print("%-18s %-12s mean=%.3f [%.3f, %.3f]   raw mean=%.3f   randinit [%.2f, %.2f]"
                  % (name, "form_ratio", r["mean"], r["min"], r["max"], q["mean"],
                     z.get("min", float("nan")), z.get("max", float("nan"))))
    print()
    for m, d in out["fertility"].items():
        print("%-28s premium %+7.2f%%  ratio %.3f" % (m, d["premium_pct"], d["ratio_guru_over_shah"]))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
