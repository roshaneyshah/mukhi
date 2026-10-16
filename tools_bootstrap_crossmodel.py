"""Cross-model bootstrap comparisons, from the saved per-draw differences.

The bootstrap stage saves each model's per-draw trained-minus-random
differences as results/bootstrap_draws_*.npz. Because every model is run with
the same --boot-seed, draw k is the same resampled sentences in every file, so
comparisons ACROSS models are paired too and need no GPU: the expensive part
was the extraction, and the draws are its reusable residue.

That is what makes this a separate step rather than a second pass of the
stage. Re-running the stage over all three models together would re-extract
every activation to recover a comparison the saved draws already contain.

Section 5.7's scale claim (XLM-R base against XLM-R large) is the comparison
this exists for. It also reports the Punjabi-against-Serbian ordering per
model, so every claim in the paper that compares two numbers has an interval.

Usage:
    python tools_bootstrap_crossmodel.py [--data flores]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# MUKHI_RESULTS lets the test suite point this at a temp directory, so a test
# run can never read or overwrite a real run's output.
RESULTS = os.environ.get("MUKHI_RESULTS") or os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

from bootstrap import summarise  # noqa: E402


# Statistic families in a draw file, by key prefix. consistency has none.
FAMILIES = {"": "consistency", "shared__": "shared", "rms__": "rms",
            "ratio__": "ratio"}


def load_draws(data):
    """Every saved draw file for this data set, keyed by model tag."""
    out = {}
    pat = os.path.join(RESULTS, "bootstrap_draws_%s_*.npz" % data)
    for path in sorted(glob.glob(pat)):
        tag = os.path.basename(path)[len("bootstrap_draws_%s_" % data):-len(".npz")]
        z = np.load(path)
        meta = ("counts_seed", "n_boot", "n_items")
        out[tag] = {"path": os.path.basename(path),
                    "counts_seed": int(z["counts_seed"][0]),
                    "n_boot": int(z["n_boot"][0]),
                    "n_items": z["n_items"] if "n_items" in z.files else None,
                    "draws": {k: z[k] for k in z.files
                              if k not in meta and not k.startswith("point__")},
                    "points": {k[len("point__"):]: z[k] for k in z.files
                               if k.startswith("point__")}}
    return out


def _trough(arr, points=None):
    """Trough layer from the FULL-SAMPLE point estimates, held fixed.

    The layer must be the one the paper quotes, which is the argmin of the
    full-sample estimates saved beside the draws. Selecting it from the mean
    over draws instead is a biased proxy and can land on a different layer,
    so the tool and the stage would then report intervals for different
    troughs. Re-minimising inside each draw would be worse still: that
    bootstraps a selected minimum and biases the interval downward.
    """
    if points is not None and len(points) == arr.shape[0]:
        li = int(np.nanargmin(np.asarray(points, dtype=float)))
    else:
        li = int(np.nanargmin(np.nanmean(arr, axis=1)))
    return li, arr[li]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="flores")
    args = ap.parse_args()

    files = load_draws(args.data)
    if not files:
        print("No bootstrap_draws_%s_*.npz in results/. Run "
              "--stage bootstrap first." % args.data)
        return 1
    print("loaded %d draw file(s):" % len(files))
    for tag, d in files.items():
        print("  %-22s seed=%d draws=%d contrasts=%s"
              % (tag, d["counts_seed"], d["n_boot"], sorted(d["draws"])))

    # Pairing across models is only valid if every file used the same
    # resample. Refuse rather than quietly compare incomparable draws.
    # The resample is reproduced from a seed in each invocation rather than
    # passed between them, so it is identical only when the seed, the draw
    # count AND the sentence count match. Guarding on the first two alone
    # would let two runs over different corpora be compared as if paired.
    seeds = {d["counts_seed"] for d in files.values()}
    nboots = {d["n_boot"] for d in files.values()}
    # Draw files written before n_items was recorded simply cannot be checked
    # on it; the guard applies to those that carry it.
    ns = {int(d["n_items"][0]) for d in files.values()
          if d.get("n_items") is not None}
    if len(ns) > 1:
        print("\nREFUSING to compare across models: the draw files were built "
              "on different sentence counts %s, so draw k is not the same "
              "resample in each." % sorted(ns))
        return 1
    if len(seeds) > 1 or len(nboots) > 1:
        print("\nREFUSING to compare across models: the draw files do not "
              "share one resample (seeds %s, draw counts %s). Re-run the "
              "stage with the same --boot-seed and --n-boot for every model."
              % (sorted(seeds), sorted(nboots)))
        return 1
    nmsg = (", n=%d" % ns.pop()) if len(ns) == 1 else \
        " (sentence count not recorded in these files)"
    print("\nall files share resample seed %d, %d draws%s: comparisons are paired"
          % (seeds.pop(), nboots.pop(), nmsg))

    out = {"data": args.data, "source_files": {t: d["path"] for t, d in files.items()},
           "per_model": {}, "across_models": {}}

    # Per model: the Punjabi trough against the Serbian trough.
    for tag, d in files.items():
        dr = d["draws"]
        pts = d.get("points", {})
        if not pts:
            print("  %s: no saved point estimates; trough layers fall back to "
                  "the draw means, which may differ from the paper's" % tag)
        per = {}
        for lab, arr in dr.items():
            li, col = _trough(arr, pts.get(lab))
            pv = float(pts[lab][li]) if lab in pts else None
            per[lab] = {"trough_layer": li, **summarise(col, point=pv)}
        # The ordering, on every statistic present: consistency (no prefix),
        # the two magnitudes, and the trained/random ratio. A claim the paper
        # keeps must hold on more than the one statistic it was found on.
        for pre in FAMILIES:
            P, S = pre + "punjabi", pre + "serbian"
            if P not in dr or S not in dr:
                continue
            pl, pc = _trough(dr[P], pts.get(P))
            sl, sc = _trough(dr[S], pts.get(S))
            pt = (float(pts[P][pl]) - float(pts[S][sl])
                  if P in pts and S in pts else None)
            key = pre + "punjabi_minus_serbian_trough"
            per[key] = {"punjabi_layer": pl, "serbian_layer": sl,
                        **summarise(pc - sc, point=pt)}
            t = per[key]
            print("%s %-11s Punjabi trough (L%d) minus Serbian trough (L%d) = "
                  "%+.4f  CI [%+.4f, %+.4f]  p=%.2g"
                  % (tag, FAMILIES[pre], pl, sl, t["point"], t["ci95"][0],
                     t["ci95"][1], t["p_boot"]))
        out["per_model"][tag] = per

    # Across models: Section 5.7's scale claim.
    pairs = [("xlm-roberta-base", "xlm-roberta-large"),
             ("xlm-roberta-large", "xlm-roberta-xl"),
             ("xlm-roberta-base", "xlm-roberta-xl")]
    print()
    for a, b in pairs:
        if a not in files or b not in files:
            print("scale comparison skipped: need both %s and %s" % (a, b))
            continue
        for lab in [pre + c for pre in FAMILIES
                    for c in ("punjabi", "serbian", "script_form", "script_form_rt")]:
            da, db = files[a]["draws"], files[b]["draws"]
            pa, pb = files[a].get("points", {}), files[b].get("points", {})
            if lab not in da or lab not in db:
                continue
            la, ca = _trough(da[lab], pa.get(lab))
            lb, cb = _trough(db[lab], pb.get(lab))
            pt = (float(pb[lab][lb]) - float(pa[lab][la])
                  if lab in pa and lab in pb else None)
            key = "%s_trough_%s_minus_%s" % (lab, b, a)
            out["across_models"][key] = {
                "%s_layer" % a: la, "%s_layer" % b: lb,
                **summarise(cb - ca, point=pt)}
            t = out["across_models"][key]
            print("scale, %-12s trough: %s (L%d) minus %s (L%d) = %+.4f  "
                  "CI [%+.4f, %+.4f]  p=%.2g"
                  % (lab, b, lb, a, la, t["point"],
                     t["ci95"][0], t["ci95"][1], t["p_boot"]))

    path = os.path.join(RESULTS, "bootstrap_crossmodel_%s.json" % args.data)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
