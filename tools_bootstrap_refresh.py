"""Recompute bootstrap summaries from the saved per-draw differences.

The draws are the expensive part and they are on disk, so a correction to how
they are summarised does not need another GPU session. This re-derives every
interval, p value and BH q value in results/bootstrap_flores_*.json from
results/bootstrap_draws_flores_*.npz, leaving the draws untouched.

Written for one such correction: the two-sided p was floored at 1/n_draws,
one step finer than B draws can express, so troughs printed 0.0005 where the
honest value is 0.001. It is reusable for any later change to summarise().

Originals are backed up as *.pre_refresh.

Usage:
    python tools_bootstrap_refresh.py [--data flores]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("MUKHI_RESULTS") or os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

from analysis import benjamini_hochberg  # noqa: E402
from bootstrap import summarise  # noqa: E402


def refresh(path, data):
    d = json.load(open(path))
    tag = [k for k in d["per_model"]][0]
    m = d["per_model"][tag]
    dp = os.path.join(RESULTS, "bootstrap_draws_%s_%s.npz" % (data, tag))
    if not os.path.exists(dp):
        print("  %s: no draw file, skipped" % tag)
        return False
    z = np.load(dp)
    changed = []

    for lab, c in m["contrasts"].items():
        if lab not in z.files:
            continue
        arr = z[lab]
        pts = z["point__" + lab] if ("point__" + lab) in z.files else None
        for li, row in enumerate(c["per_layer"]):
            before = row.get("p_boot")
            new = summarise(arr[li], point=row["point"])
            row.update({k: new[k] for k in
                        ("boot_mean", "ci95", "p_boot", "excludes_zero", "n_draws")})
            if before is not None and abs((before or 0) - row["p_boot"]) > 1e-12:
                changed.append((lab, li, before, row["p_boot"]))
        if pts is not None:
            tl = int(np.nanargmin(pts))
            c["trough"] = {"layer": tl, **summarise(arr[tl], point=float(pts[tl]))}
            c["trough_refit"] = summarise(np.nanmin(arr, axis=0))
            c["embedding"] = summarise(arr[0], point=float(pts[0]))
            c["final"] = summarise(arr[-1], point=float(pts[-1]))

    # Derived comparisons, from the same draws.
    if "punjabi" in z.files and "serbian" in z.files:
        pp, sp = z["point__punjabi"], z["point__serbian"]
        pl, sl = int(np.nanargmin(pp)), int(np.nanargmin(sp))
        m["punjabi_minus_serbian_trough"] = {
            "punjabi_layer": pl, "serbian_layer": sl,
            **summarise(z["punjabi"][pl] - z["serbian"][sl],
                        point=float(pp[pl]) - float(sp[sl]))}
    if "punjabi" in z.files and "script_form" in z.files:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(z["punjabi"]) > 1e-12,
                             z["script_form"] / z["punjabi"], np.nan)
        m["script_form_ratio"] = {
            "layer_mean": summarise(np.nanmean(ratio, axis=0)),
            "embedding": summarise(ratio[0]),
            "final": summarise(ratio[-1]),
        }
        # Per-layer intervals, which the draft once wrongly claimed to have.
        m["script_form_ratio_per_layer"] = [
            {"layer": li, **summarise(ratio[li])} for li in range(ratio.shape[0])]
        pl_pts = z["point__script_form"] / z["point__punjabi"]
        m["script_form_ratio_full_sample"] = {
            "per_layer": [float(x) for x in pl_pts],
            "layer_mean": float(np.nanmean(pl_pts)),
            "min": float(np.nanmin(pl_pts)), "max": float(np.nanmax(pl_pts)),
            "n_layers_above_one": int((pl_pts > 1.0).sum()),
        }

    # BH over this model's per-layer family, as before.
    rows = [r for c in m["contrasts"].values() for r in c["per_layer"]
            if r.get("p_boot") is not None]
    if rows:
        rej, qs = benjamini_hochberg([r["p_boot"] for r in rows], alpha=0.05)
        for r, rj, qq in zip(rows, rej, qs):
            r["q_bh"] = float(qq)
            r["significant_bh"] = bool(rj)
        d["bh_family_size"] = len(rows)

    d["refreshed_by"] = "tools_bootstrap_refresh.py"
    d["p_floor"] = "2/n_draws"
    os.replace(path, path + ".pre_refresh")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print("  %-20s %d p values changed; BH family %d; %d significant"
          % (tag, len(changed), d.get("bh_family_size", 0),
             sum(1 for r in rows if r["significant_bh"])))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="flores")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(
        RESULTS, "bootstrap_flores_*.json" if args.data == "flores"
        else "bootstrap_%s_*.json" % args.data)))
    files = [f for f in files if "crossmodel" not in f]
    if not files:
        print("no bootstrap_%s_*.json in %s" % (args.data, RESULTS))
        return 1
    print("refreshing %d file(s) from saved draws" % len(files))
    for f in files:
        refresh(f, args.data)
    print("\nNow re-run tools_bootstrap_crossmodel.py so the cross-model file "
          "uses the same floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
