#!/usr/bin/env python3
"""Test the bracket assumption of Section 5.3 from saved bootstrap draws.

The 66 to 90% script-form range assumes the share falls steadily as
information is removed from the Gurmukhi control. Four controls form a
graded series, from most information to least:

    script_form      de-voweled Gurmukhi
    script_form_dv1  plus the n merge
    script_form_dv2  plus the vowel-sign merges
    script_form_rt   Shahmukhi round-tripped back (adds rule errors)

For each model this reports the layer-averaged share per control, with a 95%
interval, and the paired difference between neighbours in the series. The
assumption holds in a model if every neighbour difference is positive.

No GPU. Reads results/bootstrap_draws_<data>_*.npz (or $MUKHI_RESULTS).
Usage: tools_bracket_check.py [flores|pumvr]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bootstrap import summarise  # noqa: E402

RESULTS = os.environ.get("MUKHI_RESULTS", os.path.join(ROOT, "results"))
SERIES = ["script_form", "script_form_dv1", "script_form_dv2", "script_form_rt"]


def share_draws(d, label):
    """Layer-mean of control/punjabi per draw, and the full-sample point."""
    num, den = d[label], d["punjabi"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(den) > 1e-12, num / den, np.nan)
        pn, pd = d["point__" + label], d["point__punjabi"]
        pr = np.where(np.abs(pd) > 1e-12, pn / pd, np.nan)
    return np.nanmean(r, axis=0), float(np.nanmean(pr))


def check(path):
    d = np.load(path)
    have = [s for s in SERIES if s in d.files]
    if "punjabi" not in d.files or len(have) < 2:
        return None
    shares, points = {}, {}
    for s in have:
        shares[s], points[s] = share_draws(d, s)
    out = {"shares": {s: summarise(shares[s], point=points[s]) for s in have},
           "steps": {}}
    for a, b in zip(have, have[1:]):
        out["steps"][f"{a}_minus_{b}"] = summarise(shares[a] - shares[b],
                                                   point=points[a] - points[b])
    steps = out["steps"].values()
    out["monotone_point"] = all(v["point"] > 0 for v in steps)
    out["monotone_ci"] = all(v["ci95"][0] > 0 for v in steps)
    out["series_present"] = have

    # The same check against each random-init seed separately. The intervals
    # above are sentence-sampling intervals around the seed MEAN, so they say
    # nothing about whether the ordering survives a different random network.
    seeds = sorted({int(k[4:k.index("__")]) for k in d.files
                    if k.startswith("seed") and k.endswith("__punjabi")})
    per_seed = {}
    for sd in seeds:
        sub = {}
        for k in d.files:
            pre = "seed%d__" % sd
            if k.startswith(pre):
                sub[k[len(pre):]] = d[k]
            elif k.startswith("point__" + pre):
                sub["point__" + k[len("point__" + pre):]] = d[k]
        hs = [x for x in have if x in sub]
        if "punjabi" not in sub or len(hs) < 2:
            continue
        sh, pt = {}, {}
        for x in hs:
            sh[x], pt[x] = share_draws(sub, x)
        st = {f"{a}_minus_{b}": summarise(sh[a] - sh[b], point=pt[a] - pt[b])
              for a, b in zip(hs, hs[1:])}
        per_seed[str(sd)] = {
            "shares": {x: summarise(sh[x], point=pt[x]) for x in hs},
            "steps": st,
            "monotone_point": all(v["point"] > 0 for v in st.values()),
            "monotone_ci": all(v["ci95"][0] > 0 for v in st.values()),
        }
    if per_seed:
        out["per_seed"] = per_seed
        out["monotone_every_seed_point"] = all(v["monotone_point"] for v in per_seed.values())
        out["monotone_every_seed_ci"] = all(v["monotone_ci"] for v in per_seed.values())
    return out


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "flores"
    files = sorted(glob.glob(os.path.join(RESULTS, f"bootstrap_draws_{data}_*.npz")))
    report = {}
    for f in files:
        tag = os.path.basename(f)[len(f"bootstrap_draws_{data}_"):-len(".npz")]
        r = check(f)
        if r is None:
            print(f"{tag}: graded series not present, skipped")
            continue
        report[tag] = r
        print(f"\n{tag}")
        for s, v in r["shares"].items():
            print(f"  {s:16s} share {v['point']:.3f}  "
                  f"CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        for k, v in r["steps"].items():
            print(f"  step {k:34s} {v['point']:+.3f}  "
                  f"CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        print(f"  falls steadily: point {r['monotone_point']}, "
              f"every CI above zero {r['monotone_ci']}")
        for sd, v in (r.get("per_seed") or {}).items():
            sh = "  ".join(f"{x.replace('script_form', 'sf')} {y['point']:.3f}"
                           for x, y in v["shares"].items())
            print(f"  seed {sd}: {sh}  | steps all > 0: {v['monotone_point']}, "
                  f"CIs above zero: {v['monotone_ci']}")
    if not report:
        print("no draw file holds the graded series")
        return 1
    path = os.path.join(RESULTS, f"bracket_check_{data}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
