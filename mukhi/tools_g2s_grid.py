"""Grid-search the Gurmukhi-to-Shahmukhi rules' ambiguous defaults on dev.

FINDING_06 and §3.2 of the draft both say the three ambiguous defaults of the
Gurmukhi-to-Shahmukhi rules were grid-searched on the PuMVR development split
and the held-out split scored once. The search itself existed only as a
hand run step, with the winner hard-coded as translit_g2s.DEFAULT_G2S, so
there was no way to check the claim. This script is that search.

The Shahmukhi-to-Gurmukhi direction already has its grid in
translit_ctx.tune_config (five defaults, 162 configurations). This is the
other direction, which has three: the nun-ghunna letter, whether gemination
is marked with shadda, and the medial realisation of e.

Scoring is dev-split only, so it cannot leak into the held-out numbers.
Writes results/g2s_grid.json.

Usage:
    python tools_g2s_grid.py
"""
import itertools
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pumvr import load_pumvr_pairs  # noqa: E402
from translit_eval import split_pairs, levenshtein  # noqa: E402
from translit_g2s import DEFAULT_G2S, gur_to_shah, normalise_shah  # noqa: E402

# The three ambiguous defaults. "digits" is in DEFAULT_G2S too but is a
# representation choice, not an ambiguity, so it is held fixed.
GRID = {
    "nn": ["ن", "ں"],
    "mark_gemination": [False, True],
    "e_medial": ["ی", "ے", "ئ"],
}


def _cer(hyps, refs):
    """Shahmukhi-normalised CER, the same measure the eval stage uses."""
    num = sum(levenshtein(normalise_shah(h), normalise_shah(r))
              for h, r in zip(hyps, refs))
    den = max(sum(len(normalise_shah(r)) for r in refs), 1)
    return num / den


def main():
    import run_all
    S, W, _ = load_pumvr_pairs(run_all.pumvr_json())
    dev, test = split_pairs(W)
    dev_refs = [s for g, s in dev]

    keys = list(GRID)
    rows = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        full = dict(DEFAULT_G2S, **cfg)
        hyps = [gur_to_shah(g, full) for g, s in dev]
        rows.append({"config": cfg, "dev_cer": _cer(hyps, dev_refs)})
    rows.sort(key=lambda r: r["dev_cer"])

    shipped = {k: DEFAULT_G2S[k] for k in keys}
    rank = next(i for i, r in enumerate(rows) if r["config"] == shipped)

    out = {
        "computed": str(date.today()),
        "script": "tools_g2s_grid.py",
        "split": "PuMVR dev words only (held-out split never scored here)",
        "n_dev_words": len(dev),
        "n_test_words": len(test),
        "n_configurations": len(rows),
        "grid": {k: [str(v) for v in GRID[k]] for k in keys},
        "shipped_config": shipped,
        "shipped_rank": rank,
        "shipped_is_dev_winner": rank == 0,
        "best": rows[0],
        "worst": rows[-1],
        "dev_cer_range": [rows[0]["dev_cer"], rows[-1]["dev_cer"]],
        "all": rows,
    }
    path = os.path.join(ROOT, "results", "g2s_grid.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("%d configurations over %d dev word pairs\n" % (len(rows), len(dev)))
    for i, r in enumerate(rows):
        mark = "  <- shipped" if r["config"] == shipped else ""
        print("%2d  %.4f  %s%s" % (i + 1, r["dev_cer"], r["config"], mark))
    print("\nshipped rank %d of %d, dev CER range %.4f to %.4f"
          % (rank + 1, len(rows), rows[0]["dev_cer"], rows[-1]["dev_cer"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
