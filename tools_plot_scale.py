#!/usr/bin/env python3
"""Figure 1: direction consistency by relative depth, trained minus random
init (three seeds), for XLM-R base, large and XL. Bands are 95% bootstrap
intervals from the saved draws. Writes results/figures/scale_depth.png."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")
FILES = [("XLM-R base", "round3/bootstrap_draws_flores_xlm-roberta-base.npz"),
         ("XLM-R large", "round3/bootstrap_draws_flores_xlm-roberta-large.npz"),
         ("XLM-R XL", "round4/bootstrap_draws_flores_xlm-roberta-xl.npz")]
SHADES = ["#9ecae1", "#3182bd", "#08306b"]
SHADES_S = ["#fdae6b", "#e6550d", "#7f2704"]


def main():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (name, f), cp, cs in zip(FILES, SHADES, SHADES_S):
        d = np.load(os.path.join(R, f))
        for lab, col, pair in (("punjabi", cp, "Punjabi"), ("serbian", cs, "Serbian")):
            pt = d["point__" + lab]
            lo, hi = np.percentile(d[lab], [2.5, 97.5], axis=1)
            x = np.linspace(0, 1, len(pt))
            ax.fill_between(x, lo, hi, color=col, alpha=0.25, linewidth=0)
            ax.plot(x, pt, color=col, lw=1.6, label=f"{pair}, {name}")
            t = int(np.argmin(pt))
            ax.plot(x[t], pt[t], "o", color=col, ms=4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("relative depth (embedding layer = 0, last layer = 1)")
    ax.set_ylabel("consistency, trained minus random")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    out = os.path.join(R, "figures", "scale_depth.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
