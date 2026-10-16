"""Figures. Colour-blind safe, no seaborn dependency, readable in greyscale."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OK = "#0173B2"; OR = "#DE8F05"; GR = "#029E73"; RD = "#CC3311"; GY = "#888888"


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_cka_curve(cka_layers, path, model_name="", contrast=""):
    L = [r["layer"] for r in cka_layers]
    d = [r["delta"] for r in cka_layers]
    a = [r["aligned"] for r in cka_layers]
    b = [r["baseline_mean"] for r in cka_layers]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(L, a, "-o", color=OK, ms=4, label="aligned pairs")
    ax.plot(L, b, "--", color=GY, lw=1.2, label="row-permuted baseline")
    ax.plot(L, d, "-s", color=RD, ms=4, label="delta (reported)")
    ax.axhline(0, color="k", lw=0.8)
    _style(ax, "layer", "CKA (unbiased)", f"{model_name}  {contrast}".strip())
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def plot_selectivity(script_layers, language_layers, path, model_name="",
                     crossover=None):
    L = [r["layer"] for r in script_layers]
    s = [r["selectivity"] for r in script_layers]
    g = [r["selectivity"] for r in language_layers]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(L, s, "-o", color=OR, ms=4, label="script selectivity")
    ax.plot(L, g, "-o", color=GR, ms=4, label="language selectivity")
    ax.axhline(0, color="k", lw=0.8)
    if crossover is not None:
        ax.axvline(crossover, color=RD, ls=":", lw=1.4)
        ax.annotate(f"crossover L{crossover}", xy=(crossover, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points",
                    color=RD, fontsize=9)
    _style(ax, "layer", "selectivity (real - control)",
           f"{model_name}  probe selectivity")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def plot_paired_displacement(pd_layers, path, model_name=""):
    """Two panels: the statistic, and the norm growth it is immune to."""
    L = [r["layer"] for r in pd_layers]
    c = [r["consistency"] for r in pd_layers]
    d = [r["delta"] for r in pd_layers]
    fl = [r["analytic_floor"] for r in pd_layers]
    na = [r["mean_norm_a"] for r in pd_layers]
    nb = [r["mean_norm_b"] for r in pd_layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(L, c, "-o", color=OK, ms=4, label="consistency")
    ax1.plot(L, d, "-s", color=RD, ms=4, label="delta (reported)")
    ax1.plot(L, fl, ":", color=GY, lw=1.2, label=r"analytic floor $1/\sqrt{n}$")
    ax1.axhline(0, color="k", lw=0.8)
    _style(ax1, "", "paired script displacement",
           f"{model_name}  script displacement (content-cancelled)")
    ax1.legend(frameon=False, fontsize=9)

    ax2.plot(L, na, "-", color=OR, lw=1.4, label="mean ||h|| Gurmukhi")
    ax2.plot(L, nb, "--", color=GR, lw=1.4, label="mean ||h|| Shahmukhi")
    _style(ax2, "layer", "activation norm", "")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def plot_fertility(comparison, path, model_names):
    """Grouped bars: tokens-per-char ratio Shahmukhi/Gurmukhi, per model."""
    ratios = [c["tpc"]["ratio_b_over_a"] for c in comparison]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(model_names))
    cols = [RD if r > 1.15 or r < 0.87 else OK for r in ratios]
    ax.bar(x, ratios, color=cols, width=0.6)
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    _style(ax, "", "tokens/char  Shahmukhi ÷ Gurmukhi", "Tokenisation premium by script")
    ax.annotate("1.0 = parity", xy=(len(x)-0.5, 1.0), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=GY, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path
