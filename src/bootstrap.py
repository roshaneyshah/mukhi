"""Paired sentence-level bootstrap for the trained-minus-random difference.

Why this exists
---------------
Every displacement the paper reports is a difference: trained consistency
minus random-init consistency at the same layer and contrast. The BH-corrected
p values in the result files attach to the sign-flip null for a SINGLE run,
which tests "is this displacement above chance", not "is the trained run above
its own random-init baseline". Reviewers called that out: the paper's central
quantity had no inferential test of its own.

This module supplies one. The design turns on an algebraic fact about the
statistic:

    consistency(S) = || mean_{i in S} delta_i || / sqrt( mean_{i in S} ||delta_i||^2 )

depends on the sentences only through sums over them. So for a bootstrap
resample with per-sentence counts c (summing to n):

    numerator   = || (c @ delta) / n ||
    denominator = sqrt( (c @ sqnorm) / n )

Both are linear in c, so B bootstrap draws cost one (B, n) @ (n, d) matmul
instead of B recomputations. That is what makes a 2,000-draw bootstrap over
every layer, contrast and model affordable inside one GPU session.

What makes it paired, and why that matters three times over
-----------------------------------------------------------
The same count matrix is used for every quantity in a draw. Because FLORES+ is
a parallel corpus, sentence i is the same sentence in Punjabi and in Serbian,
and the same sentence in the trained and random-init runs of every model. So
one draw resamples the whole family coherently, and three differences the
paper rests on become testable rather than merely reported:

  1. trained minus random, per layer and contrast (the reviewers' W3);
  2. the Punjabi trough against the Serbian trough (Section 5.4's ordering);
  3. XLM-R base against XLM-R large (Section 5.7's scale claim).

Taking the difference inside each draw cancels the shared sentence-sampling
noise that an unpaired interval would leave in. Measured on the real data the
gain is modest, not dramatic: the scale interval is 15% narrower paired and
the within-model ordering interval 4% narrower, because the per-draw
correlations are only 0.29 and 0.09 respectively. Pairing is the right thing
to do and is what makes the comparisons well defined, but both would reach
the same conclusion unpaired.

What it does NOT capture
------------------------
Sentence-sampling uncertainty only. Random-init seed uncertainty is a separate
source and is reported separately as the seed spread: where several seeds
exist, each draw uses the mean of their consistencies, so the interval
describes the mean baseline, not a single draw of weights.

Selection of the trough layer
-----------------------------
The trough layer is chosen on the full sample and then held fixed across
draws. Re-minimising inside each draw would bootstrap a selected minimum and
bias the interval downward; `trough_refit` reports that version alongside, for
contrast, and should not be read as the interval for the quoted trough.
"""
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is present wherever this runs
    torch = None


def per_item_deltas(acts_a, acts_b, normalise=True):
    """Per-sentence displacement rows for one layer.

    acts_a, acts_b: (n, d) activations for the SAME n sentences in two
    conditions, rows aligned. Returns (delta, sqnorm) with delta (n, d) and
    sqnorm (n,). The per-item L2 normalisation is applied before any
    resampling, exactly as in probes.paired_script_displacement, so the two
    agree on the full sample by construction.
    """
    A = np.asarray(acts_a, dtype=np.float64)
    B = np.asarray(acts_b, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError("paired inputs must match: %s vs %s" % (A.shape, B.shape))
    if normalise:
        na = np.linalg.norm(A, axis=1, keepdims=True)
        nb = np.linalg.norm(B, axis=1, keepdims=True)
        na[na == 0] = 1.0
        nb[nb == 0] = 1.0
        A, B = A / na, B / nb
    delta = B - A
    return delta, (delta ** 2).sum(axis=1)


def consistency_from_deltas(delta, sqnorm=None):
    """The point statistic, from delta rows. Matches paired_script_displacement."""
    delta = np.asarray(delta, dtype=np.float64)
    if sqnorm is None:
        sqnorm = (delta ** 2).sum(axis=1)
    rms = np.sqrt(np.mean(sqnorm))
    if rms <= 0:
        return float("nan")
    return float(np.linalg.norm(delta.mean(axis=0)) / rms)


def draw_counts(n, n_boot=2000, seed=0):
    """Bootstrap resample counts, (n_boot, n) int32, each row summing to n.

    Counts rather than indices because the statistic needs only sums, and a
    count matrix lets every draw be computed in one matmul. Drawn once per
    run and shared across every model, layer and contrast, which is what
    makes the comparisons paired.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n), dtype=np.int32)
    rows = np.repeat(np.arange(n_boot), n)
    np.add.at(counts, (rows, idx.ravel()), 1)
    return counts


def bootstrap_consistency(delta, sqnorm, counts, device=None):
    """Consistency under every bootstrap draw, as a (n_boot,) array.

    One (n_boot, n) @ (n, d) matmul, on the GPU when torch has one.
    """
    n = delta.shape[0]
    if counts.shape[1] != n:
        raise ValueError("counts has %d columns, deltas have %d rows"
                         % (counts.shape[1], n))
    if torch is not None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # float64 on CPU for exactness; float32 on GPU, where the matmul is
        # well conditioned (counts are small non-negative integers) and the
        # memory saving is what keeps a 25-layer model in range.
        dt = torch.float32 if str(dev).startswith("cuda") else torch.float64
        D = torch.as_tensor(np.ascontiguousarray(delta), dtype=dt, device=dev)
        Q = torch.as_tensor(np.ascontiguousarray(sqnorm), dtype=dt, device=dev)
        W = torch.as_tensor(np.ascontiguousarray(counts), dtype=dt, device=dev)
        num = torch.linalg.norm(W @ D, dim=1) / n
        den = torch.sqrt((W @ Q) / n)
        out = torch.where(den > 0, num / den,
                          torch.full_like(den, float("nan")))
        return out.double().cpu().numpy()
    W = counts.astype(np.float64)
    num = np.linalg.norm(W @ delta, axis=1) / n
    den = np.sqrt((W @ sqnorm) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def summarise(draws, point=None, alpha=0.05):
    """Percentile interval and a two-sided bootstrap p value for a difference.

    The p value is the usual bootstrap inversion: twice the smaller tail mass
    on either side of zero. It is floored at 2/n_draws, not 1/n_draws, because
    the smallest nonzero one-sided tail mass B draws can express is 1/B, so
    the finest two-sided p they can express is 2/B. Flooring at 1/B reports a
    value one step finer than the draws can resolve, which is what an earlier
    version did: with 2,000 draws it printed 0.0005 where the honest statement
    is p <= 0.001.
    """
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"point": point, "ci95": [None, None], "p_boot": None, "n_draws": 0}
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    frac_le = float(np.mean(d <= 0))
    frac_ge = float(np.mean(d >= 0))
    p = 2.0 * min(frac_le, frac_ge)
    p = float(min(1.0, max(p, 2.0 / d.size)))
    return {
        "point": float(point) if point is not None else float(d.mean()),
        "boot_mean": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_boot": p,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_draws": int(d.size),
    }


# ---------------------------------------------------------------------------
# Magnitude statistics, answering the reviewer's first point.
#
# consistency = ||mean delta|| / sqrt(mean ||delta_i||^2) measures how
# coherently the per-sentence displacements point one way, not how far apart
# the two scripts' representations are. A small, coherent displacement scores
# high; a large, incoherent one scores low. Its numerator and denominator are
# each a magnitude in their own right, on the same L2-normalised activations:
#
#   shared = ||mean delta||                 the displacement common to all
#                                           sentences (a distance between the
#                                           two conditions' centroids)
#   rms    = sqrt(mean ||delta_i||^2)        the typical per-sentence distance;
#                                           since the rows are unit vectors,
#                                           rms^2 = 2 (1 - mean cosine), so it
#                                           is a cosine-distance magnitude
#
# consistency = shared / rms. Both are linear in the resample counts, so they
# come out of the same matmul at no extra cost, and every claim can be checked
# on all three.

def point_stats(delta, sqnorm=None):
    """(consistency, shared, rms) for one layer on the full sample."""
    delta = np.asarray(delta, dtype=np.float64)
    if sqnorm is None:
        sqnorm = (delta ** 2).sum(axis=1)
    rms = float(np.sqrt(np.mean(sqnorm)))
    shared = float(np.linalg.norm(delta.mean(axis=0)))
    cons = shared / rms if rms > 0 else float("nan")
    return cons, shared, rms


def bootstrap_stats(delta, sqnorm, counts, device=None):
    """(consistency, shared, rms) under every draw, each a (n_boot,) array."""
    n = delta.shape[0]
    if counts.shape[1] != n:
        raise ValueError("counts has %d columns, deltas have %d rows"
                         % (counts.shape[1], n))
    if torch is not None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float32 if str(dev).startswith("cuda") else torch.float64
        D = torch.as_tensor(np.ascontiguousarray(delta), dtype=dt, device=dev)
        Q = torch.as_tensor(np.ascontiguousarray(sqnorm), dtype=dt, device=dev)
        W = torch.as_tensor(np.ascontiguousarray(counts), dtype=dt, device=dev)
        shared = torch.linalg.norm(W @ D, dim=1) / n
        rms = torch.sqrt((W @ Q) / n)
        cons = torch.where(rms > 0, shared / rms, torch.full_like(rms, float("nan")))
        return tuple(x.double().cpu().numpy() for x in (cons, shared, rms))
    W = counts.astype(np.float64)
    shared = np.linalg.norm(W @ delta, axis=1) / n
    rms = np.sqrt((W @ sqnorm) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        cons = np.where(rms > 0, shared / rms, np.nan)
    return cons, shared, rms
