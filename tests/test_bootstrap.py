"""Tests for the paired sentence-level bootstrap.

The whole method rests on one algebraic shortcut: that resampled consistency
can be read off weighted sums instead of recomputed. If that is wrong, every
interval is wrong and nothing else would catch it. So the first test
recomputes the statistic the slow, obvious way on explicit resampled rows and
demands agreement.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bootstrap import (bootstrap_consistency, consistency_from_deltas,  # noqa: E402
                       draw_counts, per_item_deltas, summarise)
from probes import paired_script_displacement  # noqa: E402

n_pass = n_fail = 0


def ok(cond, label, extra=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print("PASS  %s %s" % (label, extra))
    else:
        n_fail += 1
        print("FAIL  %s %s" % (label, extra))


rng = np.random.default_rng(0)
n, d = 60, 12
A = rng.normal(size=(n, d))
# A real signal: a shared direction plus item noise, so consistency is neither
# 0 nor 1 and the tests are not passing on a degenerate case.
B = A + np.array([1.0] + [0.0] * (d - 1)) + 0.4 * rng.normal(size=(n, d))

# --- the point statistic must agree with the shipped one -------------------
for norm in (True, False):
    delta, sq = per_item_deltas(A, B, normalise=norm)
    mine = consistency_from_deltas(delta, sq)
    theirs = paired_script_displacement(A, B, normalise=norm)["consistency"]
    ok(abs(mine - theirs) < 1e-12, "point statistic matches probes.py (normalise=%s)" % norm,
       "%.12f vs %.12f" % (mine, theirs))

delta, sq = per_item_deltas(A, B, normalise=True)

# --- the matmul shortcut must equal naive recomputation --------------------
counts = draw_counts(n, n_boot=50, seed=7)
ok(counts.shape == (50, n), "counts shape", str(counts.shape))
ok(np.all(counts.sum(axis=1) == n), "every draw resamples n sentences")
ok(counts.min() >= 0, "counts are non-negative")

fast = bootstrap_consistency(delta, sq, counts, device="cpu")
slow = []
for row in counts:
    idx = np.repeat(np.arange(n), row)          # explicit resampled rows
    slow.append(consistency_from_deltas(delta[idx], sq[idx]))
slow = np.array(slow)
ok(fast.shape == (50,), "bootstrap output shape", str(fast.shape))
worst = float(np.max(np.abs(fast - slow)))
ok(worst < 1e-10, "matmul shortcut equals naive recomputation", "max abs diff %.2e" % worst)

# A draw of all-ones counts must return the full-sample statistic exactly.
ones = np.ones((1, n), dtype=np.int32)
full = bootstrap_consistency(delta, sq, ones, device="cpu")[0]
pt = consistency_from_deltas(delta, sq)
ok(abs(full - pt) < 1e-12, "unit counts reproduce the point estimate",
   "%.12f vs %.12f" % (full, pt))

# --- pairing: the same counts must be reusable across quantities -----------
c2 = draw_counts(n, n_boot=50, seed=7)
ok(np.array_equal(counts, c2), "draw_counts is deterministic given a seed")
c3 = draw_counts(n, n_boot=50, seed=8)
ok(not np.array_equal(counts, c3), "a different seed gives different draws")

# A difference of two consistencies under shared draws must equal the
# difference of the two separately bootstrapped vectors, elementwise. This is
# what makes the paired interval valid.
B2 = A + 0.3 * np.array([1.0] + [0.0] * (d - 1)) + 0.4 * rng.normal(size=(n, d))
delta2, sq2 = per_item_deltas(A, B2, normalise=True)
f2 = bootstrap_consistency(delta2, sq2, counts, device="cpu")
diff = fast - f2
slow_diff = []
for row in counts:
    idx = np.repeat(np.arange(n), row)
    slow_diff.append(consistency_from_deltas(delta[idx], sq[idx])
                     - consistency_from_deltas(delta2[idx], sq2[idx]))
ok(float(np.max(np.abs(diff - np.array(slow_diff)))) < 1e-10,
   "paired difference matches draw-by-draw recomputation")

# The paired interval must be tighter than the unpaired one when the two
# quantities share sentence noise. If it is not, pairing is buying nothing and
# the design is pointless.
unpaired = bootstrap_consistency(delta2, sq2, draw_counts(n, n_boot=50, seed=99), device="cpu")
sd_paired = float(np.std(fast - f2, ddof=1))
sd_unpaired = float(np.std(fast - unpaired, ddof=1))
ok(sd_paired < sd_unpaired, "pairing reduces the spread of the difference",
   "paired %.4f < unpaired %.4f" % (sd_paired, sd_unpaired))

# --- summarise -------------------------------------------------------------
s = summarise(np.full(200, 0.5))
ok(s["ci95"][0] == 0.5 and s["ci95"][1] == 0.5, "degenerate draws give a point interval")
ok(s["p_boot"] == 2.0 / 200, "p is floored at 2/n, the finest two-sided value "
   "the draws can express", str(s["p_boot"]))
ok(s["excludes_zero"] is True, "a strictly positive interval excludes zero")

s0 = summarise(rng.normal(size=4000) * 1.0)
ok(s0["p_boot"] > 0.3, "draws centred on zero give a large p", "%.3f" % s0["p_boot"])
ok(s0["excludes_zero"] is False, "an interval straddling zero does not exclude it")

s1 = summarise(rng.normal(size=4000) * 0.01 + 1.0)
ok(s1["p_boot"] == 2.0 / 4000, "draws far from zero give exactly the smallest "
   "resolvable two-sided p", "%.2e" % s1["p_boot"])
# A one-sided floor of 1/n would be finer than the draws can express.
ok(s1["p_boot"] > 1.0 / 4000, "and not the finer one-sided floor")
ok(s1["ci95"][0] > 0, "interval sits above zero", "%.3f" % s1["ci95"][0])

s2 = summarise([np.nan, 1.0, 1.0, np.nan])
ok(s2["n_draws"] == 2, "non-finite draws are dropped", str(s2["n_draws"]))
ok(summarise([np.nan, np.nan])["n_draws"] == 0, "all-NaN draws degrade gracefully")

# --- point override --------------------------------------------------------
s3 = summarise([0.1, 0.2, 0.3], point=0.25)
ok(abs(s3["point"] - 0.25) < 1e-12, "the full-sample point estimate is kept, not the boot mean")
ok(abs(s3["boot_mean"] - 0.2) < 1e-12, "the bootstrap mean is reported separately")

# --- shape guards ----------------------------------------------------------
try:
    per_item_deltas(A, B[:10])
    ok(False, "mismatched shapes raise")
except ValueError:
    ok(True, "mismatched shapes raise")
try:
    bootstrap_consistency(delta, sq, np.ones((3, n + 1), dtype=np.int32), device="cpu")
    ok(False, "wrong count width raises")
except ValueError:
    ok(True, "wrong count width raises")

# --- a zero-displacement contrast must land at the analytic floor ----------
# Identical conditions give delta = 0 exactly, so the statistic is undefined
# rather than zero; it must come back NaN and not crash or read as 1.0.
dz, qz = per_item_deltas(A, A.copy(), normalise=True)
ok(np.isnan(consistency_from_deltas(dz, qz)), "identical conditions give NaN, not 1.0")
bz = bootstrap_consistency(dz, qz, counts, device="cpu")
ok(np.all(np.isnan(bz)), "and NaN under every draw")

# Pure isotropic noise should sit near the analytic floor 1/sqrt(n).
Z = rng.normal(size=(400, 30))
Z2 = Z + 0.5 * rng.normal(size=(400, 30))
dn, qn = per_item_deltas(Z, Z2, normalise=True)
c_noise = consistency_from_deltas(dn, qn)
floor = 1.0 / np.sqrt(400)
ok(c_noise < 8 * floor, "a no-signal contrast sits near the analytic floor",
   "%.4f vs floor %.4f" % (c_noise, floor))

# --- magnitude statistics ---------------------------------------------------
from bootstrap import bootstrap_stats, point_stats  # noqa: E402
c0, sh0, rm0 = point_stats(delta, sq)
ok(abs(c0 - consistency_from_deltas(delta, sq)) < 1e-12,
   "point_stats consistency equals the shipped statistic")
ok(abs(c0 - sh0 / rm0) < 1e-12, "consistency is shared over rms")
# For unit-normalised rows rms^2 = 2(1 - mean cosine), so rms is a
# cosine-distance magnitude; check that identity explicitly.
An = A / np.linalg.norm(A, axis=1, keepdims=True)
Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
mean_cos = float(np.mean(np.sum(An * Bn, axis=1)))
ok(abs(rm0 ** 2 - 2 * (1 - mean_cos)) < 1e-10, "rms^2 = 2(1 - mean cosine)")
bc, bsh, brm = bootstrap_stats(delta, sq, counts, device="cpu")
ok(np.max(np.abs(bc - fast)) < 1e-12, "bootstrap_stats consistency equals bootstrap_consistency")
slow_sh, slow_rm = [], []
for row in counts:
    idx = np.repeat(np.arange(n), row)
    _, s_, r_ = point_stats(delta[idx], sq[idx])
    slow_sh.append(s_); slow_rm.append(r_)
ok(np.max(np.abs(bsh - np.array(slow_sh))) < 1e-10, "shared under every draw equals naive")
ok(np.max(np.abs(brm - np.array(slow_rm))) < 1e-10, "rms under every draw equals naive")
# The reviewer's case: a small coherent shift has high consistency but small
# magnitude; a large incoherent one the reverse. The statistics must separate.
Z = rng.normal(size=(300, 20)) * 5
small = Z + 0.05 * np.r_[1.0, np.zeros(19)]
large = Z + 3.0 * rng.normal(size=(300, 20))
ds, qs_ = per_item_deltas(Z, small); dl, ql = per_item_deltas(Z, large)
cs_, _, rs_ = point_stats(ds, qs_); cl_, _, rl_ = point_stats(dl, ql)
ok(cs_ > cl_ and rs_ < rl_,
   "consistency and magnitude can disagree, so both must be reported",
   "small shift cons %.2f rms %.3f; large noise cons %.2f rms %.3f" % (cs_, rs_, cl_, rl_))

print("\n%d/%d passed" % (n_pass, n_pass + n_fail))
sys.exit(1 if n_fail else 0)
