"""
Parallel-set construction and the transliteration audit.

Proposal step 1 budgets two weeks to build the parallel set. Most of that is
unnecessary: SLPG's Punjabi Transliteration Corpus already ships 6.3M aligned
Gurmukhi/Shahmukhi sentence pairs. The work that remains is the part that
cannot be outsourced -- auditing it -- and this module is built to make a
fixed number of human judgements buy as much as possible.

The key move is `prioritise_for_audit`. Reviewing 300 uniformly sampled pairs
from a corpus with a ~5% error rate surfaces ~15 errors. Reviewing pairs where
an INDEPENDENT rule-based transliteration of the Shahmukhi side disagrees with
the corpus's own Gurmukhi side surfaces far more. Report both strata: the
reweighted estimate gives the honest corpus-level error rate, the flagged
stratum gives the error taxonomy.
"""
import csv
import hashlib
import numpy as np

from fertility import detect_script, count_words


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_slpg(split="train", max_rows=None, cache_dir=None):
    """
    Load SLPG/Punjabi_Transliteration_Corpus. Requires network access to
    huggingface.co -- blocked in some sandboxes, fine on Kaggle/Colab.

    Returns list of dicts: {"gurmukhi":..., "shahmukhi":..., "src":...}
    Column names are probed because the card does not pin them.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("SLPG/Punjabi_Transliteration_Corpus", split=split,
                          cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            "Could not reach huggingface.co to load the SLPG corpus.\n"
            f"  underlying error: {type(e).__name__}: {str(e)[:160]}\n"
            "  If this is a sandbox with an egress allowlist, huggingface.co is\n"
            "  blocked and no workaround should be attempted. Run this stage on\n"
            "  Kaggle or Colab instead, or pass a local copy via load_local().\n"
            "  `python run_all.py --stage smoke` needs no network and exercises\n"
            "  the full analysis chain on synthetic data."
        ) from e
    cols = {c.lower(): c for c in ds.column_names}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    g_col = pick("gurmukhi", "gur", "pan_guru", "target", "tgt")
    s_col = pick("shahmukhi", "shah", "pnb_arab", "source", "src")
    if g_col is None or s_col is None:
        raise RuntimeError(
            f"could not identify script columns in {ds.column_names}; "
            "pass them explicitly")

    n = len(ds) if max_rows is None else min(max_rows, len(ds))
    return [{"gurmukhi": ds[i][g_col], "shahmukhi": ds[i][s_col], "src": "slpg"}
            for i in range(n)]


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------
def filter_pairs(pairs, min_words=4, max_words=40, require_pure_script=True,
                 dedup=True):
    """
    Quality filter. Returns (kept, rejection_counts).

    min_words=4 because very short strings make CKA degenerate and fertility
    noisy. max_words=40 keeps sequences inside a 128-token window for every
    tokenizer in the model list, including the high-fertility ones.
    """
    kept, seen = [], set()
    reasons = {"short": 0, "long": 0, "script": 0, "dup": 0, "empty": 0}

    for p in pairs:
        g, s = (p.get("gurmukhi") or "").strip(), (p.get("shahmukhi") or "").strip()
        if not g or not s:
            reasons["empty"] += 1
            continue

        wg, ws = count_words(g), count_words(s)
        if min(wg, ws) < min_words:
            reasons["short"] += 1
            continue
        if max(wg, ws) > max_words:
            reasons["long"] += 1
            continue

        if require_pure_script:
            if detect_script(g) != "gurmukhi" or detect_script(s) != "shahmukhi":
                reasons["script"] += 1
                continue

        if dedup:
            key = hashlib.md5((g + "|||" + s).encode("utf-8")).hexdigest()
            if key in seen:
                reasons["dup"] += 1
                continue
            seen.add(key)

        q = dict(p)
        q["pair_id"] = hashlib.md5(g.encode("utf-8")).hexdigest()[:12]
        q["n_words_gur"], q["n_words_shah"] = wg, ws
        kept.append(q)

    return kept, reasons


def stratified_sample(pairs, n=2000, seed=0, n_bins=4):
    """
    Sample stratified by sentence length, so the evaluation set is not dominated
    by one length band. Length correlates with both fertility and CKA
    stability, so an unstratified sample can confound the depth curve.
    """
    if len(pairs) <= n:
        return list(pairs)

    lens = np.array([p["n_words_gur"] for p in pairs])
    edges = np.quantile(lens, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9

    rng = np.random.default_rng(seed)
    out, per_bin = [], n // n_bins
    for b in range(n_bins):
        idx = np.where((lens >= edges[b]) & (lens < edges[b + 1]))[0]
        take = min(per_bin, len(idx))
        out.extend(pairs[i] for i in rng.choice(idx, size=take, replace=False))

    # top up if bins were uneven
    if len(out) < n:
        chosen = {p["pair_id"] for p in out}
        rest = [p for p in pairs if p["pair_id"] not in chosen]
        rng.shuffle(rest)
        out.extend(rest[: n - len(out)])

    return out[:n]


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
AUDIT_TAU = 0.139   # 95th percentile of pair_suspicion on known-good PuMVR pairs


def prioritise_for_audit(pairs, checker, n=300, flag_frac=0.6, tau=AUDIT_TAU,
                         helper=None, seed=0):
    """
    Split the audit budget between FLAGGED and UNFLAGGED pairs.

    checker: Shahmukhi -> Gurmukhi callable that is INDEPENDENT of how the
        corpus was built. Use translit_ctx.ctx_shah_to_gur. Do NOT use the SLPG
        NMT model here: it was trained on the SLPG corpus, so its agreement with
        that corpus is circular.
    helper: optional second transliteration shown to the annotator (e.g. the
        NMT output). Displayed only; never used for stratification.

    A pair is flagged when pair_suspicion(checker(S), G) > tau. Calibrated on
    held-out PuMVR (results/audit_flag_calibration.json): 3% of correct pairs
    flagged; 100% of misaligned pairs; 53% of pairs with one word replaced;
    90% with two. The unflagged stratum exists because single-word errors slip
    under the threshold about half the time.

    Sampling is uniform WITHIN each stratum, which is what makes
    corpus_error_estimate unbiased.
    """
    from translit_eval import pair_suspicion

    rng = np.random.default_rng(seed)
    n_flag = int(n * flag_frac)

    scored = []
    for p in pairs:
        hyp = checker(p["shahmukhi"])
        scored.append((p, hyp, pair_suspicion(hyp, p["gurmukhi"])))

    flagged = [t for t in scored if t[2] > tau]
    clean = [t for t in scored if t[2] <= tau]
    rng.shuffle(flagged)
    rng.shuffle(clean)

    out = []
    for stratum, pool, k in (("flagged", flagged, n_flag),
                             ("unflagged", clean, n - min(n_flag, len(flagged)))):
        for p, hyp, sc in pool[:k]:
            q = dict(p)
            q.update(sys_a=helper(p["shahmukhi"]) if helper else "",
                     sys_b=hyp, suspicion=round(sc, 4), audit_stratum=stratum)
            out.append(q)

    return out, {"n_flagged_total": len(flagged), "n_unflagged_total": len(clean),
                 "flag_rate": len(flagged) / len(scored) if scored else np.nan,
                 "tau": tau}


AUDIT_COLUMNS = ["pair_id", "audit_stratum", "suspicion", "shahmukhi", "gurmukhi",
                 "sys_b", "sys_a", "verdict", "error_type", "notes"]

ERROR_TYPES = ["ok", "vowel", "consonant", "segmentation", "diacritic",
               "named_entity", "misalignment", "other"]


def write_audit_sheet(rows, path):
    """CSV for hand-validation. Fill `verdict` with ok/error and `error_type`."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in AUDIT_COLUMNS})
    return path


def _wilson(k, n, z=1.96):
    """Wilson score interval. Correct at small n and near 0, unlike normal approx."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score_audit(path):
    """
    Read a filled audit sheet and report error rates PER STRATUM with Wilson
    intervals, plus the error-type histogram.

    The headline corpus error rate combines both strata, weighted by the
    corpus flag rate: see `corpus_error_estimate` below.
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {"n_rows": len(rows), "strata": {}, "error_types": {}}

    for stratum in ("flagged", "unflagged"):
        sub = [r for r in rows if r.get("audit_stratum") == stratum
               and r.get("verdict", "").strip()]
        n = len(sub)
        k = sum(1 for r in sub if r["verdict"].strip().lower() not in ("ok", "correct", "1"))
        lo, hi = _wilson(k, n)
        out["strata"][stratum] = {"n_judged": n, "n_errors": k,
                                  "error_rate": (k / n) if n else np.nan,
                                  "ci95": (lo, hi)}

    for r in rows:
        et = (r.get("error_type") or "").strip().lower()
        if et:
            out["error_types"][et] = out["error_types"].get(et, 0) + 1

    return out


def corpus_error_estimate(audit_scores, flag_rate):
    """
    Combine the two strata into a corpus-level error rate.

        P(error) = P(flagged)   * P(error | flagged)
                 + P(unflagged) * P(error | unflagged)

    This is the number to put in the paper. Quoting the flagged-stratum rate as
    if it were the corpus rate would overstate the error badly, since that
    stratum was deliberately enriched for errors.
    """
    f = flag_rate
    e_f = audit_scores["strata"]["flagged"]["error_rate"]
    e_u = audit_scores["strata"]["unflagged"]["error_rate"]
    e_f = 0.0 if not np.isfinite(e_f) else e_f
    e_u = 0.0 if not np.isfinite(e_u) else e_u
    return float(f * e_f + (1 - f) * e_u)
