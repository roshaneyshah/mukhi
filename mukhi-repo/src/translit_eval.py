"""Character error rate for transliteration, with Gurmukhi-aware normalisation."""
import unicodedata
import numpy as np

NUKTA = "਼"


def normalise_gur(t, strip_nukta=False):
    """NFC, collapse whitespace; optionally drop nukta, which many writers omit."""
    t = unicodedata.normalize("NFC", " ".join((t or "").split()))
    if strip_nukta:
        t = unicodedata.normalize("NFD", t).replace(NUKTA, "")
        t = unicodedata.normalize("NFC", t)
    return t


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyps, refs, strip_nukta=False):
    """Corpus CER = total edits / total reference chars. Also exact-match rate."""
    edits = chars = exact = 0
    for h, r in zip(hyps, refs):
        h, r = normalise_gur(h, strip_nukta), normalise_gur(r, strip_nukta)
        edits += levenshtein(h, r)
        chars += len(r)
        exact += int(h == r)
    n = len(refs)
    return {"cer": edits / chars if chars else float("nan"),
            "exact_match": exact / n if n else float("nan"),
            "n": n}


def bootstrap_cer_ci(hyps, refs, n_boot=1000, seed=0, strip_nukta=False):
    """95% bootstrap CI over items, so improvements are reported with uncertainty."""
    rng = np.random.default_rng(seed)
    per = []
    for h, r in zip(hyps, refs):
        h, r = normalise_gur(h, strip_nukta), normalise_gur(r, strip_nukta)
        per.append((levenshtein(h, r), len(r)))
    per = np.array(per, float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(per), len(per))
        e, c = per[idx].sum(axis=0)
        vals.append(e / c)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def split_pairs(pairs, test_frac=0.3, seed=0):
    """Split UNIQUE (gur, shah) pairs so no test word was seen in development."""
    uniq = sorted({(p["gurmukhi"], p["shahmukhi"]) for p in pairs})
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(uniq))
    k = int(len(uniq) * test_frac)
    test = [uniq[i] for i in idx[:k]]
    dev = [uniq[i] for i in idx[k:]]
    return dev, test


def edit_ops(hyp, ref):
    """Levenshtein alignment ops as (op, hyp_char, ref_char)."""
    n, m = len(hyp), len(ref)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1,
                          D[i-1][j-1] + (hyp[i-1] != ref[j-1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + (hyp[i-1] != ref[j-1]):
            if hyp[i-1] != ref[j-1]:
                ops.append(("sub", hyp[i-1], ref[j-1]))
            i, j = i - 1, j - 1
        elif j > 0 and D[i][j] == D[i][j-1] + 1:
            ops.append(("ins", "", ref[j-1])); j -= 1
        else:
            ops.append(("del", hyp[i-1], "")); i -= 1
    return ops[::-1]


# Characters Shahmukhi routinely leaves unwritten: sihari, aunkar, addak.
UNWRITTEN = {"ਿ", "ੁ", "ੱ"}


def error_decomposition(hyps, refs):
    """Split edits into 'unwritten short vowel / gemination' (unrecoverable by
    rules) versus everything else (rule-fixable or ambiguous)."""
    unw = other = 0
    subs = {}
    for h, r in zip(hyps, refs):
        for op, a, b in edit_ops(normalise_gur(h), normalise_gur(r)):
            if op == "ins" and b in UNWRITTEN:
                unw += 1
            else:
                other += 1
                if op == "sub":
                    subs[(a, b)] = subs.get((a, b), 0) + 1
    total = unw + other
    top = sorted(subs.items(), key=lambda kv: -kv[1])[:10]
    return {"total_edits": total, "unwritten": unw,
            "unwritten_share": unw / total if total else float("nan"),
            "top_substitutions": top}


# Letters whose choice the rules cannot know from Shahmukhi. Collapsed before
# comparing, so a correct pair is not flagged for a difference the input could
# never have determined.
_COLLAPSE = str.maketrans({"ੂ": "ੋ", "ੌ": "ੋ", "ੇ": "ੀ", "ੈ": "ੀ",
                           "ਣ": "ਨ", "ਂ": "ੰ"})


def comparable(t):
    t = normalise_gur(t, strip_nukta=True)
    t = "".join(ch for ch in t if ch not in UNWRITTEN)
    return t.translate(_COLLAPSE)


def pair_suspicion(translit_hyp, corpus_gur):
    """Normalised edit distance after removing differences no rule could
    resolve. ~0 for a faithful pair; high for misaligned or corrupted pairs."""
    a, b = comparable(translit_hyp), comparable(corpus_gur)
    return levenshtein(a, b) / max(len(b), 1)


def auroc(pos, neg):
    """P(score_pos > score_neg), ties counted half. Mann-Whitney form."""
    import numpy as _np
    pos, neg = _np.asarray(pos, float), _np.asarray(neg, float)
    allv = _np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in _np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
