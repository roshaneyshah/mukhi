"""
Proposal step 5: does transliterating Shahmukhi input to Gurmukhi at
inference recover performance?

Two arms.

CONTROLLED (SIB-200 topic classification, any model). Frozen features, a
linear classifier trained on Gurmukhi and evaluated, on the SAME test
sentences, in:
    pan_Guru          in-script ceiling
    pan_Shah          cross-script zero-shot (generated Shahmukhi)
    pan_Shah_to_Guru  the intervention: Shahmukhi transliterated back
    pan_Guru_devowel  Gurmukhi carrying only Shahmukhi's information
plus pan_Shah trained-and-tested in-script, which asks whether the Shahmukhi
representation carries the label information at all.

If pan_Shah_to_Guru lands near pan_Guru_devowel and below pan_Guru, the
residual gap is information the Shahmukhi input never had (FINDING_04), not a
script-form bottleneck, and no transliteration system can close it.

NATURAL (PuMVR, decoder models). Human-written Shahmukhi. Each question's four
options are scored by length-normalised log-likelihood. Text-only, so absolute
accuracy is not comparable to PuMVR's image-conditioned numbers; the
comparison is paired across scripts on identical items, which is what matters.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correctness vectors.
    Returns (b, c, p) with b = a right & b wrong, c = a wrong & b right."""
    from scipy.stats import binomtest
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue)


def paired_bootstrap_diff(correct_a, correct_b, n_boot=2000, seed=0):
    """95% CI for acc(b) - acc(a) over the same items."""
    a = np.asarray(correct_a, float)
    b = np.asarray(correct_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------------------
# controlled arm
# ---------------------------------------------------------------------------
def frozen_transfer(train_X, train_y, test_sets, C=1.0, max_iter=3000):
    """
    Fit on train_X/train_y once; return per-test-set correctness vectors and
    accuracies. test_sets: {name: (X, y)}. Same classifier for every test set,
    so differences are attributable to the input condition alone.
    """
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=max_iter))
    clf.fit(np.asarray(train_X, np.float64), np.asarray(train_y))
    out = {}
    for name, (X, y) in test_sets.items():
        pred = clf.predict(np.asarray(X, np.float64))
        corr = pred == np.asarray(y)
        out[name] = {"acc": float(corr.mean()), "correct": corr}
    return out


def select_layer(acts_train, y_train, acts_val, y_val, C=1.0):
    """Pick the layer with best IN-SCRIPT (Gurmukhi) validation accuracy.
    Choosing on the ceiling condition avoids selecting the layer that happens
    to flatter the intervention."""
    best, scores = None, []
    for li in range(acts_train.shape[0]):
        r = frozen_transfer(acts_train[li], y_train, {"val": (acts_val[li], y_val)}, C=C)
        scores.append(r["val"]["acc"])
    best = int(np.argmax(scores))
    return best, scores


def summarise_intervention(res):
    """
    res: output of frozen_transfer with keys pan_Guru, pan_Shah,
    pan_Shah_to_Guru, pan_Guru_devowel.
    Reports gain, the gap it closes, and paired tests.
    """
    g = res["pan_Guru"]["correct"]
    s = res["pan_Shah"]["correct"]
    r = res["pan_Shah_to_Guru"]["correct"]
    d = res["pan_Guru_devowel"]["correct"]
    gap = g.mean() - s.mean()
    gain, glo, ghi = paired_bootstrap_diff(s, r)
    gap_test = mcnemar_exact(g, s)
    # "Fraction of the gap closed" is only meaningful when there IS a gap:
    # Gurmukhi significantly above Shahmukhi. Otherwise the ratio divides by
    # noise (a dry run produced 3.00 from a gap of -0.015).
    gap_ok = gap > 0 and gap_test[2] < 0.05
    return {
        "acc": {k: v["acc"] for k, v in res.items()},
        "gap_guru_minus_shah": float(gap),
        "mcnemar_guru_vs_shah": gap_test,
        "gain_restored_minus_shah": gain, "gain_ci95": [glo, ghi],
        "gap_closed_frac": float(gain / gap) if gap_ok else float("nan"),
        "gap_closed_note": "ok" if gap_ok else "undefined: no significant Guru>Shah gap",
        "mcnemar_shah_vs_restored": mcnemar_exact(s, r),
        "mcnemar_restored_vs_devowel": mcnemar_exact(r, d),
        "mcnemar_guru_vs_devowel": mcnemar_exact(g, d),
    }


# ---------------------------------------------------------------------------
# natural arm: LM option scoring
# ---------------------------------------------------------------------------
def build_mcq_prompt(question, options, script):
    """Language-neutral scaffold; only the question and options change script."""
    return f"{question}\n", [o for o in options]


def score_options(model, tok, question, options, device="cpu"):
    """
    Length-normalised log-likelihood of each option continuing the question.
    Returns an array of scores (higher is more likely).
    """
    import torch
    ctx = question.rstrip() + " "
    ctx_ids = tok(ctx, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    scores = []
    for opt in options:
        full = tok(ctx + opt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        # Option tokens are those after the shared prefix. Tokenisers can merge
        # across the boundary, so find the longest common prefix explicitly.
        k = 0
        while k < min(len(ctx_ids), len(full)) and ctx_ids[k] == full[k]:
            k += 1
        k = max(k, 1)
        with torch.no_grad():
            logits = model(full.unsqueeze(0).to(device)).logits[0].float()
        logp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = full[1:].to(device)
        tok_lp = logp[torch.arange(len(tgt)), tgt]
        opt_lp = tok_lp[k - 1:]
        scores.append(float(opt_lp.mean()) if len(opt_lp) else float("-inf"))
    return np.array(scores)


def mcq_accuracy(model, tok, items, script, device="cpu", transform=None):
    """
    items: PuMVR records. script: 'gurmukhi' | 'shahmukhi' | 'roman'.
    transform: optional callable applied to question and options (the
    intervention, e.g. ctx_shah_to_gur on the Shahmukhi items).
    Returns boolean correctness vector.
    """
    correct = []
    for it in items:
        sc = it["scripts"][script]
        q, opts, ans = sc["question"], list(sc["options"]), sc["answer"]
        gold = opts.index(ans) if ans in opts else None
        if gold is None:
            correct.append(False)
            continue
        if transform is not None:
            q, opts = transform(q), [transform(o) for o in opts]
        s = score_options(model, tok, q, opts, device=device)
        correct.append(int(np.argmax(s)) == gold)
    return np.array(correct, bool)
