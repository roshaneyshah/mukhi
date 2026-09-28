"""
Linear probes with Hewitt & Liang (2019) control tasks.

The proposal is right that skipping control tasks is the most common way
probing papers get dismissed. The implementation detail that matters: the
control labels must be assigned ONCE, per item, and reused across every layer
and every model. If you redraw them per layer you are comparing each layer
against a different control and the selectivity curve is meaningless.

selectivity = accuracy(real task) - accuracy(control task)

Both probes use identical model class, identical regularisation and identical
splits. Anything else and the difference is not attributable to the
representation.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def make_control_labels(item_ids, real_labels, seed=0):
    """
    Hewitt & Liang control task for item-level probing.

    Each distinct item_id is assigned a random label drawn from the empirical
    distribution of real_labels. Assignment is deterministic given the seed and
    independent of any representation, so the same control is reused for every
    layer and every model.

    item_ids: sequence of hashable ids, one per example. For a parallel corpus
              use the SENTENCE-PAIR id, not the row index, so that the Gurmukhi
              and Shahmukhi versions of one sentence get the SAME control
              label. Otherwise the control task leaks script identity and
              selectivity is understated.
    """
    real_labels = np.asarray(real_labels)
    classes, counts = np.unique(real_labels, return_counts=True)
    probs = counts / counts.sum()

    rng = np.random.default_rng(seed)
    uniq = sorted(set(item_ids), key=lambda x: str(x))
    mapping = {u: rng.choice(classes, p=probs) for u in uniq}

    return np.array([mapping[i] for i in item_ids]), mapping


def _fit_probe(X, y, C=1.0, n_splits=5, seed=0, max_iter=2000):
    """Cross-validated linear probe accuracy. Returns (mean_acc, std_acc)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan"), float("nan")
    # StratifiedKFold needs at least n_splits members in the smallest class
    n_splits = int(min(n_splits, counts.min()))
    if n_splits < 2:
        return float("nan"), float("nan")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs"),
        )
        clf.fit(X[tr], y[tr])
        accs.append(float(np.mean(clf.predict(X[te]) == y[te])))

    return float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)


def majority_baseline(y):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def probe_with_control(X, real_labels, control_labels, C=1.0, n_splits=5, seed=0):
    """
    Run the real probe and the control probe under identical settings.

    Returns dict with real_acc, control_acc, selectivity, majority baseline and
    fold-level spread. Report `selectivity`; report `real_acc` alone only
    alongside it.
    """
    real_acc, real_sd = _fit_probe(X, real_labels, C=C, n_splits=n_splits, seed=seed)
    ctrl_acc, ctrl_sd = _fit_probe(X, control_labels, C=C, n_splits=n_splits, seed=seed)

    return {
        "real_acc": real_acc,
        "real_sd": real_sd,
        "control_acc": ctrl_acc,
        "control_sd": ctrl_sd,
        "selectivity": real_acc - ctrl_acc,
        "majority": majority_baseline(real_labels),
        "C": C,
        "n": int(np.asarray(X).shape[0]),
        "d": int(np.asarray(X).shape[1]),
    }


def sweep_regularisation(X, real_labels, control_labels,
                         Cs=(0.001, 0.01, 0.1, 1.0, 10.0), n_splits=5, seed=0):
    """
    Hewitt & Liang show selectivity depends strongly on probe capacity. Sweeping
    C and reporting the whole curve is the defensible move; picking one C that
    flatters the result is not.
    """
    return [probe_with_control(X, real_labels, control_labels,
                               C=c, n_splits=n_splits, seed=seed) for c in Cs]


# ---------------------------------------------------------------------------
# Scale-invariant effect size
# ---------------------------------------------------------------------------
def direction_effect_size(X, labels):
    """
    Cohen's d along the class-mean-difference direction.

    WHY THIS EXISTS. Probe accuracy is not scale-invariant. If a later layer
    devotes more variance to content while holding script information exactly
    constant, script-probe accuracy falls anyway. A falling accuracy curve
    therefore does NOT by itself license the claim "orthography is abstracted
    away with depth" -- the proposal's central claim. Verified in-container:
    with the script signal literally unchanged, accuracy fell 1.000 -> 0.775
    and selectivity 0.470 -> 0.187 purely from growing content variance.

    This statistic divides the between-class mean gap by the pooled
    within-class spread ALONG THAT SAME DIRECTION, so it is invariant to
    isotropic rescaling of the layer and to added variance in orthogonal
    directions. Report it beside accuracy. When accuracy falls and d stays
    flat, the information is still there and the layer merely got noisier;
    when both fall, something was genuinely removed.

    Binary labels only.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("direction_effect_size requires exactly 2 classes")

    A = X[y == classes[0]]
    B = X[y == classes[1]]

    w = B.mean(axis=0) - A.mean(axis=0)
    nw = np.linalg.norm(w)
    if nw == 0:
        return 0.0
    w = w / nw

    a, b = A @ w, B @ w
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")

    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    if pooled == 0:
        return float("inf")
    return float(abs(b.mean() - a.mean()) / pooled)


def layer_norm_stats(X):
    """Mean L2 norm per item. Report alongside probe curves so a reader can see
    whether an accuracy decay tracks norm growth."""
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1)
    return {"mean_norm": float(n.mean()), "std_norm": float(n.std(ddof=1))}


def paired_script_displacement(X_a, X_b, normalise=True):
    """
    The measure the minimal-pair design makes possible, and which a generic
    probe throws away.

    X_a, X_b: (n, d) activations for the SAME n sentences in two scripts, rows
    aligned. Because the sentences are identical in content, the shared
    semantic component CANCELS in the per-item difference

        delta_i = X_b[i] - X_a[i]

    leaving the script displacement plus noise. If the model encodes script as
    a consistent direction, the delta_i point the same way and their mean is
    long relative to their spread.

    Returns:
      consistency  - ||mean(delta)|| / sqrt(mean ||delta_i||^2), in [0, 1].
                     1.0 = script is a perfectly consistent displacement,
                     0.0 = the two scripts differ in an item-specific way with
                     no shared direction.
      pc1_var      - fraction of delta variance on its first principal
                     component. High pc1_var with low consistency means the
                     displacement is one-dimensional but sign-flips across
                     items.
      mean_norm    - ||mean(delta)||, unnormalised, for reference.

    Unlike probe accuracy and Cohen's d, `consistency` is unaffected by
    content variance that is common to both scripts, which is exactly the
    nuisance that makes raw accuracy curves misleading with depth.
    """
    A = np.asarray(X_a, dtype=np.float64)
    B = np.asarray(X_b, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"paired inputs must match: {A.shape} vs {B.shape}")

    if normalise:
        # per-item L2 normalisation removes layerwise norm growth
        na = np.linalg.norm(A, axis=1, keepdims=True); na[na == 0] = 1.0
        nb = np.linalg.norm(B, axis=1, keepdims=True); nb[nb == 0] = 1.0
        A, B = A / na, B / nb

    delta = B - A
    mean_delta = delta.mean(axis=0)

    rms = np.sqrt((delta ** 2).sum(axis=1).mean())
    consistency = float(np.linalg.norm(mean_delta) / rms) if rms > 0 else float("nan")

    centred = delta - mean_delta
    if centred.shape[0] > 1:
        try:
            s = np.linalg.svd(centred, compute_uv=False)
            pc1 = float(s[0] ** 2 / (s ** 2).sum()) if (s ** 2).sum() > 0 else float("nan")
        except np.linalg.LinAlgError:
            pc1 = float("nan")
    else:
        pc1 = float("nan")

    return {"consistency": consistency,
            "pc1_var": pc1,
            "mean_norm": float(np.linalg.norm(mean_delta))}
