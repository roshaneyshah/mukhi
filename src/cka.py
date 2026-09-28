"""
Centered Kernel Alignment for layerwise representational similarity.

Two estimators:
  linear_cka        - Kornblith et al. (2019), biased. Fast, standard.
  unbiased_linear_cka - Song et al. (2012) HSIC_1 estimator. Use this as the
                      headline number when n is small (< ~5k). The biased
                      estimator inflates toward 1 as p/n grows, which is
                      exactly the regime of 1-2k sentences against 768-4096
                      hidden dims. Reporting the biased one alone on a small
                      parallel set is a reviewer magnet.

Both take X (n, p1) and Y (n, p2): n ALIGNED examples, rows in correspondence.
p1 and p2 need not match.
"""
import numpy as np


def _center_columns(X):
    """Subtract the per-feature mean across examples."""
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """
    Biased linear CKA.

        CKA(X, Y) = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )

    with X, Y column-centered. Returns a scalar in [0, 1].
    """
    X = _center_columns(X)
    Y = _center_columns(Y)

    # ||Y^T X||_F^2
    cross = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    nx = np.linalg.norm(X.T @ X, ord="fro")
    ny = np.linalg.norm(Y.T @ Y, ord="fro")

    denom = nx * ny
    if denom == 0.0:
        return np.nan
    return float(cross / denom)


def _hsic1(K, L):
    """
    Unbiased HSIC estimator (Song et al., 2012, Eq. 5). O(n^2).

    K, L are (n, n) gram matrices. Diagonals are zeroed internally.
    The cross term 1^T K L 1 is computed as (K 1) . (L 1), never as a matrix
    product: the naive form is O(n^3) and dominates runtime at n ~ 2000.
    """
    K = np.array(K, dtype=np.float64, copy=True)
    L = np.array(L, dtype=np.float64, copy=True)
    n = K.shape[0]
    if n < 4:
        raise ValueError("unbiased HSIC needs n >= 4, got n=%d" % n)
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    return _hsic1_zeroed(K, L)


def _hsic1_zeroed(K, L):
    """_hsic1 for grams whose diagonals are ALREADY zero. No copies."""
    n = K.shape[0]
    kr = K.sum(axis=1)
    lr = L.sum(axis=1)
    term_trace = float(np.sum(K * L))
    term_prod = float(kr.sum()) * float(lr.sum()) / ((n - 1) * (n - 2))
    term_cross = 2.0 * float(kr @ lr) / (n - 2)
    return (term_trace + term_prod - term_cross) / (n * (n - 3))


def unbiased_linear_cka(X, Y):
    """
    CKA built on the unbiased HSIC estimator with a linear kernel.

    Not guaranteed to lie in [0, 1] -- small negative values are normal when
    the true similarity is near zero and are informative, so they are NOT
    clipped. Report them as-is.
    """
    X = _center_columns(X)
    Y = _center_columns(Y)

    K = X @ X.T
    L = Y @ Y.T

    hsic_xy = _hsic1(K, L)
    hsic_xx = _hsic1(K, K)
    hsic_yy = _hsic1(L, L)

    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom == 0.0 or not np.isfinite(denom):
        return np.nan
    return float(hsic_xy / denom)


def _zero_diag_gram(X):
    X = _center_columns(X)
    K = X @ X.T
    np.fill_diagonal(K, 0.0)
    return K


def _double_centered_gram(X):
    X = _center_columns(X)
    return X @ X.T          # column-centred X gives a double-centred gram


def cka_with_baseline(X, Y, estimator="unbiased", n_permutations=200, seed=0):
    """
    The number you actually report.

    A raw CKA value is uninterpretable on its own (results/FINDING_01). This
    returns the aligned score with a row-permutation baseline that destroys the
    pairing while preserving every marginal property of both matrices.

    Grams are built once; a permutation just re-indexes L, and the
    self-similarity terms are permutation-invariant, so each permutation costs
    one O(n^2) pass. 200 permutations at n=2000 take seconds.

    Returns dict with:
        aligned, baseline_mean, baseline_std,
        delta  = aligned - baseline_mean   (the interpretable quantity)
        z      = delta / baseline_std
        p_perm = (1 + #{baseline >= aligned}) / (1 + n_permutations)
                 exact permutation p-value; its floor is 1/(1+n_permutations),
                 which is why n_permutations must be large enough to survive a
                 multiple-comparison correction over layers x contrasts.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if estimator == "unbiased":
        K = _zero_diag_gram(X)
        L = _zero_diag_gram(Y)
        hxx, hyy = _hsic1_zeroed(K, K), _hsic1_zeroed(L, L)
        denom = np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))

        def score(Lm):
            return _hsic1_zeroed(K, Lm) / denom if denom > 0 else np.nan
    else:
        K = _double_centered_gram(X)
        L = _double_centered_gram(Y)
        denom = np.linalg.norm(K) * np.linalg.norm(L)

        def score(Lm):
            return float(np.sum(K * Lm) / denom) if denom > 0 else np.nan

    aligned = float(score(L))
    baselines = np.empty(n_permutations)
    for b in range(n_permutations):
        perm = rng.permutation(n)
        baselines[b] = score(L[np.ix_(perm, perm)])

    b_mean = float(np.nanmean(baselines))
    b_std = float(np.nanstd(baselines, ddof=1)) if n_permutations > 1 else 0.0
    delta = aligned - b_mean
    p_perm = (1 + int(np.sum(baselines >= aligned))) / (1 + n_permutations)

    return {
        "aligned": aligned,
        "baseline_mean": b_mean,
        "baseline_std": b_std,
        "delta": delta,
        "z": float(delta / b_std) if b_std > 0 else float("nan"),
        "p_perm": float(p_perm),
        "estimator": estimator,
        "n": int(n),
        "n_permutations": int(n_permutations),
    }
