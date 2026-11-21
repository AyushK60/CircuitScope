"""Hand-rolled statistics (numpy only, no scipy).

- ``rankdata_average``: ranks with ties averaged (1-based).
- ``spearman``: Spearman rank correlation via average ranks; constant input
  returns None (undefined), never NaN.
- ``bootstrap_ci``: percentile CI over resamples of the values.
- ``permutation_null``: distribution of ``stat(x, y[perm])`` over random
  label permutations (the null control).
- ``topk_overlap`` / ``expected_random_overlap``: top-k set overlap and its
  random expectation k*k/n.
"""

from __future__ import annotations

import numpy as np


def rankdata_average(x) -> np.ndarray:
    """Ranks (1-based) with ties assigned the average of their ranks."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("rankdata_average expects a 1-D array")
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(arr.shape[0], dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < arr.shape[0]:
        j = i
        while j + 1 < arr.shape[0] and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks of positions i..j (0-based) averaged
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(r: np.ndarray, s: np.ndarray) -> float:
    rc = r - r.mean()
    sc = s - s.mean()
    denom = float(np.sqrt((rc * rc).sum()) * np.sqrt((sc * sc).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rc * sc).sum() / denom)


def pearson(x, y) -> float | None:
    """Pearson correlation. Returns None when either input is constant
    (correlation undefined) or the inputs are empty - never NaN."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("pearson expects two 1-D arrays of the same length")
    if a.size == 0:
        return None
    if float((a == a[0]).all()) or float((b == b[0]).all()):
        return None
    return _pearson(a, b)


def spearman(x, y) -> float | None:
    """Spearman rank correlation (average ranks). Returns None when either
    input is constant (correlation undefined) - never NaN."""
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    if float((rx == rx[0]).all()) or float((ry == ry[0]).all()):
        return None
    return _pearson(rx, ry)


def bootstrap_ci(values, stat_fn, n_boot: int, rng: np.random.Generator) -> dict:
    """Percentile bootstrap 95% CI for ``stat_fn(values)`` over resamples.

    Returns {"stat", "ci_low", "ci_high", "n_boot"}; NaN resamples (undefined
    statistic, e.g. a constant resample) are dropped and counted.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    stat = stat_fn(arr)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(arr[idx])
    valid = stats[~np.isnan(stats)]
    n_invalid = int(n_boot - valid.shape[0])
    if valid.shape[0] == 0:
        return {"stat": stat, "ci_low": None, "ci_high": None, "n_boot": n_boot, "n_invalid": n_invalid}
    return {
        "stat": stat,
        "ci_low": float(np.percentile(valid, 2.5)),
        "ci_high": float(np.percentile(valid, 97.5)),
        "n_boot": n_boot,
        "n_invalid": n_invalid,
    }


def permutation_null(x, y, n_perm: int, rng: np.random.Generator) -> dict:
    """Null distribution of ``spearman(x, y[perm])`` over random permutations.

    Returns {"p2_5", "p97_5", "n_perm", "n_invalid"}. Undefined correlations
    (constant input) are dropped and counted.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(ya.shape[0])
        v = spearman(xa, ya[perm])
        vals[i] = float("nan") if v is None else v
    valid = vals[~np.isnan(vals)]
    n_invalid = int(n_perm - valid.shape[0])
    if valid.shape[0] == 0:
        return {"p2_5": None, "p97_5": None, "n_perm": n_perm, "n_invalid": n_invalid}
    return {
        "p2_5": float(np.percentile(valid, 2.5)),
        "p97_5": float(np.percentile(valid, 97.5)),
        "n_perm": n_perm,
        "n_invalid": n_invalid,
    }


def topk_overlap(a, b, k: int) -> int:
    """|A ∩ B| where A/B are the sets of the k largest-valued indices of a/b
    (ties within the top-k boundary are broken by stable descending sort)."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if not 1 <= k <= arr_a.shape[0]:
        raise ValueError(f"k={k} out of range for n={arr_a.shape[0]}")
    top_a = set(np.argsort(-arr_a, kind="stable")[:k].tolist())
    top_b = set(np.argsort(-arr_b, kind="stable")[:k].tolist())
    return len(top_a & top_b)


def expected_random_overlap(k: int, n: int) -> float:
    """Expected |A ∩ B| for two independent random k-subsets of n items."""
    if k > n:
        raise ValueError(f"k={k} larger than n={n}")
    return k * k / n
