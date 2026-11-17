"""Stage 3 Task 4 tests: hand-computed toy cases for the stats module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.analysis.stats import (
    bootstrap_ci,
    expected_random_overlap,
    permutation_null,
    rankdata_average,
    spearman,
    topk_overlap,
)


def test_rankdata_average_no_ties():
    assert rankdata_average([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]
    assert rankdata_average([3.0, 1.0, 2.0]).tolist() == [3.0, 1.0, 2.0]


def test_rankdata_average_ties():
    # Hand-worked: values [1, 2, 2, 3] -> ranks [1, 2.5, 2.5, 4].
    assert rankdata_average([1.0, 2.0, 2.0, 3.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_perfect_and_anti():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ties_hand_value():
    # x=[1,2,3,4], y=[1,3,2,4]: Pearson of the ranks is 0.8 (hand-computed:
    # rank vectors [1,2,3,4] and [1,3,2,4]; numerator 4.0, denominator 5.0).
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)


def test_spearman_constant_input_returns_none():
    assert spearman([5, 5, 5, 5], [1, 2, 3, 4]) is None
    assert spearman([1, 2, 3, 4], [0, 0, 0, 0]) is None


def test_spearman_equals_pearson_of_ranks_reimplementation():
    """Cross-check spearman against a direct Pearson-of-ranks reimplementation
    written in the test (independent of the module's internals)."""
    rng = np.random.default_rng(0)

    def pearson(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ac, bc = a - a.mean(), b - b.mean()
        return float((ac * bc).sum() / math.sqrt(float((ac * ac).sum() * (bc * bc).sum())))

    def rank_avg(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v)
        ranks = np.empty(len(v))
        ranks[order] = np.arange(1, len(v) + 1, dtype=float)
        # average ties
        for val in np.unique(v):
            mask = v == val
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    for _ in range(20):
        x = rng.normal(size=15).round(1)  # rounding forces ties
        y = rng.normal(size=15).round(1)
        assert spearman(x, y) == pytest.approx(pearson(rank_avg(x), rank_avg(y)))


def test_bootstrap_ci_covers_true_mean_and_is_deterministic():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ci1 = bootstrap_ci(vals, np.mean, 2000, np.random.default_rng(123))
    ci2 = bootstrap_ci(vals, np.mean, 2000, np.random.default_rng(123))
    assert ci1 == ci2  # deterministic for a fixed seed
    assert ci1["ci_low"] <= vals.mean() <= ci1["ci_high"]
    # 1% percentile on a tiny known sample: mean of resample distribution
    assert ci1["stat"] == pytest.approx(5.5)


def test_bootstrap_ci_percentile_method_matches_hand_resample():
    # Two values: resamples of size 2 have means in {1, 1.5, 2}; the 2.5/97.5
    # percentiles of that discrete distribution are exact.
    vals = np.array([1.0, 2.0])
    rng = np.random.default_rng(0)
    ci = bootstrap_ci(vals, np.mean, 4000, rng)
    assert ci["ci_low"] >= 1.0 and ci["ci_high"] <= 2.0
    # with 4000 draws both extreme means (1.0 and 2.0, each p=1/4) appear:
    assert ci["ci_low"] == pytest.approx(1.0)
    assert ci["ci_high"] == pytest.approx(2.0)


def test_permutation_null_deterministic_and_bounded():
    rng = np.random.default_rng(7)
    x = np.arange(10, dtype=float)
    r1 = permutation_null(x, x, 200, np.random.default_rng(7))
    r2 = permutation_null(x, x, 200, np.random.default_rng(7))
    assert r1 == r2
    assert -1.0 <= r1["p2_5"] <= r1["p97_5"] <= 1.0
    # the true pairing (perfectly correlated) lies far above the null band
    assert spearman(x, x) > r1["p97_5"]


def test_topk_overlap_examples():
    a = [3.0, 1.0, 2.0]
    b = [3.0, 2.0, 1.0]
    assert topk_overlap(a, b, 1) == 1  # both top-1 is index 0
    assert topk_overlap(a, b, 2) == 1  # {0,2} vs {0,1}
    assert topk_overlap(a, b, 3) == 3
    assert topk_overlap([1, 2, 3], [3, 2, 1], 1) == 0  # top-1 differs (index 2 vs 0)


def test_expected_random_overlap():
    assert expected_random_overlap(5, 144) == pytest.approx(25.0 / 144.0)
    assert expected_random_overlap(10, 144) == pytest.approx(100.0 / 144.0)
    with pytest.raises(ValueError):
        expected_random_overlap(200, 144)
