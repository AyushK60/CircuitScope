"""Thresholds and formatting helpers, boundary-tested."""

from __future__ import annotations

import pytest

from backend.app.explain.thresholds import (
    dla_bucket,
    fmt_num,
    fmt_pct,
    ordinal,
    recovery_bucket,
    recovery_predicate,
)


class TestRecoveryBuckets:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.8, "most"),
            (0.79999, "large"),
            (0.4, "large"),
            (0.39999, "some"),
            (0.1, "some"),
            (0.09999, "little"),
            (0.0, "little"),
            (-0.09999, "little"),
            (-0.1, "negative"),
            (-0.5, "negative"),
        ],
    )
    def test_bucket_boundaries(self, value, expected):
        assert recovery_bucket(value) == expected

    def test_predicate_never_stronger_than_bucket(self):
        assert recovery_predicate(0.85) == "restores most of the score gap"
        assert recovery_predicate(0.5) == "restores a large part of the score gap"
        assert recovery_predicate(0.2) == "restores some of the score gap"
        assert recovery_predicate(0.0) == "has little effect on the score gap"
        assert recovery_predicate(-0.3) == "pushes the score gap the other way"

    def test_predicate_custom_gap(self):
        assert recovery_predicate(0.9, gap="the clean behaviour") == "restores most of the clean behaviour"


class TestDlaBuckets:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1.0, "strong"),
            (0.999, "weak"),
            (0.3, "weak"),
            (0.2999, "small"),
            (-1.0, "strong"),
            (-0.3, "weak"),
            (0.0, "small"),
        ],
    )
    def test_dla_bucket_boundaries(self, value, expected):
        assert dla_bucket(value) == expected


class TestFormatting:
    def test_fmt_num(self):
        assert fmt_num(0.9148, 2) == "0.91"
        assert fmt_num(-1.03, 2) == "-1.03"
        assert fmt_num(2.0, 1) == "2.0"
        assert fmt_num(None) == "n/a"
        assert fmt_num(float("nan")) == "n/a"
        assert fmt_num(float("inf")) == "n/a"
        assert fmt_num("not a number") == "n/a"

    def test_fmt_pct(self):
        assert fmt_pct(0.9148) == "91.5%"
        assert fmt_pct(0.62, 0) == "62%"
        assert fmt_pct(None) == "n/a"
        assert fmt_pct(float("nan")) == "n/a"

    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (10, "10th"),
            (11, "11th"),
            (21, "21st"),
            (111, "111th"),
            (144, "144th"),
        ],
    )
    def test_ordinal(self, n, expected):
        assert ordinal(n) == expected
