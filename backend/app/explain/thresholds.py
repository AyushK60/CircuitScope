"""Declared thresholds and formatting helpers for generated text.

Every qualitative word in generated claims comes from a bucket function here.
Changing a boundary changes the wording everywhere; the boundaries are
unit-tested at their exact values (see backend/tests/test_explain_thresholds.py).

Bucket rule (Stage 4 principle 3), on Recovery-style values r:
    r >= 0.8          -> "most"       (restores most of)
    0.4 <= r < 0.8    -> "large"      (restores a large part of)
    0.1 <= r < 0.4    -> "some"       (restores some of)
    -0.1 < r < 0.1    -> "little"     (has little effect on)
    r <= -0.1         -> "negative"   (pushes the other way)

DLA values are raw logit units, not recoveries, so they get their own declared
scale (STRONG/WEAK/SMALL). All boundaries live in this one place.
"""

from __future__ import annotations

# --- recovery-style buckets -------------------------------------------------
RECOVERY_STRONG = 0.8
RECOVERY_LARGE = 0.4
RECOVERY_SOME = 0.1
RECOVERY_NEGATIVE = -0.1

# --- direct-contribution buckets (raw logit units) ---------------------------
DLA_STRONG = 1.0
DLA_WEAK = 0.3

# --- live trust-check tolerances ---------------------------------------------
TOL_FINAL_RECOVERY = 1e-3  # final-block final-position recovery == 1
TOL_LASTBLOCK_UNCHANGED = 1e-4  # non-final positions in the last block unchanged
MIN_BASELINE_CONTRAST = 1e-3  # |D_clean - D_corrupted| gate (matches sweeps.py)

# --- readability limits (curious level; Stage 4 principle 7) ------------------
MAX_MEAN_SENTENCE_WORDS = 18
MAX_SENTENCE_WORDS = 28
MIN_FLESCH_READING_EASE = 60.0


def recovery_bucket(r: float) -> str:
    """Bucket id for a recovery-style value (the wording is derived from this)."""
    r = float(r)
    if r >= RECOVERY_STRONG:
        return "most"
    if r >= RECOVERY_LARGE:
        return "large"
    if r >= RECOVERY_SOME:
        return "some"
    if r > RECOVERY_NEGATIVE:
        return "little"
    return "negative"


def recovery_predicate(r: float, gap: str = "the score gap") -> str:
    """Verb phrase for a recovery value, composed with the gap description.
    Never stronger than the bucket allows."""
    b = recovery_bucket(r)
    if b == "most":
        return f"restores most of {gap}"
    if b == "large":
        return f"restores a large part of {gap}"
    if b == "some":
        return f"restores some of {gap}"
    if b == "little":
        return f"has little effect on {gap}"
    return f"pushes {gap} the other way"


def dla_bucket(v: float) -> str:
    """Bucket id for a direct-contribution value in raw logit units."""
    v = float(v)
    if abs(v) >= DLA_STRONG:
        return "strong"
    if abs(v) >= DLA_WEAK:
        return "weak"
    return "small"


# --- formatting helpers -------------------------------------------------------


def fmt_num(x, digits: int = 2) -> str:
    """Format a number with fixed digits; 'n/a' for None/non-finite (honest
    missing-data marker, never a zero)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return "n/a"
    return f"{v:.{digits}f}"


def fmt_pct(x, digits: int = 1) -> str:
    """Format a 0..1 fraction as a percentage string; 'n/a' when undefined."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v or v in (float("inf"), float("-inf")):
        return "n/a"
    return f"{100 * v:.{digits}f}%"


def ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 144 -> '144th'."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
