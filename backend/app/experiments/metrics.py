"""Numerical metrics for clean/corrupted/patched comparisons.

Definitions (README section 5.1):
    D        = logit(A) - logit(B)          for validated single-token candidates A, B
    Delta    = D_patched - D_corrupted
    Recovery = (D_patched - D_corrupted) / (D_clean - D_corrupted)
Recovery is NOT clamped; a near-zero denominator yields null, not a fake number.
"""

from __future__ import annotations

import torch

NEAR_ZERO_EPS = 1e-4


def logit_diff(logits: torch.Tensor, token_a: int, token_b: int) -> float:
    """D at the final sequence position. logits: [1, seq, vocab]."""
    return float(logits[0, -1, token_a] - logits[0, -1, token_b])


def compare_metrics(
    clean_logits: torch.Tensor | None,
    corrupted_logits: torch.Tensor | None,
    patched_logits: torch.Tensor | None,
    token_a: int,
    token_b: int,
) -> dict:
    out: dict = {
        "candidates": {"A": token_a, "B": token_b},
        "sign_convention": "positive D means candidate A scores above candidate B at the final position",
    }
    d_clean = d_corr = d_patch = None
    if clean_logits is not None:
        d_clean = logit_diff(clean_logits, token_a, token_b)
    if corrupted_logits is not None:
        d_corr = logit_diff(corrupted_logits, token_a, token_b)
    if patched_logits is not None:
        d_patch = logit_diff(patched_logits, token_a, token_b)
    out["D_clean"] = d_clean
    out["D_corrupted"] = d_corr
    out["D_patched"] = d_patch
    if d_patch is not None and d_corr is not None:
        out["Delta"] = d_patch - d_corr
    else:
        out["Delta"] = None
    if None not in (d_patch, d_corr, d_clean) and abs(d_clean - d_corr) > NEAR_ZERO_EPS:
        out["Recovery"] = (d_patch - d_corr) / (d_clean - d_corr)
    else:
        out["Recovery"] = None
        out["Recovery_note"] = "undefined: clean-vs-corrupted difference is near zero" if (
            d_clean is not None and abs(d_clean - (d_corr or 0.0)) <= NEAR_ZERO_EPS
        ) else "undefined: missing run"
    return out
