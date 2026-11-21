"""Exact per-component effects (denoising and ablation).

For one prompt and one corruption this module returns, for all 156 components
(144 attention heads + 12 MLP ``mlp.hook_out`` blocks), the two metrics of the
patched run — ``D_patched`` (logit difference at the final position) and
``P_patched`` (softmax probability of the IO candidate over the full vocab) —
and the normalised denoising recoveries:

    Rec_D = (D_patched - D_corr) / (D_clean - D_corr)
    Rec_P = (P_patched - P_corr) / (P_clean - P_corr)

Ablation (on the CLEAN run, necessity-style evidence) replaces a component's
output by zeros or by a leave-one-out mean over the other prompts of the same
template (two-pass design: clean activations are cached for every prompt
first, ~1 MB per prompt), and the effect is in RAW logit units:

    Abl_D = D_clean - D_ablated

The clean cache is captured once per prompt (all attn.hook_z + mlp.hook_out)
and reused for every component run via ``engine.patched_logits``. Denominator
zero (D_clean == D_corr) makes the recovery undefined: the value is None, and
the caller must exclude the (prompt, corruption) pair per the pre-registered
rule (|D_clean - D_corr| < 0.5).
"""

from __future__ import annotations

from typing import Callable

import torch

from ..config import MAX_LAYERS
from ..model_runtime import engine
from ..model_runtime.loader import ADAPTER

MAX_HEADS = 12

# Fixed component order used everywhere: heads first (l*12+h), then
# MLPs (144+l). Index 0..143 = heads, 144..155 = MLPs.
COMPONENT_NAMES = [f"head {l}.{h}" for l in range(MAX_LAYERS) for h in range(MAX_HEADS)] + [
    f"mlp.{l}" for l in range(MAX_LAYERS)
]
N_COMPONENTS = len(COMPONENT_NAMES)  # 156


def _hooks_wanted() -> set[str]:
    return {
        name
        for l in range(MAX_LAYERS)
        for name in (f"blocks.{l}.attn.hook_z", f"blocks.{l}.mlp.hook_out")
    }


def run_clean(text: str) -> tuple[torch.Tensor, dict]:
    """One clean forward pass capturing hook_z + mlp.hook_out for all layers."""
    bridge = ADAPTER.ensure_loaded()
    toks = bridge.to_tokens(text)
    wanted = _hooks_wanted()
    logits, cache = bridge.run_with_cache(toks, names_filter=lambda n: n in wanted)
    for name in wanted:
        if name not in cache:
            raise ValueError(f"hook was not captured during the clean run: {name!r}")
    return logits, cache


def token_ids(candidate_a: str, candidate_b: str) -> tuple[int, int]:
    return ADAPTER.token_id_for_str(candidate_a), ADAPTER.token_id_for_str(candidate_b)


def metrics_from_logits(logits: torch.Tensor, tok_a: int, tok_b: int) -> tuple[float, float]:
    """(D, P) at the final position: D = logit(a) - logit(b);
    P = softmax probability of candidate a over the full vocab."""
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    return float(last[tok_a] - last[tok_b]), float(probs[tok_a])


def denoise_effects(
    corrupted_text: str,
    candidate_a: str,
    candidate_b: str,
    logits_clean: torch.Tensor,
    clean_cache: dict,
    check_cancel: Callable[[], None] | None = None,
) -> dict:
    """Exact denoising effects for all 156 components: clean activations are
    copied into the corrupted run (sufficiency direction).

    Returns baselines, raw d_patched/p_patched vectors and rec_d/rec_p
    (None where the denominator is zero — the caller excludes those pairs).
    """
    tok_a, tok_b = token_ids(candidate_a, candidate_b)
    d_clean, p_clean = metrics_from_logits(logits_clean, tok_a, tok_b)
    bridge = ADAPTER.ensure_loaded()
    toks = bridge.to_tokens(corrupted_text)
    seq = toks.shape[1]
    positions = list(range(seq))

    logits_corr = bridge(toks)
    d_corr, p_corr = metrics_from_logits(logits_corr, tok_a, tok_b)
    denom_d = d_clean - d_corr
    denom_p = p_clean - p_corr

    runs = 0
    d_patched = [None] * N_COMPONENTS
    p_patched = [None] * N_COMPONENTS
    rec_d: list[float | None] = [None] * N_COMPONENTS
    rec_p: list[float | None] = [None] * N_COMPONENTS
    for l in range(MAX_LAYERS):
        for h in range(MAX_HEADS):
            if check_cancel is not None:
                check_cancel()
            logits = engine.patched_logits(
                toks, clean_cache, f"blocks.{l}.attn.hook_z", positions, [h]
            )
            d_p, p_p = metrics_from_logits(logits, tok_a, tok_b)
            idx = l * MAX_HEADS + h
            d_patched[idx], p_patched[idx] = d_p, p_p
            if denom_d != 0.0:
                rec_d[idx] = (d_p - d_corr) / denom_d
            if denom_p != 0.0:
                rec_p[idx] = (p_p - p_corr) / denom_p
            runs += 1
    for l in range(MAX_LAYERS):
        if check_cancel is not None:
            check_cancel()
        logits = engine.patched_logits(toks, clean_cache, f"blocks.{l}.mlp.hook_out", positions, None)
        d_p, p_p = metrics_from_logits(logits, tok_a, tok_b)
        idx = 144 + l
        d_patched[idx], p_patched[idx] = d_p, p_p
        if denom_d != 0.0:
            rec_d[idx] = (d_p - d_corr) / denom_d
        if denom_p != 0.0:
            rec_p[idx] = (p_p - p_corr) / denom_p
        runs += 1

    return {
        "D_clean": d_clean,
        "D_corr": d_corr,
        "P_clean": p_clean,
        "P_corr": p_corr,
        "denominator_d": denom_d,
        "denominator_p": denom_p,
        "d_patched": d_patched,
        "p_patched": p_patched,
        "rec_d": rec_d,
        "rec_p": rec_p,
        "n_runs": runs,
    }


def _ablation_runs(
    clean_text: str,
    candidate_a: str,
    candidate_b: str,
    replacements: dict[str, torch.Tensor],
    logits_clean: torch.Tensor,
    check_cancel: Callable[[], None] | None = None,
) -> list[float | None]:
    """Run one ablated pass per component (hook + heads) and return the
    Abl_D = D_clean - D_ablated vector in the fixed component order."""
    tok_a, tok_b = token_ids(candidate_a, candidate_b)
    bridge = ADAPTER.ensure_loaded()
    toks = bridge.to_tokens(clean_text)
    seq = toks.shape[1]
    positions = list(range(seq))
    d_clean, _p = metrics_from_logits(logits_clean, tok_a, tok_b)

    abl_d: list[float | None] = [None] * N_COMPONENTS
    for l in range(MAX_LAYERS):
        for h in range(MAX_HEADS):
            if check_cancel is not None:
                check_cancel()
            hook = f"blocks.{l}.attn.hook_z"
            logits = engine.ablated_logits(toks, hook, positions, [h], replacements[hook])
            d_abl, _ = metrics_from_logits(logits, tok_a, tok_b)
            abl_d[l * MAX_HEADS + h] = d_clean - d_abl
    for l in range(MAX_LAYERS):
        if check_cancel is not None:
            check_cancel()
        hook = f"blocks.{l}.mlp.hook_out"
        logits = engine.ablated_logits(toks, hook, positions, None, replacements[hook])
        d_abl, _ = metrics_from_logits(logits, tok_a, tok_b)
        abl_d[144 + l] = d_clean - d_abl
    return abl_d


def zero_ablation_effects(
    clean_text: str,
    candidate_a: str,
    candidate_b: str,
    logits_clean: torch.Tensor,
    clean_cache: dict,
    check_cancel: Callable[[], None] | None = None,
) -> dict:
    """Zero ablation on the clean run: replace each component's activation by
    zeros. Raw logit units, NOT normalised."""
    zeros = {name: torch.zeros_like(clean_cache[name]) for name in _hooks_wanted()}
    abl_d = _ablation_runs(clean_text, candidate_a, candidate_b, zeros, logits_clean, check_cancel)
    return {"abl_d": abl_d, "n_runs": N_COMPONENTS}


def mean_ablation_effects(
    clean_text: str,
    candidate_a: str,
    candidate_b: str,
    logits_clean: torch.Tensor,
    clean_cache: dict,
    family_caches: list[dict],
    check_cancel: Callable[[], None] | None = None,
) -> dict:
    """Mean ablation on the clean run: replace each component's activation by
    the leave-one-out mean over ``family_caches`` (the clean caches of the
    OTHER prompts of the same template, same token length) at the same hook
    and positions. Raises if the family is empty (e.g. a template with a
    single generated prompt in smoke tests) — the caller must skip the prompt
    and count it."""
    if not family_caches:
        raise ValueError("mean ablation needs at least one other prompt of the same template")
    want = _hooks_wanted()
    # Leave-one-out family mean, one averaged tensor per hook. All family
    # prompts must be token-aligned (same shape) with the current prompt.
    means: dict[str, torch.Tensor] = {}
    for name in want:
        shape = clean_cache[name].shape
        for fc in family_caches:
            if fc[name].shape != shape:
                raise ValueError(
                    f"family prompt is not token-aligned for {name}: "
                    f"{tuple(fc[name].shape)} vs {tuple(shape)}"
                )
        stacked = torch.stack([fc[name] for fc in family_caches], dim=0)
        means[name] = stacked.mean(dim=0)
    abl_d = _ablation_runs(clean_text, candidate_a, candidate_b, means, logits_clean, check_cancel)
    return {"abl_d": abl_d, "n_runs": N_COMPONENTS}
