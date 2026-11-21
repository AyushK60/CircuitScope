"""Logit lens: decode the residual stream after each block.

Question answered: "what would the model predict if it stopped after block i?"
For each stage the residual at the final position is taken and the final
LayerNorm + unembedding are applied EXACTLY as the model does them
(biased-variance LN with the model's own epsilon, gamma/beta, then W_U -
GPT-2 has no unembedding bias). This is a heuristic view: early-layer residual
streams are not trained to be read by the unembedding, so early "predictions"
are not what the model "was thinking".

Identity: the final stage (block 11 output == the residual entering ln_f) must
reproduce the model's real final logits.
"""

from __future__ import annotations

import torch

from ..config import MAX_LAYERS
from ..model_runtime.loader import ADAPTER

LENS_NOTE = (
    "Heuristic view: the logit lens applies the final LayerNorm and unembedding "
    "to intermediate residual streams. Early layers are not trained to be read "
    "this way, so early-stage 'predictions' are not claims about what the model "
    "was thinking - treat them as a coarse trace of how the output belief forms."
)

FINAL_TOLERANCE = 1e-3


def compute_logit_lens(
    text: str,
    candidate_a: str | None,
    candidate_b: str | None,
    top_k: int = 5,
) -> dict:
    """Compute the lens for one prompt. ``candidate_a``/``candidate_b`` may be
    None (single-token strings when given); ``top_k`` bounds the reported
    top tokens per stage (1..10)."""
    bridge = ADAPTER.ensure_loaded()
    eps = getattr(bridge.ln_final, "eps", None)
    if eps is None:
        eps = getattr(bridge.original_model.config, "layer_norm_epsilon", None)
    if eps is None:
        raise ValueError("could not find the LayerNorm epsilon on the model")
    eps = float(eps)

    tok_a = tok_b = None
    if candidate_a is not None and candidate_b is not None:
        tok_a = ADAPTER.token_id_for_str(candidate_a)
        tok_b = ADAPTER.token_id_for_str(candidate_b)

    wanted = {"hook_embed", "hook_pos_embed"} | {f"blocks.{l}.hook_out" for l in range(MAX_LAYERS)}
    toks = bridge.to_tokens(text)
    final_logits, cache = bridge.run_with_cache(toks, names_filter=lambda n: n in wanted)
    for name in wanted:
        if name not in cache:
            raise ValueError(f"hook was not captured during the run: {name!r}")

    pos = toks.shape[1] - 1
    gamma = bridge.ln_final.weight
    beta = bridge.ln_final.bias
    W_U = bridge.unembed.W_U  # [d_model, d_vocab]; no unembedding bias in GPT-2

    def lens_logits(resid: torch.Tensor) -> torch.Tensor:
        mean = resid.mean()
        s = torch.sqrt(resid.var(unbiased=False) + eps)
        normalized = (resid - mean) / s * gamma + beta
        return normalized @ W_U

    def top_tokens(logits: torch.Tensor) -> list[dict]:
        probs = torch.softmax(logits, dim=-1)
        vals, idxs = torch.topk(logits, top_k)
        tok = bridge.tokenizer
        return [
            {"token": tok.decode([int(i)]), "token_id": int(i), "logit": float(v), "prob": float(probs[int(i)])}
            for v, i in zip(vals.tolist(), idxs.tolist())
        ]

    stages: list[dict] = []
    residuals: list[tuple[str, int | None, torch.Tensor]] = [
        ("embeddings", None, cache["hook_embed"][0, pos] + cache["hook_pos_embed"][0, pos])
    ]
    residuals += [(f"block {l}", l, cache[f"blocks.{l}.hook_out"][0, pos]) for l in range(MAX_LAYERS)]

    for name, layer, resid in residuals:
        logits = lens_logits(resid)
        probs = torch.softmax(logits, dim=-1)
        stage: dict = {
            "stage": name,
            "layer": layer,
            "top": top_tokens(logits),
        }
        if tok_a is not None and tok_b is not None:
            stage["D"] = float(logits[tok_a] - logits[tok_b])
            stage["prob_a"] = float(probs[tok_a])
            stage["prob_b"] = float(probs[tok_b])
        else:
            stage["D"] = None
            stage["prob_a"] = None
            stage["prob_b"] = None
        stages.append(stage)

    model_D = float(final_logits[0, pos, tok_a] - final_logits[0, pos, tok_b]) if tok_a is not None else None
    final_lens_logits = lens_logits(residuals[-1][2])
    if tok_a is not None:
        max_abs_logit_diff_candidates = abs(stages[-1]["D"] - model_D)
    else:
        max_abs_logit_diff_candidates = None
    # Additive honesty check: the full-vocabulary logit vectors at the final stage.
    max_abs_logit_diff_full = float((final_lens_logits - final_logits[0, pos]).abs().max())
    top1_agrees = bool(int(torch.argmax(final_lens_logits)) == int(torch.argmax(final_logits[0, pos])))
    ok = top1_agrees and (
        max_abs_logit_diff_candidates is None or max_abs_logit_diff_candidates <= FINAL_TOLERANCE
    )

    out = {
        "tokens": bridge.to_str_tokens(text),
        "position": pos,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "stages": [{k: v for k, v in s.items()} for s in stages],
        "final_matches_model": {
            "max_abs_logit_diff_candidates": max_abs_logit_diff_candidates,
            "max_abs_logit_diff_full": max_abs_logit_diff_full,
            "top1_agrees": top1_agrees,
            "ok": bool(ok),
            "tolerance": FINAL_TOLERANCE,
        },
        "note": LENS_NOTE,
        "model_D": model_D,
    }
    return out
