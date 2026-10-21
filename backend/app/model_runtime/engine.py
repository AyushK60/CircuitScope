"""Inference, capture and intervention engine.

All operations execute the real model. `patch` replaces the stored clean
activation slice inside the *corrupted* run's forward pass through a forward
hook; the clean cache is copied, never mutated.
"""

from __future__ import annotations

import time

import torch

from ..config import (
    MAX_CAPTURE_HOOKS,
    MAX_CAPTURE_TOKENS,
    validate_hook_name,
)
from .loader import ADAPTER


class InterventionError(ValueError):
    pass


def topk_predictions(logits: torch.Tensor, k: int = 10) -> list[dict]:
    """Top-k next-token candidates at the final position. logits: [1, seq, vocab]."""
    bridge = ADAPTER.bridge()
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    vals, idxs = torch.topk(last, k)
    return [
        {
            "token": bridge.tokenizer.decode([int(i)]),
            "token_id": int(i),
            "logit": float(v),
            "prob": float(probs[int(i)]),
        }
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]


def predict(text: str, top_k: int = 10) -> dict:
    t0 = time.perf_counter()
    bridge = ADAPTER.bridge()
    toks = bridge.to_tokens(text)
    logits = bridge(toks)
    elapsed = time.perf_counter() - t0
    return {
        "tokens": bridge.to_str_tokens(text),
        "token_ids": toks[0].tolist(),
        "top_predictions": topk_predictions(logits, top_k),
        "elapsed_seconds": round(elapsed, 4),
        "device": ADAPTER._device,
    }


def capture(text: str, hook_names: list[str]) -> dict:
    """Capture named hooks for the given text. Bounded to short inputs."""
    if len(hook_names) > MAX_CAPTURE_HOOKS:
        raise InterventionError(f"too many hooks requested ({len(hook_names)} > {MAX_CAPTURE_HOOKS})")
    for h in hook_names:
        if not validate_hook_name(h, patchable=False):
            raise InterventionError(f"hook not supported: {h!r}")
    bridge = ADAPTER.bridge()
    toks = bridge.to_tokens(text)
    if toks.shape[1] > MAX_CAPTURE_TOKENS:
        raise InterventionError(
            f"input too long for activation capture ({toks.shape[1]} > {MAX_CAPTURE_TOKENS} tokens)"
        )
    _, cache = bridge.run_with_cache(toks, names_filter=lambda n: n in set(hook_names))
    out: dict[str, dict] = {}
    for h in hook_names:
        if h not in cache:
            raise InterventionError(f"hook did not produce a captured tensor: {h!r}")
        t = cache[h]
        # compact summary + optionally bounded raw slice for heatmaps
        entry = {
            "shape": list(t.shape),
            "mean": float(t.float().mean()),
            "std": float(t.float().std()),
            "absmax": float(t.float().abs().max()),
        }
        out[h] = entry
    return out, cache


def capture_flow(text: str) -> dict:
    """One real forward pass recording everything the flow playback needs.

    Returns exact tokens, per-block residual-stream L2 norms (measured from
    blocks.N.hook_out), and the full attention pattern for every block
    (rounded to 3 decimals for transport). This is recorded execution data,
    not a synthetic animation script.
    """
    bridge = ADAPTER.bridge()
    toks = bridge.to_tokens(text)
    if toks.shape[1] > MAX_CAPTURE_TOKENS:
        raise InterventionError("input too long for flow capture")
    n_layers = ADAPTER.metadata()["n_layers"]
    wanted = {f"blocks.{l}.hook_out" for l in range(n_layers)} | {
        f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)
    }
    logits, cache = bridge.run_with_cache(toks, names_filter=lambda n: n in wanted)
    blocks = []
    for l in range(n_layers):
        resid = cache[f"blocks.{l}.hook_out"][0]  # [seq, d_model]
        pat = cache[f"blocks.{l}.attn.hook_pattern"][0]  # [heads, q, k]
        blocks.append(
            {
                "layer": l,
                # per-token residual norm, then sequence mean — labeled aggregation
                "resid_norm_per_token": [round(float(x), 2) for x in resid.norm(dim=-1).tolist()],
                "resid_norm_mean": round(float(resid.norm(dim=-1).mean()), 2),
                "attention": [
                    [[round(float(x), 3) for x in key_row] for key_row in query_row]
                    for query_row in pat.tolist()
                ],
            }
        )
    return {
        "hook_basis": "blocks.{L}.hook_out + blocks.{L}.attn.hook_pattern (single forward pass)",
        "tokens": bridge.to_str_tokens(text),
        "token_ids": toks[0].tolist(),
        "n_layers": n_layers,
        "blocks": blocks,
        "top_predictions": topk_predictions(logits, 8),
    }


def attention_pattern(text: str, layer: int) -> dict:
    """Full attention pattern [n_heads, seq, seq] for one layer, as lists."""
    if not 0 <= layer < ADAPTER.metadata()["n_layers"]:
        raise InterventionError(f"layer out of range: {layer}")
    bridge = ADAPTER.bridge()
    toks = bridge.to_tokens(text)
    if toks.shape[1] > MAX_CAPTURE_TOKENS:
        raise InterventionError("input too long for attention capture")
    name = f"blocks.{layer}.attn.hook_pattern"
    _, cache = bridge.run_with_cache(toks, names_filter=name)
    pat = cache[name][0]  # [heads, query, key]
    return {
        "hook": name,
        "shape": list(pat.shape),
        "tokens": bridge.to_str_tokens(text),
        # round to 4 decimals for transport; full fidelity stays in the model run
        "pattern": [
            [[round(float(x), 4) for x in key_row] for key_row in query_row] for query_row in pat.tolist()
        ],
    }


def run_patched(
    corrupted_text: str,
    clean_cache: dict,
    hook_name: str,
    positions: list[int],
    heads: list[int] | None = None,
) -> dict:
    """Run the corrupted prompt with the clean activation spliced in.

    Supports tensors shaped [1, seq, d] (patch [0, pos, :]) and
    [1, seq, n_heads, d_head] (patch [0, pos, head, :] per selected head,
    all heads when ``heads`` is None).
    """
    if not validate_hook_name(hook_name, patchable=True):
        raise InterventionError(f"hook not patchable/supported: {hook_name!r}")
    if hook_name not in clean_cache:
        raise InterventionError("clean cache does not contain the requested hook")
    bridge = ADAPTER.bridge()
    src = clean_cache[hook_name].clone()  # never mutate the stored baseline
    n_seq = src.shape[1]
    for p in positions:
        if not 0 <= p < n_seq:
            raise InterventionError(
                f"position {p} out of range for {hook_name} (seq={n_seq}); "
                "clean/corrupted prompts must be token-aligned"
            )
    if src.dim() == 4:
        n_heads = src.shape[2]
        sel = heads if heads is not None else list(range(n_heads))
        for hd in sel:
            if not 0 <= hd < n_heads:
                raise InterventionError(f"head {hd} out of range (n_heads={n_heads})")

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        for p in positions:
            if z.dim() == 4:
                for hd in (heads if heads is not None else range(z.shape[2])):
                    z[0, p, hd, :] = src[0, p, hd, :]
            else:
                z[0, p, :] = src[0, p, :]
        return z

    t0 = time.perf_counter()
    logits = bridge.run_with_hooks(corrupted_text, fwd_hooks=[(hook_name, hook_fn)])
    elapsed = time.perf_counter() - t0
    return {
        "logits": logits,
        "top_predictions": topk_predictions(logits, 10),
        "elapsed_seconds": round(elapsed, 4),
        "hook": hook_name,
        "positions": positions,
        "heads": heads,
    }


def run_patched_pattern(
    corrupted_text: str,
    clean_cache: dict,
    hook_name: str,
    layer: int,
    head: int,
    query_positions: list[int],
) -> dict:
    """Replace rows (query positions) of an attention pattern for one head.

    hook_name must be blocks.{layer}.attn.hook_pattern; tensor [1, heads, seq, seq].
    """
    if not validate_hook_name(hook_name, patchable=False):
        raise InterventionError(f"hook not supported: {hook_name!r}")
    if hook_name != f"blocks.{layer}.attn.hook_pattern":
        raise InterventionError("pattern patch requires matching layer in hook name")
    if hook_name not in clean_cache:
        raise InterventionError("clean cache does not contain the requested pattern")
    bridge = ADAPTER.bridge()
    src = clean_cache[hook_name].clone()
    n_q = src.shape[2]
    for p in query_positions:
        if not 0 <= p < n_q:
            raise InterventionError(f"query position {p} out of range (seq={n_q})")

    def hook_fn(pat: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        for p in query_positions:
            pat[0, head, p, :] = src[0, head, p, :]
        return pat

    logits = bridge.run_with_hooks(corrupted_text, fwd_hooks=[(hook_name, hook_fn)])
    return {
        "logits": logits,
        "top_predictions": topk_predictions(logits, 10),
        "hook": hook_name,
        "head": head,
        "positions": query_positions,
    }
