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
    MAX_PROMPT_CHARS,
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


def continue_text(
    text: str,
    n_tokens: int = 5,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
) -> dict:
    """Autoregressive continuation with the real model.

    mode="greedy": deterministic argmax decoding.
    mode="sample": temperature softmax with optional top-k filtering; a seeded
    torch.Generator makes a run reproducible for the same seed.
    Every step re-runs the model on the growing prefix and records the measured
    candidate distribution at that position.
    """
    if not 1 <= n_tokens <= 8:
        raise InterventionError("n_tokens must be in [1, 8]")
    if mode not in ("greedy", "sample"):
        raise InterventionError("mode must be 'greedy' or 'sample'")
    if not 0.05 <= temperature <= 2.0:
        raise InterventionError("temperature must be in [0.05, 2.0]")
    if not 0 <= top_k <= 50:
        raise InterventionError("top_k must be in [0, 50] (0 = off)")
    bridge = ADAPTER.bridge()
    gen = None
    if mode == "sample":
        gen = torch.Generator()
        gen.manual_seed(seed if seed is not None else 0)
    current = text
    steps = []
    t0 = time.perf_counter()
    for i in range(n_tokens):
        toks = bridge.to_tokens(current)
        if toks.shape[1] >= MAX_CAPTURE_TOKENS:
            raise InterventionError("continuation exceeded the token limit")
        logits = bridge(toks)[0, -1]
        top = topk_predictions(logits.unsqueeze(0).unsqueeze(0), 5)
        if mode == "greedy":
            chosen_idx = int(torch.argmax(logits))
        else:
            scaled = logits / max(temperature, 1e-5)
            if top_k > 0:
                kth = torch.topk(scaled, min(top_k, scaled.shape[0])).values[-1]
                scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
            probs = torch.softmax(scaled, dim=-1)
            chosen_idx = int(torch.multinomial(probs, 1, generator=gen))
        chosen_tok = bridge.tokenizer.decode([chosen_idx])
        chosen_prob = float(torch.softmax(logits, dim=-1)[chosen_idx])
        steps.append(
            {
                "step": i + 1,
                "input_so_far": current,
                "chosen": chosen_tok,
                "chosen_id": chosen_idx,
                "chosen_prob": round(chosen_prob, 5),
                "top_predictions": top,
            }
        )
        current = current + chosen_tok
    return {
        "text_start": text,
        "text_final": current,
        "n_tokens": n_tokens,
        "steps": steps,
        "elapsed_seconds": round(time.perf_counter() - t0, 4),
        "mode": mode,
        "params": {"temperature": temperature, "top_k": top_k, "seed": seed},
        "decoding": (
            "greedy (argmax) — deterministic, no sampling"
            if mode == "greedy"
            else f"sampling · temperature {temperature} · top_k {top_k or 'off'} · seed {seed if seed is not None else 0}"
        ),
    }


def compare_candidates(logits: torch.Tensor, token_a: int, token_b: int) -> dict:
    """Measured candidate comparison at the final position."""
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    d = float(last[token_a] - last[token_b])
    return {
        "logit_a": float(last[token_a]),
        "logit_b": float(last[token_b]),
        "prob_a": float(probs[token_a]),
        "prob_b": float(probs[token_b]),
        "D": round(d, 5),
    }


def score_question_candidates(
    context: str,
    question: str,
    candidates: list[str],
    template: str | None = None,
) -> dict:
    """Question & Candidate Explorer scoring.

    Constructs a documented prompt template ending in an answer prefix, then
    scores every candidate by conditional token log-probability (teacher
    forcing). Multi-token candidates use the SUM of token log-probs under the
    same prompt prefix — never just the first token. Also reports the first-
    token probability and the length-normalized mean log-prob so the three
    metrics can be distinguished. These are continuation scores, NOT a claim
    that GPT-2 answers questions correctly.
    """
    if not 2 <= len(candidates) <= 5:
        raise InterventionError("provide between 2 and 5 candidates")
    if template and "{context}" not in template and "{question}" not in template:
        raise InterventionError("template must contain {context} and {question} placeholders")
    prompt = (
        template.format(context=context, question=question)
        if template
        else f"{context}\nQuestion: {question}\nAnswer:"
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InterventionError("constructed prompt exceeds the character limit")
    bridge = ADAPTER.bridge()
    prompt_ids = bridge.to_tokens(prompt)[0].tolist()  # includes BOS
    n_prompt = len(prompt_ids)
    scored = []
    t0 = time.perf_counter()
    for cand in candidates:
        cand_ids = bridge.tokenizer(cand, add_special_tokens=False)["input_ids"]
        if not cand_ids:
            raise InterventionError(f"candidate {cand!r} produced no tokens")
        full = torch.tensor([prompt_ids + cand_ids])
        logits = bridge(full)
        logprobs = torch.log_softmax(logits[0], dim=-1)
        # candidate token j is predicted at position n_prompt - 1 + j
        lp = [float(logprobs[n_prompt - 1 + j, tid]) for j, tid in enumerate(cand_ids)]
        first_prob = float(torch.softmax(logits[0, n_prompt - 1], dim=-1)[cand_ids[0]])
        scored.append(
            {
                "candidate": cand,
                "tokens": bridge.tokenizer.convert_ids_to_tokens(cand_ids),
                "token_ids": cand_ids,
                "n_tokens": len(cand_ids),
                "total_logprob": round(sum(lp), 4),
                "mean_logprob": round(sum(lp) / len(lp), 4),
                "first_token_prob": round(first_prob, 5),
                "per_token_logprob": [round(v, 4) for v in lp],
            }
        )
    best = max(scored, key=lambda s: s["total_logprob"])
    return {
        "prompt": prompt,
        "context": context,
        "question": question,
        "candidates": scored,
        "preferred_by_total_logprob": best["candidate"],
        "method": (
            "conditional token log-probability under the fixed prompt prefix "
            "(teacher forcing). total = sum over candidate tokens; mean = "
            "length-normalized; first = probability of the candidate's first "
            "token only. These are continuation scores, not correctness claims."
        ),
        "length_warning": len({s["n_tokens"] for s in scored}) > 1,
        "elapsed_seconds": round(time.perf_counter() - t0, 4),
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
