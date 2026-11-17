"""Direct logit attribution (DLA) at the final position.

Question answered: "how much did each component directly push the model toward
candidate A over candidate B at the final position?" This is a PARTIAL view:
it ignores effects that flow through later layers.

Math (no library shortcuts):
    r      = residual stream before the final LayerNorm at the final position
             (blocks.11.hook_out[0, -1]); it is a sum of component outputs:
             r = embed + pos_embed + sum_L (attn_out_L + mlp_out_L), and
             attn_out_L = sum_h (z_{L,h} @ W_O[L, h]) + b_O[L]
    LN(r)  = (r - mean(r)) / s * gamma + beta,   s = sqrt(var(r) + eps)
             (biased variance; eps read from the model, not hardcoded)
    w      = W_U[:, a] - W_U[:, b]   (GPT-2 has no unembedding bias)
    D      = LN(r) . w

    For any component output c (a 768-vector at the final position):
        DLA(c) = ((c - mean(c)) / s) * gamma . w      # s from the full r

The mean-subtraction is linear, so the terms telescope exactly:
sum_c (c - mean(c)) = r - mean(r). Therefore

    sum(DLA(c)) + beta . w == D

which is the reconciliation identity checked on every call.
"""

from __future__ import annotations

import torch

from ..config import MAX_LAYERS
from ..model_runtime.loader import ADAPTER

RECONCILIATION_TOLERANCE = 2e-3

NOTES = [
    "Direct effect only: ignores effects mediated by later layers.",
    "The per-component terms share the final LayerNorm scale s (computed from the full residual), "
    "so the terms are linear reads of the residual stream, not independent reruns.",
    "The sum of all terms reproduces the model's real logit difference D exactly (reconciliation); "
    "this checks the arithmetic, not the causal importance of any component.",
]


def _ln_epsilon(bridge) -> float:  # noqa: ANN001
    """Find the LayerNorm epsilon from the model itself, never hardcoded blindly."""
    eps = getattr(bridge.ln_final, "eps", None)
    if eps is None:
        eps = getattr(bridge.original_model.config, "layer_norm_epsilon", None)
    if eps is None:
        raise ValueError("could not find the LayerNorm epsilon on the model")
    return float(eps)


def compute_dla(text: str, candidate_a: str, candidate_b: str) -> dict:
    """Compute per-component DLA at the final position for one prompt.

    Runs ONE forward pass capturing every needed hook. ``candidate_a`` /
    ``candidate_b`` are single-token strings (leading spaces intact).
    """
    bridge = ADAPTER.ensure_loaded()
    tok_a = ADAPTER.token_id_for_str(candidate_a)
    tok_b = ADAPTER.token_id_for_str(candidate_b)

    wanted = {"hook_embed", "hook_pos_embed", f"blocks.{MAX_LAYERS - 1}.hook_out"} | {
        name
        for l in range(MAX_LAYERS)
        for name in (f"blocks.{l}.attn.hook_z", f"blocks.{l}.mlp.hook_out")
    }
    toks = bridge.to_tokens(text)
    logits, cache = bridge.run_with_cache(toks, names_filter=lambda n: n in wanted)
    for name in wanted:
        if name not in cache:
            raise ValueError(f"hook was not captured during the run: {name!r}")

    pos = toks.shape[1] - 1  # final position
    r = cache[f"blocks.{MAX_LAYERS - 1}.hook_out"][0, pos]
    mean_r = r.mean()
    s = torch.sqrt(r.var(unbiased=False) + _ln_epsilon(bridge))
    gamma = bridge.ln_final.weight
    beta = bridge.ln_final.bias
    W_U = bridge.unembed.W_U
    w = W_U[:, tok_a] - W_U[:, tok_b]
    # direction vector: dividing by s once here is equivalent to applying
    # ((c - mean(c)) / s * gamma) . w per component.
    direction = (gamma * w) / s

    def dla_of(c: torch.Tensor) -> float:
        return float(((c - c.mean()) @ direction).item())

    components: list[dict] = []

    embed = cache["hook_embed"][0, pos]
    pos_embed = cache["hook_pos_embed"][0, pos]
    components.append({"name": "embed", "kind": "embed", "layer": None, "head": None, "value": dla_of(embed)})
    components.append(
        {"name": "pos_embed", "kind": "pos_embed", "layer": None, "head": None, "value": dla_of(pos_embed)}
    )

    for l in range(MAX_LAYERS):
        z = cache[f"blocks.{l}.attn.hook_z"][0, pos]  # [n_heads, d_head]
        W_O = bridge.blocks[l].attn.W_O  # [n_heads, d_head, d_model]
        head_out = torch.einsum("hd,hdo->ho", z, W_O)  # [n_heads, d_model]
        for h in range(head_out.shape[0]):
            components.append(
                {
                    "name": f"head {l}.{h}",
                    "kind": "head",
                    "layer": l,
                    "head": h,
                    "value": dla_of(head_out[h]),
                }
            )
        b_O = bridge.blocks[l].attn.b_O
        components.append(
            {"name": f"attn_bias.{l}", "kind": "attn_bias", "layer": l, "head": None, "value": dla_of(b_O)}
        )
        mlp_out = cache[f"blocks.{l}.mlp.hook_out"][0, pos]
        components.append({"name": f"mlp.{l}", "kind": "mlp", "layer": l, "head": None, "value": dla_of(mlp_out)})

    components.append(
        {
            "name": "ln_final_bias",
            "kind": "ln_final_bias",
            "layer": None,
            "head": None,
            "value": float((beta @ w).item()),
        }
    )

    model_D = float(logits[0, pos, tok_a] - logits[0, pos, tok_b])
    sum_of_terms = sum(c["value"] for c in components)
    abs_error = abs(sum_of_terms - model_D)

    return {
        "tokens": bridge.to_str_tokens(text),
        "position": pos,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "D": model_D,
        "components": components,
        "reconciliation": {
            "sum_of_terms": sum_of_terms,
            "model_D": model_D,
            "abs_error": abs_error,
            "tolerance": RECONCILIATION_TOLERANCE,
            "ok": bool(abs_error <= RECONCILIATION_TOLERANCE),
        },
        "notes": list(NOTES),
    }
