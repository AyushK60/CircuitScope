"""Do exact patching, attribution patching and DLA agree?

For one task pair this module computes, for all 156 components (144 heads and
12 MLPs), three score vectors on the same scale as the study:

- exact denoising ``Rec_D`` (clean activations copied into the pair's own
  corrupted run; ``analysis/components.py``);
- attribution patching ``Rec_AtP`` (first-order approximation;
  ``analysis/atp.py``);
- direct logit attribution on the clean prompt (dollars of raw logit units,
  ``analysis/dla.py``).

The figures and claims report rank agreement on the 144 heads, top-10 overlap
against the random expectation, and Pearson correlation separately for blocks
0-5 and 6-11. Agreement is evidence about these scores on this prompt pair; it
is not evidence that any method measures the same physical thing.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import components as comps
from .atp import LIMITATION_NOTE as ATP_NOTE
from .atp import compute_atp
from .components import COMPONENT_NAMES, N_COMPONENTS
from .dla import compute_dla
from .stats import expected_random_overlap, pearson, spearman, topk_overlap

N_HEADS = 144
TOP_K = 10
GROUP_SLICES = {
    "blocks_0_5": [l * 12 + h for l in range(0, 6) for h in range(12)],
    "blocks_6_11": [l * 12 + h for l in range(6, 12) for h in range(12)],
}

EXACT_NOTE = (
    "Exact patching copies one component's clean activation into the corrupted "
    "run and re-runs the model. It is the reference here because it measures "
    "the effect directly, one component at a time. It tests sufficiency for "
    "this corruption, not necessity."
)
DLA_NOTE = (
    "Direct logit attribution reads how much each component writes straight "
    "into the final score gap. It ignores effects that pass through later "
    "blocks, so a low DLA value does not mean a component is unimportant."
)


def _clean(vec) -> list[float] | None:
    if vec is None:
        return None
    if any(v is None or not np.isfinite(float(v)) for v in vec):
        return None
    return [float(v) for v in vec]


def _rank_list(values: list[float], k: int = TOP_K) -> list[dict]:
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    out = []
    for rank, i in enumerate(order, start=1):
        c = COMPONENT_NAMES[i]
        out.append({"rank": rank, "label": c, "index": i, "value": float(values[i])})
    return out


def _group_stats(exact: list[float], other: list[float]) -> dict:
    out = {}
    for name, idx in GROUP_SLICES.items():
        ex = [exact[i] for i in idx]
        ot = [other[i] for i in idx]
        out[name] = {"pearson": pearson(ex, ot), "n": len(idx)}
    return out


def _method_stats(exact: list[float], other: list[float]) -> dict:
    ex_heads = exact[:N_HEADS]
    ot_heads = other[:N_HEADS]
    return {
        "pearson_heads": pearson(ex_heads, ot_heads),
        "spearman_heads": spearman(ex_heads, ot_heads),
        "top10_overlap": topk_overlap(ex_heads, ot_heads, TOP_K),
        "expected_random_top10": expected_random_overlap(TOP_K, N_HEADS),
        "pearson_groups": _group_stats(exact, other),
    }


def exact_head_vs_sweep(exact: list[float | None], sweep: dict) -> dict:
    """Consistency gate against a layer_head sweep for the same pair.

    Both paths patch ``blocks.{l}.attn.hook_z`` for one head at all positions,
    so the per-head recoveries must agree to within 1e-4 (same backend, same
    pass). A larger gap is reported as a failure.
    """
    rec = sweep.get("recovery") or []
    max_diff = 0.0
    compared = 0
    for l in range(min(12, len(rec))):
        for h in range(min(12, len(rec[l]))):
            v = exact[l * 12 + h]
            if v is None:
                return {"max_abs_diff": None, "tolerance": 1e-4, "ok": False, "compared": compared}
            max_diff = max(max_diff, abs(float(v) - float(rec[l][h])))
            compared += 1
    return {
        "max_abs_diff": max_diff,
        "tolerance": 1e-4,
        "ok": compared == N_HEADS and max_diff <= 1e-4,
        "compared": compared,
    }


def compute_agreement(
    clean_text: str,
    corrupted_text: str,
    candidate_a: str,
    candidate_b: str,
    progress: Callable[[float, str], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> dict:
    """Compute the three vectors and their agreement statistics."""

    def report(p: float, message: str) -> None:
        if progress is not None:
            progress(p, message)

    def cancel() -> None:
        if check_cancel is not None:
            check_cancel()

    report(0.02, "capturing the clean run")
    cancel()
    logits_clean, clean_cache = comps.run_clean(clean_text)

    report(0.08, "exact patching: all 156 components")
    exact_effect = comps.denoise_effects(
        corrupted_text,
        candidate_a,
        candidate_b,
        logits_clean,
        clean_cache,
        check_cancel=check_cancel,
    )
    exact = _clean(exact_effect["rec_d"])
    if exact is None:
        raise ValueError("exact patching produced undefined recoveries (near-zero denominator)")
    report(0.62, "attribution patching")
    cancel()
    atp = compute_atp(corrupted_text, candidate_a, candidate_b, logits_clean, clean_cache)
    atp_vec = _clean(atp["rec_atp"])
    if atp_vec is None:
        raise ValueError("attribution patching produced undefined recoveries")

    report(0.78, "direct logit attribution")
    cancel()
    dla = compute_dla(clean_text, candidate_a, candidate_b)
    dla_vec: list[float | None] = [None] * N_COMPONENTS
    for c in dla["components"]:
        if c["kind"] == "head":
            dla_vec[c["layer"] * 12 + c["head"]] = c["value"]
        elif c["kind"] == "mlp":
            dla_vec[144 + c["layer"]] = c["value"]
    if any(v is None for v in dla_vec):
        raise ValueError("direct logit attribution did not cover all 156 components")
    dla_clean = [float(v) for v in dla_vec]

    report(0.92, "computing agreement statistics")
    stats = {
        "atp": _method_stats(exact, atp_vec),
        "dla": _method_stats(exact, dla_clean),
        "procedure": (
            "Pearson over the 144 head values (values, not ranks); Spearman over "
            "the same heads; top-10 overlap on the head values; Pearson by layer "
            "group (blocks 0-5 vs 6-11); random top-10 expectation k*k/144."
        ),
    }

    labels = []
    scatter = []
    for i, name in enumerate(COMPONENT_NAMES):
        kind = "head" if i < N_HEADS else "mlp"
        layer = i // 12 if kind == "head" else i - N_HEADS
        head = i % 12 if kind == "head" else None
        labels.append({"label": name, "kind": kind, "layer": layer, "head": head})
        scatter.append(
            {
                "label": name,
                "kind": kind,
                "layer": layer,
                "head": head,
                "exact": float(exact[i]),
                "atp": float(atp_vec[i]),
                "dla": float(dla_clean[i]),
            }
        )

    return {
        "labels": labels,
        "vectors": {"exact": exact, "atp": atp_vec, "dla": dla_clean},
        "scatter": scatter,
        "stats": stats,
        "top10": {
            "exact": _rank_list(exact[:N_HEADS]),
            "atp": _rank_list(atp_vec[:N_HEADS]),
            "dla": _rank_list(dla_clean[:N_HEADS]),
        },
        "baselines": {
            "D_clean": exact_effect["D_clean"],
            "D_corrupted": exact_effect["D_corr"],
            "denominator": exact_effect["denominator_d"],
        },
        "dla_reconciliation": dla["reconciliation"],
        "notes": [EXACT_NOTE, ATP_NOTE, DLA_NOTE],
        "method_notes": {
            "exact": EXACT_NOTE,
            "atp": ATP_NOTE,
            "dla": DLA_NOTE,
        },
    }
