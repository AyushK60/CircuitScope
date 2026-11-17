"""Patching sweeps as a registered job type ("sweep").

Two modes, both denoising (clean activations are copied into the corrupted
run, as in the existing Playground — this direction is fixed and reported in
every result):

- ``layer_position``: for layer L in 0..11 and position p in 0..seq-1, patch
  blocks.{L}.hook_out at [p]. Grid 12 x seq.
- ``layer_head``: for layer L and head h, patch blocks.{L}.attn.hook_z for
  head h at ALL positions. Grid 12 x 12.

The clean cache (all needed hooks) is captured ONCE and reused for every
cell; the corrupted baseline runs once. Metric per cell: D_patched and
Recovery = (D_patched - D_corrupted) / (D_clean - D_corrupted).
"""

from __future__ import annotations

import time

import torch

from ..config import MAX_LAYERS
from ..model_runtime import engine
from ..model_runtime.loader import ADAPTER
from ..provenance import collect as collect_provenance
from ..tasks import registry as task_registry
from .jobs import ExperimentValidationError, JobContext, register_job_type

MIN_BASELINE_DIFF = 1e-3  # below this the sweep is rejected
UNSTABLE_BASELINE_DIFF = 0.5  # below this Recovery is unstable (warning)
DIRECTION = "clean -> corrupted (denoising)"


def validate_sweep_config(cfg: dict) -> dict:
    """Validate a sweep spec against the task registry; returns the normalized
    config persisted with the job (must contain job_type: "sweep")."""
    task_id = cfg.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ExperimentValidationError("task_id is required")
    task = task_registry.get_task(task_id)
    if task is None:
        raise ExperimentValidationError(f"unknown task: {task_id!r}")
    validation = task_registry.validate_task(task)
    if not validation.get("ok"):
        raise ExperimentValidationError(
            f"task {task_id} failed registry validation: {'; '.join(validation.get('errors', []))}"
        )
    pair_id = cfg.get("pair_id")
    pair = task.pairs[0] if pair_id is None else next((p for p in task.pairs if p.id == pair_id), None)
    if pair is None:
        raise ExperimentValidationError(f"unknown pair {pair_id!r} for task {task_id!r}")
    mode = cfg.get("mode")
    if mode not in ("layer_position", "layer_head"):
        raise ExperimentValidationError("mode must be 'layer_position' or 'layer_head'")
    return {
        "task_id": task_id,
        "pair_id": pair.id,
        "mode": mode,
        "clean_text": pair.clean_text,
        "corrupted_text": pair.corrupted_text,
        "candidate_a": pair.candidate_a,
        "candidate_b": pair.candidate_b,
        "job_type": "sweep",  # persisted so history can tell sweeps apart
    }


def check_baselines(d_clean: float, d_corrupted: float) -> list[str]:
    """Gate the sweep on the baseline contrast. Returns warnings; raises when
    the sweep would be meaningless (near-identical baselines)."""
    denominator = d_clean - d_corrupted
    warnings: list[str] = []
    if abs(denominator) < MIN_BASELINE_DIFF:
        raise ExperimentValidationError(
            f"clean and corrupted baselines are nearly identical (|D_clean - D_corrupted| = "
            f"{abs(denominator):.2e} < {MIN_BASELINE_DIFF}); Recovery would be meaningless. "
            "Provide a corrupted prompt that actually changes the model's answer."
        )
    if abs(denominator) < UNSTABLE_BASELINE_DIFF:
        warnings.append(
            f"|D_clean - D_corrupted| = {abs(denominator):.3f} < {UNSTABLE_BASELINE_DIFF}: "
            "Recovery values are unstable (small denominators amplify noise)."
        )
    return warnings


def run_sweep(cfg: dict, ctx: JobContext) -> dict:
    norm = validate_sweep_config(cfg)
    bridge = ADAPTER.ensure_loaded()
    tok_a = ADAPTER.token_id_for_str(norm["candidate_a"])
    tok_b = ADAPTER.token_id_for_str(norm["candidate_b"])

    toks_clean = bridge.to_tokens(norm["clean_text"])
    toks_corr = bridge.to_tokens(norm["corrupted_text"])
    seq = toks_clean.shape[1]
    if toks_clean.shape != toks_corr.shape:
        raise ExperimentValidationError(
            f"clean/corrupted prompts are not token-aligned: {toks_clean.shape[1]} vs {toks_corr.shape[1]} tokens"
        )

    ctx.report(0.02, "capturing clean cache (one pass, all hooks)")
    # One clean forward pass captures every hook both modes need.
    wanted = {f"blocks.{l}.hook_out" for l in range(MAX_LAYERS)} | {
        f"blocks.{l}.attn.hook_z" for l in range(MAX_LAYERS)
    }
    logits_clean, clean_cache = bridge.run_with_cache(toks_clean, names_filter=lambda n: n in wanted)
    for name in wanted:
        if name not in clean_cache:
            raise ExperimentValidationError(f"hook was not captured during the clean run: {name!r}")

    ctx.report(0.04, "running corrupted baseline")
    logits_corr = bridge(toks_corr)

    def d_of(logits: torch.Tensor) -> float:
        return float(logits[0, -1, tok_a] - logits[0, -1, tok_b])

    d_clean = d_of(logits_clean)
    d_corrupted = d_of(logits_corr)
    denominator = d_clean - d_corrupted
    warnings = check_baselines(d_clean, d_corrupted)

    mode = norm["mode"]
    if mode == "layer_position":
        col_labels = bridge.to_str_tokens(norm["clean_text"])
        cells = [(l, p) for l in range(MAX_LAYERS) for p in range(seq)]
    else:
        col_labels = [f"head {h}" for h in range(bridge.cfg.n_heads)]
        cells = [(l, h) for l in range(MAX_LAYERS) for h in range(bridge.cfg.n_heads)]
    total = len(cells)

    recovery = [[0.0] * len(col_labels) for _ in range(MAX_LAYERS)]
    d_patched = [[0.0] * len(col_labels) for _ in range(MAX_LAYERS)]

    t0 = time.perf_counter()
    for i, (l, c) in enumerate(cells):
        ctx.check_cancel()
        if mode == "layer_position":
            hook = f"blocks.{l}.hook_out"
            positions = [c]
            heads = None
            label = f"layer {l}, position {c}"
        else:
            hook = f"blocks.{l}.attn.hook_z"
            positions = list(range(seq))
            heads = [c]
            label = f"layer {l}, head {c}"
        logits = engine.patched_logits(toks_corr, clean_cache, hook, positions, heads)
        d_p = d_of(logits)
        d_patched[l][c] = d_p
        recovery[l][c] = (d_p - d_corrupted) / denominator
        ctx.report((i + 1) / total, f"{label} ({i + 1}/{total})")
    elapsed = time.perf_counter() - t0

    return {
        "job_type": "sweep",
        "mode": mode,
        "task_id": norm["task_id"],
        "pair_id": norm["pair_id"],
        "direction": DIRECTION,
        "tokens_clean": bridge.to_str_tokens(norm["clean_text"]),
        "tokens_corrupted": bridge.to_str_tokens(norm["corrupted_text"]),
        "candidate_a": norm["candidate_a"],
        "candidate_b": norm["candidate_b"],
        "baselines": {
            "D_clean": d_clean,
            "D_corrupted": d_corrupted,
            "denominator": denominator,
            "warnings": warnings,
        },
        "row_labels": [f"block {l}" for l in range(MAX_LAYERS)],
        "col_labels": col_labels,
        "recovery": recovery,
        "d_patched": d_patched,
        "n_runs": total,
        "elapsed_seconds": round(elapsed, 4),
        "provenance": {"elapsed_seconds": round(elapsed, 4), **collect_provenance()},
    }


# Registering at import time; routes.py imports this module so the app and the
# tests share one registry.
register_job_type("sweep", validate_sweep_config, run_sweep)
