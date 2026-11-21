"""The "agreement" job type (about one prompt pair).

Runs the exact per-component denoising pass, attribution patching, and DLA on
the same task pair, then stores the three vectors, scatter data, statistics,
top-10 lists and provenance. POST /api/agreement; poll
GET /api/experiments/{job_id}. The one-heavy-job rule is shared with every
other job type.
"""

from __future__ import annotations

import time

from ..analysis.agreement import compute_agreement
from ..model_runtime.loader import ADAPTER
from ..provenance import collect as collect_provenance
from ..tasks import registry as task_registry
from .jobs import ExperimentValidationError, JobContext, register_job_type


def validate_agreement_config(cfg: dict) -> dict:
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
    return {
        "task_id": task_id,
        "pair_id": pair.id,
        "clean_text": pair.clean_text,
        "corrupted_text": pair.corrupted_text,
        "candidate_a": pair.candidate_a,
        "candidate_b": pair.candidate_b,
        "job_type": "agreement",
    }


def run_agreement(cfg: dict, ctx: JobContext) -> dict:
    norm = validate_agreement_config(cfg)
    t0 = time.perf_counter()
    ADAPTER.ensure_loaded()
    ctx.report(0.01, "starting agreement measurement")
    result = compute_agreement(
        norm["clean_text"],
        norm["corrupted_text"],
        norm["candidate_a"],
        norm["candidate_b"],
        progress=ctx.report,
        check_cancel=ctx.check_cancel,
    )
    elapsed = time.perf_counter() - t0
    return {
        "job_type": "agreement",
        "task_id": norm["task_id"],
        "pair_id": norm["pair_id"],
        "direction": "denoising Rec_D vs Rec_AtP vs direct logit attribution (DLA)",
        "candidate_a": norm["candidate_a"],
        "candidate_b": norm["candidate_b"],
        **result,
        "provenance": {"elapsed_seconds": round(elapsed, 4), **collect_provenance()},
        "elapsed_seconds": round(elapsed, 4),
    }


def register() -> None:
    register_job_type("agreement", validate_agreement_config, run_agreement)
