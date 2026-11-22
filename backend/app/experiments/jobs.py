"""Experiment execution: validation, the clean/corrupted/patched pipeline,
and a bounded serial worker (one heavy experiment at a time).

Job types are registered in JOB_TYPES (name -> validate/run). The original
pipeline is registered as type "intervention"; ctx.report()/ctx.check_cancel()
give run functions progress reporting and cooperative cancellation without
string-matching exception messages.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..config import (
    MAX_LAYERS,
    MAX_PROMPT_CHARS,
    MAX_TOP_K,
    S2_POSITION,
    validate_hook_name,
)
from ..model_runtime import engine
from ..model_runtime.loader import ADAPTER
from ..provenance import collect as collect_provenance
from ..storage import db
from .metrics import compare_metrics


class ExperimentValidationError(ValueError):
    pass


class JobCancelled(Exception):
    """Raised via ctx.check_cancel() when the job's cancel flag is set.
    Deliberately NOT a subclass of ExperimentValidationError."""


@dataclass
class JobRecord:
    job_id: str
    config: dict
    status: str = "queued"  # queued | running | complete | failed | cancelled
    results: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    type: str = "intervention"
    progress: float = 0.0
    progress_message: str = ""


def validate_experiment_config(cfg: dict) -> dict:
    """Validate an experiment spec; returns a normalized config dict."""
    clean = cfg.get("clean_text") or ""
    corr = cfg.get("corrupted_text") or ""
    if not clean or not corr:
        raise ExperimentValidationError("clean_text and corrupted_text are required")
    if len(clean) > MAX_PROMPT_CHARS or len(corr) > MAX_PROMPT_CHARS:
        raise ExperimentValidationError("prompt exceeds MAX_PROMPT_CHARS")
    hook = cfg.get("hook")
    if not hook or not validate_hook_name(hook, patchable=True):
        raise ExperimentValidationError(f"hook not patchable/supported: {hook!r}")
    positions = cfg.get("positions")
    if not isinstance(positions, list) or not positions or not all(
        isinstance(p, int) and p >= 0 for p in positions
    ):
        raise ExperimentValidationError("positions must be a non-empty list of non-negative ints")
    heads = cfg.get("heads")
    if heads is not None:
        if not isinstance(heads, list) or not all(isinstance(h, int) and 0 <= h < MAX_LAYERS for h in heads):
            raise ExperimentValidationError(f"heads must be ints in [0, {MAX_LAYERS})")
    cand_a = cfg.get("candidate_a")
    cand_b = cfg.get("candidate_b")
    if not cand_a or not cand_b:
        raise ExperimentValidationError("candidate_a and candidate_b (leading-space strings) are required")
    return {
        "clean_text": clean,
        "corrupted_text": corr,
        "hook": hook,
        "positions": positions,
        "heads": heads,
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "top_k": min(int(cfg.get("top_k", 8)), MAX_TOP_K),
    }


def execute_experiment(
    cfg: dict,
    cancel_flag: threading.Event | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Run clean -> corrupted -> patched with the real model; return full results.

    ``progress(value, message)`` is called at fixed milestones (0.05 after
    validation, 0.2 after the clean run, 0.5 after the corrupted run, 0.9 after
    the patched run); the runner reports 1.0 when the job completes. The return
    value is unchanged.
    """
    norm = validate_experiment_config(cfg)
    bridge = ADAPTER.ensure_loaded()
    tok = bridge.tokenizer

    def check_cancel():
        if cancel_flag is not None and cancel_flag.is_set():
            raise JobCancelled("cancelled")

    def report(value: float, message: str) -> None:
        if progress is not None:
            progress(value, message)

    check_cancel()
    # Candidate validity: both must be single tokens with leading spaces intact.
    tid_a = ADAPTER.token_id_for_str(norm["candidate_a"])
    tid_b = ADAPTER.token_id_for_str(norm["candidate_b"])

    # Paired-input validity: token lengths must match; positions must be in range.
    toks_clean = bridge.to_tokens(norm["clean_text"])
    toks_corr = bridge.to_tokens(norm["corrupted_text"])
    if toks_clean.shape != toks_corr.shape:
        raise ExperimentValidationError(
            f"clean/corrupted prompts are not token-aligned: {toks_clean.shape[1]} vs {toks_corr.shape[1]} tokens"
        )
    if norm["clean_text"] == norm["corrupted_text"]:
        raise ExperimentValidationError(
            "clean and corrupted prompts are identical - the patch would have no meaningful contrast. "
            "Provide a corrupted variant (e.g. swap the subject) as in the John/Mary preset."
        )
    seq = toks_clean.shape[1]
    if any(p >= seq for p in norm["positions"]):
        raise ExperimentValidationError(f"position out of range for sequence length {seq}")

    report(0.05, "validated - starting clean run")
    t0 = time.perf_counter()
    check_cancel()
    logits_clean, clean_cache = bridge.run_with_cache(
        toks_clean, names_filter=lambda n: n == norm["hook"]
    )
    if norm["hook"] not in clean_cache:
        raise ExperimentValidationError("requested hook was not captured during clean run")
    report(0.2, "clean run complete")
    check_cancel()
    logits_corr = bridge(toks_corr)
    report(0.5, "corrupted run complete")
    check_cancel()
    patched = engine.run_patched(
        toks_corr,
        clean_cache,
        norm["hook"],
        norm["positions"],
        norm["heads"],
    )
    report(0.9, "patched run complete")
    elapsed = time.perf_counter() - t0

    metrics = compare_metrics(logits_clean, logits_corr, patched["logits"], tid_a, tid_b)
    return {
        "config": norm,
        "provenance": {
            "model_id": "gpt2",
            "library": "transformer_lens TransformerBridge",
            "device": ADAPTER._device,
            "torch": torch.__version__,
            "elapsed_seconds": round(elapsed, 4),
            **collect_provenance(),
        },
        "tokens_clean": bridge.to_str_tokens(norm["clean_text"]),
        "tokens_corrupted": bridge.to_str_tokens(norm["corrupted_text"]),
        "runs": {
            "clean": engine.topk_predictions(logits_clean, norm["top_k"]),
            "corrupted": engine.topk_predictions(logits_corr, norm["top_k"]),
            "patched": patched["top_predictions"],
        },
        "metrics": metrics,
        "patch": {
            "hook": norm["hook"],
            "positions": norm["positions"],
            "heads": norm["heads"],
            "source": "clean run activation (cloned, never mutated)",
        },
    }


class JobContext:
    """Passed to job-type run functions: clamped monotonic progress reporting
    and cooperative cancellation."""

    def __init__(self, runner: "ExperimentRunner", job: JobRecord, cancel_flag: threading.Event | None) -> None:
        self._runner = runner
        self._job = job
        self.job_id = job.job_id
        self.config = job.config
        self.cancel_flag = cancel_flag  # read by run fns that forward it downstream

    def report(self, progress: float, message: str) -> None:
        """Clamp to [0, 1]; never let the stored progress decrease."""
        try:
            p = max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError):
            p = 0.0
        with self._runner._lock:
            if p < self._job.progress:
                p = self._job.progress
            self._job.progress = p
            self._job.progress_message = str(message)

    def check_cancel(self) -> None:
        if self.cancel_flag is not None and self.cancel_flag.is_set():
            raise JobCancelled("cancelled")


# Job-type registry: name -> {"validate": fn(cfg), "run": fn(config, ctx)}.
JOB_TYPES: dict[str, dict] = {}


def register_job_type(name: str, validate_fn: Callable, run_fn: Callable) -> None:
    JOB_TYPES[name] = {"validate": validate_fn, "run": run_fn}


def _run_intervention(cfg: dict, ctx: JobContext) -> dict:
    return execute_experiment(cfg, cancel_flag=ctx.cancel_flag, progress=ctx.report)


# The original clean/corrupted/patched pipeline, registered as a job type.
register_job_type("intervention", validate_experiment_config, _run_intervention)


class ExperimentRunner:
    """Bounded serial worker: one heavy experiment at a time."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel_flags: dict[str, threading.Event] = {}

    def submit(self, cfg: dict, job_type: str = "intervention") -> JobRecord:
        spec = JOB_TYPES.get(job_type)
        if spec is None:
            raise ExperimentValidationError(f"unknown job type: {job_type!r}")
        norm = spec["validate"](cfg)  # fail fast on invalid specs
        with self._lock:
            running = any(j.status in ("queued", "running") for j in self._jobs.values())
            if running:
                raise ExperimentValidationError("an experiment is already running; wait for it to finish")
            job = JobRecord(job_id=uuid.uuid4().hex[:12], config=norm, type=job_type)
            self._jobs[job.job_id] = job
            self._cancel_flags[job.job_id] = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(job.job_id,), daemon=True)
        self._thread.start()
        return job

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            flag = self._cancel_flags[job_id]
            job.status = "running"
        ctx = JobContext(self, job, flag)
        try:
            # Lookup inside the try: an unknown job type must fail the job
            # (and release the worker), never kill the thread and leave the
            # job stuck in "running" forever.
            spec = JOB_TYPES.get(job.type)
            if spec is None:
                raise ExperimentValidationError(f"unknown job type: {job.type!r}")
            results = spec["run"](job.config, ctx)
            ctx.report(1.0, "complete")
            # Persist BEFORE marking complete, so "complete" always implies the
            # record is retrievable from history. If persistence fails, the job
            # is still marked complete (with the stderr warning kept).
            try:
                db.save_experiment(job.config, results, "complete", exp_id=job_id)
            except Exception as e:
                import sys
                print(f"PERSIST WARNING: experiment results not saved: {type(e).__name__}: {e}", file=sys.stderr)
            with self._lock:
                job.results = results
                job.status = "complete"
                job.finished_at = time.time()
        except JobCancelled:
            with self._lock:
                job.status = "cancelled"
                job.error = "cancelled"
                job.finished_at = time.time()
        except ExperimentValidationError as e:
            with self._lock:
                job.status = "failed"
                job.error = str(e)
                job.finished_at = time.time()
        except Exception as e:  # surface the real failure
            with self._lock:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.finished_at = time.time()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            flag = self._cancel_flags.get(job_id)
            if job is None or flag is None:
                return False
            if job.status in ("queued", "running"):
                flag.set()
                return True
            return False

    def recent(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]


RUNNER = ExperimentRunner()
