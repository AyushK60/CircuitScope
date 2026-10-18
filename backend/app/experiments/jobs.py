"""Experiment execution: validation, the clean/corrupted/patched pipeline,
and a bounded serial worker (one heavy experiment at a time).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

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
from .metrics import compare_metrics


class ExperimentValidationError(ValueError):
    pass


@dataclass
class JobRecord:
    job_id: str
    config: dict
    status: str = "queued"  # queued | running | complete | failed | cancelled
    results: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


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


def execute_experiment(cfg: dict, cancel_flag: threading.Event | None = None) -> dict:
    """Run clean -> corrupted -> patched with the real model; return full results."""
    norm = validate_experiment_config(cfg)
    bridge = ADAPTER.ensure_loaded()
    tok = bridge.tokenizer

    def check_cancel():
        if cancel_flag is not None and cancel_flag.is_set():
            raise ExperimentValidationError("cancelled")

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
    seq = toks_clean.shape[1]
    if any(p >= seq for p in norm["positions"]):
        raise ExperimentValidationError(f"position out of range for sequence length {seq}")

    t0 = time.perf_counter()
    check_cancel()
    logits_clean, clean_cache = bridge.run_with_cache(
        toks_clean, names_filter=lambda n: n == norm["hook"]
    )
    if norm["hook"] not in clean_cache:
        raise ExperimentValidationError("requested hook was not captured during clean run")
    check_cancel()
    logits_corr = bridge(toks_corr)
    check_cancel()
    patched = engine.run_patched(
        toks_corr,
        clean_cache,
        norm["hook"],
        norm["positions"],
        norm["heads"],
    )
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


class ExperimentRunner:
    """Bounded serial worker: one heavy experiment at a time."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel_flags: dict[str, threading.Event] = {}

    def submit(self, cfg: dict) -> JobRecord:
        norm = validate_experiment_config(cfg)  # fail fast on invalid specs
        with self._lock:
            running = any(j.status in ("queued", "running") for j in self._jobs.values())
            if running:
                raise ExperimentValidationError("an experiment is already running; wait for it to finish")
            job = JobRecord(job_id=uuid.uuid4().hex[:12], config=norm)
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
        try:
            results = execute_experiment(job.config, cancel_flag=flag)
            with self._lock:
                job.results = results
                job.status = "complete"
                job.finished_at = time.time()
        except ExperimentValidationError as e:
            with self._lock:
                job.status = "cancelled" if "cancelled" in str(e) else "failed"
                job.error = str(e)
                job.finished_at = time.time()
        except Exception as e:  # surface real failures honestly
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
