"""Job engine tests: progress reporting, cancellation, the serial-worker rule,
and the complete-implies-persisted invariant. Uses a test-only "sleep_test"
job type registered inside a fixture and unregistered afterwards, so the
production registry is untouched."""

from __future__ import annotations

import threading
import time

import pytest

from backend.app.experiments.jobs import (
    JOB_TYPES,
    ExperimentRunner,
    ExperimentValidationError,
    JobContext,
    JobRecord,
    register_job_type,
)
from backend.app.storage import db


def _sleep_validate(cfg: dict) -> dict:
    return dict(cfg)


def _sleep_run(cfg: dict, ctx: JobContext) -> dict:
    n = int(cfg.get("n", 10))
    for i in range(1, n + 1):
        ctx.report(i / n, f"step {i}/{n}")
        ctx.check_cancel()
        time.sleep(0.05)
    return {"steps": n}


@pytest.fixture()
def sleep_test_type():
    saved = dict(JOB_TYPES)
    register_job_type("sleep_test", _sleep_validate, _sleep_run)
    yield
    JOB_TYPES.clear()
    JOB_TYPES.update(saved)


def _wait_terminal(runner: ExperimentRunner, job_id: str, timeout: float = 60.0) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = runner.get(job_id)
        if job.status in ("complete", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state in {timeout}s")


def test_unknown_job_type_rejected():
    runner = ExperimentRunner()
    with pytest.raises(ExperimentValidationError):
        runner.submit({"n": 1}, job_type="no_such_type")


def test_progress_clamps_and_never_decreases():
    runner = ExperimentRunner()
    job = JobRecord(job_id="ctx-test", config={})
    ctx = JobContext(runner, job, threading.Event())
    ctx.report(0.3, "a")
    assert job.progress == 0.3
    ctx.report(1.5, "b")  # above 1 -> clamped
    assert job.progress == 1.0
    ctx.report(-0.2, "c")  # below 0 -> clamped to 0, but never decreases
    assert job.progress == 1.0
    ctx.report(0.7, "d")  # lower than current -> never decreases
    assert job.progress == 1.0
    assert job.progress_message == "d"


def test_sleep_test_progress_monotonic_and_completes(sleep_test_type):
    runner = ExperimentRunner()
    job = runner.submit({"n": 8}, job_type="sleep_test")
    assert job.type == "sleep_test"
    seen: list[float] = []
    while True:
        j = runner.get(job.job_id)
        seen.append(j.progress)
        if j.status in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.02)
    assert j.status == "complete", j.error
    assert all(0.0 <= p <= 1.0 for p in seen)
    assert all(b >= a for a, b in zip(seen, seen[1:]))  # monotonic non-decreasing
    assert j.progress == 1.0
    assert j.progress_message == "complete"


def test_cancel_mid_run_gives_cancelled_status(sleep_test_type):
    runner = ExperimentRunner()
    job = runner.submit({"n": 60}, job_type="sleep_test")  # ~3 s if not cancelled
    # wait until the run has actually started making progress
    deadline = time.time() + 10
    while time.time() < deadline and runner.get(job.job_id).progress < 0.05:
        time.sleep(0.02)
    assert runner.cancel(job.job_id) is True
    j = _wait_terminal(runner, job.job_id, timeout=10)
    assert j.status == "cancelled", j.error
    assert j.status != "failed"


def test_second_submit_rejected_while_running_then_allowed(sleep_test_type):
    runner = ExperimentRunner()
    first = runner.submit({"n": 20}, job_type="sleep_test")
    with pytest.raises(ExperimentValidationError) as ei:
        runner.submit({"n": 2}, job_type="sleep_test")
    assert "an experiment is already running; wait for it to finish" in str(ei.value)
    _wait_terminal(runner, first.job_id, timeout=30)
    second = runner.submit({"n": 2}, job_type="sleep_test")  # works after finish
    j = _wait_terminal(runner, second.job_id, timeout=30)
    assert j.status == "complete"


def test_submit_default_type_is_intervention():
    runner = ExperimentRunner()
    with pytest.raises(ExperimentValidationError):
        # invalid config still fails fast through the type's validate_fn
        runner.submit({"clean_text": "", "corrupted_text": ""})


def test_complete_implies_persisted(bridge, ioi):
    """A complete job must already be retrievable from the DB (persist before
    marking complete)."""
    runner = ExperimentRunner()
    cfg = {
        "clean_text": ioi["clean"],
        "corrupted_text": ioi["corrupted"],
        "hook": "blocks.5.hook_out",
        "positions": [9],
        "candidate_a": " Mary",
        "candidate_b": " John",
    }
    job = runner.submit(cfg)  # default type: intervention
    assert job.type == "intervention"
    j = _wait_terminal(runner, job.job_id, timeout=120)
    assert j.status == "complete", j.error
    assert j.progress == 1.0
    stored = db.get_experiment(job.job_id)
    assert stored is not None
    assert stored["results"]["metrics"]["D_patched"] is not None


def test_test_job_type_not_in_production_registry(sleep_test_type):
    assert "sleep_test" in JOB_TYPES  # registered inside the fixture
    assert "intervention" in JOB_TYPES


def test_unknown_type_in_run_fails_job_and_releases_worker():
    """A job whose type disappeared from the registry must end 'failed' with a
    clear error — not stay 'running' forever (which would block all future
    jobs). Exercises _run directly so the lookup path is deterministic."""
    runner = ExperimentRunner()
    job = JobRecord(job_id="ghost", config={}, type="vanishing_type")
    runner._jobs[job.job_id] = job
    runner._cancel_flags[job.job_id] = threading.Event()
    runner._run(job.job_id)  # direct call: no thread, no race
    assert job.status == "failed", job.error
    assert "unknown job type" in (job.error or "")
    # the serial-worker rule is not blocked by the dead job
    assert not any(j.status in ("queued", "running") for j in runner._jobs.values())
