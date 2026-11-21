"""Sweep tests: mathematical invariants of the patching sweep grids,
the job-engine contract (progress, cancellation, one-heavy-job rule), API
validation, and determinism. All numbers are measured on the real model.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.app.experiments.jobs import (
    RUNNER,
    ExperimentValidationError,
    JobRecord,
)
from backend.app.experiments.sweeps import check_baselines
from backend.app.main import create_app

TASK = "ioi_john_mary"


@pytest.fixture(scope="module")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def _wait_terminal(job_id: str, timeout: float = 300.0) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = RUNNER.get(job_id)
        assert job is not None
        if job.status in ("complete", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state in {timeout}s")


def _submit(mode: str) -> str:
    job = RUNNER.submit({"task_id": TASK, "mode": mode}, job_type="sweep")
    assert job.type == "sweep"
    return job.job_id


# ---- full sweep invariants (layer_position) ----


def test_layer_position_invariants(bridge):
    """Patching block 11 at the final position must fully restore the clean
    logit difference (Recovery 1); patching block 11 at any non-final position
    must not move the final logits at all (nothing comes after block 11)."""
    job = _submit("layer_position")
    j = _wait_terminal(job)
    assert j.status == "complete", j.error
    res = j.results
    assert res["job_type"] == "sweep" and res["mode"] == "layer_position"
    assert res["n_runs"] == 168
    d_corr = res["baselines"]["D_corrupted"]
    seq = len(res["col_labels"])
    # final position of the last block
    assert res["recovery"][11][seq - 1] == pytest.approx(1.0, abs=1e-3)
    assert res["d_patched"][11][seq - 1] == pytest.approx(res["baselines"]["D_clean"], abs=1e-3)
    # non-final positions of the last block cannot affect the final logits
    for p in range(seq - 1):
        assert abs(res["d_patched"][11][p] - d_corr) <= 1e-4, f"position {p}"
    # monotonic progress ended at exactly 1.0
    assert j.progress == 1.0


def test_layer_head_grid_shape_and_progress(bridge):
    job = _submit("layer_head")
    j = _wait_terminal(job)
    assert j.status == "complete", j.error
    res = j.results
    assert res["n_runs"] == 144
    assert len(res["recovery"]) == 12 and len(res["recovery"][0]) == 12
    assert res["col_labels"] == [f"head {h}" for h in range(12)]
    assert res["row_labels"] == [f"block {l}" for l in range(12)]
    assert res["direction"] == "clean -> corrupted (denoising)"
    assert res["baselines"]["warnings"] == []
    assert j.progress == 1.0 and j.progress_message == "complete"


def test_full_residual_patch_restores_clean(bridge, ioi):
    """A full-residual patch (all positions) at any layer must reproduce the
    clean run exactly, because every later block reads only clean inputs.
    Run through the EXISTING execute_experiment pipeline (job type
    "intervention") for a few layers."""
    from backend.app.experiments.jobs import execute_experiment

    for layer in (0, 5, 11):
        cfg = {
            "clean_text": ioi["clean"],
            "corrupted_text": ioi["corrupted"],
            "hook": f"blocks.{layer}.hook_out",
            "positions": list(range(14)),
            "candidate_a": " Mary",
            "candidate_b": " John",
        }
        m = execute_experiment(cfg)["metrics"]
        assert m["Recovery"] == pytest.approx(1.0, abs=5e-3), f"layer {layer}"
        assert m["D_patched"] == pytest.approx(m["D_clean"], abs=5e-3)


def test_spot_check_against_existing_pipeline(bridge, ioi):
    """Three cells of each mode must match the existing execute_experiment
    pipeline (same hook / positions / heads) within 1e-4."""
    from backend.app.experiments.jobs import execute_experiment

    lp = _wait_terminal(_submit("layer_position")).results
    lh = _wait_terminal(_submit("layer_head")).results

    cells = [(2, 4), (7, 9), (10, 0)]
    for l, p in cells:
        cfg = {
            "clean_text": ioi["clean"],
            "corrupted_text": ioi["corrupted"],
            "hook": f"blocks.{l}.hook_out",
            "positions": [p],
            "candidate_a": " Mary",
            "candidate_b": " John",
        }
        d = execute_experiment(cfg)["metrics"]["D_patched"]
        assert d == pytest.approx(lp["d_patched"][l][p], abs=1e-4), f"layer_position ({l},{p})"

    for l, h in ((5, 4), (9, 6), (1, 0)):
        cfg = {
            "clean_text": ioi["clean"],
            "corrupted_text": ioi["corrupted"],
            "hook": f"blocks.{l}.attn.hook_z",
            "positions": list(range(14)),
            "heads": [h],
            "candidate_a": " Mary",
            "candidate_b": " John",
        }
        d = execute_experiment(cfg)["metrics"]["D_patched"]
        assert d == pytest.approx(lh["d_patched"][l][h], abs=1e-4), f"layer_head ({l},{h})"


def test_sweep_determinism(bridge):
    job_a = _submit("layer_head")
    a = _wait_terminal(job_a).results
    job_b = _submit("layer_head")
    b = _wait_terminal(job_b).results
    assert a["recovery"] == b["recovery"]
    assert a["d_patched"] == b["d_patched"]


# ---- job-engine contract ----


def test_progress_monotonic_and_ends_at_one(bridge):
    job_id = _submit("layer_position")
    seen: list[float] = []
    while True:
        j = RUNNER.get(job_id)
        seen.append(j.progress)
        if j.status in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert j.status == "complete"
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert j.progress == 1.0


def test_cancel_mid_sweep_ends_cancelled(bridge):
    job_id = _submit("layer_position")
    deadline = time.time() + 30
    while time.time() < deadline and RUNNER.get(job_id).progress < 0.1:
        time.sleep(0.02)
    assert RUNNER.cancel(job_id) is True
    j = _wait_terminal(job_id, timeout=60)
    assert j.status == "cancelled", j.error
    assert j.status != "failed"


def test_second_sweep_rejected_while_one_running(bridge):
    job_id = _submit("layer_position")
    with pytest.raises(ExperimentValidationError, match="already running"):
        _submit("layer_head")
    RUNNER.cancel(job_id)
    _wait_terminal(job_id, timeout=60)


# ---- API surface ----


def test_api_sweep_unknown_task_rejected(client):
    r = client.post("/api/sweeps", json={"task_id": "no_such_task", "mode": "layer_position"})
    assert r.status_code == 422
    assert "unknown task" in r.json()["detail"].lower()


def test_api_sweep_bad_mode_rejected(client):
    r = client.post("/api/sweeps", json={"task_id": TASK, "mode": "layerXposition"})
    assert r.status_code == 422


def test_api_sweep_unknown_pair_rejected(client):
    r = client.post("/api/sweeps", json={"task_id": TASK, "pair_id": "nope", "mode": "layer_position"})
    assert r.status_code == 422
    assert "pair" in r.json()["detail"].lower()


def test_api_sweep_lifecycle_and_persisted_config(client, bridge):
    r = client.post("/api/sweeps", json={"task_id": TASK, "mode": "layer_head"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(1200):
        s = client.get(f"/api/experiments/{job_id}").json()
        if s["status"] in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert s["status"] == "complete", s.get("error")
    assert s["type"] == "sweep"
    assert s["results"]["job_type"] == "sweep"
    # the persisted config must carry job_type so history can tell it apart
    saved = client.get("/api/experiments").json()["saved"]
    rec = next(x for x in saved if x["id"] == job_id)
    assert rec["config"]["job_type"] == "sweep"


# ---- baseline gate ----


def test_identical_baselines_rejected_honestly():
    with pytest.raises(ExperimentValidationError) as ei:
        check_baselines(2.5128, 2.5128)
    assert "Recovery would be meaningless" in str(ei.value)


def test_near_identical_baselines_rejected():
    with pytest.raises(ExperimentValidationError):
        check_baselines(2.5128, 2.51285)


def test_small_but_usable_baseline_warns():
    warnings = check_baselines(2.0, 1.6)  # |denominator| = 0.4 < 0.5
    assert len(warnings) == 1 and "unstable" in warnings[0]


def test_healthy_baseline_has_no_warnings():
    assert check_baselines(2.5128, -4.1768) == []


def test_sweep_result_reports_real_baselines(bridge):
    job_id = _submit("layer_position")
    res = _wait_terminal(job_id).results
    b = res["baselines"]
    assert abs(b["denominator"] - (b["D_clean"] - b["D_corrupted"])) < 1e-9
    assert abs(b["denominator"]) >= 0.5  # IOI pair is far from degenerate
