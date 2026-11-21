"""Tests for the "study" job end to end (small n=4),
progress/cancel/persistence contract, validation, exclusions, determinism.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.app.experiments.jobs import RUNNER, JobRecord
from backend.app.main import create_app
from backend.app.storage import db

N4_SECONDS = 600.0  # generous: an n=4 study runs ~790 model runs per prompt


@pytest.fixture(scope="module")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def _wait_terminal(job_id: str, timeout: float = N4_SECONDS) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = RUNNER.get(job_id)
        assert job is not None
        if job.status in ("complete", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach a terminal state in {timeout}s")


def test_api_invalid_params_422(client):
    for body in (
        {"n_prompts": 3},  # below the minimum
        {"n_prompts": 101},  # above the maximum
        {"n_prompts": "many"},
        {"n_prompts": 8, "seed": -1},
    ):
        r = client.post("/api/study", json=body)
        assert r.status_code == 422, f"{body} -> {r.status_code}"


def test_api_defaults_accepted(client):
    r = client.post("/api/study", json={})
    assert r.status_code == 200
    # a study was submitted with defaults (n=40 is far too long for tests);
    # cancel it immediately.
    body = r.json()
    assert set(body) == {"job_id", "status"}
    client.post(f"/api/experiments/{body['job_id']}/cancel")
    _wait_terminal(body["job_id"])
    assert RUNNER.get(body["job_id"]).status == "cancelled"


def test_study_n4_end_to_end(client):
    progress_seen: list[float] = []
    job = RUNNER.submit({"n_prompts": 4, "seed": 5}, job_type="study")
    assert job.type == "study"
    deadline = time.time() + N4_SECONDS
    last = -1.0
    while time.time() < deadline:
        j = RUNNER.get(job.job_id)
        progress_seen.append(j.progress)
        assert j.progress >= last - 1e-9, "progress decreased"
        last = j.progress
        if j.status in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert j.status == "complete", j.error
    assert j.progress == pytest.approx(1.0)
    assert len(progress_seen) > 2

    res = j.results
    assert res["job_type"] == "study"
    assert res["params"] == {"n_prompts": 4, "seed": 5}
    assert res["direction"] and res["metrics"] and res["exclusion_rule"]
    assert len(res["per_prompt"]) == 4
    assert len(res["baselines_per_prompt"]) == 4
    assert res["prompt_set"]["rejected_draws"] >= 0
    assert 0.0 <= res["model_accuracy"] <= 1.0
    # every prompt's vectors have 156 entries
    for row in res["per_prompt"]:
        for kind in ("swap", "third", "resample"):
            if row["rec_d"][kind] is not None:
                assert len(row["rec_d"][kind]) == 156
        if row["atp"] is not None:
            assert len(row["atp"]) == 156
        if row["dla"] is not None:
            assert len(row["dla"]) == 156
        if row["abl_zero"] is not None:
            assert len(row["abl_zero"]) == 156
    # summary structure (pre-registered sections)
    rq1 = res["summary"]["rq1"]["all_prompts"]
    for method in ("atp", "dla"):
        for group in ("heads", "mlps", "all"):
            assert "signed" in rq1["methods"][method][group]
            assert "abs" in rq1["methods"][method][group]
    assert set(rq1["null"]) == {"heads", "mlps", "all"}
    rq2 = res["summary"]["rq2"]
    assert len(rq2["condition_names"]) == 8
    assert len(rq2["spearman"]["matrix"]) == 8
    assert rq2["topk"]["expected_random"]["k5"] == pytest.approx(25.0 / 144.0)
    assert "mean_ranks_by_group" in res["summary"]["anchors"]
    assert res["elapsed_seconds"] > 0
    assert res["provenance"]["git_commit"]


def test_study_persisted_before_complete_and_in_history(client):
    job = RUNNER.submit({"n_prompts": 4, "seed": 6}, job_type="study")
    j = _wait_terminal(job.job_id)
    assert j.status == "complete", j.error
    stored = db.get_experiment(job.job_id)
    assert stored is not None
    assert stored["status"] == "complete"
    assert stored["config"]["job_type"] == "study"
    assert stored["results"]["job_type"] == "study"
    # present in the history list endpoint
    r = client.get("/api/experiments")
    ids = [s["id"] for s in r.json()["saved"]]
    assert job.job_id in ids


def test_study_cancelled_leaves_no_partial_results(client):
    job = RUNNER.submit({"n_prompts": 40, "seed": 0}, job_type="study")
    deadline = time.time() + 120
    while time.time() < deadline and RUNNER.get(job.job_id).progress < 0.03:
        time.sleep(0.05)
    assert RUNNER.cancel(job.job_id) is True
    j = _wait_terminal(job.job_id)
    assert j.status == "cancelled"
    assert j.results is None
    stored = db.get_experiment(job.job_id)
    assert stored is None or stored.get("status") != "complete"


def test_exclusion_rule_applied_and_counted():
    """The recorded exclusions must match the rule recomputed from the stored
    per-prompt baselines (uses the completed n=4 seed=5 study)."""
    jobs = [j for j in RUNNER.recent(50) if j.type == "study" and j.status == "complete"]
    assert jobs, "no completed study available"
    res = jobs[0].results
    recomputed = {
        kind: sum(
            1
            for row in res["per_prompt"]
            if row["baselines"][kind].get("excluded")
        )
        for kind in ("swap", "third", "resample")
    }
    assert res["exclusions"]["swap"] == recomputed["swap"]
    assert res["exclusions"]["third"] == recomputed["third"]
    assert res["exclusions"]["resample"] == recomputed["resample"]
    # every excluded pair has |D_clean - D_corr| < 0.5 (or is missing)
    for row in res["per_prompt"]:
        for kind in ("swap", "third", "resample"):
            b = row["baselines"][kind]
            if b.get("excluded"):
                assert b.get("missing") or b["abs_D_diff"] < 0.5
            else:
                assert b["abs_D_diff"] >= 0.5


def test_study_determinism_same_seed():
    a = RUNNER.submit({"n_prompts": 4, "seed": 9}, job_type="study")
    ja = _wait_terminal(a.job_id)
    assert ja.status == "complete", ja.error
    b = RUNNER.submit({"n_prompts": 4, "seed": 9}, job_type="study")
    jb = _wait_terminal(b.job_id)
    assert jb.status == "complete", jb.error
    assert ja.results["summary"] == jb.results["summary"]
    assert ja.results["per_prompt"] == jb.results["per_prompt"]
    assert ja.results["model_accuracy"] == jb.results["model_accuracy"]


def test_second_study_rejected_while_one_running():
    job = RUNNER.submit({"n_prompts": 4, "seed": 11}, job_type="study")
    try:
        with pytest.raises(Exception):
            RUNNER.submit({"n_prompts": 4, "seed": 12}, job_type="study")
    finally:
        RUNNER.cancel(job.job_id)
        _wait_terminal(job.job_id)
