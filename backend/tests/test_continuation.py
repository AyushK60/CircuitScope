"""Tests for continuation, candidate comparison, and history endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.experiments.jobs import ExperimentValidationError
from backend.app.main import create_app
from backend.app.storage import db


@pytest.fixture(scope="module")
def client(bridge):
    with TestClient(create_app()) as c:
        yield c


def test_continue_greedy_deterministic(client):
    req = {"text": "The capital of France is", "n_tokens": 3}
    r1 = client.post("/api/continue", json=req)
    r2 = client.post("/api/continue", json=req)
    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.json(), r2.json()
    assert b1["text_final"] == b2["text_final"], "greedy decoding must be deterministic"
    assert len(b1["steps"]) == 3
    assert b1["steps"][0]["input_so_far"] == req["text"]
    # each step's input grows by the previous chosen token
    assert b1["steps"][1]["input_so_far"] == req["text"] + b1["steps"][0]["chosen"]
    assert b1["text_final"] == req["text"] + "".join(s["chosen"] for s in b1["steps"])
    assert "greedy" in b1["decoding"]


def test_continue_bounds(client):
    assert client.post("/api/continue", json={"text": "Hi", "n_tokens": 0}).status_code == 422
    assert client.post("/api/continue", json={"text": "Hi", "n_tokens": 9}).status_code == 422


def test_predict_candidate_comparison_matches_reference(client):
    r = client.post(
        "/api/predict",
        json={
            "text": "John and Mary went to the shop. John gave a book to",
            "candidate_a": " Mary",
            "candidate_b": " John",
        },
    )
    assert r.status_code == 200
    c = r.json()["comparison"]
    # reference value from scripts/reference_patch.py / test_reference_agreement
    assert c["D"] == pytest.approx(2.5128, abs=5e-3)
    assert c["prob_a"] > c["prob_b"]


def test_predict_candidate_validation_honest(client):
    r = client.post(
        "/api/predict",
        json={"text": "Hi there", "candidate_a": " shopkeeper", "candidate_b": " John"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "comparison" not in body
    assert "not a single token" in body["comparison_error"]


def test_history_lists_saved_experiments(client):
    saved_before = len(db.list_experiments())
    exp_id = db.save_experiment(
        config={"hook": "blocks.5.hook_out"}, results={"metrics": {"D": 1.0}}, status="complete"
    )
    saved = db.list_experiments()
    assert len(saved) == saved_before + 1
    assert saved[0]["id"] == exp_id
    assert saved[0]["status"] == "complete"


def test_sample_mode_seeded_reproducible(client):
    req = {"text": "John and Mary went to the shop.", "n_tokens": 4, "mode": "sample", "temperature": 0.9, "top_k": 20, "seed": 1234}
    r1 = client.post("/api/continue", json=req)
    r2 = client.post("/api/continue", json=req)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["text_final"] == r2.json()["text_final"], "same seed must reproduce"
    assert "sampling" in r1.json()["decoding"]
    # different seed should (virtually always) produce a different continuation
    r3 = client.post("/api/continue", json={**req, "seed": 99})
    assert r3.json()["text_final"] != r1.json()["text_final"]


def test_sample_mode_param_bounds(client):
    ok = client.post("/api/continue", json={"text": "Hi", "n_tokens": 1, "mode": "sample", "temperature": 0.05, "top_k": 1})
    assert ok.status_code == 200
    bad_t = client.post("/api/continue", json={"text": "Hi", "n_tokens": 1, "temperature": 3.0})
    assert bad_t.status_code == 422
    bad_m = client.post("/api/continue", json={"text": "Hi", "n_tokens": 1, "mode": "beam"})
    assert bad_m.status_code == 422
    bad_k = client.post("/api/continue", json={"text": "Hi", "n_tokens": 1, "top_k": 99})
    assert bad_k.status_code == 422


def test_question_scoring_single_token(client):
    r = client.post(
        "/api/question",
        json={
            "context": "The father asked his son for water because the father was sick.",
            "question": "Who was sick?",
            "candidates": [" father", " son"],
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d["prompt"].endswith("Answer:")
    for c in d["candidates"]:
        assert c["n_tokens"] == 1
        # for single-token candidates, total == the single token's logprob
        assert c["total_logprob"] == c["per_token_logprob"][0]
    assert d["candidates"][0]["candidate"] == " father"


def test_question_scoring_multi_token_uses_sum(client):
    r = client.post(
        "/api/question",
        json={
            "context": "The father asked his son for water.",
            "question": "Who was asked for water?",
            "candidates": [" the father", " his son"],
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d["length_warning"] is False or d["length_warning"] is True  # present
    for c in d["candidates"]:
        if c["n_tokens"] > 1:
            # total must be the sum of ALL candidate token logprobs, not just the first
            assert c["total_logprob"] == pytest.approx(sum(c["per_token_logprob"]), abs=1e-3)
            assert c["mean_logprob"] == pytest.approx(c["total_logprob"] / c["n_tokens"], abs=1e-3)
            assert abs(c["first_token_prob"]) < 1.0


def test_question_validation(client):
    too_few = client.post("/api/question", json={"context": "a", "question": "b", "candidates": [" x"]})
    assert too_few.status_code == 422
    bad_template = client.post(
        "/api/question",
        json={"context": "a", "question": "b", "candidates": [" x", " y"], "template": "no placeholders"},
    )
    assert bad_template.status_code == 422


def test_identical_prompts_rejected(client):
    from backend.app.experiments.jobs import execute_experiment
    with pytest.raises(ExperimentValidationError, match="identical"):
        execute_experiment(
            {
                "clean_text": "John and Mary went to the shop. John gave a book to",
                "corrupted_text": "John and Mary went to the shop. John gave a book to",
                "hook": "blocks.5.hook_out",
                "positions": [9],
                "candidate_a": " Mary",
                "candidate_b": " John",
            }
        )


def test_completed_experiment_is_persisted_with_results(client):
    r = client.post(
        "/api/experiments",
        json={
            "clean_text": "John and Mary went to the shop. John gave a book to",
            "corrupted_text": "John and Mary went to the shop. Mary gave a book to",
            "hook": "blocks.5.hook_out",
            "positions": [9],
            "candidate_a": " Mary",
            "candidate_b": " John",
        },
    )
    import time
    job_id = r.json()["job_id"]
    for _ in range(240):
        st = client.get(f"/api/experiments/{job_id}").json()
        if st["status"] in ("complete", "failed"):
            break
        time.sleep(0.25)
    assert st["status"] == "complete"
    # the runner persists asynchronously right after completion; poll briefly
    saved = None
    for _ in range(40):
        saved = db.get_experiment(job_id)
        if saved is not None and saved["results"] is not None:
            break
        time.sleep(0.1)
    assert saved is not None and saved["status"] == "complete"
    assert saved["results"] is not None
    assert saved["results"]["metrics"]["Recovery"] == pytest.approx(0.9148, abs=5e-3)
