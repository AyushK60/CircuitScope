"""Tests for continuation, candidate comparison, and history endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.storage import db


@pytest.fixture(scope="module")
def client(bridge):
    db.init("outputs/test_experiments.db")
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
