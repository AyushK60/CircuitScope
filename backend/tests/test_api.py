"""API integration tests: health, predict, inspect, experiment lifecycle,
validation errors, exports, persistence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.storage import db


@pytest.fixture(scope="session")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]["n_layers"] == 12


def test_health_revision_hash_is_40_hex(client):
    """Stage 6 A1: metadata() exposes the real model revision as revision_hash,
    additively; the legacy `revision` (tokenizer name/path) is unchanged."""
    r = client.get("/api/health")
    assert r.status_code == 200
    model = r.json()["model"]
    assert model["revision"] == "gpt2"  # legacy field, unchanged for old clients
    h = model["revision_hash"]
    assert isinstance(h, str)
    assert len(h) == 40
    assert all(c in "0123456789abcdef" for c in h)


def test_models(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    assert r.json()["loaded"] is True


def test_predict(client):
    r = client.post("/api/predict", json={"text": "John and Mary went to the shop. John gave a book to"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["top_predictions"][0]["token"].strip() == "Mary"
    assert body["tokenization"]["tokens"][0] == "<|endoftext|>"
    assert len(body["prediction"]["top_predictions"]) <= 8


def test_predict_rejects_oversized_input(client):
    r = client.post("/api/predict", json={"text": "x" * 600})
    assert r.status_code == 422  # pydantic bound


def test_inspect_attention(client):
    r = client.post("/api/inspect", json={"text": "Hello world", "layer": 5})
    assert r.status_code == 200
    att = r.json()["attention"]
    assert att["shape"] == [12, 3, 3]
    assert abs(sum(att["pattern"][0][0]) - 1.0) < 0.01  # attention rows sum to 1


def test_inspect_invalid_hook(client):
    r = client.post("/api/inspect", json={"text": "Hi", "hooks": ["evil.hook"]})
    assert r.status_code == 200
    assert "activations_error" in r.json()


def test_experiment_lifecycle(client):
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
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # poll until done (bounded)
    import time

    for _ in range(240):
        s = client.get(f"/api/experiments/{job_id}").json()
        if s["status"] in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.25)
    assert s["status"] == "complete", s.get("error")
    m = s["results"]["metrics"]
    assert m["D_patched"] > 0 > m["D_corrupted"]
    assert 0.5 < m["Recovery"] < 1.5

    # export json and csv
    ej = client.get(f"/api/experiments/{job_id}/export")
    assert ej.status_code == 200 and ej.json()["metrics"] == m
    ec = client.get(f"/api/experiments/{job_id}/export?fmt=csv")
    assert ec.status_code == 200
    assert "D_patched" in ec.text


def test_experiment_invalid_hook_rejected(client):
    r = client.post(
        "/api/experiments",
        json={
            "clean_text": "a b",
            "corrupted_text": "a c",
            "hook": "blocks.99.hook_out",
            "positions": [0],
            "candidate_a": " a",
            "candidate_b": " b",
        },
    )
    assert r.status_code == 422


def test_experiment_misaligned_prompts_fail_honestly(client):
    r = client.post(
        "/api/experiments",
        json={
            "clean_text": "John and Mary went to the shop. John gave a book to",
            "corrupted_text": "John and Mary went to the shop. Mary gave a book",
            "hook": "blocks.5.hook_out",
            "positions": [9],
            "candidate_a": " Mary",
            "candidate_b": " John",
        },
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    import time

    for _ in range(120):
        s = client.get(f"/api/experiments/{job_id}").json()
        if s["status"] in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.25)
    assert s["status"] == "failed"
    assert "token-aligned" in s["error"]


def test_experiment_persistence(client):
    saved = db.list_experiments()
    assert isinstance(saved, list)
