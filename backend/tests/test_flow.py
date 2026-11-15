"""Tests for the flow-capture endpoint (playback data source)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.model_runtime import engine
from backend.app.model_runtime.loader import ADAPTER
from backend.app.storage import db


@pytest.fixture(scope="module")
def client(bridge):
    with TestClient(create_app()) as c:
        yield c


def test_capture_flow_structure(bridge):
    flow = engine.capture_flow("John and Mary went to the shop. John gave a book to")
    assert len(flow["tokens"]) == 14
    assert flow["token_ids"][0] == 50256  # BOS prepended by the bridge
    assert len(flow["blocks"]) == 12
    for i, b in enumerate(flow["blocks"]):
        assert b["layer"] == i
        assert len(b["resid_norm_per_token"]) == 14
        assert b["resid_norm_mean"] > 0
        assert len(b["attention"]) == 12  # heads
        assert len(b["attention"][0]) == 14  # query positions
        assert len(b["attention"][0][0]) == 14  # key positions
        # attention rows sum to ~1 (rounded to 3 decimals → tolerance 0.01)
        row_sum = sum(b["attention"][0][0])
        assert abs(row_sum - 1.0) < 0.01


def test_capture_flow_norms_increase(bridge):
    """Residual norm should grow as blocks add contributions (measured, robust)."""
    flow = engine.capture_flow("Hello world")
    means = [b["resid_norm_mean"] for b in flow["blocks"]]
    assert means[-1] > means[0], f"residual norm should accumulate: {means}"


def test_flow_api(client):
    r = client.post("/api/flow", json={"text": "Hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_layers"] == 12
    assert len(body["blocks"]) == 12
    assert body["top_predictions"]
    assert "hook_basis" in body


def test_flow_api_rejects_oversized(client):
    r = client.post("/api/flow", json={"text": "x" * 600})
    assert r.status_code == 422
