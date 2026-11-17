"""Stage 2 analysis tests: direct logit attribution (DLA) and the logit lens.

These are mathematical invariant tests, not copies of output numbers: the
reconciliation identity and the sign-flip-under-candidate-swap property pin
down the implementation's correctness.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.analysis.dla import compute_dla
from backend.app.analysis.lens import compute_logit_lens
from backend.app.main import create_app

CAND_A = " Mary"
CAND_B = " John"


@pytest.fixture(scope="module")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---- DLA: unit-level invariants ----


def test_dla_reconciliation_clean(bridge):
    d = compute_dla(
        "John and Mary went to the shop. John gave a book to", CAND_A, CAND_B
    )
    assert d["reconciliation"]["ok"] is True
    assert d["reconciliation"]["abs_error"] <= 2e-3


def test_dla_reconciliation_corrupted(bridge):
    d = compute_dla(
        "John and Mary went to the shop. Mary gave a book to", CAND_A, CAND_B
    )
    assert d["reconciliation"]["ok"] is True
    assert d["reconciliation"]["abs_error"] <= 2e-3


def test_dla_component_census(bridge):
    d = compute_dla(
        "John and Mary went to the shop. John gave a book to", CAND_A, CAND_B
    )
    kinds: dict[str, list] = {}
    for c in d["components"]:
        kinds.setdefault(c["kind"], []).append(c)
    assert len(kinds["head"]) == 144  # 12 layers x 12 heads
    assert len(kinds["mlp"]) == 12
    assert len(kinds["attn_bias"]) == 12
    assert len(kinds["embed"]) == 1
    assert len(kinds["pos_embed"]) == 1
    assert len(kinds["ln_final_bias"]) == 1
    # every head name is head L.H with matching layer/head fields
    for c in kinds["head"]:
        assert c["name"] == f"head {c['layer']}.{c['head']}"
    assert len(d["components"]) == 144 + 12 + 12 + 1 + 1 + 1


def test_dla_swap_candidates_negates_every_value(bridge):
    text = "John and Mary went to the shop. John gave a book to"
    ab = compute_dla(text, CAND_A, CAND_B)
    ba = compute_dla(text, CAND_B, CAND_A)
    assert ab["D"] == pytest.approx(-ba["D"], abs=1e-9)
    for c1, c2 in zip(ab["components"], ba["components"]):
        assert c1["name"] == c2["name"]
        assert c1["value"] == pytest.approx(-c2["value"], abs=1e-9)


def test_dla_notes_present(bridge):
    d = compute_dla(
        "John and Mary went to the shop. John gave a book to", CAND_A, CAND_B
    )
    assert any("Direct effect only" in n for n in d["notes"])


# ---- DLA: API ----


def test_dla_endpoint_clean_and_corrupted(client):
    for run in ("clean", "corrupted"):
        r = client.post("/api/analysis/dla", json={"task_id": "ioi_john_mary", "run": run})
        assert r.status_code == 200
        body = r.json()
        assert body["run"] == run
        assert body["pair_id"] == "default"
        assert body["tokens"][0] == "<|endoftext|>"
        assert body["reconciliation"]["ok"] is True
        assert body["reconciliation"]["abs_error"] <= 2e-3
        # candidate A is " Mary" (the recipient in the clean prompt)
        assert body["candidate_a"] == " Mary" and body["candidate_b"] == " John"


def test_dla_endpoint_unknown_task_404(client):
    r = client.post("/api/analysis/dla", json={"task_id": "does_not_exist", "run": "clean"})
    assert r.status_code == 404


def test_dla_endpoint_bad_run_422(client):
    r = client.post("/api/analysis/dla", json={"task_id": "ioi_john_mary", "run": "sideways"})
    assert r.status_code == 422


# ---- logit lens ----


def test_lens_block11_matches_real_logits(bridge):
    out = compute_logit_lens(
        "John and Mary went to the shop. John gave a book to", CAND_A, CAND_B, 5
    )
    fm = out["final_matches_model"]
    assert fm["ok"] is True
    assert fm["max_abs_logit_diff_candidates"] <= 1e-3
    assert fm["top1_agrees"] is True
    # the model's real D and the lens block-11 D agree
    assert out["stages"][-1]["D"] == pytest.approx(out["model_D"], abs=1e-3)


def test_lens_embeddings_stage_exists(bridge):
    out = compute_logit_lens(
        "John and Mary went to the shop. John gave a book to", CAND_A, CAND_B, 5
    )
    names = [s["stage"] for s in out["stages"]]
    assert names[0] == "embeddings"
    assert names == ["embeddings"] + [f"block {i}" for i in range(12)]
    assert all(len(s["top"]) == 5 for s in out["stages"])
    # probabilities are over the full vocab and sum-consistent for the top token
    s0 = out["stages"][0]
    assert 0.0 <= s0["top"][0]["prob"] <= 1.0


def test_lens_without_candidates_still_reports_stages(bridge):
    out = compute_logit_lens("The cat sat on the", None, None, 3)
    assert len(out["stages"]) == 13
    assert out["final_matches_model"]["ok"] is True
    assert out["stages"][0]["D"] is None


def test_lens_note_present(bridge):
    out = compute_logit_lens("The cat sat on the", None, None, 5)
    assert "Heuristic view" in out["note"]


def test_lens_endpoint_multitoken_candidate_422(client):
    r = client.post(
        "/api/analysis/logit_lens",
        json={"text": "hello world", "candidate_a": " apple pie", "candidate_b": " John"},
    )
    assert r.status_code == 422
    assert "single token" in r.json()["detail"]


def test_lens_endpoint_single_candidate_422(client):
    # providing only one candidate must be rejected, not silently ignored
    r = client.post("/api/analysis/logit_lens", json={"text": "hello", "candidate_a": " Mary"})
    assert r.status_code == 422


def test_lens_endpoint_ok(client):
    r = client.post(
        "/api/analysis/logit_lens",
        json={
            "text": "John and Mary went to the shop. John gave a book to",
            "candidate_a": " Mary",
            "candidate_b": " John",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["position"] == 13
    assert body["final_matches_model"]["ok"] is True
    assert "provenance" in body
