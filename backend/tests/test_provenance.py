"""Provenance tests: collect() shape, git-failure resilience, API merge with
old keys preserved, and back-compat for stored records without provenance."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from backend.app import provenance
from backend.app.main import create_app
from backend.app.storage import db

EXPECTED_KEYS = {
    "model_id",
    "model_revision",
    "device",
    "dtype",
    "versions",
    "git_commit",
    "timestamp_utc",
}

IOI_CFG = {
    "clean_text": "John and Mary went to the shop. John gave a book to",
    "corrupted_text": "John and Mary went to the shop. Mary gave a book to",
    "hook": "blocks.5.hook_out",
    "positions": [9],
    "candidate_a": " Mary",
    "candidate_b": " John",
}


@pytest.fixture(scope="module")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_collect_has_all_keys():
    p = provenance.collect()
    assert EXPECTED_KEYS <= set(p)
    assert set(p["versions"]) == {"torch", "transformer_lens", "transformers", "python"}
    assert p["versions"]["torch"]
    # timestamp is ISO-8601 and parseable
    ts = p["timestamp_utc"]
    assert ts.endswith("Z")
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert dt.tzinfo is not None
    # real revision: 40-hex commit hash, not the tokenizer's "gpt2" string
    assert p["model_revision"] != "gpt2"
    assert len(p["model_revision"]) == 40
    assert all(c in "0123456789abcdef" for c in p["model_revision"])


def test_git_failure_is_swallowed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("subprocess unavailable")

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    provenance._clear_static_cache()
    try:
        p = provenance.collect()  # must not raise
        assert p["git_commit"] == "unknown"
        assert EXPECTED_KEYS <= set(p)  # the rest of the record is intact
    finally:
        provenance._clear_static_cache()  # repopulate truthfully for later tests


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_git_commit_is_not_cached(monkeypatch):
    """A changed HEAD must show up in the next collect() call (no stale hash
    for the life of the process). Static fields stay cached and identical."""

    calls = {"n": 0}

    def fake_run(args, **kwargs):
        argv = list(args)
        if "rev-parse" in argv:
            calls["n"] += 1
            short = "aaaaaaa1" if calls["n"] == 1 else "bbbbbbb2"
            return _FakeCompleted(0, short + "\n")
        if "status" in argv:
            return _FakeCompleted(0, "")  # clean tree both times
        return _FakeCompleted(1, "")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    p1 = provenance.collect()
    p2 = provenance.collect()
    assert p1["git_commit"] == "aaaaaaa1"
    assert p2["git_commit"] == "bbbbbbb2"
    assert p1["git_commit"] != p2["git_commit"]
    # the genuinely static fields are still cached (identical between calls)
    assert p1["model_id"] == p2["model_id"]
    assert p1["model_revision"] == p2["model_revision"]
    assert p1["versions"] == p2["versions"]


def test_api_experiment_provenance_old_and_new_keys(client):
    r = client.post("/api/experiments", json=IOI_CFG)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    import time

    for _ in range(240):
        s = client.get(f"/api/experiments/{job_id}").json()
        if s["status"] in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.25)
    assert s["status"] == "complete", s.get("error")
    prov = s["results"]["provenance"]
    # old keys, unchanged names
    for old in ("model_id", "library", "device", "torch", "elapsed_seconds"):
        assert old in prov, old
    assert prov["library"] == "transformer_lens TransformerBridge"
    assert isinstance(prov["elapsed_seconds"], float)
    # new keys
    for new in ("model_revision", "dtype", "versions", "git_commit", "timestamp_utc"):
        assert new in prov, new


def test_old_record_without_provenance_still_loads(client):
    legacy_results = {"config": IOI_CFG, "metrics": {"D_patched": 1.0}}  # no provenance
    exp_id = "legacy000001"
    db.save_experiment(IOI_CFG, legacy_results, "complete", exp_id=exp_id)
    r = client.get(f"/api/experiments/{exp_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert "provenance" not in body["results"]
