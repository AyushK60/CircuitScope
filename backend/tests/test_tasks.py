"""Task registry tests: file loading (malformed files reported, never fatal),
per-pair validation against the real tokenizer, and the /api/tasks routes."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.tasks import registry
from backend.app.tasks.registry import get_load_errors, get_task, load_all, validate_task


def make_task(**overrides) -> dict:
    """Minimal valid task; overrides applied at the top level or via pair__x."""
    pair = {
        "id": "default",
        "clean_text": "John and Mary went to the shop. John gave a book to",
        "corrupted_text": "John and Mary went to the shop. Mary gave a book to",
        "candidate_a": " Mary",
        "candidate_b": " John",
        "positions_of_interest": {"second_subject": 9, "final": 13},
        "recommended_hooks": [
            {"hook": "blocks.5.hook_out", "positions": [9], "heads": None, "note": "n"}
        ],
    }
    base = {
        "id": "test_task",
        "name": "Test task",
        "description": "d",
        "source": {"citation": "c", "url": "u"},
        "expected_behavior": "e",
        "pairs": [pair],
    }
    for key, value in overrides.items():
        if key.startswith("pair__"):
            pair[key[len("pair__") :]] = value
        else:
            base[key] = value
    return base


def write_task(tmp_path, data, name="t.json") -> None:
    if isinstance(data, str):
        (tmp_path / name).write_text(data, encoding="utf-8")
    else:
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")


# --- loading ---------------------------------------------------------------


def test_ioi_loads_and_validates(bridge):
    load_all()  # default repo directory
    task = get_task("ioi_john_mary")
    assert task is not None
    assert len(task.pairs) == 1
    v = validate_task(task)
    assert v["ok"] is True, v["errors"]
    pair = v["pairs"][0]
    assert pair["candidate_a_id"] is not None
    assert pair["candidate_b_id"] is not None
    assert pair["n_tokens_clean"] == pair["n_tokens_corrupted"]


def test_invalid_json_reported_not_fatal(tmp_path):
    write_task(tmp_path, "{ not json ]")
    load_all(tmp_path)
    assert get_task("test_task") is None
    errs = get_load_errors()
    assert len(errs) == 1
    assert "JSONDecodeError" in errs[0]["error"]
    assert errs[0]["file"] == "t.json"


def test_duplicate_id_keeps_first(tmp_path):
    write_task(tmp_path, make_task(name="first"), name="a.json")
    write_task(tmp_path, make_task(name="second"), name="b.json")
    load_all(tmp_path)
    assert get_task("test_task") is not None
    assert get_task("test_task").name == "first"
    assert any("duplicate" in e["error"] for e in get_load_errors())


def test_bad_id_chars_reported(tmp_path):
    write_task(tmp_path, make_task(id="IOI John!"))
    load_all(tmp_path)
    assert get_task("IOI John!") is None
    assert len(get_load_errors()) == 1


def test_schema_missing_field_reported(tmp_path):
    bad = make_task()
    del bad["expected_behavior"]
    write_task(tmp_path, bad)
    load_all(tmp_path)
    assert len(get_load_errors()) == 1


# --- per-pair validation (needs the tokenizer) ------------------------------


def test_multi_token_candidate_rejected(bridge, tmp_path):
    # STAGE_1.md suggested " Constantinople", but the real GPT-2 tokenizer
    # resolves it to a single token (id 46369) — recorded in V2_PROGRESS
    # "Questions". " apple pie" is genuinely multi-token, so the real
    # tokenizer rejects it (AssertionError → ValueError in ADAPTER).
    write_task(tmp_path, make_task(pair__candidate_a=" apple pie"))
    load_all(tmp_path)
    v = validate_task(get_task("test_task"))
    assert v["ok"] is False
    assert any("not a single token" in e for e in v["errors"])
    assert v["pairs"][0]["candidate_a_id"] is None


def test_misaligned_pair_rejected(bridge, tmp_path):
    write_task(
        tmp_path,
        make_task(pair__corrupted_text="John and Mary went to the shop. Mary gave book to"),
    )
    load_all(tmp_path)
    v = validate_task(get_task("test_task"))
    assert v["ok"] is False
    assert any("token counts differ" in e for e in v["errors"])


def test_out_of_range_position_rejected(bridge, tmp_path):
    write_task(tmp_path, make_task(pair__positions_of_interest={"second_subject": 9, "final": 99}))
    load_all(tmp_path)
    v = validate_task(get_task("test_task"))
    assert v["ok"] is False
    assert any("out of range" in e for e in v["errors"])


def test_bad_hook_name_rejected(bridge, tmp_path):
    write_task(
        tmp_path,
        make_task(
            pair__recommended_hooks=[
                {"hook": "blocks.99.hook_out", "positions": [9], "heads": None, "note": "n"}
            ]
        ),
    )
    load_all(tmp_path)
    v = validate_task(get_task("test_task"))
    assert v["ok"] is False
    assert any("not patchable" in e for e in v["errors"])


def test_validation_is_cached(bridge, tmp_path):
    write_task(tmp_path, make_task())
    load_all(tmp_path)
    task = get_task("test_task")
    v1 = validate_task(task)
    v2 = validate_task(task)
    assert v1 is v2  # same cached object


# --- API ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_api_tasks_index(client):
    load_all()  # restore the real registry after the tmp-dir tests above
    r = client.get("/api/tasks")
    assert r.status_code == 200
    body = r.json()
    ids = [t["id"] for t in body["tasks"]]
    assert "ioi_john_mary" in ids
    ioi = next(t for t in body["tasks"] if t["id"] == "ioi_john_mary")
    assert ioi["n_pairs"] == 1
    assert "load_errors" in body


def test_api_task_detail_validated(client):
    r = client.get("/api/tasks/ioi_john_mary")
    assert r.status_code == 200
    body = r.json()
    assert body["pairs"][0]["candidate_a"] == " Mary"
    assert body["validation"]["ok"] is True
    pair = body["validation"]["pairs"][0]
    assert pair["candidate_a_id"] is not None
    assert pair["n_tokens_clean"] == pair["n_tokens_corrupted"]


def test_api_task_unknown_id_404(client):
    assert client.get("/api/tasks/nope").status_code == 404


def test_api_task_path_traversal_id_404(client):
    assert client.get("/api/tasks/../x").status_code == 404
