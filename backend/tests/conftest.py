import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.model_runtime.loader import ADAPTER  # noqa: E402
from backend.app.storage import db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _isolated_db(tmp_path_factory):
    """Tests never touch the real outputs/experiments.db."""
    db.init(str(tmp_path_factory.mktemp("db") / "test_experiments.db"))


@pytest.fixture(scope="session")
def bridge():
    return ADAPTER.ensure_loaded()


@pytest.fixture(scope="session")
def ioi():
    clean = "John and Mary went to the shop. John gave a book to"
    corrupted = "John and Mary went to the shop. Mary gave a book to"
    tok_a = ADAPTER.token_id_for_str(" Mary")
    tok_b = ADAPTER.token_id_for_str(" John")
    return {
        "clean": clean,
        "corrupted": corrupted,
        "tok_a": tok_a,
        "tok_b": tok_b,
        "s2_position": 9,
        "end_position": 13,
    }


# Every test whose fixture closure needs the real model is tagged
# with the `model` marker (registered in pytest.ini), so CI can run
# `pytest -m "not model"` as a fast job. Fixture-based tagging means existing
# tests are untouched: `bridge`/`ioi` are the model fixtures, and the per-file
# `client`/job fixtures depend on `bridge`, which `item.fixturenames` (the
# transitive closure) already includes.
#
# Two exceptions that request no model fixture but still need the model
# environment (found by running the fast subset):
# - test_study.py: its fixture-less tests (exclusion rule, determinism, worker
#   rejection) read the completed studies that only the module's own model
#   tests leave in RUNNER - the module only makes sense as one model-marked
#   unit.
# - test_provenance.py::test_collect_has_all_keys asserts a real 40-hex HF
#   model revision, which requires the downloaded weights to be cached.
_MODEL_FIXTURES = {"bridge", "ioi"}
_MODEL_MODULES = {"test_study"}
_MODEL_SINGLE_TESTS = {"test_provenance.py::test_collect_has_all_keys"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        module = Path(str(item.fspath)).stem
        if (
            _MODEL_FIXTURES.intersection(item.fixturenames)
            or module in _MODEL_MODULES
            or any(item.nodeid.endswith(t) for t in _MODEL_SINGLE_TESTS)
        ):
            item.add_marker(pytest.mark.model)
