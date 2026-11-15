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
