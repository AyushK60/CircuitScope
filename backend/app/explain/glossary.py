"""Glossary loader: examples/glossary.json is the single source shared with
the frontend (frontend/src/lib/glossary.ts imports the same file).

The backend validates that every {{term:ID|display}} marker used in a template
exists in the glossary (templates.py checks at import; tests re-check).
"""

from __future__ import annotations

import json
import pathlib
from functools import lru_cache

GLOSSARY_PATH = pathlib.Path(__file__).resolve().parents[3] / "examples" / "glossary.json"

LEVELS = ("curious", "student", "expert")


@lru_cache(maxsize=1)
def load_glossary() -> dict:
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def terms() -> dict:
    return load_glossary()["terms"]


def analogies() -> list[dict]:
    return load_glossary()["analogies"]


def has_term(term_id: str) -> bool:
    return term_id in terms()


def term_display(term_id: str) -> str:
    return terms()[term_id]["display"]


def analogy_for_term(term_id: str) -> dict | None:
    """The glossary entry's own analogy (text + breaks_when), if it has one."""
    t = terms().get(term_id)
    if not t:
        return None
    a = t.get("analogy")
    return dict(a) if a else None


def analogy_by_id(analogy_id: str) -> dict | None:
    for a in analogies():
        if a["id"] == analogy_id:
            return dict(a)
    return None
