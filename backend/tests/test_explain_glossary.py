"""Glossary consistency — backend side.

- every term has all three levels with non-empty text
- curious definitions pass the principle-7 lint (banned words + readability)
- analogies have a mandatory breaks_when line
- every {{term:...}} marker in backend templates exists in the glossary
- every glossary term id referenced in frontend source exists
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.app.explain import glossary, lint
from backend.app.explain.templates import TEMPLATES, TERM_MARKER_RE

FRONTEND_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"
GLOSSARY_JSON = (
    Path(__file__).resolve().parents[2] / "examples" / "glossary.json"
)


class TestGlossaryData:
    def test_file_exists_and_loads(self):
        data = glossary.load_glossary()
        assert data["terms"] and isinstance(data["analogies"], list)

    def test_every_term_has_three_levels(self):
        for tid, t in glossary.terms().items():
            for level in ("curious", "student", "expert"):
                assert t.get(level), f"term {tid} missing {level} definition"
            assert t.get("display"), f"term {tid} missing display"

    def test_see_also_ids_exist(self):
        for tid, t in glossary.terms().items():
            for other in t.get("see_also", []):
                assert other in glossary.terms(), f"{tid} -> unknown see_also {other}"

    def test_curious_definitions_pass_lint(self):
        problems = []
        for tid, t in glossary.terms().items():
            problems += [f"{tid}: {p}" for p in lint.lint_text(t["curious"], "curious")]
        assert problems == []

    def test_analogies_have_breaks_when(self):
        for a in glossary.analogies():
            assert a.get("id") and a.get("text") and a.get("title")
            assert a.get("breaks_when"), f"analogy {a['id']} missing breaks_when"
            assert a.get("term_id") in glossary.terms()

    def test_required_terms_present(self):
        required = {
            "token", "probability", "score", "score_gap", "residual_stream",
            "attention_head", "mlp", "block", "patching", "ablation", "recovery",
            "direct_contribution", "logit_lens", "attribution_patching",
            "rank_correlation", "confidence_interval", "control", "prompt_set",
            "corrupted_prompt", "baseline",
        }
        assert required <= set(glossary.terms())

    def test_required_analogies_present(self):
        ids = {a["id"] for a in glossary.analogies()}
        assert {"shared_whiteboard", "spotlight", "note_swap", "covered_answer", "half_finished"} <= ids


class TestTemplateMarkers:
    def test_all_template_markers_known(self):
        # import-time check already raises; re-assert explicitly per template
        for key, spec in TEMPLATES.items():
            for level, text in spec["levels"].items():
                for m in TERM_MARKER_RE.finditer(text):
                    assert glossary.has_term(m.group(1)), f"{key}[{level}]: {m.group(1)}"

    def test_marker_display_text_nonempty(self):
        for spec in TEMPLATES.values():
            for text in spec["levels"].values():
                for m in TERM_MARKER_RE.finditer(text):
                    display = m.group(0).split("|", 1)[1].rstrip("}")
                    assert display.strip(), m.group(0)

    def test_placeholders_declared_cover_all_levels(self):
        for key, spec in TEMPLATES.items():
            used = set()
            for text in spec["levels"].values():
                used |= {m.group(1) for m in re.finditer(r"\$([a-z][a-z0-9_]*)", text)}
            assert set(spec["placeholders"]) == used, (
                f"{key}: declared {spec['placeholders']} vs used {sorted(used)}"
            )


class TestFrontendTermIds:
    def test_term_ids_used_in_frontend_exist(self):
        literal_pattern = re.compile(r'<Term\s+[^>]*?id="([a-z_]+)"')
        dynamic_pattern = re.compile(r"<Term\s")
        used: set[str] = set()
        dynamic_uses = 0
        for p in FRONTEND_SRC.rglob("*.tsx"):
            if "__tests__" in p.parts or p.name.endswith(".test.tsx"):
                continue
            src = p.read_text(encoding="utf-8")
            used |= set(literal_pattern.findall(src))
            dynamic_uses += len(dynamic_pattern.findall(src))
        # Term components are rendered (dynamically from template markers, whose
        # ids are validated against the glossary by the template tests)
        assert dynamic_uses > 0 or used, "no <Term ...> usages found in frontend source"
        unknown = used - set(glossary.terms())
        assert not unknown, f"frontend uses unknown glossary terms: {unknown}"
