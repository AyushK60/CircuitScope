"""Claim lint — banned words, readability limits, structure."""

from __future__ import annotations

import pytest

from backend.app.explain import lint
from backend.app.explain.thresholds import (
    MAX_MEAN_SENTENCE_WORDS,
    MAX_SENTENCE_WORDS,
    MIN_FLESCH_READING_EASE,
)


def claim(text, level="curious", claim_type="measured", evidence=None, cid="test.claim"):
    return {
        "id": cid,
        "level": level,
        "text": text,
        "claim_type": claim_type,
        "evidence": evidence or ["data.value"],
    }


class TestBannedWords:
    @pytest.mark.parametrize(
        "word", ["proves", "proof", "understands", "knows", "thinks", "reasons", "decides", "causes"]
    )
    def test_single_banned_words_flagged(self, word):
        assert lint.lint_text(f"This {word} the circuit.", "expert")

    def test_case_insensitive(self):
        assert lint.lint_text("This PROVES the claim.", "expert")

    def test_whole_word_only(self):
        # "proofs" is not the whole word "proof"; "thinking" is not "thinks"
        assert lint.lint_text("The proofs multiply.", "expert") == []
        assert lint.lint_text("It was thinking ahead.", "expert") == []

    @pytest.mark.parametrize("phrase", ["responsible for", "is the reason", "explains why"])
    def test_banned_phrases_flagged(self, phrase):
        assert lint.lint_text(f"This head is {phrase} the answer.", "expert")

    def test_markers_do_not_trip_the_checker(self):
        text = 'The {{term:score_gap|score gap}} is 2.51.'
        assert lint.lint_text(text, "curious") == []


class TestMarkers:
    def test_strip_markers_yields_display_text(self):
        assert (
            lint.strip_markers('The {{term:score_gap|score gap}} grows.')
            == "The score gap grows."
        )

    def test_unknown_term_flagged(self):
        assert lint.lint_claim(claim("x {{term:nope|nope}} y"))


class TestReadability:
    def test_short_simple_text_passes(self):
        text = "The test copies one signal. Then the model runs again."
        assert lint.lint_text(text, "curious") == []

    def test_long_sentence_fails(self):
        text = " ".join(["word"] * (MAX_SENTENCE_WORDS + 1)) + "."
        assert any("longest sentence" in p for p in lint.lint_text(text, "curious"))

    def test_high_mean_sentence_length_fails(self):
        # many sentences of 20 words each: mean 20 > 18, each sentence <= 28
        sent = " ".join(["word"] * 20) + "."
        text = sent * 3
        assert any("mean sentence words" in p for p in lint.lint_text(text, "curious"))

    def test_flesch_boundary(self):
        # a polysyllabic sentence with few words per sentence pushes Flesch down
        text = "Probability simultaneously characterized terminology."  # 4 words, heavy
        r = lint.readability(text)
        assert r["flesch_reading_ease"] < MIN_FLESCH_READING_EASE
        assert any("Flesch" in p for p in lint.lint_text(text, "curious"))

    def test_readability_stats_sane(self):
        r = lint.readability("One two three. Four five six seven eight.")
        assert r["n_sentences"] == 2
        assert r["n_words"] == 8
        assert r["max_sentence_words"] == 5
        # per the documented syllable heuristic "seven" counts as 2 syllables
        assert r["flesch_reading_ease"] == pytest.approx(107.6)

    def test_idiom_banned_only_at_curious(self):
        text = "Let's take a deep dive into the black box."
        assert any("idiom" in p for p in lint.lint_text(text, "curious"))
        assert lint.lint_text(text, "expert") == []


class TestClaimStructure:
    def test_missing_evidence_flagged(self):
        c = {"id": "x", "level": "curious", "text": "Some text.", "claim_type": "measured", "evidence": []}
        assert any("evidence" in p for p in lint.lint_claim(c))

    def test_bad_claim_type_flagged(self):
        assert lint.lint_claim(claim("Text.", claim_type="guess"))

    def test_bad_level_flagged(self):
        assert lint.lint_claim(claim("Text.", level="poetic"))

    def test_duplicate_id_level_pair_flagged(self):
        a = claim("One.", cid="dup")
        b = claim("Two.", cid="dup")
        assert any("duplicate" in p for p in lint.lint_claims([a, b]))

    def test_same_id_different_levels_ok(self):
        claims = [claim("One.", cid="x", level=lvl) for lvl in ("curious", "student", "expert")]
        assert lint.lint_claims(claims) == []


class TestMarkdownLint:
    def test_prose_lines_only(self):
        md = "# Heading\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\nShort prose here. Fine.\n"
        assert lint.lint_markdown(md, "curious") == []

    def test_banned_word_in_prose(self):
        md = "This proves the point.\n"
        assert lint.lint_markdown(md, "curious")

    def test_long_sentence_in_claim_bullet(self):
        # readability is scoped to generated claim bullets
        md = "- [measured] " + " ".join(["word"] * 40) + ".\n"
        assert lint.lint_markdown(md, "curious")

    def test_long_prose_line_without_claim_marker_not_readability_checked(self):
        # structural prose (not generated claims) is exempt from the readability
        # limits but still banned-word-checked
        md = "model revision: " + "a" * 200 + "\n"
        assert lint.lint_markdown(md, "curious") == []


def test_declared_limits_match_thresholds_module():
    # the lint uses the same declared constants (single source of truth)
    import backend.app.explain.lint as l

    assert l.MAX_MEAN_SENTENCE_WORDS == MAX_MEAN_SENTENCE_WORDS
    assert l.MAX_SENTENCE_WORDS == MAX_SENTENCE_WORDS
    assert l.MIN_FLESCH_READING_EASE == MIN_FLESCH_READING_EASE
