"""Claim lint.

``lint_claims`` is used BOTH by the tests and at runtime: a report job that
produces a lint failure fails honestly instead of shipping bad text.

Checks per claim:
  - structural: id / level / claim_type present and valid, evidence non-empty.
  - banned words and phrases (principle 4), case-insensitive, whole word for
    single words, substring for fixed phrases. Glossary markers are stripped
    first (the marker syntax itself must not trip the checker).
  - glossary markers well-formed and known.
  - `curious` level only (principle 7): readability limits
    (mean sentence words <= 18, longest sentence <= 28 words, Flesch reading
    ease >= 60) and the banned-idiom list.

Syllable heuristic (documented for CONCEPTS_STAGE4.md): a word's syllable
count is the number of vowel groups ([aeiouy]+) in the lowercased word,
minimum 1, minus one if the word ends in a silent 'e' (ends with 'e', has
more than one vowel group, and the 'e' is not part of the final group as in
"le"). Digits count as one-syllable words. This is a heuristic, not a
pronunciation model; the limits were chosen so that ordinary short English
passes with margin.
"""

from __future__ import annotations

import re

from . import glossary
from .banned_words import BANNED_IDIOMS, BANNED_PHRASES, BANNED_WORDS
from .templates import CLAIM_TYPES, LEVELS, TERM_MARKER_RE
from .thresholds import (
    MAX_MEAN_SENTENCE_WORDS,
    MAX_SENTENCE_WORDS,
    MIN_FLESCH_READING_EASE,
)

TERM_FULL_RE = re.compile(r"\{\{term:[a-z_]+\|[^{}]*\}\}")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

_VOWEL_GROUPS_RE = re.compile(r"[aeiouy]+")


def strip_markers(text: str) -> str:
    """Replace {{term:id|display}} markers with their display text."""

    def sub(m: re.Match) -> str:
        inner = m.group(0)
        return inner[inner.index("|") + 1 : -2]

    return TERM_FULL_RE.sub(sub, text)


def count_syllables(word: str) -> int:
    w = word.lower()
    groups = _VOWEL_GROUPS_RE.findall(w)
    n = max(1, len(groups))
    if len(w) > 2 and w.endswith("e") and n > 1 and not w.endswith(
        ("le", "ee", "ye", "ie", "ue", "oe")
    ):
        n -= 1
    return n


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def readability(text: str) -> dict:
    """Word/sentence statistics + Flesch reading ease over stripped text."""
    clean = strip_markers(text)
    sents = sentences(clean)
    words: list[str] = []
    per_sentence: list[int] = []
    for s in sents:
        w = WORD_RE.findall(s)
        per_sentence.append(len(w))
        words.extend(w)
    n_syll = sum(count_syllables(w) for w in words)
    n_words = len(words)
    n_sents = max(1, len(sents))
    asl = n_words / n_sents
    asw = n_syll / n_words if n_words else 0.0
    flesch = 206.835 - 1.015 * asl - 84.6 * asw if n_words else 206.835
    return {
        "n_sentences": len(sents),
        "n_words": n_words,
        "mean_sentence_words": round(asl, 2),
        "max_sentence_words": max(per_sentence) if per_sentence else 0,
        "flesch_reading_ease": round(flesch, 1),
    }


def _banned_word_problems(text: str) -> list[str]:
    clean = strip_markers(text).lower()
    problems: list[str] = []
    for w in BANNED_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", clean):
            problems.append(f"banned word: {w!r}")
    for p in BANNED_PHRASES:
        if p in clean:
            problems.append(f"banned phrase: {p!r}")
    return problems


def _marker_problems(text: str) -> list[str]:
    problems: list[str] = []
    for m in re.finditer(r"\{\{[^}]*\}\}", text):
        if not TERM_MARKER_RE.fullmatch(m.group(0)):
            problems.append(f"malformed glossary marker: {m.group(0)!r}")
        else:
            term_id = TERM_MARKER_RE.fullmatch(m.group(0)).group(1)
            if not glossary.has_term(term_id):
                problems.append(f"unknown glossary term: {term_id!r}")
    return problems


def lint_text(text: str, level: str) -> list[str]:
    """Lint one generated text at one level (banned words at every level;
    readability + idioms at `curious`)."""
    problems = _banned_word_problems(text)
    if level == "curious":
        clean = strip_markers(text).lower()
        for idiom in BANNED_IDIOMS:
            if idiom in clean:
                problems.append(f"banned idiom (curious): {idiom!r}")
        r = readability(text)
        if r["mean_sentence_words"] > MAX_MEAN_SENTENCE_WORDS:
            problems.append(
                f"curious mean sentence words {r['mean_sentence_words']} > {MAX_MEAN_SENTENCE_WORDS}"
            )
        if r["max_sentence_words"] > MAX_SENTENCE_WORDS:
            problems.append(
                f"curious longest sentence {r['max_sentence_words']} words > {MAX_SENTENCE_WORDS}"
            )
        if r["flesch_reading_ease"] < MIN_FLESCH_READING_EASE:
            problems.append(
                f"curious Flesch reading ease {r['flesch_reading_ease']} < {MIN_FLESCH_READING_EASE}"
            )
    return problems


def lint_claim(claim: dict) -> list[str]:
    problems: list[str] = []
    cid = claim.get("id", "<no id>")
    if not claim.get("id"):
        problems.append("claim missing id")
    level = claim.get("level")
    if level not in LEVELS:
        problems.append(f"{cid}: bad level {level!r}")
    if claim.get("claim_type") not in CLAIM_TYPES:
        problems.append(f"{cid}: bad claim_type {claim.get('claim_type')!r}")
    ev = claim.get("evidence")
    if not isinstance(ev, list) or not ev:
        problems.append(f"{cid}: evidence must be a non-empty list")
    text = claim.get("text", "")
    if not isinstance(text, str) or not text.strip():
        problems.append(f"{cid}: empty text")
    problems.extend(f"{cid}: {p}" for p in _marker_problems(text))
    if level in LEVELS:
        problems.extend(f"{cid}: {p}" for p in lint_text(text, level))
    return problems


def lint_claims(claims: list[dict]) -> list[str]:
    """Lint a list of claim objects; returns a list of problems (empty = pass).
    Uniqueness is per (id, level): each claim instance appears once per level."""
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for c in claims:
        problems.extend(lint_claim(c))
        key = (c.get("id", "?"), c.get("level", "?"))
        if key in seen:
            problems.append(f"{key[0]}[{key[1]}]: duplicate claim id/level pair")
        seen.add(key)
    return problems


def lint_markdown(md: str, level: str = "curious") -> list[str]:
    """Lint the exported Markdown. Banned words apply to every prose line
    (headings and table rows excluded); the curious readability limits apply
    to the generated claim bullets ("- [claim_type] ..."), which are the only
    generated sentences in the document - structural metadata (provenance
    hashes, evidence-tier labels, fixed trust-panel lines) is not generated
    claim text."""
    problems: list[str] = []
    prose_lines: list[str] = []
    claim_lines: list[str] = []
    for line in md.splitlines():
        s = line.strip()
        if not s or s.startswith("|") or s.startswith("#"):
            continue
        prose_lines.append(s)
        if s.startswith("- ["):
            claim_lines.append(s[len("- [") :].partition("]")[2].strip())
    problems.extend(_banned_word_problems("\n".join(prose_lines)))
    if level == "curious":
        for line in claim_lines:
            problems.extend(lint_text(line, "curious"))
    return problems
