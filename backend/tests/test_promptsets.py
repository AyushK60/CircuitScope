"""Stage 3 Task 1 tests: the deterministic prompt-set generator.

All checks run against the REAL tokenizer (loaded via the session bridge).
"""

from __future__ import annotations

import pytest

from backend.app.analysis import promptsets
from backend.app.model_runtime.loader import ADAPTER


def test_determinism_same_seed(bridge):
    a = promptsets.generate(12, 7)
    b = promptsets.generate(12, 7)
    assert a["prompts"] == b["prompts"]
    assert a["rejected_draws"] == b["rejected_draws"]
    assert a["template_counts"] == b["template_counts"]


def test_different_seed_differs(bridge):
    a = promptsets.generate(12, 7)
    b = promptsets.generate(12, 8)
    assert a["prompts"] != b["prompts"]


def test_record_fields(bridge):
    out = promptsets.generate(6, 0)
    for p in out["prompts"]:
        for key in (
            "id", "template_id", "names", "place", "obj", "s2_role", "clean_text",
            "candidate_a", "candidate_b", "corruptions", "s2_position",
            "end_position", "n_tokens",
        ):
            assert key in p, f"missing field {key}"
        assert set(p["names"]) == {"A", "B", "C"}
        assert set(p["corruptions"]) == {"swap", "third", "resample"}


def test_all_words_single_token(bridge):
    out = promptsets.generate(20, 3)
    for p in out["prompts"]:
        for name in p["names"].values():
            ADAPTER.token_id_for_str(f" {name}")  # raises if multi-token
        ADAPTER.token_id_for_str(f" {p['place']}")
        ADAPTER.token_id_for_str(f" {p['obj']}")
        ADAPTER.token_id_for_str(p["candidate_a"])
        ADAPTER.token_id_for_str(p["candidate_b"])


def test_alignment_of_all_three_corruptions_100_prompts(bridge):
    out = promptsets.generate(100, 11)
    assert len(out["prompts"]) == 100
    for p in out["prompts"]:
        for kind in ("swap", "third", "resample"):
            text = p["corruptions"][kind]
            assert text is not None, f"{kind} missing for {p['id']}"
            n = promptsets._token_count(text)
            assert n == p["n_tokens"], (
                f"{p['id']} {kind} misaligned: {n} vs {p['n_tokens']}"
            )


def test_template_and_s2_role_balance(bridge):
    out = promptsets.generate(36, 5)
    assert out["template_counts"] == {
        "template_1": 12, "template_2": 12, "template_3": 12
    }
    assert out["s2_role_counts"] == {"A": 18, "B": 18}


def test_io_is_not_s(bridge):
    out = promptsets.generate(20, 2)
    for p in out["prompts"]:
        assert p["candidate_a"] != p["candidate_b"]
        io, s = p["candidate_a"].strip(), p["candidate_b"].strip()
        # S is one of A/B; IO is the other
        assert s in (p["names"]["A"], p["names"]["B"])
        assert io in (p["names"]["A"], p["names"]["B"])
        assert io != s


def test_s2_position_points_at_s2_name(bridge):
    out = promptsets.generate(20, 4)
    for p in out["prompts"]:
        toks = ADAPTER.bridge().to_str_tokens(p["clean_text"])
        s2 = p["candidate_b"].strip()
        assert toks[p["s2_position"]].strip() == s2
        assert p["end_position"] == len(toks) - 1
        assert p["n_tokens"] == len(toks)


def test_swap_flips_subject_and_third_uses_c(bridge):
    out = promptsets.generate(9, 6)
    for p in out["prompts"]:
        s2 = p["candidate_b"].strip()
        io = p["candidate_a"].strip()
        # swap: the second subject position now holds the OTHER name
        toks_swap = ADAPTER.bridge().to_str_tokens(p["corruptions"]["swap"])
        assert toks_swap[p["s2_position"]].strip() == io
        # third: the second subject position now holds C
        toks_third = ADAPTER.bridge().to_str_tokens(p["corruptions"]["third"])
        assert toks_third[p["s2_position"]].strip() == p["names"]["C"]
        # clean still holds S2
        toks_clean = ADAPTER.bridge().to_str_tokens(p["clean_text"])
        assert toks_clean[p["s2_position"]].strip() == s2


def test_resample_uses_next_prompt_of_same_template(bridge):
    out = promptsets.generate(9, 6)
    prompts = out["prompts"]
    for i, p in enumerate(prompts):
        # find the cyclic next prompt of the same template
        nxt = None
        for j in range(1, len(prompts) + 1):
            cand = prompts[(i + j) % len(prompts)]
            if cand["template_id"] == p["template_id"]:
                nxt = cand
                break
        assert p["corruptions"]["resample"] == nxt["clean_text"]


def test_invalid_n_rejected(bridge):
    with pytest.raises(ValueError):
        promptsets.generate(0, 0)


def test_default_pair_template1_layout(bridge):
    """The generator's template 1 must reproduce the default IOI token layout:
    S2 at position 9, 14 tokens total (BOS included)."""
    out = promptsets.generate(3, 1)
    p = next(p for p in out["prompts"] if p["template_id"] == 1)
    assert p["s2_position"] == 9
    assert p["n_tokens"] == 14
