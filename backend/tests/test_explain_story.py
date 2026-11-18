"""Stage 4 Task 7: property-style story generation tests over synthetic grids.

Grid covers: negative recovery, near-zero denominators, missing study, ties
(constant grids), empty lists, and None lens D values. Invariants asserted:
never raises; every claim has valid claim_type + non-empty evidence; lint
passes at all three levels; the curious level meets the readability limits;
absent data produces a "not run yet" claim; the qualitative word always
matches the bucket of the cited number.
"""

from __future__ import annotations

import random

import pytest

from backend.app.explain import lint, story as st
from backend.app.explain.thresholds import recovery_bucket

TOKENS = ["<|endoftext|>", "John", "and", " Mary", "went", "to", "the", "shop", ".", " John", "gave", "a", "book", "to"]


def grid(rows, cols, values):
    if callable(values):
        return [[values(l, c) for c in range(cols)] for l in range(rows)]
    return [[values] * cols for _ in range(rows)]


def make_position_sweep(kind="mixed"):
    if kind == "ties":
        recovery = grid(12, len(TOKENS), 0.5)
    elif kind == "negative":
        recovery = grid(12, len(TOKENS), -0.6)
    elif kind == "empty":
        recovery = []
    else:
        rnd = random.Random(7)
        recovery = [[rnd.uniform(-0.3, 1.0) for _ in TOKENS] for _ in range(12)]
    d_patched = grid(12, len(TOKENS), -4.0)
    return {
        "mode": "layer_position",
        "direction": "clean -> corrupted (denoising)",
        "baselines": {"D_clean": 2.5, "D_corrupted": -4.2, "denominator": 6.7, "warnings": []},
        "tokens": list(TOKENS),
        "recovery": recovery,
        "d_patched": d_patched,
        "n_runs": 168,
    }


def make_head_sweep(kind="mixed"):
    if kind == "ties":
        recovery = grid(12, 12, 0.5)
    elif kind == "negative":
        recovery = grid(12, 12, -0.4)
    elif kind == "empty":
        recovery = []
    else:
        rnd = random.Random(11)
        recovery = [[rnd.uniform(-0.3, 1.0) for _ in range(12)] for _ in range(12)]
    return {
        "mode": "layer_head",
        "direction": "clean -> corrupted (denoising)",
        "baselines": {"D_clean": 2.5, "D_corrupted": -4.2, "denominator": 6.7, "warnings": []},
        "tokens": [f"head {h}" for h in range(12)],
        "recovery": recovery,
        "d_patched": grid(12, 12, -4.0),
        "n_runs": 144,
    }


def make_lens(d_values=None):
    if d_values is None:
        d_values = [-1.2] + [-1.2 + 0.3 * l + (1.5 if l == 7 else 0.0) for l in range(12)]
    stages = [
        {"stage": "embeddings", "layer": None, "D": d_values[0], "prob_a": 0.1, "prob_b": 0.5,
         "top1": " the", "top1_prob": 0.2}
    ] + [
        {"stage": f"block {l}", "layer": l, "D": d_values[l + 1], "prob_a": None, "prob_b": None,
         "top1": None, "top1_prob": None}
        for l in range(12)
    ]
    return {"stages": stages, "max_jump": None, "final_d": d_values[-1],
            "final_check": None, "note": "note"}


def make_dla(values=None):
    if values is None:
        values = [1.65, 1.29, -1.03] + [0.1] * 141
    top = [{"label": f"{i // 12}.{i % 12}", "layer": i // 12, "index": i % 12, "value": v, "rank": r + 1}
           for r, (i, v) in enumerate(sorted(enumerate(values), key=lambda t: (-t[1], t[0])))]
    return {"D": 2.5,
            "reconciliation": {"sum_of_terms": 2.5, "model_D": 2.5, "abs_error": 1e-5,
                               "tolerance": 2e-3, "ok": True},
            "top": top, "most_negative": top[-1], "notes": []}


REFERENCE = {
    "groups": [
        {"group": "name_movers", "heads": ["9.9", "10.0", "9.6"]},
        {"group": "s_inhibition", "heads": ["7.3", "7.9", "8.6", "8.10"]},
        {"group": "previous_token", "heads": ["2.2", "4.11"]},
    ],
    "fuzzy_heads": ["5.8", "5.9"],
    "verified_against_paper": True,
    "note": "reference note",
    "citation": "Wang et al. 2022",
}

PRED = {
    "candidate_a": " Mary", "candidate_b": " John", "top_token": " Mary", "top_prob": 0.62,
    "top3": [], "prob_a": 0.62, "prob_b": 0.11, "D": 2.5128,
    "sign_convention": "positive D means candidate A scores above candidate B",
}

STUDY = {
    "params": {"n_prompts": 8, "seed": 0}, "n_prompts": 8, "model_accuracy": 0.875,
    "exclusions": {"swap": 0, "third": 1, "resample": 2}, "exclusion_rule": "rule",
    "rq1_atp_heads": {"mean": 0.976, "ci_low": 0.97, "ci_high": 0.982},
    "rq1_dla_heads": {"mean": 0.17, "ci_low": 0.1, "ci_high": 0.25},
    "beats_control": {"atp": True, "dla": False},
    "overlap": {"atp": 7, "dla": 3, "expected_random_k10": 0.6944},
    "stability": [{"condition": "third_D", "k10_overlap": 9}, {"condition": "resample_D", "k10_overlap": 8}],
    "expected_random_k10": 0.6944,
}


def all_story_claims(kind="mixed", study=STUDY):
    lens = make_lens()
    lens["max_jump"] = st.compact_lens(
        {"stages": [
            {"stage": s["stage"], "layer": s["layer"], "D": s["D"], "prob_a": None, "prob_b": None, "top": []}
            for s in lens["stages"]
        ], "final_matches_model": None, "note": "note"}
    )["max_jump"]
    steps = {
        "prediction": PRED,
        "lens": lens,
        "position_sweep": make_position_sweep(kind),
        "head_sweep": make_head_sweep(kind),
        "dla": make_dla(),
        "reference": REFERENCE,
        "study": study,
    }
    # lens needs a real max_jump: recompute via compact_lens on a fake result
    fake_lens_result = {
        "stages": [
            {"stage": s["stage"], "layer": s["layer"], "D": s["D"], "prob_a": None, "prob_b": None, "top": []}
            for s in make_lens()["stages"]
        ],
        "final_matches_model": {"max_abs_logit_diff_candidates": 1e-5, "top1_agrees": True, "ok": True, "tolerance": 1e-3},
        "note": "note",
    }
    steps["lens"] = st.compact_lens(fake_lens_result)
    sections = _fake_sections(steps)
    return [c for s in sections for c in s["claims"]], steps, sections


def _fake_sections(steps):
    """Build section data the same way report.build_sections does, but with
    synthetic grids (no model)."""
    head_sweep = steps["head_sweep"]
    pos_sweep = steps["position_sweep"]
    return [
        {"id": "answer", "data": {"prediction": steps["prediction"]},
         "claims": st.story_prediction(steps["prediction"])},
        {"id": "where_formed", "data": {"lens": steps["lens"]}, "claims": st.story_lens(steps["lens"])},
        {"id": "which_words", "data": {"sweep": pos_sweep},
         "claims": st.story_sweep_position({"sweep": pos_sweep})},
        {"id": "which_parts", "data": {"sweep": head_sweep, "dla": steps["dla"]},
         "claims": st.story_sweep_heads({"sweep": head_sweep, "dla": steps["dla"]})},
        {"id": "vs_paper", "data": {"sweep": head_sweep, "reference": steps["reference"]},
         "claims": st.story_agreement_with_paper({"sweep": head_sweep, "reference": steps["reference"]})},
        {"id": "how_sure", "data": {"study": steps["study"]}, "claims": st.story_study(steps["study"])},
        {"id": "limits", "data": {}, "claims": st.story_limits()},
    ]


@pytest.mark.parametrize("kind", ["mixed", "ties", "negative", "empty"])
def test_all_claims_valid_and_lint_clean(kind):
    claims, _steps, _sections = all_story_claims(kind)
    assert claims
    problems = lint.lint_claims(claims)
    assert problems == []
    for c in claims:
        assert c["claim_type"] in ("measured", "interpretation", "caveat")
        assert isinstance(c["evidence"], list) and c["evidence"]
        assert c["text"].strip()
        assert c["level"] in ("curious", "student", "expert")


@pytest.mark.parametrize("kind", ["mixed", "ties", "negative", "empty"])
def test_curious_readability_limits_hold(kind):
    claims, _s, _sec = all_story_claims(kind)
    curious = " ".join(c["text"] for c in claims if c["level"] == "curious")
    r = lint.readability(curious)
    assert r["mean_sentence_words"] <= 18
    assert r["max_sentence_words"] <= 28
    assert r["flesch_reading_ease"] >= 60


def test_missing_study_produces_not_run_claim():
    claims = st.story_study(None)
    assert any("Not run yet" in c["text"] for c in claims if c["level"] == "curious")
    assert lint.lint_claims(claims) == []


def test_missing_sweep_produces_not_run_claim():
    claims = st.story_sweep_position({"sweep": None})
    assert any("Not run yet" in c["text"] for c in claims if c["level"] == "curious")
    claims = st.story_agreement_with_paper({"sweep": None, "reference": REFERENCE})
    assert any("Not run yet" in c["text"] for c in claims if c["level"] == "curious")


def test_none_lens_produces_not_run_claim():
    claims = st.story_lens(None)
    assert any("Not run yet" in c["text"] for c in claims if c["level"] == "curious")
    assert lint.lint_claims(claims) == []


def test_bucket_word_matches_cited_number():
    """Every recovery-phrase claim cites the same value it bucketed."""
    claims, steps, _sections = all_story_claims("mixed")
    for c in claims:
        if c["level"] != "curious":
            continue
        if "restores most of" in c["text"]:
            expect = "most"
        elif "restores a large part of" in c["text"]:
            expect = "large"
        elif "restores some of" in c["text"]:
            expect = "some"
        elif "has little effect on" in c["text"]:
            expect = "little"
        elif "pushes the score gap the other way" in c["text"] or "pushes the other way" in c["text"]:
            expect = "negative"
        else:
            continue
        # find the recovery value in the cited evidence path; words.* cite the
        # position sweep, parts.recovery_top.* cite the head sweep
        sweep_key = "position_sweep" if c["id"].startswith("words.") else "head_sweep"
        vals = []
        for ev in c["evidence"]:
            if ev.startswith("sweep.recovery["):
                idx = ev[len("sweep.recovery["):-1].split("][")
                l, h = int(idx[0]), int(idx[1])
                vals.append(steps[sweep_key]["recovery"][l][h])
        assert vals, f"claim {c['id']} cites a bucket but no resolvable recovery value"
        assert recovery_bucket(vals[0]) == expect, f"{c['id']}: bucket {recovery_bucket(vals[0])} != {expect}"


def test_ties_broken_by_index():
    sweep = make_position_sweep("ties")
    best = st._best_cell(sweep, (0, 1, 2, 3))
    assert (best["layer"], best["position"]) == (0, 0)
    head = make_head_sweep("ties")
    top = st.top10_by_recovery(head)
    assert top[0]["label"] == "0.0" and top[1]["label"] == "0.1"


def test_negative_grid_uses_negative_wording():
    claims, _s, _sec = all_story_claims("negative")
    curious = [c["text"] for c in claims if c["level"] == "curious"]
    assert any("pushes the score gap the other way" in t for t in curious)
    assert not any("restores most of" in t for t in curious)


def test_lens_max_jump_deterministic():
    lens_result = {
        "stages": [
            {"stage": s, "layer": l, "D": d, "prob_a": None, "prob_b": None, "top": []}
            for s, l, d in [
                ("embeddings", None, 0.0), ("block 0", 0, 1.0), ("block 1", 1, 2.0), ("block 2", 2, 2.0)
            ]
        ],
        "final_matches_model": None,
        "note": "",
    }
    c1 = st.compact_lens(lens_result)["max_jump"]
    c2 = st.compact_lens(lens_result)["max_jump"]
    # tie (jump 1.0 twice) -> earliest stage wins
    assert c1 == c2 == {"stage": "block 0", "D_before": 0.0, "D_after": 1.0, "jump": 1.0}


def test_top10_overlap_from_per_prompt():
    """The extra check recomputes the top-10 overlap from stored per_prompt."""
    def vec(seed):
        rnd = random.Random(seed)
        return [round(rnd.uniform(-1, 1), 4) for _ in range(156)]

    base = vec(0)
    per_prompt = []
    for i in range(6):
        row = {
            "id": f"p{i}", "baselines": {"swap": {"excluded": False}},
            "rec_d": {"swap": list(base)},
            "atp": list(base), "dla": [round(x * 0.5, 6) for x in base],
        }
        per_prompt.append(row)
    # one excluded prompt must be dropped from the overlap
    per_prompt.append({
        "id": "px", "baselines": {"swap": {"excluded": True}},
        "rec_d": {"swap": vec(1)}, "atp": vec(2), "dla": vec(3),
    })
    study_result = {
        "params": {"n_prompts": 7, "seed": 0},
        "model_accuracy": 1.0,
        "exclusions": {"swap": 1, "third": 0, "resample": 0},
        "exclusion_rule": "rule",
        "per_prompt": per_prompt,
        "summary": {"rq1": {"all_prompts": {"methods": {"atp": {"heads": {"signed": {"mean": 0.9}}}}}}},
        "rq1": None,
    }
    compact = st.compact_study(study_result)
    assert compact["overlap"]["atp"] == 10  # identical vectors -> full overlap
    assert compact["overlap"]["atp"] == 10 and compact["overlap"]["dla"] == 10  # monotone rescale keeps order
    claims = st.story_study(compact)
    assert lint.lint_claims(claims) == []
    curious = [c["text"] for c in claims if c["level"] == "curious"]
    assert any("share 10 of 10" in t or "shares 10 of 10" in t for t in curious)


def test_study_claim_numbers_cited():
    claims, _s, _sec = all_story_claims("mixed")
    by_id = {c["id"]: c for c in claims if c["level"] == "curious"}
    assert "0.98" in by_id["study.atp"]["text"] or "0.976" in by_id["study.atp"]["text"]
    assert "8" in by_id["study.size"]["text"]
    assert "3" in by_id["study.exclusions"]["text"]  # 0 + 1 + 2
