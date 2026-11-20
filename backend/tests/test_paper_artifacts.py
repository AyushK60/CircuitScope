"""Stage 6 B1: the committed results files and the generated paper tables stay
in agreement.

Recomputes headline numbers from the stored `per_prompt` rows of
results/study_seed{0,1}.json (the committed copies of the two studies of
record) with the statistic definitions of backend/app/analysis/stats.py, and
asserts they equal (a) the stored `summary` and (b) the numbers rendered in
the generated docs/paper/tables.md. No model, no backend: this is a check of
committed artefacts, so it runs in the fast subset.

Import note: scripts/ is not a package; paper_artifacts.py is imported by
path so the recomputation code under test is exactly the code that generated
the tables.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_MD = REPO_ROOT / "docs" / "paper" / "tables.md"

_spec = importlib.util.spec_from_file_location("paper_artifacts", SCRIPTS_DIR / "paper_artifacts.py")
paper_artifacts = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("paper_artifacts", paper_artifacts)
_spec.loader.exec_module(paper_artifacts)

RESULTS = {
    seed: json.loads((RESULTS_DIR / f"study_seed{seed}.json").read_text(encoding="utf-8"))
    for seed in (0, 1)
}


def _exploratory_row(method: str) -> list[str]:
    """The labelled exploratory row for `method` in tables.md: | AtP | sp | pe | ov / 10 |."""
    text = TABLES_MD.read_text(encoding="utf-8")
    match = re.search(rf"^\| {method} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) / 10 \|$", text, re.MULTILINE)
    assert match is not None, f"exploratory row for {method} not found in tables.md"
    return match.groups()


def test_atp_spearman_recomputed_matches_summary_and_tables() -> None:
    """Headline 1: mean per-prompt AtP Spearman over the 144 heads.

    Tolerance: the stored per_prompt rows are rounded to 6 decimal places at
    storage time, while the stored summary was computed from full-precision
    values before that rounding, so recomputation from the rows differs from
    the stored mean by pure storage rounding (observed ~1e-7; the Stage 3
    verification used 5e-7 for the same reason on the averaged vector).
    1e-6 is far below anything scientifically meaningful.
    """
    for seed, res in RESULTS.items():
        recomputed = paper_artifacts.headline_numbers(res)["atp_spearman_mean"]
        stored = res["summary"]["rq1"]["all_prompts"]["methods"]["atp"]["heads"]["signed"]["mean"]
        assert recomputed == pytest.approx(stored, abs=1e-6), f"seed {seed}: recomputed vs stored summary"
    seed0_row = _exploratory_row("AtP")
    assert seed0_row[0] == "0.976"  # the value rendered in tables.md (3 dp of the stored mean)
    assert f"{RESULTS[0]['summary']['rq1']['all_prompts']['methods']['atp']['heads']['signed']['mean']:.3f}" == seed0_row[0]


def test_dla_top10_overlap_recomputed_matches_tables() -> None:
    """Headline 2: mean per-prompt DLA top-10 overlap vs the exact ranking.

    The overlap is not part of the stored summary (it is the labelled
    exploratory view), so the reference value is the generated tables.md,
    which paper_artifacts.py derives from the same recomputation — here the
    recomputation is re-done independently by this test module's own
    topk-overlap use of the same imported functions over the raw rows.
    """
    res = RESULTS[0]
    recomputed = paper_artifacts.headline_numbers(res)["dla_top10_overlap_mean"]
    pp = res["per_prompt"]
    exact = [row["rec_d"]["swap"][: paper_artifacts.N_HEADS] for row in pp]
    dla = [row["dla"][: paper_artifacts.N_HEADS] for row in pp]
    overlaps = [paper_artifacts.topk_overlap(e, d, 10) for e, d in zip(exact, dla)]
    assert recomputed == pytest.approx(sum(overlaps) / len(overlaps), abs=1e-12)
    row = _exploratory_row("DLA")
    assert f"{recomputed:.2f}" == row[2]
    assert row[2] == "4.00"  # random expectation is 0.69, so 4.00 is ~6x chance


def test_accuracy_recomputed_matches_summary_and_tables() -> None:
    """Headline 3: task accuracy as the fraction of prompts with D_clean > 0."""
    for seed, res in RESULTS.items():
        recomputed = paper_artifacts.headline_numbers(res)
        assert recomputed["accuracy"] == pytest.approx(res["model_accuracy"], abs=1e-12)
        n = res["params"]["n_prompts"]
        assert recomputed["n_correct"] == round(res["model_accuracy"] * n)
    tables = TABLES_MD.read_text(encoding="utf-8")
    assert "0.95 (38/40)" in tables, "seed-0 accuracy row missing from tables.md"
    assert "0.975 (39/40)" in tables, "seed-1 accuracy row missing from tables.md"


def test_top10_heads_identical_across_seeds_and_in_tables() -> None:
    expected = ["5.5", "8.10", "8.6", "9.9", "7.9", "3.0", "7.3", "6.9", "10.0", "9.7"]
    names_by_seed = {}
    for seed, res in RESULTS.items():
        vec = res["summary"]["rq2"]["prompt_averaged_vectors"]["swap_D"]
        names_by_seed[seed] = paper_artifacts.topk_names(vec, 10)
    assert names_by_seed[0] == expected
    assert names_by_seed[1] == expected, "top-10 by prompt-averaged Rec_D[swap] must be identical across seeds"
    tables = TABLES_MD.read_text(encoding="utf-8")
    assert f"Seed 0: {', '.join(expected)}." in tables
    assert f"Seed 1: {', '.join(expected)}." in tables


def test_exploratory_pearson_matches_tables_and_is_labelled() -> None:
    """The exploratory nuance (Pearson + top-10) is present and labelled."""
    stats = paper_artifacts.exploratory_stats(RESULTS[0])
    atp_row = _exploratory_row("AtP")
    dla_row = _exploratory_row("DLA")
    assert f"{stats['methods']['atp']['pearson_mean']:.3f}" == atp_row[1] == "0.969"
    assert f"{stats['methods']['dla']['pearson_mean']:.3f}" == dla_row[1] == "0.540"
    assert f"{stats['methods']['atp']['top10_overlap_mean']:.2f}" == atp_row[2] == "9.45"
    assert f"{stats['methods']['dla']['top10_overlap_mean']:.2f}" == dla_row[2] == "4.00"
    tables = TABLES_MD.read_text(encoding="utf-8")
    assert "EXPLORATORY" in tables, "the exploratory table must be labelled"


def test_exploratory_dla_spearman_column_reproduces_pre_registered_value() -> None:
    """The exploratory table's Spearman column is the pre-registered RQ1 number
    (tolerance as in test_atp_spearman_recomputed_matches_summary_and_tables:
    6-decimal storage rounding of per_prompt inputs)."""
    stats = paper_artifacts.exploratory_stats(RESULTS[0])
    stored = RESULTS[0]["summary"]["rq1"]["all_prompts"]["methods"]["dla"]["heads"]["signed"]["mean"]
    assert stats["methods"]["dla"]["spearman_mean"] == pytest.approx(stored, abs=1e-6)
    row = _exploratory_row("DLA")
    assert row[0] == f"{stored:.3f}" == "0.170"


def test_results_files_are_the_studies_of_record() -> None:
    """The committed files carry the identity and provenance of the two studies."""
    for seed, res in RESULTS.items():
        assert res["job_type"] == "study"
        assert res["params"]["seed"] == seed
        assert res["params"]["n_prompts"] == 40
    assert RESULTS[0]["provenance"]["git_commit"].startswith("a57b0f9")
    assert (
        RESULTS[0]["provenance"]["model_revision"]
        == RESULTS[1]["provenance"]["model_revision"]
        == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    assert len(RESULTS[0]["per_prompt"]) == len(RESULTS[1]["per_prompt"]) == 40
