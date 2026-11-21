"""The agreement job, declared wording boundaries, the layer_head
consistency gate, and the Report section integration.
"""

from __future__ import annotations

import time

import pytest

from backend.app.analysis.agreement import N_HEADS, N_COMPONENTS, exact_head_vs_sweep
from backend.app.api import routes as _routes  # noqa: F401  (registers job types)
from backend.app.experiments.jobs import RUNNER, JobRecord
from backend.app.experiments.sweeps import run_sweep
from backend.app.explain import lint as explain_lint
from backend.app.explain.story import compact_agreement, story_agreement_methods
from backend.app.explain.thresholds import agreement_bucket, agreement_phrase


def _wait(job_id: str, timeout: float = 300.0) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = RUNNER.get(job_id)
        if job is not None and job.status in ("complete", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


@pytest.fixture(scope="module")
def agreement(bridge):
    job = RUNNER.submit({"task_id": "ioi_john_mary"}, job_type="agreement")
    done = _wait(job.job_id, timeout=180)
    assert done.status == "complete", done.error
    return done


@pytest.fixture(scope="module")
def report_job(bridge):
    job = RUNNER.submit({"task_id": "ioi_john_mary"}, job_type="report")
    done = _wait(job.job_id, timeout=300)
    assert done.status == "complete", done.error
    return done


def _texts(claims, level="curious"):
    return " ".join(c["text"] for c in claims if c["level"] == level)


def _synthetic(pearson: float, early: float = 0.8, late: float = 0.8) -> dict:
    block = {
        "pearson_heads": pearson,
        "spearman_heads": pearson,
        "top10_overlap": 5,
        "expected_random_top10": 0.6944,
        "pearson_groups": {
            "blocks_0_5": {"pearson": early, "n": 72},
            "blocks_6_11": {"pearson": late, "n": 72},
        },
    }
    return {
        "stats": {
            "atp": block,
            "dla": dict(block),
            "procedure": "test",
        },
        "top10": {"exact": [], "atp": [], "dla": []},
        "notes": [],
    }


@pytest.mark.parametrize(
    "rho,bucket,phrase",
    [
        (1.0, "close", "lines up closely with"),
        (0.9, "close", "lines up closely with"),
        (0.8999999, "rough", "lines up roughly with"),
        (0.6, "rough", "lines up roughly with"),
        (0.5999999, "loose", "lines up only loosely with"),
        (0.0, "loose", "lines up only loosely with"),
    ],
)
def test_agreement_phrase_boundaries(rho, bucket, phrase):
    assert agreement_bucket(rho) == bucket
    assert phrase in agreement_phrase(rho, "the exact test")


def test_group_sentence_only_above_declared_difference():
    quiet = story_agreement_methods(_synthetic(0.8, early=0.8, late=0.7))
    assert "blocks 0 to 5" not in _texts(quiet)
    loud = story_agreement_methods(_synthetic(0.8, early=0.9, late=0.6))
    assert "blocks 0 to 5" in _texts(loud)
    assert "blocks 6 to 11" in _texts(loud)
    reversed_ = story_agreement_methods(_synthetic(0.8, early=0.5, late=0.85))
    assert "blocks 6 to 11" in _texts(reversed_)


def test_story_claims_are_lint_clean_at_all_levels():
    claims = story_agreement_methods(_synthetic(0.75))
    assert {c["level"] for c in claims} == {"curious", "student", "expert"}
    assert explain_lint.lint_claims(claims) == []


def test_agreement_result_shapes(agreement):
    res = agreement.results
    assert res["job_type"] == "agreement"
    assert agreement.config["job_type"] == "agreement"
    assert len(res["labels"]) == N_COMPONENTS
    assert len(res["scatter"]) == N_COMPONENTS
    for key in ("exact", "atp", "dla"):
        assert len(res["vectors"][key]) == N_COMPONENTS
        assert all(isinstance(v, float) for v in res["vectors"][key])
        assert len(res["top10"][key]) == 10
    for method in ("atp", "dla"):
        stats = res["stats"][method]
        assert stats["pearson_heads"] is not None
        assert stats["spearman_heads"] is not None
        assert 0 <= stats["top10_overlap"] <= 10
        assert set(stats["pearson_groups"]) == {"blocks_0_5", "blocks_6_11"}
    assert res["dla_reconciliation"]["ok"] is True
    assert res["provenance"]["git_commit"]
    print(
        "\nagreement: AtP r="
        f"{res['stats']['atp']['pearson_heads']:.3f} top10={res['stats']['atp']['top10_overlap']}, "
        f"DLA r={res['stats']['dla']['pearson_heads']:.3f} top10={res['stats']['dla']['top10_overlap']}"
    )


def test_consistency_gate_against_stage2_head_sweep(agreement):
    class _Ctx:
        cancel_flag = None

        def report(self, progress, message):
            pass

        def check_cancel(self):
            pass

    sweep = run_sweep(
        {"task_id": "ioi_john_mary", "pair_id": None, "mode": "layer_head"}, _Ctx()
    )
    gate = exact_head_vs_sweep(agreement.results["vectors"]["exact"], sweep)
    assert gate["compared"] == N_HEADS
    assert gate["ok"] is True, gate
    assert gate["max_abs_diff"] <= 1e-4


def test_compact_and_story_for_real_result(agreement):
    compact = compact_agreement(agreement.results)
    assert len(compact["exact_heads"]) == N_HEADS
    claims = story_agreement_methods(compact)
    assert explain_lint.lint_claims(claims) == []
    text = _texts(claims)
    assert "lines up" in text
    assert "top 10" in text.lower() or "top-10" in text.lower()


def test_report_section_has_live_consistency_check(report_job):
    sections = {s["id"]: s for s in report_job.results["sections"]}
    section = sections["methods_agree"]
    assert section["analogy"] is None
    assert section["data"]["scatter"] and section["data"]["stats"]
    checks = {c["id"]: c for c in section["trust"]["checks"]}
    assert "agreement_exact_vs_head_sweep" in checks
    assert checks["agreement_exact_vs_head_sweep"]["ok"] is True
    assert checks["agreement_exact_vs_head_sweep"]["value"] <= 1e-4
    assert checks["agreement_dla_reconciliation"]["ok"] is True
    assert section["trust"]["not_shown"]
    assert explain_lint.lint_claims(section["claims"]) == []
