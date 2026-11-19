"""Stage 4 Task 7: the "report" job end to end (real model, temp DB), export
markdown, study-backed section with a fixture study, /api/explain, evidence
re-derivation from stored section data.

No test touches the real outputs/experiments.db (conftest isolates it).
"""

from __future__ import annotations

import re
import time

import pytest
from fastapi.testclient import TestClient

from backend.app.experiments.jobs import RUNNER, JobRecord
from backend.app.experiments.sweeps import MIN_BASELINE_DIFF
from backend.app.main import create_app
from backend.app.storage import db

REPORT_SECONDS = 300.0


@pytest.fixture(scope="module")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def _wait_terminal(job_id: str, timeout: float = REPORT_SECONDS) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = RUNNER.get(job_id)
        assert job is not None
        if job.status in ("complete", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach a terminal state in {timeout}s")


def _resolve(data, path):
    """Resolve an evidence path like 'sweep.recovery[9][6]' inside section data."""
    key, _, rest = path.partition(".")
    node = data[key]
    if rest:
        for tok in re.findall(r"\[[^\]]+\]|[^.\[]+", rest):
            if tok.startswith("["):
                node = node[int(tok[1:-1])]
            else:
                node = node[tok]
    return node


@pytest.fixture(scope="module")
def report_job(bridge):
    job = RUNNER.submit({"task_id": "ioi_john_mary"}, job_type="report")
    progress_seen: list[float] = []
    deadline = time.time() + REPORT_SECONDS
    last = -1.0
    while time.time() < deadline:
        j = RUNNER.get(job.job_id)
        progress_seen.append(j.progress)
        assert j.progress >= last - 1e-9, "progress decreased"
        last = j.progress
        if j.status in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.05)
    return j


def test_report_completes_with_all_sections(report_job):
    assert report_job.status == "complete", report_job.error
    assert report_job.progress == pytest.approx(1.0)
    res = report_job.results
    assert res["job_type"] == "report"
    ids = [s["id"] for s in res["sections"]]
    assert ids == [
        "answer",
        "where_formed",
        "which_words",
        "which_parts",
        "methods_agree",
        "vs_paper",
        "how_sure",
        "limits",
    ]
    by_id = {s["id"]: s for s in res["sections"]}
    # no study attached: how_sure must say so, not invent numbers
    assert any("Not run yet" in c["text"] for c in by_id["how_sure"]["claims"] if c["level"] == "curious")
    # each section carries question, evidence tier, trust
    for s in res["sections"]:
        assert s["title"] and s["question"] and s["evidence_tier"] in (
            "single prompt, live", "multi-prompt study, saved",
        )
        assert set(s["trust"]) == {"measured", "against", "checks", "warnings", "not_shown"}
    # claims: lint-clean by construction; verify independently
    from backend.app.explain import lint as explain_lint

    all_claims = [c for s in res["sections"] for c in s["claims"]]
    assert explain_lint.lint_claims(all_claims) == []
    curious = " ".join(c["text"] for c in all_claims if c["level"] == "curious")
    stats = explain_lint.readability(curious)
    assert stats["mean_sentence_words"] <= 18 and stats["max_sentence_words"] <= 28
    assert stats["flesch_reading_ease"] >= 60
    print(f"\nreport curious readability: {stats}")
    assert res["generated_at"] and res["provenance"]["git_commit"]
    # honest disagreement section present (single-prompt, live tier for vs_paper)
    assert by_id["vs_paper"]["evidence_tier"] == "single prompt, live"


def test_report_every_evidence_path_resolves(report_job):
    res = report_job.results
    for s in res["sections"]:
        for c in s["claims"]:
            for ev in c["evidence"]:
                if ev.startswith("fixed:") or ev.startswith("procedure:"):
                    continue
                _resolve(s["data"], ev)  # raises if unresolvable


def test_report_evidence_rederivation(report_job):
    """Re-derive >= 5 cited numbers directly from sections[].data and compare
    with the numbers in the claim text after formatting."""
    res = report_job.results
    by_id = {s["id"]: s for s in res["sections"]}
    from backend.app.explain.lint import strip_markers
    from backend.app.explain.thresholds import fmt_num, fmt_pct

    claims = {}
    for s in res["sections"]:
        for c in s["claims"]:
            if c["level"] == "curious":
                claims.setdefault(c["id"], (c, s))

    checked = 0

    def check(cid, formatted, frag=None):
        nonlocal checked
        c, s = claims[cid]
        text = strip_markers(c["text"])
        assert formatted in text, f"{cid}: {formatted!r} not in {text!r}"
        assert frag is None or frag in text
        checked += 1

    pred = by_id["answer"]["data"]["prediction"]
    check("answer.top_token", fmt_pct(pred["top_prob"]))
    check("answer.gap", fmt_num(pred["D"], 2))
    lens = by_id["where_formed"]["data"]["lens"]
    check("lens.biggest_jump", fmt_num(lens["max_jump"]["D_after"], 2))
    words = by_id["which_words"]["claims"]
    early = next(c for c in words if c["level"] == "curious" and c["id"] == "words.early")
    ev = next(e for e in early["evidence"] if e.startswith("sweep.recovery["))
    idx = ev[len("sweep.recovery["):-1].split("][")
    v = by_id["which_words"]["data"]["sweep"]["recovery"][int(idx[0])][int(idx[1])]
    check("words.early", fmt_num(v, 2))
    parts = by_id["which_parts"]["claims"]
    rec1 = next(c for c in parts if c["level"] == "curious" and c["id"] == "parts.recovery_top.1")
    ev = next(e for e in rec1["evidence"] if e.startswith("sweep.recovery["))
    idx = ev[len("sweep.recovery["):-1].split("][")
    v = by_id["which_parts"]["data"]["sweep"]["recovery"][int(idx[0])][int(idx[1])]
    check("parts.recovery_top.1", fmt_num(v, 2))
    dla = by_id["which_parts"]["data"]["dla"]
    check("parts.dla_top.1", fmt_num(dla["top"][0]["value"], 2))
    check("parts.reconciliation", fmt_num(dla["reconciliation"]["abs_error"], 4))
    # trust checks computed live from stored data
    checks = {c["id"]: c for s in res["sections"] for c in s["trust"]["checks"]}
    assert checks["final_cell_recovery"]["ok"] is True
    assert checks["last_block_other_positions"]["ok"] is True
    assert checks["dla_reconciliation"]["ok"] is True
    assert checks["baseline_contrast"]["ok"] is True
    assert checks["baseline_contrast"]["value"] >= MIN_BASELINE_DIFF
    assert checked >= 5


def test_report_persisted_with_job_type_and_history(client, report_job):
    stored = db.get_experiment(report_job.job_id)
    assert stored is not None and stored["status"] == "complete"
    assert stored["config"]["job_type"] == "report"
    r = client.get("/api/experiments")
    assert report_job.job_id in [s["id"] for s in r.json()["saved"]]


def test_export_markdown(client, report_job):
    r = client.get(f"/api/report/{report_job.job_id}/export?fmt=md&level=curious")
    assert r.status_code == 200
    md = r.text
    assert md.startswith("# Circuit Report")
    assert "## Provenance" in md
    assert "| measure | value |" in md
    from backend.app.explain import lint as explain_lint

    assert explain_lint.lint_markdown(md, "curious") == []
    # other levels exported too; unknown level is a 422
    assert client.get(f"/api/report/{report_job.job_id}/export?fmt=md&level=expert").status_code == 200
    assert client.get(f"/api/report/{report_job.job_id}/export?fmt=md&level=nope").status_code == 422
    assert client.get(f"/api/report/{report_job.job_id}/export?fmt=json").status_code == 400


def test_export_404_for_non_report_jobs(client):
    r = client.post("/api/experiments", json={
        "clean_text": "John and Mary went to the shop. John gave a book to",
        "corrupted_text": "John and Mary went to the shop. Mary gave a book to",
        "hook": "blocks.5.hook_out", "positions": [9],
        "candidate_a": " Mary", "candidate_b": " John",
    })
    exp_id = r.json()["job_id"]
    assert client.get(f"/api/report/{exp_id}/export?fmt=md").status_code == 404
    assert client.get("/api/report/deadbeef/export?fmt=md").status_code == 404
    RUNNER.cancel(exp_id)
    _wait_terminal(exp_id)


def test_study_job_id_validation_422(client):
    # unknown id
    r = client.post("/api/report", json={"task_id": "ioi_john_mary", "study_job_id": "nope123"})
    assert r.status_code == 422
    # not a study job: run a fast intervention job and reference it
    r = client.post("/api/experiments", json={
        "clean_text": "John and Mary went to the shop. John gave a book to",
        "corrupted_text": "John and Mary went to the shop. Mary gave a book to",
        "hook": "blocks.5.hook_out", "positions": [9],
        "candidate_a": " Mary", "candidate_b": " John",
    })
    exp_id = r.json()["job_id"]
    _wait_terminal(exp_id)
    assert RUNNER.get(exp_id).status == "complete"
    r = client.post("/api/report", json={"task_id": "ioi_john_mary", "study_job_id": exp_id})
    assert r.status_code == 422
    assert "not a study" in r.json()["detail"]
    # incomplete study: submit a long one and reference it while running
    study_id = None
    try:
        r = client.post("/api/study", json={"n_prompts": 40, "seed": 3})
        study_id = r.json()["job_id"]
        deadline = time.time() + 30
        while time.time() < deadline and RUNNER.get(study_id).status == "queued":
            time.sleep(0.1)
        r = client.post("/api/report", json={"task_id": "ioi_john_mary", "study_job_id": study_id})
        assert r.status_code == 422
        assert "not complete" in r.json()["detail"]
    finally:
        if study_id:
            RUNNER.cancel(study_id)
            _wait_terminal(study_id)


def test_report_cancelled(client):
    job = RUNNER.submit({"task_id": "ioi_john_mary"}, job_type="report")
    deadline = time.time() + 30
    while time.time() < deadline and RUNNER.get(job.job_id).status == "queued":
        time.sleep(0.05)
    assert RUNNER.cancel(job.job_id) is True
    j = _wait_terminal(job.job_id)
    assert j.status == "cancelled"
    assert j.results is None


def _write_study_fixture(study_id: str) -> None:
    """A small synthetic COMPLETED study in the temp DB (never a real run)."""
    import random

    rnd = random.Random(42)
    base = [round(rnd.uniform(-1, 1), 6) for _ in range(156)]
    atp = [round(x * 0.98 + 0.01, 6) for x in base]
    dla = [round(x * 0.3, 6) for x in base]
    per_prompt = []
    for i in range(6):
        row = {
            "id": f"p{i}", "template_id": 1, "s2_role": "S2_is_IO",
            "D_clean": 3.0, "P_clean": 0.8,
            "baselines": {
                "swap": {"D_corr": -2.0, "P_corr": 0.1, "abs_D_diff": 5.0, "excluded": False},
                "third": {"D_corr": -1.0, "P_corr": 0.2, "abs_D_diff": 4.0, "excluded": i == 0},
                "resample": {"D_corr": -1.5, "P_corr": 0.15, "abs_D_diff": 4.5, "excluded": i < 2},
            },
            "rec_d": {"swap": list(base), "third": list(base), "resample": list(base)},
            "rec_p": {"swap": list(base), "third": list(base), "resample": list(base)},
            "atp": list(atp), "dla": list(dla),
            "abl_zero": list(base), "abl_mean": list(base),
        }
        per_prompt.append(row)
    rho_ci = {"mean": 0.9, "ci_low": 0.85, "ci_high": 0.94, "n": 6, "n_invalid": 0}
    results = {
        "job_type": "study",
        "params": {"n_prompts": 6, "seed": 0},
        "exclusion_rule": "|D_clean - D_corr| < 0.5 per (prompt, corruption) pair",
        "model_accuracy": 1.0,
        "exclusions": {"swap": 0, "third": 1, "resample": 2, "abl_zero": 0, "abl_mean": 0},
        "per_prompt": per_prompt,
        "summary": {
            "rq1": {
                "all_prompts": {
                    "methods": {
                        "atp": {"heads": {"signed": rho_ci}},
                        "dla": {"heads": {"signed": {"mean": 0.2, "ci_low": 0.1, "ci_high": 0.3, "n": 6, "n_invalid": 0}}},
                    },
                    "beats_control": {"atp": {"heads": True}, "dla": {"heads": False}},
                }
            },
            "rq2": {
                "condition_names": ["swap_D", "swap_P", "third_D", "third_P",
                                    "resample_D", "resample_P", "abl_zero_D", "abl_mean_D"],
                "topk": {"k10": {"matrix": [[10 if a == b else 7 for b in range(8)] for a in range(8)]}},
            },
        },
    }
    db.save_experiment({"n_prompts": 6, "seed": 0, "job_type": "study"}, results, "complete", exp_id=study_id)


@pytest.fixture(scope="module")
def study_report_job(bridge):
    study_id = "fixtstudy01"
    _write_study_fixture(study_id)
    job = RUNNER.submit({"task_id": "ioi_john_mary", "study_job_id": study_id}, job_type="report")
    j = _wait_terminal(job.job_id)
    return j


def test_report_with_study_shows_how_sure_and_overlap(study_report_job):
    assert study_report_job.status == "complete", study_report_job.error
    res = study_report_job.results
    how = next(s for s in res["sections"] if s["id"] == "how_sure")
    assert how["evidence_tier"] == "multi-prompt study, saved"
    curious = {c["id"]: c["text"] for c in how["claims"] if c["level"] == "curious"}
    assert "study.not_run" not in curious
    assert "6" in curious["study.size"]
    assert "100.0%" in curious["study.accuracy"]
    assert "3" in curious["study.exclusions"]
    # pre-registered Spearman numbers AND the recomputed top-10 overlap, labelled extra check
    assert "0.900" in curious["study.atp"]
    assert "Extra check" in curious["study.overlap_atp"]
    # the fixture's AtP vector is an order-preserving rescale of the exact
    # vector, so the recomputed top-10 overlap is exactly 10 of 10
    assert "10 of 10" in curious["study.overlap_atp"]
    assert "7 of 10" in curious["study.stable"]  # stored rq2 k10 matrix row
    # exclusions live check passes
    checks = {c["id"]: c for c in how["trust"]["checks"]}
    assert checks["study_exclusions_recount"]["ok"] is True
    from backend.app.explain import lint as explain_lint

    assert explain_lint.lint_claims(how["claims"]) == []


def test_explain_endpoint_on_completed_sweep(client, bridge):
    job = RUNNER.submit({"task_id": "ioi_john_mary", "mode": "layer_position"}, job_type="sweep")
    j = _wait_terminal(job.job_id, timeout=120)
    assert j.status == "complete", j.error
    r = client.post("/api/explain", json={"job_id": j.job_id, "level": "curious"})
    assert r.status_code == 200
    body = r.json()
    assert body["job_type"] == "sweep" and body["level"] == "curious"
    assert body["claims"] and all(c["level"] == "curious" for c in body["claims"])
    assert body["lint"]["ok"] is True
    # invalid level -> 422 (pydantic pattern)
    assert client.post("/api/explain", json={"job_id": j.job_id, "level": "poetic"}).status_code == 422
    # unknown job -> 404
    assert client.post("/api/explain", json={"job_id": "deadbeef", "level": "curious"}).status_code == 404


def test_explain_endpoint_on_study_fixture(client):
    r = client.post("/api/explain", json={"job_id": "fixtstudy01", "level": "curious"})
    assert r.status_code == 200
    claims = r.json()["claims"]
    assert any("share 10 of 10" in c["text"] or "shares 10 of 10" in c["text"] for c in claims)


def test_explain_rejects_non_sweep_study_jobs(client, report_job):
    r = client.post("/api/explain", json={"job_id": report_job.job_id, "level": "curious"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Stage 4 review regressions (found by reading the generated report cold)
# ---------------------------------------------------------------------------


def test_which_words_table_cells_match_claims_and_exclude_identity(report_job):
    """The stored/table 'late' cell must be the same non-trivial cell the claim
    describes (the last-block, last-word identity cell always equals 1.0)."""
    sec = next(s for s in report_job.results["sections"] if s["id"] == "which_words")
    sweep = sec["data"]["sweep"]
    last_l = len(sweep["recovery"]) - 1
    last_p = len(sweep["recovery"][last_l]) - 1
    for key in ("early", "late"):
        cell = sec["data"][key]
        assert (cell["layer"], cell["position"]) != (last_l, last_p)
    late = sec["data"]["late"]
    curious = {c["id"]: c["text"] for c in sec["claims"] if c["level"] == "curious"}
    assert f"{late['recovery']:.2f}" in curious["words.late"]


def test_vs_paper_top10_rows_carry_groups(report_job):
    sec = next(s for s in report_job.results["sections"] if s["id"] == "vs_paper")
    assert all("groups" in e for e in sec["data"]["top10"])
    # at least one of our top-10 heads is a known head on the default pair
    assert any(e["groups"] for e in sec["data"]["top10"])
    curious = {c["id"]: c["text"] for c in sec["claims"] if c["level"] == "curious"}
    assert "random list of 10 would hold about" in curious["paper.overall"]


def test_study_sensitivity_claim_reports_metric_and_ablation_overlap(study_report_job):
    how = next(s for s in study_report_job.results["sections"] if s["id"] == "how_sure")
    curious = {c["id"]: c["text"] for c in how["claims"] if c["level"] == "curious"}
    assert "study.sensitivity" in curious
    # fixture matrix: off-diagonal overlaps are 7 of 10
    assert "7 of 10" in curious["study.sensitivity"]
    assert "does not stay the same" in curious["study.sensitivity"]
