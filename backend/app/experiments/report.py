"""The "report" job type — the Circuit Report.

Pipeline (progress reported per step): prediction (clean prompt, candidates)
-> logit lens -> layer x position sweep -> layer x head sweep -> DLA (clean)
-> optional study load (must exist, be complete, have job_type == "study";
validated at submit time -> 422) -> story generation -> claim lint.

Every sentence is a template from app.explain filled with measured values;
``lint_claims`` runs at the end — a lint failure fails the job honestly
(experiment error), it never ships bad text.

Results contain sections (id, title, question, claims, compact data,
evidence_tier, trust), provenance and generated_at (provenance timestamp only;
templates never read it).
"""

from __future__ import annotations

import json
import pathlib

from ..analysis.agreement import compute_agreement, exact_head_vs_sweep
from ..analysis.dla import compute_dla
from ..analysis.lens import compute_logit_lens
from ..explain import lint as explain_lint
from ..explain import story as explain_story
from ..explain.glossary import analogy_by_id
from ..explain.templates import LEVELS
from ..explain.thresholds import (
    MIN_BASELINE_CONTRAST,
    TOL_FINAL_RECOVERY,
    TOL_LASTBLOCK_UNCHANGED,
    fmt_num,
)
from ..model_runtime import engine
from ..model_runtime.loader import ADAPTER
from ..provenance import collect as collect_provenance
from ..storage import db
from ..tasks import registry as task_registry
from .jobs import RUNNER, ExperimentValidationError, JobContext, register_job_type
from .sweeps import DIRECTION, check_baselines, run_sweep

REFERENCE_HEADS_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "examples" / "reference" / "ioi_wang2022_heads.json"
)


def _load_reference() -> dict | None:
    try:
        with open(REFERENCE_HEADS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def validate_report_config(cfg: dict) -> dict:
    """Validate a report spec against the task registry (+ optional study job).
    Returns the normalized config persisted with the job."""
    task_id = cfg.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ExperimentValidationError("task_id is required")
    task = task_registry.get_task(task_id)
    if task is None:
        raise ExperimentValidationError(f"unknown task: {task_id!r}")
    validation = task_registry.validate_task(task)
    if not validation.get("ok"):
        raise ExperimentValidationError(
            f"task {task_id} failed registry validation: {'; '.join(validation.get('errors', []))}"
        )
    pair_id = cfg.get("pair_id")
    pair = task.pairs[0] if pair_id is None else next((p for p in task.pairs if p.id == pair_id), None)
    if pair is None:
        raise ExperimentValidationError(f"unknown pair {pair_id!r} for task {task_id!r}")

    study_job_id = cfg.get("study_job_id")
    if study_job_id is not None:
        if not isinstance(study_job_id, str) or not study_job_id:
            raise ExperimentValidationError("study_job_id must be a non-empty string or null")
        job = RUNNER.get(study_job_id)
        stored = db.get_experiment(study_job_id)
        if job is None and stored is None:
            raise ExperimentValidationError(f"unknown study_job_id: {study_job_id!r}")
        if job is not None and job.type != "study":
            raise ExperimentValidationError(f"job {study_job_id} is a {job.type!r} job, not a study")
        if stored is not None and (stored.get("config") or {}).get("job_type") not in (None, "study"):
            raise ExperimentValidationError(
                f"job {study_job_id} is a {(stored.get('config') or {}).get('job_type')!r} job, not a study"
            )
        status = job.status if job is not None else (stored or {}).get("status")
        if status != "complete":
            raise ExperimentValidationError(
                f"study job {study_job_id} is not complete (status: {status})"
            )
        results = (job.results if job is not None and job.results else None) or (
            (stored or {}).get("results") if stored else None
        )
        if not results:
            raise ExperimentValidationError(f"study job {study_job_id} has no stored results")

    return {
        "task_id": task_id,
        "pair_id": pair.id,
        "study_job_id": study_job_id,
        "clean_text": pair.clean_text,
        "corrupted_text": pair.corrupted_text,
        "candidate_a": pair.candidate_a,
        "candidate_b": pair.candidate_b,
        "job_type": "report",  # persisted so history can tell reports apart
    }


class _PhaseCtx:
    """Rescales a sub-run's progress into a slice of the report's progress
    (JobContext.report is monotonic-clamped, so nested sweeps must not fight
    over the absolute scale)."""

    def __init__(self, inner: JobContext, base: float, span: float) -> None:
        self._inner = inner
        self._base = base
        self._span = span

    def report(self, progress: float, message: str) -> None:
        p = max(0.0, min(1.0, float(progress)))
        self._inner.report(self._base + p * self._span, message)

    def check_cancel(self) -> None:
        self._inner.check_cancel()


SECTION_TITLES = {
    "answer": "What did the model answer?",
    "where_formed": "Where does the answer form?",
    "which_words": "Which words matter?",
    "which_parts": "Which parts of the model matter?",
    "methods_agree": "Do the three methods agree?",
    "vs_paper": "How does this compare with the published circuit?",
    "how_sure": "How sure can we be?",
    "limits": "What this does not show",
}

SECTION_QUESTIONS = {
    "answer": "Which next token does the model pick on the clean prompt, and by how much?",
    "where_formed": "After which block does the score gap grow the most?",
    "which_words": "Which input words restore the answer when patched, early vs late in the model?",
    "which_parts": "Which attention heads restore the answer (patching), and which push it directly (DLA)?",
    "methods_agree": "Do exact patching, attribution patching and direct logit attribution rank the same components?",
    "vs_paper": "How do our top heads line up with the published IOI circuit groups?",
    "how_sure": "What do many prompts say beyond this one example?",
    "limits": "What was measured, and what was deliberately not claimed?",
}

SECTION_ANALOGY = {
    "where_formed": "shared_whiteboard",
    "which_words": "note_swap",
    "which_parts": "spotlight",
    "how_sure": "covered_answer",
}


def _check(cid: str, label: str, value, tolerance, ok: bool, comparison: str) -> dict:
    return {
        "id": cid,
        "label": label,
        "value": value,
        "tolerance": tolerance,
        "ok": bool(ok),
        "comparison": comparison,
    }


def _baseline_contrast_check(sweep: dict) -> dict:
    b = sweep["baselines"]
    v = abs(b["denominator"])
    return _check(
        "baseline_contrast",
        "|D_clean − D_corrupted| must exceed the rejection gate",
        round(v, 6),
        MIN_BASELINE_CONTRAST,
        v >= MIN_BASELINE_CONTRAST,
        ">= gate (else the sweep is rejected)",
    )


def _sweep_invariant_checks(sweep: dict) -> list[dict]:
    """Live sweep invariants (computed from the stored grid, never hardcoded):
    final-block final-position recovery == 1 within 1e-3; non-final positions
    in the last block unchanged within 1e-4."""
    rec = sweep["recovery"]
    d_patched = sweep["d_patched"]
    d_corr = sweep["baselines"]["D_corrupted"]
    checks = []
    last_row, last_col = len(rec) - 1, len(rec[-1]) - 1
    v_final = rec[last_row][last_col]
    checks.append(
        _check(
            "final_cell_recovery",
            "final-block final-position recovery equals 1",
            round(v_final, 6),
            TOL_FINAL_RECOVERY,
            abs(v_final - 1.0) <= TOL_FINAL_RECOVERY,
            "|value − 1| <= tolerance",
        )
    )
    max_drift = 0.0
    for p in range(last_col):
        max_drift = max(max_drift, abs(d_patched[last_row][p] - d_corr))
    checks.append(
        _check(
            "last_block_other_positions",
            "patching non-final positions of the last block leaves D unchanged",
            round(max_drift, 8),
            TOL_LASTBLOCK_UNCHANGED,
            max_drift <= TOL_LASTBLOCK_UNCHANGED,
            "max drift <= tolerance",
        )
    )
    return checks


def _load_study(cfg: dict) -> dict | None:
    if not cfg.get("study_job_id"):
        return None
    stored = db.get_experiment(cfg["study_job_id"])
    if stored is None or stored.get("status") != "complete":
        raise ExperimentValidationError(
            f"study job {cfg['study_job_id']} is no longer complete at report time"
        )
    results = stored.get("results") or {}
    if (stored.get("config") or {}).get("job_type") != "study" or results.get("job_type") != "study":
        raise ExperimentValidationError(f"job {cfg['study_job_id']} is not a study")
    return results


def build_sections(cfg: dict, steps: dict) -> list[dict]:
    """Assemble the report sections from the compact step data (also used by
    POST /api/explain, which serves a subset of the same story functions)."""
    pred, lens = steps["prediction"], steps["lens"]
    pos_sweep, head_sweep, dla = steps["position_sweep"], steps["head_sweep"], steps["dla"]
    reference = steps["reference"]
    study = steps["study"]

    live_tier = "single prompt, live"
    study_tier = "multi-prompt study, saved"

    sections: list[dict] = []

    # ---- answer -------------------------------------------------------------
    answer_data = {"prediction": pred}
    answer_claims = explain_story.story_prediction(pred)
    answer_checks = []
    if pred.get("D") is not None and lens.get("final_d") is not None:
        diff = abs(pred["D"] - lens["final_d"])
        answer_checks.append(
            _check(
                "prediction_matches_lens_final",
                "the prediction's score gap matches the lens final stage",
                round(diff, 8),
                1e-3,
                diff <= 1e-3,
                "|difference| <= tolerance",
            )
        )
    sections.append(
        {
            "id": "answer",
            "title": SECTION_TITLES["answer"],
            "question": SECTION_QUESTIONS["answer"],
            "claims": answer_claims,
            "data": answer_data,
            "evidence_tier": live_tier,
            "analogy": None,
            "trust": {
                "measured": "One forward pass on the clean prompt: the top next token and the two candidate probabilities are read at the final position.",
                "against": [
                    {"label": "D (score gap, clean)", "value": pred.get("D")},
                    {
                        "label": "P(candidate A)",
                        "value": pred.get("prob_a"),
                    },
                ],
                "checks": answer_checks,
                "warnings": [],
                "not_shown": [
                    "One prompt shows what the model did here, not what it does in general.",
                    "A high probability is a measurement of this run, not a claim about the model's inner life.",
                ],
            },
        }
    )

    # ---- where_formed (logit lens) -------------------------------------------
    lens_data = {"lens": lens}
    lens_checks = []
    fc = lens.get("final_check") or {}
    if fc.get("max_abs_logit_diff_candidates") is not None:
        lens_checks.append(
            _check(
                "lens_final_identity",
                "the lens final stage reproduces the model's real logits",
                round(fc["max_abs_logit_diff_candidates"], 8),
                fc.get("tolerance"),
                bool(fc.get("ok")),
                "candidate-logit diff <= tolerance (and top-1 agrees)",
            )
        )
    sections.append(
        {
            "id": "where_formed",
            "title": SECTION_TITLES["where_formed"],
            "question": SECTION_QUESTIONS["where_formed"],
            "claims": explain_story.story_lens(lens),
            "data": lens_data,
            "evidence_tier": live_tier,
            "analogy": analogy_by_id(SECTION_ANALOGY["where_formed"]),
            "trust": {
                "measured": "The logit lens applies the final LayerNorm and unembedding to the residual stream after each block, reading the score gap per stage.",
                "against": [
                    {"label": "model's real D (final logits)", "value": pred.get("D")},
                    {"label": "lens final-stage D", "value": lens.get("final_d")},
                ],
                "checks": lens_checks,
                "warnings": [],
                "not_shown": [
                    "The lens is a heuristic view: early-stage readouts are not trained, so they are not the model's intermediate decisions.",
                    "A growing score gap across blocks is a trace of the readout, not a causal claim about any block.",
                ],
            },
        }
    )

    # ---- which_words (layer x position sweep) ---------------------------------
    _rec = pos_sweep.get("recovery") or []
    _identity = {(len(_rec) - 1, len(_rec[-1]) - 1)} if _rec and _rec[-1] else set()
    early = explain_story._best_cell(pos_sweep, explain_story.EARLY_BLOCKS, exclude=_identity) or {}
    late = explain_story._best_cell(pos_sweep, explain_story.LATE_BLOCKS, exclude=_identity) or {}
    words_data = {"sweep": pos_sweep, "early": early, "late": late}
    pos_checks = _sweep_invariant_checks(pos_sweep) + [_baseline_contrast_check(pos_sweep)]
    sections.append(
        {
            "id": "which_words",
            "title": SECTION_TITLES["which_words"],
            "question": SECTION_QUESTIONS["which_words"],
            "claims": explain_story.story_sweep_position({"sweep": pos_sweep}),
            "data": words_data,
            "evidence_tier": live_tier,
            "analogy": analogy_by_id(SECTION_ANALOGY["which_words"]),
            "trust": {
                "measured": "For every (block, word) cell the clean block-output activation is copied into the corrupted run (denoising), and Recovery is recorded.",
                "against": [
                    {"label": "D_clean", "value": pos_sweep["baselines"]["D_clean"]},
                    {"label": "D_corrupted", "value": pos_sweep["baselines"]["D_corrupted"]},
                    {"label": "denominator (contrast)", "value": pos_sweep["baselines"]["denominator"]},
                ],
                "checks": pos_checks,
                "warnings": list(pos_sweep["baselines"].get("warnings") or []),
                "not_shown": [
                    "Recovery from a full-residual patch does not say which part inside the block carried the effect.",
                    "One strong cell does not make the word 'the' location of the behaviour; other cells contribute too.",
                ],
            },
        }
    )

    # ---- which_parts (head sweep + DLA) ----------------------------------------
    head_top = explain_story.top10_by_recovery(head_sweep)[:3] if head_sweep.get("recovery") else []
    dla_compact = dla
    parts_data = {
        "sweep": head_sweep,
        "dla": dla_compact,
        "top_recovery": head_top,
        "top_dla": (dla_compact or {}).get("top", [])[:3],
    }
    parts_checks = [_baseline_contrast_check(head_sweep)]
    rec_ = (dla_compact or {}).get("reconciliation") or {}
    if rec_.get("abs_error") is not None:
        parts_checks.append(
            _check(
                "dla_reconciliation",
                "DLA terms sum to the model's real D",
                round(rec_["abs_error"], 8),
                rec_.get("tolerance"),
                bool(rec_.get("ok")),
                "abs error <= tolerance",
            )
        )
    sections.append(
        {
            "id": "which_parts",
            "title": SECTION_TITLES["which_parts"],
            "question": SECTION_QUESTIONS["which_parts"],
            "claims": explain_story.story_sweep_heads({"sweep": head_sweep, "dla": dla_compact}),
            "data": parts_data,
            "evidence_tier": live_tier,
            "analogy": analogy_by_id(SECTION_ANALOGY["which_parts"]),
            "trust": {
                "measured": "Two tests per head: a denoising patch of the whole head output (Recovery), and direct logit attribution (the direct push).",
                "against": [
                    {"label": "D_clean (head sweep)", "value": head_sweep["baselines"]["D_clean"]},
                    {"label": "D_corrupted (head sweep)", "value": head_sweep["baselines"]["D_corrupted"]},
                    {"label": "model D (DLA)", "value": (dla_compact or {}).get("D")},
                ],
                "checks": parts_checks,
                "warnings": list(head_sweep["baselines"].get("warnings") or []),
                "not_shown": [
                    "The patching test shows sufficiency (enough to restore), not necessity (needed).",
                    "DLA ignores effects through later blocks; it is a partial view of each head's role.",
                    "A high rank on one test does not imply a high rank on the other; they answer different questions.",
                ],
            },
        }
    )

    # ---- methods_agree (exact vs AtP vs DLA) ------------------------------------
    agreement = steps.get("agreement")
    agreement_compact = explain_story.compact_agreement(agreement) if agreement else None
    agree_data = agreement_compact or {}
    agree_checks = []
    if agreement_compact:
        gate = exact_head_vs_sweep(agreement_compact.get("exact_heads") or [], head_sweep)
        agree_checks.append(
            _check(
                "agreement_exact_vs_head_sweep",
                "exact agreement values equal the report's layer x head sweep",
                gate.get("max_abs_diff"),
                gate.get("tolerance"),
                bool(gate.get("ok")),
                "max |difference| <= tolerance for all 144 heads",
            )
        )
        rec_a = agreement_compact.get("dla_reconciliation") or {}
        if rec_a.get("abs_error") is not None:
            agree_checks.append(
                _check(
                    "agreement_dla_reconciliation",
                    "the agreement run's DLA terms sum to its real D",
                    round(rec_a["abs_error"], 8),
                    rec_a.get("tolerance"),
                    bool(rec_a.get("ok")),
                    "abs error <= tolerance",
                )
            )
        if (agreement_compact.get("baselines") or {}).get("denominator") is not None:
            agree_checks.append(
                _baseline_contrast_check({"baselines": agreement_compact["baselines"]})
            )
    stats = (agreement_compact or {}).get("stats") or {}
    agree_against = [
        {
            "label": "D_clean (agreement run)",
            "value": (agreement_compact or {}).get("baselines", {}).get("D_clean"),
        },
        {
            "label": "D_corrupted (agreement run)",
            "value": (agreement_compact or {}).get("baselines", {}).get("D_corrupted"),
        },
        {
            "label": "exact vs AtP Pearson (heads)",
            "value": (stats.get("atp") or {}).get("pearson_heads"),
        },
        {
            "label": "exact vs DLA Pearson (heads)",
            "value": (stats.get("dla") or {}).get("pearson_heads"),
        },
    ]
    sections.append(
        {
            "id": "methods_agree",
            "title": SECTION_TITLES["methods_agree"],
            "question": SECTION_QUESTIONS["methods_agree"],
            "claims": explain_story.story_agreement_methods(agreement_compact),
            "data": agree_data,
            "evidence_tier": live_tier,
            "analogy": None,
            "trust": {
                "measured": "Three score vectors for all 156 components on this pair: exact denoising reruns, attribution patching (a first-order approximation), and direct logit attribution on the clean prompt.",
                "against": agree_against,
                "checks": agree_checks,
                "warnings": [],
                "not_shown": [
                    "Agreement between scores does not show the three methods measure one shared mechanism.",
                    "The comparison uses one prompt pair and the pair's own corruption.",
                    "DLA ignores effects that pass through later blocks; exact patching includes them.",
                ],
            },
        }
    )

    # ---- vs_paper ---------------------------------------------------------------
    top10 = explain_story.top10_by_recovery(head_sweep) if head_sweep.get("recovery") else []
    if reference and reference.get("groups"):
        _membership = {g["group"]: set(g.get("heads", [])) for g in reference["groups"]}
        for _e in top10:
            _e["groups"] = [gid for gid, hs in _membership.items() if _e["label"] in hs]
    paper_data = {
        "sweep": head_sweep,
        "reference": reference,
        "top10": top10,
    }
    ref_checks = []
    if reference:
        ref_checks.append(
            _check(
                "reference_loaded",
                "the known-heads reference file loaded with groups",
                len(reference.get("groups") or []),
                None,
                len(reference.get("groups") or []) >= 1,
                ">= 1 group",
            )
        )
    sections.append(
        {
            "id": "vs_paper",
            "title": SECTION_TITLES["vs_paper"],
            "question": SECTION_QUESTIONS["vs_paper"],
            "claims": explain_story.story_agreement_with_paper({"sweep": head_sweep, "reference": reference}),
            "data": paper_data,
            "evidence_tier": live_tier,
            "analogy": None,
            "trust": {
                "measured": "The top 10 heads by denoising recovery are counted against the published IOI group lists (Wang et al. 2022 reference file).",
                "against": [
                    {
                        "label": "reference groups",
                        "value": len(reference.get("groups") or []) if reference else 0,
                        "note": "examples/reference/ioi_wang2022_heads.json",
                    }
                ],
                "checks": ref_checks,
                "warnings": [],
                "not_shown": [
                    "Belonging to a group does not show the head plays the paper's role in this run.",
                    "The reference list is an anchor; group membership is a comparison, not a replication of the paper's method.",
                ],
            },
        }
    )

    # ---- how_sure (optional study) ----------------------------------------------
    study_summary = explain_story.compact_study(study) if study else None
    how_data = {"study": study_summary}
    how_checks = []
    how_against = []
    if study_summary:
        # live check: recount exclusions from per_prompt data and compare
        per = study.get("per_prompt") or []
        recounted = {
            kind: sum(1 for row in per if (row.get("baselines") or {}).get(kind, {}).get("excluded"))
            for kind in ("swap", "third", "resample")
        }
        stored_ex = study_summary.get("exclusions") or {}
        mismatch = sum(1 for k in recounted if int(stored_ex.get(k) or 0) != recounted[k])
        how_checks.append(
            _check(
                "study_exclusions_recount",
                "exclusion counts recomputed from stored per-prompt baselines match the stored summary",
                mismatch,
                0,
                mismatch == 0,
                "mismatch == 0",
            )
        )
        how_against = [
            {
                "label": "expected random top-10 overlap",
                "value": study_summary.get("expected_random_k10"),
                "note": "control: two random top-10 lists of 144 heads",
            },
            {
                "label": "excluded pairs (swap/third/resample)",
                "value": f"{stored_ex.get('swap', 0)}/{stored_ex.get('third', 0)}/{stored_ex.get('resample', 0)}",
            },
        ]
    sections.append(
        {
            "id": "how_sure",
            "title": SECTION_TITLES["how_sure"],
            "question": SECTION_QUESTIONS["how_sure"],
            "claims": explain_story.story_study(study_summary),
            "data": how_data,
            "evidence_tier": study_tier if study_summary else live_tier,
            "analogy": analogy_by_id(SECTION_ANALOGY["how_sure"]) if study_summary else None,
            "trust": {
                "measured": "The attached study ran the pre-registered analysis over a generated prompt set. Only the top-10 overlap is recomputed here, from the stored per-prompt data.",
                "against": how_against,
                "checks": how_checks,
                "warnings": [],
                "not_shown": [
                    "The study covers the generated IOI prompt set; it does not prove the circuit works the same on free text.",
                    "Bootstrap intervals describe sampling over prompts, not uncertainty about the model's internals.",
                ],
            },
        }
    )

    # ---- limits ---------------------------------------------------------------
    sections.append(
        {
            "id": "limits",
            "title": SECTION_TITLES["limits"],
            "question": SECTION_QUESTIONS["limits"],
            "claims": explain_story.story_limits(),
            "data": {},
            "evidence_tier": live_tier,
            "analogy": None,
            "trust": {
                "measured": "Fixed limitation statements attached to every report; they are caveats, not measurements.",
                "against": [],
                "checks": [],
                "warnings": [],
                "not_shown": [
                    "The limits section is fixed text; it contains no measured numbers by design.",
                ],
            },
        }
    )
    return sections


def run_report(cfg: dict, ctx: JobContext) -> dict:
    norm = validate_report_config(cfg)  # re-validated inside the job (cheap)
    ctx.report(0.02, "prediction (clean prompt, candidates)")
    bridge = ADAPTER.ensure_loaded()
    tok_a = ADAPTER.token_id_for_str(norm["candidate_a"])
    tok_b = ADAPTER.token_id_for_str(norm["candidate_b"])
    toks_clean = bridge.to_tokens(norm["clean_text"])
    logits_clean = bridge(toks_clean)
    top3 = engine.topk_predictions(logits_clean, 3)
    comparison = engine.compare_candidates(logits_clean, tok_a, tok_b)
    pred = explain_story.compact_prediction(
        norm["candidate_a"], norm["candidate_b"], top3, comparison
    )

    ctx.report(0.06, "logit lens")
    lens = explain_story.compact_lens(
        compute_logit_lens(norm["clean_text"], norm["candidate_a"], norm["candidate_b"], 5)
    )

    ctx.report(0.10, "layer x position sweep")
    pos_sweep = explain_story.compact_sweep(
        run_sweep(
            {
                "task_id": norm["task_id"],
                "pair_id": norm["pair_id"],
                "mode": "layer_position",
                "job_type": "sweep",
            },
            _PhaseCtx(ctx, 0.10, 0.35),
        )
    )

    ctx.report(0.45, "layer x head sweep")
    head_sweep = explain_story.compact_sweep(
        run_sweep(
            {
                "task_id": norm["task_id"],
                "pair_id": norm["pair_id"],
                "mode": "layer_head",
                "job_type": "sweep",
            },
            _PhaseCtx(ctx, 0.45, 0.35),
        )
    )

    ctx.report(0.80, "direct logit attribution")
    dla = explain_story.compact_dla(
        compute_dla(norm["clean_text"], norm["candidate_a"], norm["candidate_b"])
    )

    ctx.report(0.82, "agreement: exact vs attribution patching vs DLA")
    agreement = compute_agreement(
        norm["clean_text"],
        norm["corrupted_text"],
        norm["candidate_a"],
        norm["candidate_b"],
        progress=_PhaseCtx(ctx, 0.82, 0.10).report,
        check_cancel=ctx.check_cancel,
    )

    ctx.report(0.94, "loading study")
    study = _load_study(norm)
    reference = explain_story.compact_reference(_load_reference())

    ctx.report(0.96, "writing the report")
    steps = {
        "prediction": pred,
        "lens": lens,
        "position_sweep": pos_sweep,
        "head_sweep": head_sweep,
        "dla": dla,
        "agreement": agreement,
        "reference": reference,
        "study": study,
    }
    sections = build_sections(cfg, steps)

    # lint ALL claims at ALL levels; a failure fails the job honestly.
    all_claims = [c for s in sections for c in s["claims"]]
    problems = explain_lint.lint_claims(all_claims)
    curious_claims = [c["text"] for c in all_claims if c["level"] == "curious"]
    curious_stats = explain_lint.readability(" ".join(curious_claims))
    if problems:
        raise ExperimentValidationError(
            "report lint failed; refusing to ship generated text: " + "; ".join(problems[:10])
        )

    ctx.report(0.97, "lint ok — assembling results")
    prov = collect_provenance()
    return {
        "job_type": "report",
        "task_id": norm["task_id"],
        "pair_id": norm["pair_id"],
        "study_job_id": norm["study_job_id"],
        "direction": DIRECTION,
        "candidate_a": norm["candidate_a"],
        "candidate_b": norm["candidate_b"],
        "tokens_clean": bridge.to_str_tokens(norm["clean_text"]),
        "sections": sections,
        "lint": {"ok": True, "curious_readability": curious_stats},
        "provenance": prov,
        "generated_at": prov.get("timestamp_utc"),
    }


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def export_markdown(results: dict, level: str = "curious") -> str:
    """Render the stored report results as Markdown (claims at the requested
    level, key-number tables, trust panels, provenance footer)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level!r}")
    lines: list[str] = []
    lines.append("# Circuit Report")
    lines.append("")
    lines.append(
        f"Task: `{results['task_id']}` · pair: `{results['pair_id']}` · "
        f"direction: {results['direction']}"
    )
    lines.append(f"Generated: {results.get('generated_at')} (provenance timestamp only)")
    lines.append(f"Reading level: {level}")
    lines.append("")
    for s in results["sections"]:
        lines.append(f"## {s['title']}")
        lines.append("")
        lines.append(f"*Question: {s['question']}*")
        lines.append(f"*Evidence: {s['evidence_tier']}*")
        lines.append("")
        for c in s["claims"]:
            if c["level"] != level:
                continue
            lines.append(f"- [{c['claim_type']}] {explain_lint.strip_markers(c['text'])}")
        lines.append("")
        tbl = _section_table(s)
        if tbl:
            lines.extend(tbl)
            lines.append("")
        t = s.get("trust") or {}
        if t:
            lines.append("### Why trust this?")
            lines.append("")
            if t.get("measured"):
                lines.append(f"- What was measured: {t['measured']}")
            for a in t.get("against") or []:
                v = a.get("value")
                v_s = fmt_num(v, 4) if isinstance(v, (int, float)) else str(v)
                note = f" ({a['note']})" if a.get("note") else ""
                lines.append(f"- Against what: {a['label']} = {v_s}{note}")
            for c in t.get("checks") or []:
                tol = f" (tolerance {c['tolerance']})" if c.get("tolerance") is not None else ""
                status = "ok" if c["ok"] else "**FAILED**"
                lines.append(f"- Check: {c['label']}: {c['value']}{tol} — {status}")
            for w in t.get("warnings") or []:
                lines.append(f"- Warning: {w}")
            for n in t.get("not_shown") or []:
                lines.append(f"- Does NOT show: {n}")
            lines.append("")
        if s.get("analogy"):
            a = s["analogy"]
            lines.append(f"> Think of it like {a['title']} — an analogy, not a description of the model.")
            lines.append(">")
            lines.append(f"> {a['text']}")
            lines.append(">")
            lines.append(f"> Where the analogy breaks: {a['breaks_when']}")
            lines.append("")
    p = results.get("provenance") or {}
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- model revision: {p.get('model_revision')}")
    lines.append(f"- torch: {p.get('versions', {}).get('torch')} · transformer_lens: {p.get('versions', {}).get('transformer_lens')} · transformers: {p.get('versions', {}).get('transformers')}")
    lines.append(f"- git commit: {p.get('git_commit')}")
    lines.append(f"- timestamp (UTC): {p.get('timestamp_utc')}")
    lines.append("")
    return "\n".join(lines)


def _section_table(s: dict) -> list[str]:
    """A compact key-numbers table per section (from sections[].data only)."""
    sid, data = s["id"], s.get("data") or {}
    rows: list[tuple[str, str]] = []

    def add(label, v, digits=4):
        rows.append((label, fmt_num(v, digits) if isinstance(v, (int, float)) and v is not None else str(v)))

    if sid == "answer" and data.get("prediction"):
        p = data["prediction"]
        add("top token probability", p.get("top_prob"))
        add("P(candidate A)", p.get("prob_a"))
        add("P(candidate B)", p.get("prob_b"))
        add("score gap D", p.get("D"), 2)
    elif sid == "where_formed" and data.get("lens"):
        for st in data["lens"]["stages"]:
            add(st["stage"], st["D"], 2)
    elif sid == "which_words" and data.get("sweep"):
        b = data["sweep"]["baselines"]
        add("D_clean", b["D_clean"], 2)
        add("D_corrupted", b["D_corrupted"], 2)
        e, l = data.get("early") or {}, data.get("late") or {}
        if e:
            rows.append(("strongest early cell", f"block {e.get('layer')} · {e.get('token')} · recovery {fmt_num(e.get('recovery'), 2)}"))
        if l:
            rows.append(("strongest late cell (trivial last-block, last-word cell excluded)", f"block {l.get('layer')} · {l.get('token')} · recovery {fmt_num(l.get('recovery'), 2)}"))
    elif sid == "which_parts" and data.get("sweep"):
        for e in data.get("top_recovery") or []:
            rows.append((f"recovery rank {e['rank']}", f"head {e['label']} · {fmt_num(e['recovery'], 2)}"))
        for e in data.get("top_dla") or []:
            rows.append((f"DLA rank {e['rank']}", f"head {e['label']} · {fmt_num(e['value'], 2)}"))
    elif sid == "methods_agree" and data.get("stats"):
        a = data
        stats = a.get("stats") or {}
        for method, label in (("atp", "attribution patching"), ("dla", "direct logit attribution")):
            block = stats.get(method) or {}
            add(f"{label}: Pearson over 144 heads", block.get("pearson_heads"), 3)
            add(f"{label}: Spearman over 144 heads", block.get("spearman_heads"), 3)
            add(
                f"{label}: top-10 overlap with exact (random {fmt_num(block.get('expected_random_top10'), 1)})",
                block.get("top10_overlap"),
                0,
            )
            groups = block.get("pearson_groups") or {}
            for group, glabel in (("blocks_0_5", "blocks 0-5"), ("blocks_6_11", "blocks 6-11")):
                add(f"{label}: Pearson {glabel}", (groups.get(group) or {}).get("pearson"), 3)
        for i, entry in enumerate((a.get("top10") or {}).get("exact") or []):
            if i >= 3:
                break
            rows.append((f"exact top-10 rank {entry['rank']}", f"{entry['label']} · {fmt_num(entry['value'], 2)}"))
    elif sid == "vs_paper" and data.get("top10"):
        for e in data["top10"]:
            rows.append((f"top-10 rank {e['rank']}", f"head {e['label']} · recovery {fmt_num(e['recovery'], 2)} · groups: {', '.join(e.get('groups') or []) or 'none'}"))
    elif sid == "how_sure" and data.get("study"):
        st = data["study"]
        add("n prompts", st.get("n_prompts"), 0)
        add("model accuracy", st.get("model_accuracy"))
        ov = st.get("overlap") or {}
        add("top-10 overlap exact vs AtP (extra check)", ov.get("atp"), 0)
        add("top-10 overlap exact vs DLA (extra check)", ov.get("dla"), 0)
    if not rows:
        return []
    out = ["| measure | value |", "| --- | --- |"]
    out.extend(f"| {label} | {v} |" for label, v in rows)
    return out


def register() -> None:
    register_job_type("report", validate_report_config, run_report)
