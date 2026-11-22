"""FastAPI routes. Localhost-only product API; strict allowlists; plain error messages."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from ..analysis.dla import compute_dla
from ..analysis.lens import compute_logit_lens
from ..explain import lint as explain_lint
from ..explain import story as report_story
from ..config import (
    CAPTURABLE_HOOKS,
    MAX_LAYERS,
    MAX_PROMPT_CHARS,
    MODEL_ID,
    PATCHABLE_HOOKS,
    validate_hook_name,
)
from ..experiments import sweeps  # noqa: F401  (registers the "sweep" job type)
from ..experiments import agreement as agreement_job
from ..experiments import report as report_job
from ..experiments import study as study_job
from ..experiments import verify as verify_job
from ..experiments.jobs import RUNNER, ExperimentValidationError

study_job.register()  # registers the "study" job type (validate/run)
report_job.register()  # registers the "report" job type (validate/run)
agreement_job.register()  # registers the "agreement" job type (validate/run)
verify_job.register()  # registers the "verify" job type (validate/run)
from ..model_runtime import engine
from ..model_runtime.loader import ADAPTER, ModelNotLoadedError
from ..provenance import collect as provenance_collect
from ..reproduce.generate import (
    KINDS as REPRODUCE_KINDS,
    dla_data,
    filename as reproduce_filename,
    generate as generate_reproduce_script,
    intervention_data,
    lens_data,
    report_data,
    study_subset_data,
    sweep_data,
)
from ..storage import db
from ..tasks import registry as task_registry
from ..tasks.registry import TASK_ID_PATTERN

router = APIRouter(prefix="/api")

# Task definition files are read once at startup. Reading JSON here can never
# trigger a model load; malformed files are recorded, never raised.
task_registry.load_all()


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    top_k: int = Field(default=8, ge=1, le=20)
    candidate_a: str | None = Field(default=None, max_length=32)
    candidate_b: str | None = Field(default=None, max_length=32)


class QuestionRequest(BaseModel):
    context: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    question: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    candidates: list[str] = Field(min_length=2, max_length=5)
    template: str | None = Field(default=None, max_length=MAX_PROMPT_CHARS)


class ContinueRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    n_tokens: int = Field(default=5, ge=1, le=8)
    mode: str = Field(default="greedy", pattern="^(greedy|sample)$")
    temperature: float = Field(default=1.0, ge=0.05, le=2.0)
    top_k: int = Field(default=0, ge=0, le=50)
    seed: int | None = Field(default=None, ge=0)


class InspectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    hooks: list[str] = Field(default_factory=list)
    layer: int | None = Field(default=None, ge=0, le=MAX_LAYERS - 1)


class ExperimentRequest(BaseModel):
    clean_text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    corrupted_text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    hook: str
    positions: list[int]
    heads: list[int] | None = None
    candidate_a: str
    candidate_b: str
    top_k: int = Field(default=8, ge=1, le=20)


@router.get("/health")
def health() -> dict:
    try:
        meta = ADAPTER.metadata()
        return {"status": "ok", "model_loaded": True, "model": meta}
    except Exception:
        return {"status": "ok", "model_loaded": False, "model": None}


@router.get("/models")
def models() -> dict:
    return {
        "supported": [{"id": MODEL_ID, "name": "GPT-2 Small", "approx_params_M": 124}],
        "loaded": ADAPTER.is_loaded,
    }


@router.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        meta = ADAPTER.metadata()
        tok = ADAPTER.tokenize(req.text)
        pred = engine.predict(req.text, req.top_k)
        response = {"model": meta, "tokenization": tok, "prediction": pred}
        # optional measured candidate comparison (both must be single tokens)
        if req.candidate_a and req.candidate_b:
            try:
                id_a = ADAPTER.token_id_for_str(req.candidate_a)
                id_b = ADAPTER.token_id_for_str(req.candidate_b)
                import torch

                bridge = ADAPTER.bridge()
                logits = bridge(bridge.to_tokens(req.text))
                response["comparison"] = {
                    **engine.compare_candidates(logits, id_a, id_b),
                    "candidate_a": req.candidate_a,
                    "candidate_b": req.candidate_b,
                    "sign_convention": "positive D means candidate A scores above candidate B at the final position",
                }
            except ValueError as e:
                response["comparison_error"] = str(e)
        return response
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/question")
def question(req: QuestionRequest) -> dict:
    try:
        meta = ADAPTER.metadata()
        return {"model": meta, **engine.score_question_candidates(
            req.context, req.question, req.candidates, req.template
        )}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}")


@router.post("/continue")
def continue_text(req: ContinueRequest) -> dict:
    try:
        meta = ADAPTER.metadata()
        return {
            "model": meta,
            **engine.continue_text(
                req.text, req.n_tokens, req.mode, req.temperature, req.top_k, req.seed
            ),
        }
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/flow")
def flow(req: PredictRequest) -> dict:
    """Recorded single-pass flow data for the computation playback."""
    try:
        meta = ADAPTER.metadata()
        return {"model": meta, **engine.capture_flow(req.text)}
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/inspect")
def inspect(req: InspectRequest) -> dict:
    try:
        meta = ADAPTER.metadata()
        tok = ADAPTER.tokenize(req.text)
        response: dict[str, Any] = {"model": meta, "tokenization": tok}
        if req.hooks:
            try:
                summaries, _cache = engine.capture(req.text, req.hooks)
                response["activations"] = summaries
            except Exception as e:
                response["activations_error"] = str(e)
        if req.layer is not None:
            try:
                response["attention"] = engine.attention_pattern(req.text, req.layer)
            except Exception as e:
                response["attention_error"] = str(e)
        return response
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/experiments")
def create_experiment(req: ExperimentRequest) -> dict:
    try:
        job = RUNNER.submit(req.model_dump())
    except ExperimentValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


@router.get("/experiments/{job_id}")
def experiment_status(job_id: str) -> dict:
    job = RUNNER.get(job_id)
    if job is None:
        stored = db.get_experiment(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return stored
    return {
        "id": job.job_id,
        "status": job.status,
        "config": job.config,
        "results": job.results,
        "error": job.error,
        "type": job.type,
        "progress": job.progress,
        "progress_message": job.progress_message,
    }


@router.post("/experiments/{job_id}/cancel")
def cancel_experiment(job_id: str) -> dict:
    ok = RUNNER.cancel(job_id)
    return {"job_id": job_id, "cancel_requested": ok}


@router.get("/experiments")
def experiments() -> dict:
    return {"jobs": [j.job_id for j in RUNNER.recent()], "saved": db.list_experiments()}


@router.delete("/experiments/{exp_id}")
def delete_experiment(exp_id: str) -> dict:
    deleted = db.delete_experiment(exp_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="experiment not found")
    return {"deleted": True, "id": exp_id}


@router.get("/experiments/{job_id}/export")
def export_experiment(job_id: str, fmt: str = "json") -> Any:
    job = RUNNER.get(job_id)
    if job is not None and job.results is not None:
        payload = job.results
    else:
        stored = db.get_experiment(job_id)
        if stored is None or stored.get("results") is None:
            raise HTTPException(status_code=404, detail="no completed results for this experiment")
        payload = stored["results"]
    if fmt == "json":
        return payload
    if fmt == "csv":
        # Candidate-level CSV of the three runs' top predictions + metrics.
        lines = ["run,rank,token,token_id,logit,prob"]
        for run_name in ("clean", "corrupted", "patched"):
            for i, cand in enumerate(payload["runs"][run_name]):
                lines.append(
                    f"{run_name},{i},{cand['token']!r},{cand['token_id']},{cand['logit']:.6f},{cand['prob']:.8f}"
                )
        m = payload["metrics"]
        lines.append(f"metric,D_clean,{m['D_clean']:.6f},,")
        lines.append(f"metric,D_corrupted,{m['D_corrupted']:.6f},,")
        lines.append(f"metric,D_patched,{m['D_patched']:.6f},,")
        lines.append(f"metric,Delta,{m['Delta']:.6f},,")
        if m["Recovery"] is not None:
            lines.append(f"metric,Recovery,{m['Recovery']:.6f},,")
        from fastapi import Response

        return Response(content="\n".join(lines), media_type="text/csv")
    raise HTTPException(status_code=400, detail="fmt must be json or csv")


@router.get("/hooks")
def hooks() -> dict:
    return {"patchable": sorted(PATCHABLE_HOOKS), "capturable": sorted(CAPTURABLE_HOOKS)}


class SweepRequest(BaseModel):
    task_id: str
    pair_id: str | None = None
    mode: str


class DlaRequest(BaseModel):
    task_id: str
    pair_id: str | None = None
    run: str = Field(default="clean", pattern="^(clean|corrupted)$")


class LogitLensRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    candidate_a: str | None = Field(default=None, max_length=32)
    candidate_b: str | None = Field(default=None, max_length=32)
    top_k: int = Field(default=5, ge=1, le=10)


def _task_for_analysis(task_id: str, pair_id: str | None):
    """Resolve a task pair for the analysis endpoints: 404 for unknown
    task/pair, 422 when the pair fails registry validation."""
    if not re.fullmatch(TASK_ID_PATTERN, task_id):
        raise HTTPException(status_code=404, detail="task not found")
    task = task_registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    pair = task.pairs[0] if pair_id is None else next((p for p in task.pairs if p.id == pair_id), None)
    if pair is None:
        raise HTTPException(status_code=404, detail=f"pair not found for task {task_id}")
    validation = task_registry.validate_task(task)
    pair_validation = next((p for p in validation.get("pairs", []) if p["id"] == pair.id), None)
    if pair_validation is None or not pair_validation.get("ok"):
        detail = "; ".join((pair_validation or {}).get("errors", []) or validation.get("errors", []))
        raise HTTPException(status_code=422, detail=f"task pair failed validation: {detail}")
    return pair


@router.post("/analysis/dla")
def analysis_dla(req: DlaRequest) -> dict:
    """Direct logit attribution for one task pair's clean or corrupted prompt."""
    task = _task_for_analysis(req.task_id, req.pair_id)
    try:
        text = task.clean_text if req.run == "clean" else task.corrupted_text
        result = compute_dla(text, task.candidate_a, task.candidate_b)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "task_id": req.task_id,
        "pair_id": task.id,
        "run": req.run,
        **result,
        "provenance": provenance_collect(),
    }


@router.post("/analysis/logit_lens")
def analysis_logit_lens(req: LogitLensRequest) -> dict:
    """Logit lens over the stages of one free-text prompt."""
    if (req.candidate_a is None) != (req.candidate_b is None):
        raise HTTPException(status_code=422, detail="candidate_a and candidate_b must be provided together")
    try:
        result = compute_logit_lens(req.text, req.candidate_a, req.candidate_b, req.top_k)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {**result, "provenance": provenance_collect()}


REPRODUCE_JOB_KINDS = {
    "intervention": "intervention",
    "sweep": "sweep",
    "study_subset": "study",
    "report": "report",
}


class ReproduceRequest(BaseModel):
    kind: str
    job_id: str | None = None
    task_id: str | None = None
    pair_id: str | None = None
    run: str | None = Field(default=None, pattern="^(clean|corrupted)$")
    text: str | None = Field(default=None, min_length=1, max_length=MAX_PROMPT_CHARS)
    candidate_a: str | None = Field(default=None, max_length=32)
    candidate_b: str | None = Field(default=None, max_length=32)
    mode: str = Field(default="sample", pattern="^(sample|full)$")


def _completed_result_for_reproduce(job_id: str, expected_type: str) -> tuple[dict, dict]:
    """Load a COMPLETED job of the expected type (live record first, then DB).

    Returns (results, config). A wrong type or an unfinished job is a 422, an
    unknown id is a 404. The client never supplies the numbers.
    """
    job = RUNNER.get(job_id)
    if job is not None:
        status, results, config, job_type = job.status, job.results, job.config, job.type
    else:
        stored = db.get_experiment(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="job not found")
        status = stored.get("status")
        results = stored.get("results")
        config = stored.get("config") or {}
        job_type = config.get("job_type") or ("intervention" if expected_type == "intervention" else "unknown")
    if job_type != expected_type:
        raise HTTPException(
            status_code=422,
            detail=f"job {job_id} is a {job_type!r} job; this reproduction kind needs a {expected_type!r} job",
        )
    if status != "complete" or not results:
        raise HTTPException(status_code=422, detail=f"job {job_id} has no completed results")
    return results, config


@router.post("/reproduce")
def reproduce(req: ReproduceRequest) -> Any:
    """Return a standalone Python script that recomputes the displayed numbers.

    The server only GENERATES text here; it never executes the script. All
    inputs are validated against the existing limits and embedded as one JSON
    literal by app.reproduce.generate.
    """
    if req.kind not in REPRODUCE_KINDS:
        raise HTTPException(status_code=422, detail=f"unknown reproduce kind: {req.kind!r}")

    if req.kind in REPRODUCE_JOB_KINDS:
        if not req.job_id:
            raise HTTPException(status_code=422, detail="job_id is required for this reproduce kind")
        results, config = _completed_result_for_reproduce(
            req.job_id, REPRODUCE_JOB_KINDS[req.kind]
        )
        try:
            if req.kind == "intervention":
                data = intervention_data(config, results)
            elif req.kind == "sweep":
                data = sweep_data(
                    mode=results["mode"],
                    clean_text=config["clean_text"],
                    corrupted_text=config["corrupted_text"],
                    candidate_a=config["candidate_a"],
                    candidate_b=config["candidate_b"],
                    baselines=results["baselines"],
                    recovery=results["recovery"],
                    d_patched=results["d_patched"],
                    provenance=results.get("provenance"),
                )
            elif req.kind == "study_subset":
                data = study_subset_data(results)
                if not data["prompts"]:
                    raise ValueError("this study has no usable prompt with stored swap/AtP values")
            else:  # report
                data = report_data(results, config)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"cannot build this reproduction: {e}")
        id_for_name = req.job_id
    else:
        # Synchronous analyses: recompute the "stored" numbers at request time
        # with the same functions the analysis endpoints call.
        if req.task_id:
            pair = _task_for_analysis(req.task_id, req.pair_id)
            text = pair.clean_text if (req.run or "clean") == "clean" else pair.corrupted_text
            cand_a, cand_b = pair.candidate_a, pair.candidate_b
            id_for_name = pair.id
        else:
            if not req.text:
                raise HTTPException(status_code=422, detail="text (or task_id) is required")
            text, cand_a, cand_b = req.text, req.candidate_a, req.candidate_b
            id_for_name = "text"
        if (cand_a is None) != (cand_b is None):
            raise HTTPException(
                status_code=422, detail="candidate_a and candidate_b must be provided together"
            )
        if req.kind == "dla" and (cand_a is None or cand_b is None):
            raise HTTPException(status_code=422, detail="DLA reproduction needs both candidates")
        try:
            for cand in (cand_a, cand_b):
                if cand is not None:
                    ADAPTER.token_id_for_str(cand)
            if req.kind == "dla":
                result = compute_dla(text, cand_a, cand_b)
                data = dla_data(
                    text=text,
                    candidate_a=cand_a,
                    candidate_b=cand_b,
                    dla_result=result,
                    provenance=provenance_collect(),
                )
            else:
                result = compute_logit_lens(text, cand_a, cand_b, 5)
                data = lens_data(
                    text=text,
                    candidate_a=cand_a,
                    candidate_b=cand_b,
                    lens_result=result,
                    provenance=provenance_collect(),
                )
        except ModelNotLoadedError:
            raise HTTPException(status_code=503, detail="model not loaded yet")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    try:
        script = generate_reproduce_script(req.kind, data, req.mode)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return Response(
        content=script,
        media_type="text/x-python",
        headers={
            "Content-Disposition": f'attachment; filename="{reproduce_filename(req.kind, id_for_name)}"'
        },
    )


@router.post("/sweeps")
def create_sweep(req: SweepRequest) -> dict:
    """Submit a patching sweep job (one heavy job at a time, shared with
    experiments). Poll GET /api/experiments/{job_id} for status/results."""
    try:
        job = RUNNER.submit(req.model_dump(), job_type="sweep")
    except ExperimentValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


class StudyRequest(BaseModel):
    n_prompts: int = Field(default=40, ge=4, le=100)
    seed: int = Field(default=0, ge=0)


@router.post("/study")
def create_study(req: StudyRequest) -> dict:
    """Submit a study job (one heavy job at a time, shared with
    experiments and sweeps). Poll GET /api/experiments/{job_id} for
    status/progress/results."""
    try:
        job = RUNNER.submit(req.model_dump(), job_type="study")
    except ExperimentValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


class AgreementRequest(BaseModel):
    task_id: str
    pair_id: str | None = None


@router.post("/agreement")
def create_agreement(req: AgreementRequest) -> dict:
    """Submit an agreement job (exact vs AtP vs DLA). One heavy job at
    a time, shared with experiments/sweeps/studies/reports."""
    try:
        job = RUNNER.submit(req.model_dump(), job_type="agreement")
    except ExperimentValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


class VerifyRequest(BaseModel):
    check_id: str


@router.post("/verify")
def create_verify(req: VerifyRequest) -> dict:
    """Submit one verification check. The server supplies every input; the
    request carries only an allowlisted check id."""
    try:
        job = RUNNER.submit({"check_id": req.check_id}, job_type="verify")
    except ExperimentValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


@router.get("/verify/status")
def verify_status() -> dict:
    """Latest stored result per check, with script-hash staleness."""
    return verify_job.verify_status()


class ReportRequest(BaseModel):
    task_id: str
    pair_id: str | None = None
    study_job_id: str | None = None


@router.post("/report")
def create_report(req: ReportRequest) -> dict:
    """Submit a Circuit Report job (one heavy job at a time, shared
    with experiments/sweeps/studies). Poll GET /api/experiments/{job_id}."""
    try:
        job = RUNNER.submit(req.model_dump(), job_type="report")
    except ExperimentValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


@router.get("/report/{job_id}/export")
def export_report(job_id: str, fmt: str = "md", level: str = "curious") -> Any:
    """Markdown export of a completed report at the requested reading level.
    404 for unknown ids and for non-report jobs."""
    if fmt != "md":
        raise HTTPException(status_code=400, detail="fmt must be md")
    job = RUNNER.get(job_id)
    if job is not None and job.type != "report":
        raise HTTPException(status_code=404, detail="not a report job")
    stored = db.get_experiment(job_id) if job is None else None
    if stored is not None and (stored.get("config") or {}).get("job_type") != "report":
        raise HTTPException(status_code=404, detail="not a report job")
    payload = (job.results if job is not None else None) or (stored or {}).get("results")
    if not payload:
        raise HTTPException(status_code=404, detail="no completed report for this id")
    from fastapi import Response

    from ..explain.templates import LEVELS
    from ..experiments.report import export_markdown

    if level not in LEVELS:
        raise HTTPException(status_code=422, detail="level must be curious, student or expert")
    md = export_markdown(payload, level)
    problems = explain_lint.lint_markdown(md, level)
    if problems:
        raise HTTPException(status_code=500, detail="export lint failed: " + "; ".join(problems[:10]))
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="report_{job_id}_{level}.md"'},
    )


class ExplainRequest(BaseModel):
    job_id: str
    level: str = Field(default="curious", pattern="^(curious|student|expert)$")


@router.post("/explain")
def explain_job(req: ExplainRequest) -> dict:
    """Plain-words claims for a COMPLETED sweep or study job. The server reads
    the stored results; the client never supplies data."""
    job = RUNNER.get(req.job_id)
    results = None
    if job is not None and job.results is not None:
        results = job.results
    else:
        stored = db.get_experiment(req.job_id)
        if stored is not None:
            results = stored.get("results")
    if not results:
        raise HTTPException(status_code=404, detail="no completed results for this job id")
    job_type = results.get("job_type")
    if job_type == "sweep":
        compact = report_story.compact_sweep(results)
        if results.get("mode") == "layer_position":
            claims = report_story.story_sweep_position({"sweep": compact})
        else:
            claims = report_story.story_sweep_heads({"sweep": compact, "dla": None})
    elif job_type == "study":
        claims = report_story.story_study(report_story.compact_study(results))
    elif job_type == "agreement":
        claims = report_story.story_agreement_methods(report_story.compact_agreement(results))
    else:
        raise HTTPException(
            status_code=422,
            detail=f"job type {job_type!r} cannot be explained (sweep, study or agreement only)",
        )
    problems = explain_lint.lint_claims(claims)
    if problems:
        raise HTTPException(status_code=500, detail="explain lint failed: " + "; ".join(problems[:10]))
    return {
        "job_id": req.job_id,
        "job_type": job_type,
        "level": req.level,
        "claims": [c for c in claims if c["level"] == req.level],
        "lint": {"ok": True},
    }


@router.get("/tasks")
def tasks_index() -> dict:
    return {
        "tasks": task_registry.list_tasks(),
        "load_errors": task_registry.get_load_errors(),
    }


@router.get("/tasks/{task_id}")
def task_detail(task_id: str) -> dict:
    # Reject malformed ids (e.g. path fragments) with 404, never 500. The
    # pair-level validation needs the tokenizer, so it runs lazily here (cached).
    if not re.fullmatch(TASK_ID_PATTERN, task_id):
        raise HTTPException(status_code=404, detail="task not found")
    task = task_registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {**task.model_dump(), "validation": task_registry.validate_task(task)}
