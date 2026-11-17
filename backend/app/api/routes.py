"""FastAPI routes. Localhost-only product API; strict allowlists; honest errors."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..analysis.dla import compute_dla
from ..analysis.lens import compute_logit_lens
from ..config import (
    CAPTURABLE_HOOKS,
    MAX_LAYERS,
    MAX_PROMPT_CHARS,
    MODEL_ID,
    PATCHABLE_HOOKS,
    validate_hook_name,
)
from ..experiments import sweeps  # noqa: F401  (registers the "sweep" job type)
from ..experiments import study as study_job
from ..experiments.jobs import RUNNER, ExperimentValidationError

study_job.register()  # registers the "study" job type (validate/run)
from ..model_runtime import engine
from ..model_runtime.loader import ADAPTER, ModelNotLoadedError
from ..provenance import collect as provenance_collect
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
    """Submit a Stage 3 study job (one heavy job at a time, shared with
    experiments and sweeps). Poll GET /api/experiments/{job_id} for
    status/progress/results."""
    try:
        job = RUNNER.submit(req.model_dump(), job_type="study")
    except ExperimentValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


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
