"""FastAPI routes. Localhost-only product API; strict allowlists; honest errors."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import (
    CAPTURABLE_HOOKS,
    MAX_LAYERS,
    MAX_PROMPT_CHARS,
    MODEL_ID,
    PATCHABLE_HOOKS,
    validate_hook_name,
)
from ..experiments.jobs import RUNNER, ExperimentValidationError
from ..model_runtime import engine
from ..model_runtime.loader import ADAPTER, ModelNotLoadedError
from ..storage import db

router = APIRouter(prefix="/api")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    top_k: int = Field(default=8, ge=1, le=20)
    candidate_a: str | None = Field(default=None, max_length=32)
    candidate_b: str | None = Field(default=None, max_length=32)


class ContinueRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    n_tokens: int = Field(default=5, ge=1, le=8)


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


@router.post("/continue")
def continue_text(req: ContinueRequest) -> dict:
    try:
        meta = ADAPTER.metadata()
        return {"model": meta, **engine.continue_text(req.text, req.n_tokens)}
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
    }


@router.post("/experiments/{job_id}/cancel")
def cancel_experiment(job_id: str) -> dict:
    ok = RUNNER.cancel(job_id)
    return {"job_id": job_id, "cancel_requested": ok}


@router.get("/experiments")
def experiments() -> dict:
    return {"jobs": [j.job_id for j in RUNNER.recent()], "saved": db.list_experiments()}


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
