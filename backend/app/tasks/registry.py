"""Task registry: load task definition files and validate them against the
real tokenizer.

- ``load_all()`` reads every ``*.json`` in the tasks directory (built from the
  repo root, never from request input). A malformed or schema-invalid file is
  recorded in ``load_errors`` and skipped - it must never crash app startup.
- ``validate_task()`` checks each pair against the real model tokenizer
  (single-token candidates, token alignment, position ranges, hook
  allowlist). It needs the model, so it runs lazily on request and the result
  is cached per task id; loading itself never triggers a model load.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..config import validate_hook_name
from ..model_runtime.loader import ADAPTER

TASK_ID_PATTERN = r"^[a-z0-9_]+$"


class TaskSource(BaseModel):
    citation: str
    url: str


class RecommendedHook(BaseModel):
    hook: str
    positions: list[int]
    heads: list[int] | None = None
    note: str = ""


class TaskPair(BaseModel):
    id: str
    clean_text: str
    corrupted_text: str
    candidate_a: str
    candidate_b: str
    positions_of_interest: dict[str, int]
    recommended_hooks: list[RecommendedHook]


class Task(BaseModel):
    id: str
    name: str
    description: str
    source: TaskSource
    expected_behavior: str
    pairs: list[TaskPair]


def default_tasks_dir() -> Path:
    """Repository-rooted examples/tasks directory (backend/app/tasks → root)."""
    return Path(__file__).resolve().parents[3] / "examples" / "tasks"


# Module-level registry state. load_all() is the only writer.
_REGISTRY: dict[str, Task] = {}
_LOAD_ERRORS: list[dict[str, str]] = []
_VALIDATION_CACHE: dict[str, dict] = {}


def load_all(tasks_dir: Path | str | None = None) -> None:
    """(Re)load every task file in ``tasks_dir``. Malformed files are recorded
    in load_errors and skipped; duplicate ids keep the first occurrence."""
    directory = Path(tasks_dir) if tasks_dir is not None else default_tasks_dir()
    registry: dict[str, Task] = {}
    errors: list[dict[str, str]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("task file must be a JSON object")
                task_id = raw.get("id")
                if not isinstance(task_id, str) or not re.fullmatch(TASK_ID_PATTERN, task_id):
                    raise ValueError(f"task id must match {TASK_ID_PATTERN}: {task_id!r}")
                task = Task.model_validate(raw)
                if task.id in registry:
                    raise ValueError(f"duplicate task id: {task.id!r} (keeping the first)")
                registry[task.id] = task
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                errors.append({"file": path.name, "error": f"{type(e).__name__}: {e}"})
    _REGISTRY.clear()
    _REGISTRY.update(registry)
    _LOAD_ERRORS.clear()
    _LOAD_ERRORS.extend(errors)
    _VALIDATION_CACHE.clear()  # cached validation results refer to the old files


def get_task(task_id: str) -> Task | None:
    return _REGISTRY.get(task_id)


def get_load_errors() -> list[dict[str, str]]:
    return list(_LOAD_ERRORS)


def list_tasks() -> list[dict]:
    return [
        {"id": t.id, "name": t.name, "description": t.description, "n_pairs": len(t.pairs)}
        for t in _REGISTRY.values()
    ]


def validate_task(task: Task) -> dict:
    """Validate every pair against the real tokenizer/model. Cached per task
    id (the cache is cleared whenever load_all() reloads)."""
    cached = _VALIDATION_CACHE.get(task.id)
    if cached is not None:
        return cached

    bridge = ADAPTER.ensure_loaded()
    errors: list[str] = []
    pairs_out: list[dict] = []

    for pair in task.pairs:
        perrors: list[str] = []
        cand_ids: dict[str, int | None] = {}
        for key in ("candidate_a", "candidate_b"):
            try:
                cand_ids[key] = ADAPTER.token_id_for_str(getattr(pair, key))
            except ValueError as e:
                cand_ids[key] = None
                perrors.append(f"{key}: {e}")

        toks_clean = bridge.to_tokens(pair.clean_text)
        toks_corr = bridge.to_tokens(pair.corrupted_text)
        n_clean = int(toks_clean.shape[1])
        n_corr = int(toks_corr.shape[1])
        if n_clean != n_corr:
            perrors.append(f"clean/corrupted token counts differ: {n_clean} vs {n_corr}")
        if pair.clean_text == pair.corrupted_text:
            perrors.append("clean and corrupted texts are identical - no contrast")
        seq_len = n_clean

        for name, pos in pair.positions_of_interest.items():
            if pos >= seq_len:
                perrors.append(f"position_of_interest {name}={pos} out of range for {seq_len} tokens")
        for rh in pair.recommended_hooks:
            if not validate_hook_name(rh.hook, patchable=True):
                perrors.append(f"recommended hook is not patchable/supported: {rh.hook!r}")
            for pos in rh.positions:
                if pos >= seq_len:
                    perrors.append(f"hook {rh.hook} position {pos} out of range for {seq_len} tokens")

        pairs_out.append(
            {
                "id": pair.id,
                "ok": not perrors,
                "errors": perrors,
                "n_tokens_clean": n_clean,
                "n_tokens_corrupted": n_corr,
                "candidate_a_id": cand_ids["candidate_a"],
                "candidate_b_id": cand_ids["candidate_b"],
            }
        )
        errors.extend(f"pair {pair.id}: {e}" for e in perrors)

    result = {"ok": not errors, "errors": errors, "pairs": pairs_out}
    _VALIDATION_CACHE[task.id] = result
    return result
