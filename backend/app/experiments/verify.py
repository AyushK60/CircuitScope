"""Stage 5 Task 2: the "verify" job type and the check registry.

Each entry in ``CHECKS`` names one of the repository's independent reference
scripts. A verify job computes the API-side numbers IN PROCESS with the same
functions the API endpoints call, writes them to a temp directory under
``outputs/verify_tmp/<job_id>/``, and then runs the reference script with a
FIXED argv (no user string ever reaches the command line, no shell). The
script's own ``max |api - reference|`` line and exit code decide pass/fail;
a failure is reported honestly and is never rounded away.

``feedforward`` is listed but not runnable from the server: the browser Neural
Lab needs browser-side input, as the Stage 5 spec states.

The reference scripts hardcode the default IOI pair (clean/corrupted, the two
candidate strings, and the blocks.5.hook_out patch at position 9). The
constants below mirror them; ``_reference_pair()`` checks that the task
registry still matches instead of silently verifying a different prompt.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time

from ..analysis.dla import compute_dla
from ..analysis.lens import compute_logit_lens
from ..provenance import collect as collect_provenance
from ..storage import db
from ..tasks import registry as task_registry
from .jobs import (
    ExperimentValidationError,
    JobCancelled,
    JobContext,
    execute_experiment,
    register_job_type,
)
from .sweeps import run_sweep

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TEMP_ROOT = REPO_ROOT / "outputs" / "verify_tmp"
VERIFY_TIMEOUT = 600.0
TAIL_LINES = 60

REFERENCE_TASK_ID = "ioi_john_mary"
REFERENCE_CLEAN = "John and Mary went to the shop. John gave a book to"
REFERENCE_CORRUPTED = "John and Mary went to the shop. Mary gave a book to"
REFERENCE_CAND_A = " Mary"
REFERENCE_CAND_B = " John"
REFERENCE_HOOK = "blocks.5.hook_out"
REFERENCE_POSITIONS = [9]

MAX_DIFF_RE = re.compile(r"max\s+\|api[^\n]*?=\s*([-+0-9.eE]+)")


# ---------------------------------------------------------------------------
# fixed inputs (server-side; never client data)
# ---------------------------------------------------------------------------


def _reference_pair():
    """The registry's default pair, checked against the reference scripts."""
    task = task_registry.get_task(REFERENCE_TASK_ID)
    if task is None or not task.pairs:
        raise ExperimentValidationError(
            f"verification needs the task {REFERENCE_TASK_ID!r}, which is not loaded"
        )
    pair = task.pairs[0]
    expected = (
        REFERENCE_CLEAN,
        REFERENCE_CORRUPTED,
        REFERENCE_CAND_A,
        REFERENCE_CAND_B,
    )
    actual = (pair.clean_text, pair.corrupted_text, pair.candidate_a, pair.candidate_b)
    if actual != expected:
        raise ExperimentValidationError(
            f"task {REFERENCE_TASK_ID!r} no longer matches the reference scripts' fixed prompt; "
            "the verification panel would compare different inputs"
        )
    return pair


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class _PhaseCtx:
    """Forward cancel and rescale progress into a slice of the verify job."""

    def __init__(self, inner, base: float, span: float) -> None:
        self._inner = inner
        self._base = base
        self._span = span

    def report(self, progress: float, message: str) -> None:
        value = max(0.0, min(1.0, float(progress)))
        self._inner.report(self._base + value * self._span, message)

    def check_cancel(self) -> None:
        self._inner.check_cancel()

    @property
    def cancel_flag(self):
        return self._inner.cancel_flag


# ---------------------------------------------------------------------------
# API-side inputs, computed in process
# ---------------------------------------------------------------------------


def _prepare_dla_lens(tmpdir: pathlib.Path, ctx) -> list[str]:
    pair = _reference_pair()
    ctx.report(0.08, "computing DLA in process")
    dla = compute_dla(pair.clean_text, pair.candidate_a, pair.candidate_b)
    ctx.check_cancel()
    ctx.report(0.24, "computing the logit lens in process")
    lens = compute_logit_lens(pair.clean_text, pair.candidate_a, pair.candidate_b, 5)
    dla_path, lens_path = tmpdir / "dla.json", tmpdir / "lens.json"
    _write_json(dla_path, dla)
    _write_json(lens_path, lens)
    return ["--dla-json", str(dla_path), "--lens-json", str(lens_path)]


def _prepare_sweep(tmpdir: pathlib.Path, ctx) -> list[str]:
    _reference_pair()
    ctx.report(0.02, "computing the layer x position sweep in process")
    position = run_sweep(
        {"task_id": REFERENCE_TASK_ID, "pair_id": None, "mode": "layer_position"},
        _PhaseCtx(ctx, 0.02, 0.46),
    )
    ctx.check_cancel()
    ctx.report(0.50, "computing the layer x head sweep in process")
    head = run_sweep(
        {"task_id": REFERENCE_TASK_ID, "pair_id": None, "mode": "layer_head"},
        _PhaseCtx(ctx, 0.50, 0.42),
    )
    pos_path, head_path = tmpdir / "layer_position.json", tmpdir / "layer_head.json"
    _write_json(pos_path, position)
    _write_json(head_path, head)
    return ["--layer-position-json", str(pos_path), "--layer-head-json", str(head_path)]


def _latest_study() -> dict:
    rows = db.latest_completed_with_job_type("study")
    if not rows or not rows[0].get("results"):
        raise ExperimentValidationError(
            "needs a completed study: run one on the Study page first"
        )
    results = rows[0]["results"]
    if not (results.get("prompt_set") or {}).get("prompts"):
        raise ExperimentValidationError("the most recent study has no stored prompt set")
    return results


def _prepare_components(tmpdir: pathlib.Path, ctx) -> list[str]:
    ctx.report(0.10, "loading the most recent completed study")
    study = _latest_study()
    path = tmpdir / "study.json"
    _write_json(path, study)
    return ["--api-json", str(path)]


def _prepare_atp(tmpdir: pathlib.Path, ctx) -> list[str]:
    ctx.report(0.10, "loading the most recent completed study")
    study = _latest_study()
    path = tmpdir / "study.json"
    _write_json(path, study)
    return ["--study-json", str(path)]


def _prepare_patch(tmpdir: pathlib.Path, ctx) -> list[str]:
    _reference_pair()
    ctx.report(0.06, "running the patched experiment in process")
    result = execute_experiment(
        {
            "clean_text": REFERENCE_CLEAN,
            "corrupted_text": REFERENCE_CORRUPTED,
            "hook": REFERENCE_HOOK,
            "positions": list(REFERENCE_POSITIONS),
            "heads": None,
            "candidate_a": REFERENCE_CAND_A,
            "candidate_b": REFERENCE_CAND_B,
        },
        cancel_flag=ctx.cancel_flag,
        progress=_PhaseCtx(ctx, 0.06, 0.50).report,
    )
    path = tmpdir / "intervention.json"
    _write_json(path, result)
    return ["--api-json", str(path)]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

CHECKS: dict[str, dict] = {
    "dla_lens": {
        "id": "dla_lens",
        "title": "Direct logit attribution and the logit lens",
        "what": (
            "Recomputes all 171 DLA terms, the reconciliation identity and every "
            "lens stage's score gap with raw HuggingFace forward hooks."
        ),
        "script": "scripts/reference_dla_lens.py",
        "tolerance": 2e-3,
        "runtime": "about 15-30 seconds",
        "prepare": _prepare_dla_lens,
        "runnable": True,
    },
    "sweep": {
        "id": "sweep",
        "title": "Both patching sweeps",
        "what": (
            "Recomputes every cell of the layer x position and layer x head "
            "denoising sweeps, plus both baselines."
        ),
        "script": "scripts/reference_sweep.py",
        "tolerance": 5e-3,
        "runtime": "about 2-4 minutes",
        "prepare": _prepare_sweep,
        "runnable": True,
    },
    "components": {
        "id": "components",
        "title": "Exact per-component effects",
        "what": (
            "Recomputes Rec_D, Rec_P, zero ablation and a mean-ablation subset for "
            "all 156 components on the first 3 usable prompts of the latest study."
        ),
        "script": "scripts/reference_components.py",
        "tolerance": 5e-3,
        "runtime": "about 2-5 minutes",
        "prepare": _prepare_components,
        "runnable": True,
    },
    "atp": {
        "id": "atp",
        "title": "Attribution patching",
        "what": (
            "Recomputes Rec_AtP for all 156 components on the first 3 usable "
            "prompts of the latest study."
        ),
        "script": "scripts/reference_atp.py",
        "tolerance": 2e-3,
        "runtime": "about 30-90 seconds",
        "prepare": _prepare_atp,
        "runnable": True,
    },
    "patch": {
        "id": "patch",
        "title": "Patched-run numbers",
        "what": (
            "Recomputes D_clean, D_corrupted, D_patched and Recovery for the "
            "default pair's blocks.5.hook_out patch at position 9."
        ),
        "script": "scripts/reference_patch.py",
        "tolerance": 5e-3,
        "runtime": "about 10 seconds",
        "prepare": _prepare_patch,
        "runnable": True,
    },
    "feedforward": {
        "id": "feedforward",
        "title": "Neural Lab feedforward",
        "what": (
            "The feedforward reference needs an input typed into the browser "
            "Neural Lab, so it is a manual check and is not runnable here."
        ),
        "script": "scripts/reference_feedforward.py",
        "tolerance": None,
        "runtime": "manual (needs browser input)",
        "prepare": None,
        "runnable": False,
    },
}

CHECK_ORDER = ("dla_lens", "sweep", "components", "atp", "patch", "feedforward")


def validate_verify_config(cfg: dict) -> dict:
    check_id = cfg.get("check_id")
    if not isinstance(check_id, str) or not check_id:
        raise ExperimentValidationError("check_id is required")
    spec = CHECKS.get(check_id)
    if spec is None:
        raise ExperimentValidationError(f"unknown verification check: {check_id!r}")
    if not spec.get("runnable"):
        raise ExperimentValidationError(
            f"{check_id!r} is a manual check and cannot be run from the server"
        )
    return {"check_id": check_id, "job_type": "verify"}


# ---------------------------------------------------------------------------
# subprocess execution
# ---------------------------------------------------------------------------


def script_sha256(relative_path: str) -> str | None:
    try:
        return hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
    except OSError:
        return None


def _tail(text: str, lines: int = TAIL_LINES) -> str:
    out = text.splitlines()
    if len(out) <= lines:
        return "\n".join(out)
    return "\n".join(out[-lines:])


def _terminate(proc: subprocess.Popen) -> None:
    """Terminate, then kill after a short grace period."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def extract_max_diff(text: str) -> float | None:
    """The largest number on any ``max |api - reference| ... =`` line."""
    values = []
    for match in MAX_DIFF_RE.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return max(values) if values else None


def _run_reference(
    spec: dict, args: list[str], ctx, timeout: float, log_path: pathlib.Path
) -> tuple[int | None, str, bool]:
    """Run one reference script with a fixed argv. Returns (exit, output, timed_out).

    stdout and stderr go to one temp file (never a pipe), so a long script
    cannot deadlock on a full buffer. Cancellation terminates then kills.
    """
    argv = [sys.executable, str(REPO_ROOT / spec["script"]), *args]
    started = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        timed_out = False
        try:
            while proc.poll() is None:
                if ctx.cancel_flag is not None and ctx.cancel_flag.is_set():
                    _terminate(proc)
                    raise JobCancelled("cancelled")
                elapsed = time.time() - started
                if elapsed > timeout:
                    timed_out = True
                    _terminate(proc)
                    break
                ctx.report(0.55 + min(0.40, 0.40 * elapsed / timeout), "running the reference script")
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                _terminate(proc)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return (None if timed_out else proc.returncode), text, timed_out


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------


def run_verify(cfg: dict, ctx: JobContext) -> dict:
    norm = validate_verify_config(cfg)
    spec = CHECKS[norm["check_id"]]
    timeout = float(spec.get("timeout", VERIFY_TIMEOUT))
    tmpdir = TEMP_ROOT / ctx.job_id
    started = time.perf_counter()
    prov = collect_provenance()
    code_sha = script_sha256(spec["script"])
    base = {
        "job_type": "verify",
        "job_id": ctx.job_id,
        "check_id": spec["id"],
        "title": spec["title"],
        "tolerance": spec["tolerance"],
        "script": spec["script"],
        "script_sha256": code_sha,
        "git_commit": prov.get("git_commit"),
        "model_revision": prov.get("model_revision"),
        "versions": prov.get("versions"),
        "checked_at": prov.get("timestamp_utc"),
    }
    try:
        tmpdir.mkdir(parents=True, exist_ok=True)
        ctx.report(0.02, f"preparing {spec['title']}")
        args = spec["prepare"](tmpdir, ctx)
        ctx.check_cancel()
        ctx.report(0.55, "running the reference script")
        log_path = tmpdir / "reference.log"
        exit_code, output, timed_out = _run_reference(spec, args, ctx, timeout, log_path)
        elapsed = round(time.perf_counter() - started, 3)
        if timed_out:
            return {
                **base,
                "ok": False,
                "exit_code": None,
                "max_diff": None,
                "elapsed_seconds": elapsed,
                "stdout_tail": _tail(output),
                "error": f"timed out after {timeout:.0f}s",
            }
        max_diff = extract_max_diff(output)
        if max_diff is None:
            return {
                **base,
                "ok": False,
                "exit_code": exit_code,
                "max_diff": None,
                "elapsed_seconds": elapsed,
                "stdout_tail": _tail(output),
                "error": "the reference script printed no max |api - reference| line",
            }
        ok = exit_code == 0 and max_diff <= float(spec["tolerance"])
        return {
            **base,
            "ok": bool(ok),
            "exit_code": exit_code,
            "max_diff": float(max_diff),
            "elapsed_seconds": elapsed,
            "stdout_tail": _tail(output),
            "error": None if ok else f"reference script exited {exit_code} or exceeded tolerance",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# status for the UI
# ---------------------------------------------------------------------------


def verify_status() -> dict:
    """Latest stored result per registered check, with script staleness."""
    rows = db.latest_completed_with_job_type("verify")
    latest: dict[str, dict] = {}
    for row in rows:
        check_id = (row.get("config") or {}).get("check_id")
        if check_id and row.get("results") and check_id not in latest:
            latest[check_id] = row["results"]
    prov = collect_provenance()
    checks = []
    for check_id in CHECK_ORDER:
        spec = CHECKS[check_id]
        stored = latest.get(check_id)
        current_sha = script_sha256(spec["script"])
        entry = {
            "check_id": check_id,
            "title": spec["title"],
            "what": spec["what"],
            "tolerance": spec["tolerance"],
            "runtime": spec["runtime"],
            "manual": not spec.get("runnable", False),
            "script": spec["script"],
            "current_script_sha256": current_sha,
        }
        if stored is None:
            entry.update(
                {
                    "status": "never run",
                    "ok": None,
                    "stale": False,
                    "result": None,
                }
            )
        else:
            entry.update(
                {
                    "status": "pass" if stored.get("ok") else "fail",
                    "ok": bool(stored.get("ok")),
                    "stale": bool(
                        current_sha and stored.get("script_sha256") != current_sha
                    ),
                    "result": stored,
                }
            )
        checks.append(entry)
    return {"checks": checks, "current_git_commit": prov.get("git_commit")}


def register() -> None:
    register_job_type("verify", validate_verify_config, run_verify)
