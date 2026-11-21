"""The verification job never accepts user paths or options.

Tests cover the fixed argv, rejection of unknown/manual/injected ids, temp-dir
cleanup on timeout and cancellation, the one-heavy-job 422, staleness, and one
real reference check end to end.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.app.experiments import verify
from backend.app.experiments.jobs import (
    JOB_TYPES,
    RUNNER,
    JobCancelled,
    register_job_type,
)
from backend.app.main import create_app
from backend.app.storage import db

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class FakeCtx:
    def __init__(self, job_id: str = "verifytest01", cancel_flag=None) -> None:
        self.job_id = job_id
        self.config = {}
        self.cancel_flag = cancel_flag
        self.messages: list[tuple[float, str]] = []

    def report(self, progress: float, message: str) -> None:
        self.messages.append((progress, message))

    def check_cancel(self) -> None:
        if self.cancel_flag is not None and self.cancel_flag.is_set():
            raise JobCancelled("cancelled")


@pytest.fixture(scope="module")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_fixed_argv_contains_only_server_values(monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        kwargs["stdout"].write("max |api - reference| = 1.0e-06 (tol 5e-03)\n")
        return FakeProc()

    monkeypatch.setattr(verify.subprocess, "Popen", fake_popen)
    log = tmp_path / "log.txt"
    spec = {"script": "scripts/reference_patch.py"}
    args = ["--api-json", str(tmp_path / "input.json")]
    code, output, timed_out = verify._run_reference(spec, args, FakeCtx(), 5.0, log)
    assert code == 0 and timed_out is False
    assert captured["argv"] == [
        sys.executable,
        str(REPO_ROOT / "scripts/reference_patch.py"),
        "--api-json",
        str(tmp_path / "input.json"),
    ]
    assert verify.extract_max_diff(output) == pytest.approx(1e-6)


@pytest.mark.parametrize(
    "check_id",
    [
        "../../../python",
        "dla_lens --help",
        "dla_lens;rm -rf /",
        "unknown",
        "",
        "feedforward",  # manual only
    ],
)
def test_unknown_manual_and_injection_ids_rejected(client, check_id):
    r = client.post("/api/verify", json={"check_id": check_id})
    assert r.status_code == 422, r.text


def test_status_lists_every_check_and_marks_stale(client):
    r = client.get("/api/verify/status")
    assert r.status_code == 200
    body = r.json()
    ids = [c["check_id"] for c in body["checks"]]
    assert ids == list(verify.CHECK_ORDER)
    ff = next(c for c in body["checks"] if c["check_id"] == "feedforward")
    assert ff["manual"] is True and ff["status"] == "never run"

    # A stored result from an older script must read stale, without hiding it.
    db.save_experiment(
        {"job_type": "verify", "check_id": "dla_lens"},
        {
            "job_type": "verify",
            "check_id": "dla_lens",
            "script": "scripts/reference_dla_lens.py",
            "script_sha256": "0" * 64,
            "ok": True,
            "max_diff": 1e-6,
            "tolerance": 2e-3,
            "elapsed_seconds": 1.0,
            "checked_at": "2026-01-01T00:00:00Z",
            "git_commit": "abc123",
            "exit_code": 0,
            "stdout_tail": "",
            "error": None,
        },
        "complete",
    )
    stale = next(
        c for c in client.get("/api/verify/status").json()["checks"] if c["check_id"] == "dla_lens"
    )
    assert stale["status"] == "pass"
    assert stale["stale"] is True
    assert stale["result"]["git_commit"] == "abc123"


def _register_sleeping_check(tmp_path, seconds: float, timeout: float, body: str) -> dict:
    script = tmp_path / "sleep_check.py"
    script.write_text(body, encoding="utf-8")
    spec = {
        "id": "test_sleep",
        "title": "test sleeper",
        "what": "test only",
        "script": str(script),
        "tolerance": 1e-6,
        "runtime": "test",
        "prepare": lambda tmpdir, ctx: [],
        "runnable": True,
        "timeout": timeout,
    }
    verify.CHECKS["test_sleep"] = spec
    return spec


def test_timeout_returns_failed_record_and_removes_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "TEMP_ROOT", tmp_path / "verify_tmp")
    _register_sleeping_check(
        tmp_path, 10.0, 0.5, "import time\ntime.sleep(10)\nprint('done')\n"
    )
    try:
        ctx = FakeCtx(job_id="timeoutjob1")
        result = verify.run_verify({"check_id": "test_sleep"}, ctx)
        assert result["ok"] is False
        assert result["exit_code"] is None
        assert "timed out" in (result["error"] or "")
        assert not (tmp_path / "verify_tmp" / "timeoutjob1").exists()
    finally:
        verify.CHECKS.pop("test_sleep", None)


def test_cancel_terminates_subprocess_and_removes_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "TEMP_ROOT", tmp_path / "verify_tmp")
    _register_sleeping_check(
        tmp_path, 10.0, 30.0, "import time\ntime.sleep(10)\nprint('done')\n"
    )
    flag = threading.Event()
    ctx = FakeCtx(job_id="canceljob1", cancel_flag=flag)

    def cancel_soon():
        time.sleep(0.4)
        flag.set()

    t = threading.Thread(target=cancel_soon)
    t.start()
    try:
        with pytest.raises(JobCancelled):
            verify.run_verify({"check_id": "test_sleep"}, ctx)
        assert not (tmp_path / "verify_tmp" / "canceljob1").exists()
    finally:
        t.join(timeout=5)
        verify.CHECKS.pop("test_sleep", None)


def test_concurrent_heavy_job_is_422(client):
    def validate(cfg):
        return {"job_type": "stage5_sleep"}

    def run(cfg, ctx):
        for i in range(100):
            ctx.report(i / 100, "sleeping")
            time.sleep(0.01)
        return {"ok": True}

    register_job_type("stage5_sleep", validate, run)
    job = RUNNER.submit({}, job_type="stage5_sleep")
    try:
        r = client.post("/api/verify", json={"check_id": "dla_lens"})
        assert r.status_code == 422
        assert "already running" in r.json()["detail"]
    finally:
        RUNNER.cancel(job.job_id)
        deadline = time.time() + 10
        while time.time() < deadline:
            j = RUNNER.get(job.job_id)
            if j and j.status in ("cancelled", "complete", "failed"):
                break
            time.sleep(0.05)
        JOB_TYPES.pop("stage5_sleep", None)


def test_real_patch_check_passes_and_cleans_temp(bridge):
    ctx = FakeCtx(job_id="realpatch01")
    result = verify.run_verify({"check_id": "patch"}, ctx)
    assert result["ok"] is True, result
    assert result["exit_code"] == 0
    assert result["max_diff"] is not None and result["max_diff"] <= result["tolerance"]
    assert result["script_sha256"]
    assert result["git_commit"]
    assert not (verify.TEMP_ROOT / "realpatch01").exists()
    print(f"\npatch verify: max_diff={result['max_diff']} elapsed={result['elapsed_seconds']}s")


# ---------------------------------------------------------------------------
# review regression: staleness must not depend on line endings
# ---------------------------------------------------------------------------


def test_script_sha256_ignores_crlf_conversion(tmp_path, monkeypatch):
    """git on Windows (core.autocrlf) rewrites LF to CRLF on checkout; an
    unchanged script must keep the same hash, a real edit must not."""
    monkeypatch.setattr(verify, "REPO_ROOT", tmp_path)
    p = tmp_path / "s.py"
    p.write_bytes(b"print('a')\nprint('b')\n")
    lf = verify.script_sha256("s.py")
    p.write_bytes(b"print('a')\r\nprint('b')\r\n")
    assert verify.script_sha256("s.py") == lf
    p.write_bytes(b"print('a')\nprint('c')\n")
    assert verify.script_sha256("s.py") != lf
