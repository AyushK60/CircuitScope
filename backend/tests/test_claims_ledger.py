"""Stage 6 B2: the claims ledger checks pass, and a tampered claim fails.

Runs scripts/check_claims.py as a subprocess (standard library only, so `-I`
is safe) against the committed docs/paper/claims.json + results/*.json, then
proves the checker actually detects a wrong number by running it again with
one deliberately altered expected value. Also asserts the generated
CLAIMS_LEDGER.md stays in sync with claims.json. No model, no backend: fast
subset.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_CLAIMS = REPO_ROOT / "scripts" / "check_claims.py"
CLAIMS = REPO_ROOT / "docs" / "paper" / "claims.json"
LEDGER = REPO_ROOT / "docs" / "paper" / "CLAIMS_LEDGER.md"


def _run(claims_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(CHECK_CLAIMS), "--claims", str(claims_path), "--repo-root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_check_claims_passes_on_committed_files() -> None:
    proc = _run(CLAIMS)
    assert proc.returncode == 0, f"check_claims failed on the committed files:\n{proc.stdout}\n{proc.stderr}"
    spec = json.loads(CLAIMS.read_text(encoding="utf-8"))
    n = len(spec["claims"])
    assert f"{n}/{n} claims pass" in proc.stdout


def test_deliberately_altered_expected_value_fails(tmp_path) -> None:
    spec = json.loads(CLAIMS.read_text(encoding="utf-8"))
    target = next(c for c in spec["claims"] if c["id"] == "rq1-atp-heads-agreement")
    target["expected_value"] = 0.99  # a materially different (wrong) number
    tampered = tmp_path / "claims_tampered.json"
    tampered.write_text(json.dumps(spec, indent=2), encoding="utf-8", newline="\n")
    proc = _run(tampered)
    assert proc.returncode != 0, "the checker accepted an altered expected value"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("rq1-atp-heads-agreement"))
    assert "FAIL" in line


def test_ledger_is_in_sync_with_claims_json() -> None:
    spec = json.loads(CLAIMS.read_text(encoding="utf-8"))
    text = LEDGER.read_text(encoding="utf-8")
    for claim in spec["claims"]:
        header = f"## {claim['id']} — PASS — {claim['status']}"
        assert header in text, f"ledger missing (or not PASS) for {claim['id']}: expected header {header!r}"
        assert claim["statement"] in text
        assert claim["would_be_falsified_by"] in text
    assert "EXPLORATORY" in text or "exploratory" in text
