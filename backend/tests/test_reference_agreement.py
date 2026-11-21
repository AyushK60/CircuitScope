"""M2 gate: backend patched-run numbers must match the independent reference
implementation (scripts/reference_patch.py - raw HF transformers + forward
hooks, no TransformerLens).

Reference values recorded 2026-10-30 on this machine/versions; re-run the
script to regenerate if dependencies change. Tolerance 5e-3 absolute (fp32
CPU is deterministic; tolerance covers cross-version drift).
"""

from __future__ import annotations

import pytest

from backend.app.experiments.jobs import execute_experiment

REFERENCE = {
    "D_clean": 2.5128,
    "D_corrupted": -4.1768,
    "D_patched": 1.9426,
    "Recovery": 0.9148,
}
TOL = 5e-3


def test_backend_matches_reference(bridge, ioi):
    cfg = {
        "clean_text": ioi["clean"],
        "corrupted_text": ioi["corrupted"],
        "hook": "blocks.5.hook_out",
        "positions": [9],
        "candidate_a": " Mary",
        "candidate_b": " John",
    }
    m = execute_experiment(cfg)["metrics"]
    for key, ref in REFERENCE.items():
        assert m[key] == pytest.approx(ref, abs=TOL), f"{key}: {m[key]} vs reference {ref}"
