"""Stage 5 Task 1: reproduce scripts are text, self-contained, and honest.

Every kind is generated from a real small result, compiled, import-checked,
and executed in a subprocess in ``--mode sample``. A tampered stored number
must make the script exit non-zero. No test lets the server execute the
generated text; only the test process does.
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.analysis import components as comps
from backend.app.analysis.atp import compute_atp
from backend.app.analysis.dla import compute_dla
from backend.app.analysis.lens import compute_logit_lens
from backend.app.analysis.promptsets import generate as generate_prompts
from backend.app.experiments.jobs import RUNNER, JobRecord
from backend.app.main import create_app
from backend.app.provenance import collect as collect_provenance
from backend.app.reproduce.generate import (
    dla_data,
    generate,
    intervention_data,
    KINDS,
    lens_data,
    report_data,
    study_subset_data,
    sweep_data,
)
from backend.app.storage import db

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN = "John and Mary went to the shop. John gave a book to"
CORRUPTED = "John and Mary went to the shop. Mary gave a book to"
CAND_A = " Mary"
CAND_B = " John"
ALLOWED_IMPORTS = {"argparse", "json", "sys", "torch", "transformers"}
SCRIPT_TIMEOUT = 900


@pytest.fixture(scope="module")
def client(bridge):
    app = create_app()
    with TestClient(app) as c:
        yield c


def _wait(job_id: str, timeout: float = 300.0) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = RUNNER.get(job_id)
        if job is not None and job.status in ("complete", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


@pytest.fixture(scope="module")
def intervention_job(bridge, client):
    r = client.post(
        "/api/experiments",
        json={
            "clean_text": CLEAN,
            "corrupted_text": CORRUPTED,
            "hook": "blocks.5.hook_out",
            "positions": [9],
            "heads": None,
            "candidate_a": CAND_A,
            "candidate_b": CAND_B,
        },
    )
    assert r.status_code == 200, r.text
    job = _wait(r.json()["job_id"], timeout=120)
    assert job.status == "complete", job.error
    return job


@pytest.fixture(scope="module")
def sweep_job(bridge):
    job = RUNNER.submit(
        {"task_id": "ioi_john_mary", "pair_id": None, "mode": "layer_position"},
        job_type="sweep",
    )
    done = _wait(job.job_id, timeout=300)
    assert done.status == "complete", done.error
    return done


@pytest.fixture(scope="module")
def report_job(bridge):
    job = RUNNER.submit({"task_id": "ioi_john_mary"}, job_type="report")
    done = _wait(job.job_id, timeout=300)
    assert done.status == "complete", done.error
    return done


@pytest.fixture(scope="module")
def study_results(bridge):
    """A real 2-prompt study result, stored like a completed study job.

    The study job validation requires at least 4 prompts, so this fixture calls
    the same analysis functions the study job calls for 2 prompts and stores
    the result in the isolated test DB.
    """
    prompt_set = generate_prompts(2, 0)
    per_prompt = []
    for p in prompt_set["prompts"]:
        swap = p["corruptions"]["swap"]
        assert swap
        logits_clean, cache = comps.run_clean(p["clean_text"])
        effect = comps.denoise_effects(
            swap, p["candidate_a"], p["candidate_b"], logits_clean, cache
        )
        atp = compute_atp(
            swap, p["candidate_a"], p["candidate_b"], logits_clean, cache
        )
        per_prompt.append(
            {
                "id": p["id"],
                "template_id": p["template_id"],
                "s2_role": p["s2_role"],
                "D_clean": effect["D_clean"],
                "P_clean": effect["P_clean"],
                "baselines": {"swap": {"D_corr": effect["D_corr"], "excluded": False}},
                "rec_d": {"swap": effect["rec_d"]},
                "rec_p": {"swap": effect["rec_p"]},
                "atp": atp["rec_atp"],
                "dla": None,
                "abl_zero": None,
                "abl_mean": None,
            }
        )
    results = {
        "job_type": "study",
        "params": {"n_prompts": 2, "seed": 0},
        "direction": "denoising",
        "metrics": {},
        "exclusion_rule": "test fixture",
        "prompt_set": prompt_set,
        "model_accuracy": 1.0,
        "exclusions": {"swap": 0, "third": 0, "resample": 0, "abl_mean": 0},
        "per_prompt": per_prompt,
        "baselines_per_prompt": [],
        "summary": {},
        "provenance": collect_provenance(),
        "elapsed_seconds": 0.0,
    }
    exp_id = "stage5study1"
    db.save_experiment(
        {"job_type": "study", "n_prompts": 2, "seed": 0, "params": {"n_prompts": 2, "seed": 0}},
        results,
        "complete",
        exp_id=exp_id,
    )
    return exp_id, results


def _sweep_payload(job: JobRecord) -> dict:
    res = job.results
    return sweep_data(
        mode=res["mode"],
        clean_text=CLEAN,
        corrupted_text=CORRUPTED,
        candidate_a=CAND_A,
        candidate_b=CAND_B,
        baselines=res["baselines"],
        recovery=res["recovery"],
        d_patched=res["d_patched"],
        provenance=res.get("provenance"),
    )


@pytest.fixture(scope="module")
def payloads(intervention_job, sweep_job, study_results, report_job):
    _, study = study_results
    dla = compute_dla(CLEAN, CAND_A, CAND_B)
    lens = compute_logit_lens(CLEAN, CAND_A, CAND_B, 5)
    prov = collect_provenance()
    return {
        "intervention": (
            intervention_data(intervention_job.config, intervention_job.results),
            ("D_clean", "Recovery"),
        ),
        "sweep": (_sweep_payload(sweep_job), ("baseline D_clean", "recovery")),
        "study_subset": (study_subset_data(study), ("Rec_D", "Rec_AtP")),
        "report": (
            report_data(report_job.results, report_job.config),
            ("dla: DLA embed", "lens: lens block 11 D", "positions:", "heads:"),
        ),
        "dla": (
            dla_data(
                text=CLEAN,
                candidate_a=CAND_A,
                candidate_b=CAND_B,
                dla_result=dla,
                provenance=prov,
            ),
            ("DLA embed", "DLA reconciliation"),
        ),
        "lens": (
            lens_data(
                text=CLEAN,
                candidate_a=CAND_A,
                candidate_b=CAND_B,
                lens_result=lens,
                provenance=prov,
            ),
            ("lens block 11 D", "final identity"),
        ),
    }


def _imports(text: str) -> set[str]:
    tree = ast.parse(text)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def embedded_data(text: str) -> dict:
    """Recover the ONE JSON literal from a generated script without running it."""
    tree = ast.parse(text)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DATA"
        ):
            call = node.value
            assert isinstance(call, ast.Call)
            literal = call.args[0]
            return json.loads(ast.literal_eval(literal))
    raise AssertionError("generated script has no DATA literal")


@pytest.mark.parametrize("kind", list(KINDS))
def test_generated_scripts_compile_and_import_only_allowed(kind, payloads):
    data, _ = payloads[kind]
    script = generate(kind, data, "sample")
    compile(script, f"<{kind}>", "exec")
    assert _imports(script) <= ALLOWED_IMPORTS
    payload = embedded_data(script)
    assert payload["mode_default"] == "sample"
    assert "tolerances" in payload and "dla_term" in payload["tolerances"]
    # one literal: the DATA assignment uses json.loads(<one python string>)
    assert script.count("json.loads(") == 1


@pytest.mark.parametrize("kind", list(KINDS))
def test_sample_script_runs_and_prints_expected_quantities(kind, payloads, tmp_path):
    data, expected = payloads[kind]
    script = generate(kind, data, "sample")
    path = tmp_path / f"repro_{kind}.py"
    path.write_text(script, encoding="utf-8")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(path), "--mode", "sample"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT,
    )
    elapsed = time.time() - started
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "PASS:" in proc.stdout
    for needle in expected:
        assert needle in proc.stdout, f"{kind} output missing {needle!r}"
    print(f"{kind} sample script: {elapsed:.1f}s")


def test_hostile_strings_stay_data(payloads):
    data, _ = payloads["sweep"]
    hostile = 'x"; import os; os.system("echo pwned"); """\n\\\u2603 \'quoted\''
    tampered = copy.deepcopy(data)
    tampered["clean_text"] = hostile
    script = generate("sweep", tampered, "sample")
    compile(script, "<hostile>", "exec")
    parsed = embedded_data(script)
    assert parsed["clean_text"] == hostile
    assert "import os" not in script.split("DATA =", 1)[0]


def test_tampering_with_stored_number_fails(intervention_job, tmp_path):
    data = intervention_data(intervention_job.config, intervention_job.results)
    tampered = copy.deepcopy(data)
    tampered["stored"]["metrics"]["D_clean"] += 0.5
    script = generate("intervention", tampered, "sample")
    path = tmp_path / "tampered.py"
    path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(path), "--mode", "sample"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT,
    )
    assert proc.returncode != 0
    assert "FAIL:" in proc.stdout


def test_reproduce_endpoint_rejects_unknown_kind(client):
    r = client.post("/api/reproduce", json={"kind": "everything"})
    assert r.status_code == 422
    assert "unknown reproduce kind" in r.json()["detail"]


def test_reproduce_endpoint_rejects_wrong_job_type(client, intervention_job):
    r = client.post(
        "/api/reproduce", json={"kind": "sweep", "job_id": intervention_job.job_id}
    )
    assert r.status_code == 422
    assert "expected a sweep" in r.json()["detail"] or "needs a 'sweep' job" in r.json()["detail"]


def test_reproduce_endpoint_needs_job_id(client):
    r = client.post("/api/reproduce", json={"kind": "sweep"})
    assert r.status_code == 422
    r2 = client.post("/api/reproduce", json={"kind": "dla", "task_id": "ioi_john_mary", "run": "clean"})
    assert r2.status_code == 200


def test_reproduce_endpoint_returns_script_and_filename(client, sweep_job):
    r = client.post(
        "/api/reproduce", json={"kind": "sweep", "job_id": sweep_job.job_id, "mode": "full"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/x-python")
    assert r.headers["content-disposition"].startswith("attachment; filename=")
    assert "circuitscope_repro_sweep_" in r.headers["content-disposition"]
    assert "Embedded default mode: full" in r.text
    compile(r.text, "<endpoint>", "exec")


def test_reproduce_dla_and_lens_are_computed_at_request_time(client):
    dla = client.post(
        "/api/reproduce",
        json={"kind": "dla", "task_id": "ioi_john_mary", "pair_id": None, "run": "clean"},
    )
    assert dla.status_code == 200, dla.text
    assert "DLA embed" in dla.text or "DLA " in dla.text
    lens = client.post(
        "/api/reproduce",
        json={
            "kind": "lens",
            "text": CLEAN,
            "candidate_a": CAND_A,
            "candidate_b": CAND_B,
        },
    )
    assert lens.status_code == 200, lens.text
    assert "lens" in lens.text
