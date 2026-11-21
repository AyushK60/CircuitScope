"""Tests for exact per-component effects.

- Consistency gate: per-head Rec_D[swap] must equal the layer_head
  sweep results within 1e-4 for the default IOI pair.
- Ablation sanity: replacing a component's activation by ITSELF changes
  nothing; zero ablation actually intervenes; helpers do not mutate caches.
"""

from __future__ import annotations

import time

import pytest
import torch

from backend.app.analysis import components as comps
from backend.app.experiments import sweeps  # noqa: F401  (registers the "sweep" job type)
from backend.app.experiments.jobs import RUNNER
from backend.app.tasks import registry as task_registry

task_registry.load_all()  # the consistency gate resolves the task from the registry
from backend.app.config import MAX_LAYERS
from backend.app.model_runtime import engine
from backend.app.model_runtime.loader import ADAPTER

IOI_CLEAN = "John and Mary went to the shop. John gave a book to"
IOI_SWAP = "John and Mary went to the shop. Mary gave a book to"
CAND_A = " Mary"
CAND_B = " John"
TOL = 1e-4


def _wait_sweep(job_id, timeout=300.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = RUNNER.get(job_id)
        if job is not None and job.status in ("complete", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("sweep did not finish in time")


def test_consistency_gate_with_stage2_layer_head_sweep(bridge):
    """For the default IOI pair, per-head Rec_D[swap] from components.py must
    equal the layer_head sweep within 1e-4 (pre-registered gate)."""
    job = RUNNER.submit({"task_id": "ioi_john_mary", "mode": "layer_head"}, job_type="sweep")
    j = _wait_sweep(job.job_id)
    assert j.status == "complete", j.error
    sweep_recovery = j.results["recovery"]

    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    eff = comps.denoise_effects(IOI_SWAP, CAND_A, CAND_B, logits_clean, cache)
    max_diff = 0.0
    for l in range(MAX_LAYERS):
        for h in range(12):
            ref = sweep_recovery[l][h]
            ours = eff["rec_d"][l * 12 + h]
            assert ours is not None
            max_diff = max(max_diff, abs(ours - ref))
    assert max_diff <= TOL, f"consistency gate failed: max |diff| = {max_diff:.2e} > {TOL}"


def test_both_metrics_from_same_runs(bridge):
    """D and P for every component come from the same patched runs (the spec's
    requirement): they are computed together and None only when denominators
    are zero."""
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    eff = comps.denoise_effects(IOI_SWAP, CAND_A, CAND_B, logits_clean, cache)
    assert eff["denominator_d"] != 0.0 and eff["denominator_p"] != 0.0
    assert len(eff["rec_d"]) == comps.N_COMPONENTS == 156
    assert len(eff["rec_p"]) == 156
    assert all(v is not None for v in eff["rec_d"])
    assert all(v is not None for v in eff["rec_p"])
    # component order: heads first, MLPs last
    assert comps.COMPONENT_NAMES[0] == "head 0.0"
    assert comps.COMPONENT_NAMES[143] == "head 11.11"
    assert comps.COMPONENT_NAMES[144] == "mlp.0"
    assert comps.COMPONENT_NAMES[155] == "mlp.11"


def test_ablated_logits_identity_replacement_changes_nothing(bridge):
    """Replacing a component's activation by its own clean values must
    reproduce the clean logits (sanity for the ablation helper)."""
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    hook = "blocks.5.attn.hook_z"
    logits_same = engine.ablated_logits(
        IOI_CLEAN, hook, list(range(cache[hook].shape[1])), None, cache[hook].clone()
    )
    assert torch.allclose(logits_clean, logits_same, atol=1e-4)


def test_zero_ablation_intervenes_and_is_finite(bridge):
    """Zero-ablating every component must actually change the logits for at
    least one component (a no-op everywhere would mean the helper is broken)
    and every Abl_D must be finite."""
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    out = comps.zero_ablation_effects(IOI_CLEAN, CAND_A, CAND_B, logits_clean, cache)
    abl = out["abl_d"]
    assert len(abl) == 156
    assert all(v is not None and torch.isfinite(torch.tensor(v)) for v in abl)
    assert any(abs(v) > 1e-6 for v in abl)


def test_mean_ablation_needs_family(bridge):
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    with pytest.raises(ValueError):
        comps.mean_ablation_effects(IOI_CLEAN, CAND_A, CAND_B, logits_clean, cache, [])


def test_mean_ablation_with_real_family(bridge):
    """Mean ablation with a one-prompt family (the leave-one-out mean of that
    single other prompt) must run and stay finite."""
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    other_logits, other_cache = comps.run_clean(IOI_SWAP)  # same token count
    out = comps.mean_ablation_effects(
        IOI_CLEAN, CAND_A, CAND_B, logits_clean, cache, [other_cache]
    )
    assert len(out["abl_d"]) == 156
    assert all(v is not None for v in out["abl_d"])


def test_zero_and_mean_helpers_do_not_mutate_caches(bridge):
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    _, other_cache = comps.run_clean(IOI_SWAP)
    snapshot = {k: v.clone() for k, v in cache.items()}
    other_snapshot = {k: v.clone() for k, v in other_cache.items()}

    comps.zero_ablation_effects(IOI_CLEAN, CAND_A, CAND_B, logits_clean, cache)
    comps.mean_ablation_effects(
        IOI_CLEAN, CAND_A, CAND_B, logits_clean, cache, [other_cache]
    )
    for k in cache:
        assert torch.equal(cache[k], snapshot[k]), f"cache mutated: {k}"
    for k in other_cache:
        assert torch.equal(other_cache[k], other_snapshot[k]), f"family cache mutated: {k}"


def test_interpolated_patch_alpha_zero_is_corrupted_alpha_one_is_clean(bridge):
    """The linearity-gate helper's endpoints must be exact: alpha=0 leaves the
    corrupted run unchanged, alpha=1 equals the exact clean patch."""
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    tok_a = ADAPTER.token_id_for_str(CAND_A)
    tok_b = ADAPTER.token_id_for_str(CAND_B)

    def d_of(logits):
        return float(logits[0, -1, tok_a] - logits[0, -1, tok_b])

    toks = ADAPTER.bridge().to_tokens(IOI_SWAP)
    d_corr = d_of(ADAPTER.bridge()(toks))
    d0 = d_of(engine.interpolated_patch_logits(toks, cache, "blocks.5.attn.hook_z", list(range(14)), [3], 0.0))
    d1 = d_of(engine.interpolated_patch_logits(toks, cache, "blocks.5.attn.hook_z", list(range(14)), [3], 1.0))
    assert d0 == pytest.approx(d_corr, abs=1e-4)
    exact = engine.patched_logits(toks, cache, "blocks.5.attn.hook_z", list(range(14)), [3])
    # alpha=1 recomputes src via a_corr + 1.0*(src - a_corr): identical in exact
    # arithmetic, so equality holds to float32 rounding noise only.
    assert d1 == pytest.approx(d_of(exact), abs=1e-4)


def test_ablated_logits_rejects_bad_hook(bridge):
    zeros = torch.zeros(1, 14, 12, 64)
    with pytest.raises(Exception):
        engine.ablated_logits(IOI_CLEAN, "blocks.99.attn.hook_z", [0], [0], zeros)
