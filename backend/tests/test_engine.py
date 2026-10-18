"""Scientific engine tests: real model, real capture, real patch, no-op controls.

Numerical tolerance: GPT-2 fp32 CPU inference is deterministic run-to-run on the
same machine/library versions, but we still use documented tolerances
(1e-4 absolute for D values) rather than exact equality.
"""

from __future__ import annotations

import pytest
import torch

from backend.app.config import PATCHABLE_HOOKS, validate_hook_name
from backend.app.experiments.jobs import (
    ExperimentValidationError,
    execute_experiment,
    validate_experiment_config,
)
from backend.app.experiments.metrics import compare_metrics, logit_diff
from backend.app.model_runtime import engine
from backend.app.model_runtime.loader import ADAPTER

TOL = 1e-4


def test_metadata_matches_gpt2_small(bridge):
    meta = ADAPTER.metadata()
    assert meta["n_layers"] == 12
    assert meta["n_heads"] == 12
    assert meta["d_model"] == 768
    assert meta["d_head"] == 64
    assert meta["d_vocab"] == 50257
    assert meta["model_id"] == "gpt2"


def test_tokenization_leading_space(bridge):
    toks = ADAPTER.tokenize("John gave a book to Mary")
    assert " Mary" in "".join(toks["tokens"]) or any("Mary" in t for t in toks["tokens"])
    # " Mary" must be a single token for the IOI candidates to be valid
    assert ADAPTER.token_id_for_str(" Mary") is not None
    assert ADAPTER.token_id_for_str(" John") is not None
    try:
        ADAPTER.token_id_for_str(" shopkeeper")  # multi-token candidate
        multi = False
    except ValueError:
        multi = True
    assert multi, "multi-token candidate should be rejected"


def test_predict_matches_raw_hf(bridge):
    """Cross-check the bridge's logits against raw Hugging Face transformers.

    TransformerLens prepends a BOS token; HF does not, so compare per-position
    logits on the shared positions via the HF-with-BOS encoding.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    text = "The capital of France is"
    hf_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()
    with torch.no_grad():
        enc = hf_tok(text, return_tensors="pt")
        # TransformerBridge prepends BOS; do the same for HF so encodings match.
        ids_bos = torch.tensor([[50256] + enc["input_ids"][0].tolist()])
        hf_logits = hf_model(input_ids=ids_bos).logits
        pred = engine.predict(text, top_k=5)
        b = ADAPTER.bridge()
        tl_logits = b(b.to_tokens(text))
    assert tl_logits.shape == hf_logits.shape
    assert torch.allclose(tl_logits[0], hf_logits[0], atol=2e-3)
    # top prediction must agree with HF's top prediction at the final position
    hf_top = hf_tok.decode([int(hf_logits[0, -1].argmax())])
    assert pred["top_predictions"][0]["token"] == hf_top


def test_baseline_reproducible(bridge):
    a = engine.predict("John and Mary went to the shop. John gave a book to")
    b = engine.predict("John and Mary went to the shop. John gave a book to")
    da, db = a["top_predictions"][0]["logit"], b["top_predictions"][0]["logit"]
    assert abs(da - db) < TOL


def test_capture_shapes(bridge):
    summaries, cache = engine.capture(
        "Hello world", ["blocks.5.hook_out", "blocks.5.attn.hook_pattern", "hook_embed"]
    )
    assert summaries["blocks.5.hook_out"]["shape"] == [1, 3, 768]
    assert summaries["blocks.5.attn.hook_pattern"]["shape"] == [1, 12, 3, 3]
    assert summaries["hook_embed"]["shape"] == [1, 3, 768]


def test_unsupported_hook_rejected(bridge):
    import pytest

    with pytest.raises(Exception):
        engine.capture("Hello", ["made.up.hook"])
    with pytest.raises(Exception):
        engine.capture("Hello", ["blocks.5.attn.hook_result"])  # in hook_dict but not capturable


def test_noop_patch_preserves_output(bridge, ioi):
    """Patching an activation into itself must reproduce the baseline within tolerance."""
    _, cache = bridge_run_cache(ioi["corrupted"])
    logits_noop = engine.run_patched(
        ioi["corrupted"], cache, "blocks.5.hook_out", [ioi["s2_position"]]
    )["logits"]
    logits_base = bridge_run(ioi["corrupted"])
    assert abs(logit_diff(logits_noop, ioi["tok_a"], ioi["tok_b"]) - logit_diff(logits_base, ioi["tok_a"], ioi["tok_b"])) < TOL


def bridge_run(text):
    b = ADAPTER.bridge()
    return b(b.to_tokens(text))


def bridge_run_cache(text):
    b = ADAPTER.bridge()
    return b.run_with_cache(b.to_tokens(text))


def test_real_patch_changes_computation(bridge, ioi):
    """The verified IOI patch: clean resid@block5@S2 -> corrupted run must move D strongly."""
    _, clean_cache = bridge_run_cache(ioi["clean"])
    logits_corr = bridge_run(ioi["corrupted"])
    patched = engine.run_patched(
        ioi["corrupted"], clean_cache, "blocks.5.hook_out", [ioi["s2_position"]]
    )
    d_corr = logit_diff(logits_corr, ioi["tok_a"], ioi["tok_b"])
    d_patch = logit_diff(patched["logits"], ioi["tok_a"], ioi["tok_b"])
    # Verified reference values (see scripts/reference_patch.py): D_clean ~ +2.51,
    # D_corrupted ~ -4.18, patched recovery ~0.9.
    assert d_corr < 0, "corrupted run should prefer ' John'"
    assert d_patch > 0, "patched run should move preference toward ' Mary'"
    assert d_patch != d_corr


def test_clean_cache_not_mutated(bridge, ioi):
    _, clean_cache = bridge_run_cache(ioi["clean"])
    before = clean_cache["blocks.5.hook_out"].clone()
    engine.run_patched(ioi["corrupted"], clean_cache, "blocks.5.hook_out", [ioi["s2_position"]])
    assert torch.equal(clean_cache["blocks.5.hook_out"], before)


def test_hooks_do_not_leak(bridge, ioi):
    """After a patched run, a plain run must equal the pre-patch baseline (hooks reset)."""
    _, clean_cache = bridge_run_cache(ioi["clean"])
    base = bridge_run(ioi["corrupted"])
    engine.run_patched(ioi["corrupted"], clean_cache, "blocks.5.hook_out", [ioi["s2_position"]])
    after = bridge_run(ioi["corrupted"])
    assert torch.allclose(base, after, atol=1e-5)


def test_invalid_positions_rejected(bridge, ioi):
    _, clean_cache = bridge_run_cache(ioi["clean"])
    import pytest

    with pytest.raises(Exception):
        engine.run_patched(ioi["corrupted"], clean_cache, "blocks.5.hook_out", [999])


def test_metric_math(bridge):
    import torch as t

    clean = t.tensor([[[0.0, 3.0]]])
    corr = t.tensor([[[-1.0, 1.0]]])
    patch = t.tensor([[[0.0, 2.0]]])
    m = compare_metrics(clean, corr, patch, token_a=1, token_b=0)
    assert abs(m["D_clean"] - 3.0) < 1e-6
    assert abs(m["D_corrupted"] - 2.0) < 1e-6
    assert abs(m["D_patched"] - 2.0) < 1e-6
    assert abs(m["Delta"] - 0.0) < 1e-6
    assert abs(m["Recovery"] - 0.0) < 1e-6
    # near-zero denominator guard
    m2 = compare_metrics(clean, clean, patch, token_a=1, token_b=0)
    assert m2["Recovery"] is None


def test_execute_experiment_end_to_end(bridge, ioi):
    cfg = {
        "clean_text": ioi["clean"],
        "corrupted_text": ioi["corrupted"],
        "hook": "blocks.5.hook_out",
        "positions": [ioi["s2_position"]],
        "candidate_a": " Mary",
        "candidate_b": " John",
    }
    res = execute_experiment(cfg)
    assert res["runs"]["clean"][0]["token"].strip() == "Mary"
    assert res["runs"]["corrupted"][0]["token"].strip() == "John"
    m = res["metrics"]
    assert m["D_corrupted"] < 0 < m["D_patched"]
    assert 0.5 < m["Recovery"] < 1.5


def test_validate_experiment_config_rejects_bad_hooks():
    with pytest.raises(ExperimentValidationError):
        validate_experiment_config(
            {
                "clean_text": "a",
                "corrupted_text": "b",
                "hook": "not.a.hook",
                "positions": [0],
                "candidate_a": " x",
                "candidate_b": " y",
            }
        )


def test_hook_allowlist_membership():
    assert "blocks.5.hook_out" in PATCHABLE_HOOKS
    assert "blocks.11.mlp.hook_out" in PATCHABLE_HOOKS
    assert not validate_hook_name("blocks.5.attn.hook_result", patchable=True)
    assert not validate_hook_name("blocks.12.hook_out", patchable=True)
