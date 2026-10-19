# SCIENTIFIC VALIDATION

All numbers measured 2026-10-30, GPT-2 Small (124.4M params), torch 2.15.0+cpu,
transformer-lens 4.3.0, transformers 5.22.0, Windows 11, Python 3.11.9.

## 1. Bridge correctness (cross-checked against raw HF)

TransformerBridge prepends BOS (50256). With HF given the same BOS-prefixed
encoding, final-position logits agree within 2e-3 max abs difference
(`backend/tests/test_engine.py::test_predict_matches_raw_hf`).

## 2. Independent reference agreement (M2 gate)

`scripts/reference_patch.py` implements the identical patch with raw HF
transformers + PyTorch hooks — no TransformerLens:

| Metric | Reference | Backend (pytest) | Agreement |
|---|---|---|---|
| D_clean | 2.5128 | 2.5128 ± 5e-3 | ✓ |
| D_corrupted | −4.1768 | −4.1768 ± 5e-3 | ✓ |
| D_patched | 1.9426 | 1.9426 ± 5e-3 | ✓ |
| Recovery | 0.9148 | 0.9148 ± 5e-3 | ✓ |

Patch: `blocks.5.hook_out` (residual after block 5), position 9 (second
subject " John"), IOI prompts from `examples/prepared_ioi_examples.json`.

## 3. Controls

- **No-op patch** (patching an activation into itself): output equals baseline
  within 1e-4 (`test_noop_patch_preserves_output`).
- **Baseline reproducibility**: repeated runs identical within 1e-4.
- **Hook cleanup**: post-patch plain run equals pre-patch baseline within 1e-5
  (`test_hooks_do_not_leak`).
- **Baseline immutability**: clean cache bit-identical after patched run
  (`test_clean_cache_not_mutated`).

## 4. Negative cases (honest failure)

- Multi-token candidates rejected (` shopkeeper` → ValueError).
- Mismatched clean/corrupted token lengths → experiment `failed` with
  "token-aligned" error, surfaced in UI.
- Unsupported hooks rejected server-side (422); `attn.hook_result` excluded
  because TransformerLens 4.3.0 does not capture it in `run_with_cache`.

## 5. Verified intervention sweep (blocks.5–10 resid @ pos 9 / pos 13)

Position 9 (S2): recovery 0.92 (L5), 0.88 (L6), 0.67 (L7), then decays.
Position 13 (END): 0.78 (L8), 0.95 (L9), 0.97 (L10).
MLP-out patches: ≈0 recovery. Consistent with known IOI circuit structure;
presented in-app as measurements, not claims of mechanism.

## 6. Reproduction

```bat
.venv\Scripts\python -m pytest backend\tests -q
.venv\Scripts\python scripts\reference_patch.py
```

Tolerance policy: fp32 CPU inference is deterministic on a fixed stack;
5e-3 absolute covers cross-version drift. Never loosen a test to hide a bug —
re-measure and update reference values with a dated note instead.
