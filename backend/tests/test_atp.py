"""Tests for attribution patching.

- Linearity gate: the central difference of D around alpha=0 (per unit of full
  patch) must match AtP_D for all 156 components with Pearson r > 0.999, and
  the 30 largest |AtP_D| components agree within max(5e-3, 2% relative).
- No grad leak: after compute_atp, a normal request still runs without
  building a graph and no hooks/state leak.
"""

from __future__ import annotations

import math

import pytest
import torch

from backend.app.analysis import components as comps
from backend.app.analysis.atp import compute_atp
from backend.app.model_runtime import engine
from backend.app.model_runtime.loader import ADAPTER

IOI_CLEAN = "John and Mary went to the shop. John gave a book to"
IOI_SWAP = "John and Mary went to the shop. Mary gave a book to"
CAND_A = " Mary"
CAND_B = " John"
ALPHA = 0.01


def _pearson(x, y) -> float:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc * yc).sum() / math.sqrt(float((xc * xc).sum() * (yc * yc).sum())))


def test_atp_linearity_gate(bridge):
    """Central difference (D(+alpha) - D(-alpha)) / (2*alpha) ≈ AtP_D."""
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    atp = compute_atp(IOI_SWAP, CAND_A, CAND_B, logits_clean, cache)
    tok_a = ADAPTER.token_id_for_str(CAND_A)
    tok_b = ADAPTER.token_id_for_str(CAND_B)

    def d_of(logits):
        return float(logits[0, -1, tok_a] - logits[0, -1, tok_b])

    seq = cache["blocks.0.attn.hook_z"].shape[1]
    positions = list(range(seq))
    central = []
    for l in range(12):
        for h in range(12):
            hook = f"blocks.{l}.attn.hook_z"
            d_plus = d_of(engine.interpolated_patch_logits(IOI_SWAP, cache, hook, positions, [h], ALPHA))
            d_minus = d_of(engine.interpolated_patch_logits(IOI_SWAP, cache, hook, positions, [h], -ALPHA))
            central.append((d_plus - d_minus) / (2 * ALPHA))
    for l in range(12):
        hook = f"blocks.{l}.mlp.hook_out"
        d_plus = d_of(engine.interpolated_patch_logits(IOI_SWAP, cache, hook, positions, None, ALPHA))
        d_minus = d_of(engine.interpolated_patch_logits(IOI_SWAP, cache, hook, positions, None, -ALPHA))
        central.append((d_plus - d_minus) / (2 * ALPHA))

    assert len(central) == 156
    r = _pearson(atp["atp_d"], central)
    assert r > 0.999, f"linearity gate failed: Pearson r = {r:.6f}"

    # 30 largest |AtP_D| components: agree within max(5e-3, 2% relative)
    order = sorted(range(156), key=lambda i: -abs(atp["atp_d"][i]))[:30]
    worst = 0.0
    for i in order:
        ref = central[i]
        got = atp["atp_d"][i]
        tol = max(5e-3, 0.02 * abs(ref))
        worst = max(worst, abs(got - ref))
        assert abs(got - ref) <= tol, (
            f"component {i} ({comps.COMPONENT_NAMES[i]}): AtP {got:.6f} vs central diff {ref:.6f} "
            f"(tol {tol:.6f})"
        )


def test_atp_no_grad_leak(bridge):
    """After compute_atp: grad mode back to disabled; a normal forward builds
    no graph; a patched run's outputs also require no grad."""
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    compute_atp(IOI_SWAP, CAND_A, CAND_B, logits_clean, cache)

    assert torch.is_grad_enabled() is False  # global grad mode restored

    bridge_ = ADAPTER.bridge()
    logits = bridge_(bridge_.to_tokens(IOI_CLEAN))
    assert logits.requires_grad is False
    patched = engine.patched_logits(IOI_SWAP, cache, "blocks.5.attn.hook_z", [9], [0])
    assert patched.requires_grad is False


def test_atp_denominator_and_recovery_shape(bridge):
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    atp = compute_atp(IOI_SWAP, CAND_A, CAND_B, logits_clean, cache)
    assert len(atp["atp_d"]) == 156
    assert len(atp["rec_atp"]) == 156
    assert atp["denominator"] == pytest.approx(atp["D_clean"] - atp["D_corr"], abs=1e-6)
    assert atp["denominator"] != 0.0
    assert all(v is not None for v in atp["rec_atp"])
    assert "first-order" in atp["note"]


def test_interpolated_patch_logits_rejects_bad_hook(bridge):
    logits_clean, cache = comps.run_clean(IOI_CLEAN)
    with pytest.raises(Exception):
        engine.interpolated_patch_logits(IOI_SWAP, cache, "blocks.0.attn.hook_result", [0], None, 0.5)
