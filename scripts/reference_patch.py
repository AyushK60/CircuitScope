"""Independent reference implementation of the IOI activation patch.

Uses raw Hugging Face transformers + PyTorch forward hooks only — no
TransformerLens — so the backend's patched-run numbers can be verified against
a genuinely separate code path.

Patching semantics verified against the backend:
  backend "blocks.{L}.hook_out" (residual stream after block L)
    == HF GPT2Model transformer.h[L] output hidden state at chosen positions.

Usage: python scripts/reference_patch.py  -> prints D_clean, D_corrupted, D_patched.
"""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CLEAN = "John and Mary went to the shop. John gave a book to"
CORRUPTED = "John and Mary went to the shop. Mary gave a book to"
LAYER = 5
POSITION = 9  # second subject " John"
CAND_A = " Mary"
CAND_B = " John"


def main() -> dict:
    torch.set_grad_enabled(False)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    tid_a = tok(CAND_A, add_special_tokens=False)["input_ids"]
    tid_b = tok(CAND_B, add_special_tokens=False)["input_ids"]
    assert len(tid_a) == 1 and len(tid_b) == 1, "candidates must be single tokens"
    tid_a, tid_b = tid_a[0], tid_b[0]

    def d_of(logits: torch.Tensor) -> float:
        return float(logits[0, -1, tid_a] - logits[0, -1, tid_b])

    # TransformerBridge prepends BOS (50256); do the same here so token
    # positions match the backend exactly.
    def encode(text: str) -> dict[str, torch.Tensor]:
        ids = [50256] + tok(text, add_special_tokens=False)["input_ids"]
        return {"input_ids": torch.tensor([ids])}

    enc_clean = encode(CLEAN)
    enc_corr = encode(CORRUPTED)
    assert enc_clean["input_ids"].shape == enc_corr["input_ids"].shape

    out_clean = model(**enc_clean)
    out_corr = model(**enc_corr)
    d_clean, d_corr = d_of(out_clean.logits), d_of(out_corr.logits)

    # Capture the clean residual at (LAYER, POSITION) using a forward hook.
    captured: dict[str, torch.Tensor] = {}

    def clean_hook(module, args, output):
        # transformers 5.x GPT2Block forward hook returns the hidden-state tensor
        # directly: [batch, seq, d_model]
        captured["src"] = output[0, POSITION, :].detach().clone()

    handle = model.transformer.h[LAYER].register_forward_hook(clean_hook, with_kwargs=False)
    model(**enc_clean)
    handle.remove()

    def patch_hook(module, args, output):
        hs = output.clone()
        hs[0, POSITION, :] = captured["src"]
        return hs

    handle = model.transformer.h[LAYER].register_forward_hook(patch_hook)
    out_patched = model(**enc_corr)
    handle.remove()
    d_patched = d_of(out_patched.logits)

    recovery = (d_patched - d_corr) / (d_clean - d_corr)
    print(f"D_clean     {d_clean:.4f}")
    print(f"D_corrupted {d_corr:.4f}")
    print(f"D_patched   {d_patched:.4f}")
    print(f"Recovery    {recovery:.4f}")
    return {
        "D_clean": d_clean,
        "D_corrupted": d_corr,
        "D_patched": d_patched,
        "Recovery": recovery,
    }


if __name__ == "__main__":
    main()
