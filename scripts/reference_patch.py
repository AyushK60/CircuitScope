"""Independent reference implementation of the IOI activation patch.

Uses raw Hugging Face transformers + PyTorch forward hooks only - no
TransformerLens - so the backend's patched-run numbers can be verified against
a genuinely separate code path.

Patching semantics verified against the backend:
  backend "blocks.{L}.hook_out" (residual stream after block L)
    == HF GPT2Model transformer.h[L] output hidden state at chosen positions.

Usage:
  python scripts/reference_patch.py
      # prints D_clean, D_corrupted, D_patched, Recovery for the prepared IOI
      # example (the original CLI behaviour, unchanged)
  python scripts/reference_patch.py --api-json FILE
      # compare against a saved experiment result (GET
      # /api/experiments/{id} or the results dict itself). Exits non-zero on
      # any difference > TOL. This is the offline mode the verification panel
      # uses; the file supplies every input, so no client string reaches the
      # argv.

Tolerance: 5e-3 absolute on D_clean, D_corrupted, D_patched and Recovery.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CLEAN = "John and Mary went to the shop. John gave a book to"
CORRUPTED = "John and Mary went to the shop. Mary gave a book to"
LAYER = 5
POSITION = 9  # second subject " John"
CAND_A = " Mary"
CAND_B = " John"
BOS_ID = 50256
TOL = 5e-3
N_HEADS = 12
D_HEAD = 64


def load_result(path: str) -> dict:
    """Accept either a full experiment record or a bare results dict."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if "results" in payload and isinstance(payload["results"], dict):
        return payload["results"]
    return payload


def parse_hook(name: str) -> tuple[str, int]:
    parts = name.split(".")
    layer = int(parts[1])
    if name.endswith(".attn.hook_z"):
        return "z", layer
    if name.endswith(".attn.hook_out"):
        return "attn_out", layer
    if name.endswith(".mlp.hook_out"):
        return "mlp_out", layer
    if name.endswith(".hook_out"):
        return "block_out", layer
    raise SystemExit(f"unsupported hook for reference_patch: {name!r}")


def unwrap(output):
    return output[0] if isinstance(output, tuple) else output


def rewrap(output, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    return new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-json",
        dest="api_json",
        default=None,
        help="saved experiment results; compare against the reference implementation",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    cfg: dict = {
        "clean_text": CLEAN,
        "corrupted_text": CORRUPTED,
        "candidate_a": CAND_A,
        "candidate_b": CAND_B,
        "hook": f"blocks.{LAYER}.hook_out",
        "positions": [POSITION],
        "heads": None,
    }
    metrics_stored = None
    if args.api_json:
        result = load_result(args.api_json)
        cfg = dict(result.get("config") or cfg)
        patch = result.get("patch") or {}
        cfg["hook"] = patch.get("hook", cfg.get("hook"))
        cfg["positions"] = list(patch.get("positions") or cfg.get("positions") or [])
        cfg["heads"] = patch.get("heads", cfg.get("heads"))
        metrics_stored = result.get("metrics")
        if not metrics_stored:
            raise SystemExit("--api-json file has no metrics to compare against")

    tid_a = tok(cfg["candidate_a"], add_special_tokens=False)["input_ids"]
    tid_b = tok(cfg["candidate_b"], add_special_tokens=False)["input_ids"]
    assert len(tid_a) == 1 and len(tid_b) == 1, "candidates must be single tokens"
    tid_a, tid_b = tid_a[0], tid_b[0]

    def d_of(logits: torch.Tensor) -> float:
        return float(logits[0, -1, tid_a] - logits[0, -1, tid_b])

    # TransformerBridge prepends BOS (50256); do the same here so token
    # positions match the backend exactly.
    def encode(text: str) -> dict[str, torch.Tensor]:
        ids = [BOS_ID] + tok(text, add_special_tokens=False)["input_ids"]
        return {"input_ids": torch.tensor([ids])}

    enc_clean = encode(cfg["clean_text"])
    enc_corr = encode(cfg["corrupted_text"])
    assert enc_clean["input_ids"].shape == enc_corr["input_ids"].shape

    out_clean = model(**enc_clean)
    out_corr = model(**enc_corr)
    d_clean, d_corr = d_of(out_clean.logits), d_of(out_corr.logits)

    kind, layer = parse_hook(cfg["hook"])
    positions = list(cfg["positions"])
    heads = cfg.get("heads")
    if heads is None:
        heads = list(range(N_HEADS))

    captured: dict[str, torch.Tensor] = {}

    def capture_fwd(module, fargs, output):
        captured["src"] = unwrap(output).detach().clone()
        return output

    def capture_pre(module, fargs):
        captured["src"] = fargs[0].detach().clone()

    if kind == "z":
        handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(capture_pre)
    elif kind == "attn_out":
        handle = model.transformer.h[layer].attn.register_forward_hook(capture_fwd)
    elif kind == "mlp_out":
        handle = model.transformer.h[layer].mlp.register_forward_hook(capture_fwd)
    else:
        handle = model.transformer.h[layer].register_forward_hook(capture_fwd)
    try:
        model(**enc_clean)
    finally:
        handle.remove()

    if kind == "z":
        def patch_hook(module, fargs):
            x = fargs[0].clone()
            for p in positions:
                for h in heads:
                    x[0, p, h * D_HEAD:(h + 1) * D_HEAD] = captured["src"][
                        0, p, h * D_HEAD:(h + 1) * D_HEAD
                    ]
            return (x,) + tuple(fargs[1:])

        handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch_hook)
    else:
        def patch_hook(module, fargs, output):  # noqa: ANN001
            out = unwrap(output).clone()
            for p in positions:
                out[0, p, :] = captured["src"][0, p, :]
            return rewrap(output, out)

        if kind == "attn_out":
            handle = model.transformer.h[layer].attn.register_forward_hook(patch_hook)
        elif kind == "mlp_out":
            handle = model.transformer.h[layer].mlp.register_forward_hook(patch_hook)
        else:
            handle = model.transformer.h[layer].register_forward_hook(patch_hook)
    try:
        out_patched = model(**enc_corr)
    finally:
        handle.remove()
    d_patched = d_of(out_patched.logits)

    recovery = (d_patched - d_corr) / (d_clean - d_corr) if (d_clean - d_corr) != 0.0 else None
    reference = {
        "D_clean": d_clean,
        "D_corrupted": d_corr,
        "D_patched": d_patched,
        "Recovery": recovery,
    }

    if not args.api_json:
        print(f"D_clean     {d_clean:.4f}")
        print(f"D_corrupted {d_corr:.4f}")
        print(f"D_patched   {d_patched:.4f}")
        print(f"Recovery    {recovery:.4f}")
        return 0

    problems: list[str] = []
    max_diff = 0.0
    for key, ref in reference.items():
        stored = metrics_stored.get(key)
        if stored is None and ref is None:
            continue
        if stored is None or ref is None:
            problems.append(f"{key}: stored={stored!r} reference={ref!r}")
            continue
        diff = abs(float(stored) - float(ref))
        max_diff = max(max_diff, diff)
        if diff > TOL:
            problems.append(f"{key}: stored {stored:.6f} vs reference {ref:.6f} (|diff| {diff:.2e})")
    print(f"max |api - reference| = {max_diff:.3e} (tol {TOL})")
    if problems:
        print("\nFAILURES:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: patched-run numbers within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
