"""Independent reference implementation of both patching sweep modes, written
SEPARATELY from the backend implementation.

Uses raw Hugging Face transformers + PyTorch forward hooks only — no
TransformerLens, no backend imports.

Reimplementation:
  - layer_position: forward hook on transformer.h[L] replaces the block output
    hidden state at position p with the clean run's value (denoising: clean
    activations are copied into the corrupted run).
  - layer_head: forward PRE-hook on transformer.h[L].attn.c_proj replaces the
    c_proj input slice h*64:(h+1)*64 (head h's post-attention values) at ALL
    positions with the clean run's values.
  GPT-2 Conv1D: c_proj.weight is [768, 768] and head h owns input rows
  h*64:(h+1)*64, so replacing that input slice is exactly replacing head h.

Compares the full recovery and d_patched grids with the backend sweep job's
results (tolerance <= 5e-3 on every cell).

Usage:
  python scripts/reference_sweep.py                # live backend on 127.0.0.1:8000
      (submits both sweep modes to the backend and polls until complete)
  python scripts/reference_sweep.py --layer-position-json FILE --layer-head-json FILE
      # compare against saved sweep results JSON files instead
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CLEAN = "John and Mary went to the shop. John gave a book to"
CORRUPTED = "John and Mary went to the shop. Mary gave a book to"
CAND_A = " Mary"
CAND_B = " John"
BOS_ID = 50256
TOL = 5e-3
BASE_URL = "http://127.0.0.1:8000"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-position-json", dest="lp_json", default=None)
    parser.add_argument("--layer-head-json", dest="lh_json", default=None)
    parser.add_argument("--base-url", dest="base_url", default=BASE_URL)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    tid_a = tok(CAND_A, add_special_tokens=False)["input_ids"][0]
    tid_b = tok(CAND_B, add_special_tokens=False)["input_ids"][0]

    def encode(text: str) -> torch.Tensor:
        ids = [BOS_ID] + tok(text, add_special_tokens=False)["input_ids"]
        return torch.tensor([ids])

    enc_clean = encode(CLEAN)
    enc_corr = encode(CORRUPTED)
    assert enc_clean.shape == enc_corr.shape
    seq = enc_clean.shape[1]

    def d_of(logits: torch.Tensor) -> float:
        return float(logits[0, -1, tid_a] - logits[0, -1, tid_b])

    out_clean = model(enc_clean)
    out_corr = model(enc_corr)
    d_clean, d_corrupted = d_of(out_clean.logits), d_of(out_corr.logits)
    denom = d_clean - d_corrupted
    print(f"reference baselines: D_clean={d_clean:.4f} D_corrupted={d_corrupted:.4f} denom={denom:.4f}")

    def run_sweep(mode: str) -> dict[str, list[list[float]]]:
        """Reference sweep: one clean capture per mode, one patched run per cell."""
        captured: dict[int, torch.Tensor] = {}
        handles: list = []
        for l in range(n_layers):
            if mode == "layer_position":

                def fwd_hook(module, args, output, _l=l):  # noqa: ANN001
                    captured[_l] = output.detach().clone()
                    return output

                handles.append(model.transformer.h[l].register_forward_hook(fwd_hook))
            else:

                def pre_hook(module, args, _l=l):  # noqa: ANN001
                    captured[_l] = args[0].detach().clone()
                    return None

                handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre_hook))
        model(enc_clean)
        for h in handles:
            h.remove()

        recovery = [[0.0] * (seq if mode == "layer_position" else n_heads) for _ in range(n_layers)]
        d_patched = [[0.0] * len(recovery[0]) for _ in range(n_layers)]
        n_cells = sum(len(r) for r in recovery)
        done = 0
        for l in range(n_layers):
            for c in range(len(recovery[l])):
                if mode == "layer_position":

                    def patch_hook(module, args, output, _l=l, _c=c):  # noqa: ANN001
                        hs = output.clone()
                        hs[0, _c, :] = captured[_l][0, _c, :]
                        return hs

                    handle = model.transformer.h[l].register_forward_hook(patch_hook)
                else:

                    def patch_hook_pre(module, args, _l=l, _c=c):  # noqa: ANN001
                        x = args[0].clone()
                        x[..., :, _c * d_head : (_c + 1) * d_head] = captured[_l][
                            ..., :, _c * d_head : (_c + 1) * d_head
                        ]
                        return (x,) + tuple(args[1:])

                    handle = model.transformer.h[l].attn.c_proj.register_forward_pre_hook(patch_hook_pre)
                logits = model(enc_corr).logits
                handle.remove()
                dp = d_of(logits)
                d_patched[l][c] = dp
                recovery[l][c] = (dp - d_corrupted) / denom
                done += 1
                if done % 24 == 0 or done == n_cells:
                    print(f"  {mode}: {done}/{n_cells} cells")
        return {"recovery": recovery, "d_patched": d_patched}

    print("running reference sweeps (raw HF, one forward pass per cell)...")
    ref_lp = run_sweep("layer_position")
    ref_lh = run_sweep("layer_head")

    # ---- fetch backend sweep results ----
    def api_call(path: str, payload: dict | None = None) -> dict:
        if payload is None:
            with urllib.request.urlopen(args.base_url + path, timeout=600) as resp:
                return json.load(resp)
        req = urllib.request.Request(
            args.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)

    def get_sweep(mode: str, saved_path: str | None) -> dict:
        if saved_path:
            with open(saved_path, encoding="utf-8") as f:
                return json.load(f)
        sub = api_call("/api/sweeps", {"task_id": "ioi_john_mary", "pair_id": None, "mode": mode})
        job_id = sub["job_id"]
        print(f"backend sweep {mode}: job {job_id}, polling...")
        for _ in range(1200):
            job = api_call(f"/api/experiments/{job_id}")
            if job["status"] in ("complete", "failed", "cancelled"):
                break
            time.sleep(0.25)
        if job["status"] != "complete":
            raise RuntimeError(f"backend sweep {mode} did not complete: {job.get('error')}")
        return job["results"]

    api_lp = get_sweep("layer_position", args.lp_json)
    api_lh = get_sweep("layer_head", args.lh_json)

    failures: list[str] = []
    max_diff = 0.0
    for label, api_res, ref in (("layer_position", api_lp, ref_lp), ("layer_head", api_lh, ref_lh)):
        api_rec = api_res["recovery"]
        api_dp = api_res["d_patched"]
        if len(api_rec) != n_layers or len(api_rec[0]) != len(ref["recovery"][0]):
            failures.append(f"{label}: grid shape mismatch api {len(api_rec)}x{len(api_rec[0])}")
            continue
        for l in range(n_layers):
            for c in range(len(api_rec[l])):
                for key, ref_grid in (("recovery", ref["recovery"]), ("d_patched", ref["d_patched"])):
                    diff = abs(api_res[key][l][c] - ref_grid[l][c])
                    max_diff = max(max_diff, diff)
                    if diff > TOL:
                        failures.append(f"{label} {key}[{l}][{c}]: api {api_res[key][l][c]:.6f} vs ref {ref_grid[l][c]:.6f} (|diff| {diff:.2e})")
        # baselines must agree too
        for key, ref_v in (("D_clean", d_clean), ("D_corrupted", d_corrupted)):
            diff = abs(api_res["baselines"][key] - ref_v)
            max_diff = max(max_diff, diff)
            if diff > TOL:
                failures.append(f"{label} baseline {key}: api {api_res['baselines'][key]:.6f} vs ref {ref_v:.6f}")

    print("comparison:")
    print(f"  max |api - reference| over all cells and baselines = {max_diff:.3e} (tol {TOL})")
    if failures:
        print("\nFAILURES:")
        for f in failures[:20]:
            print(f"  - {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1
    print("\nOK: all sweep cells within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
