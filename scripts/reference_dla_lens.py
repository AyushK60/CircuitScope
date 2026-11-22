"""Independent reference implementation of direct logit attribution (DLA) and
the logit lens, written SEPARATELY from the backend implementation.

Uses raw Hugging Face transformers + PyTorch forward hooks only - no
TransformerLens, no backend imports - so the backend's /api/analysis/dla and
/api/analysis/logit_lens numbers are verified against a genuinely separate
code path.

Key HF facts used here (GPT-2):
  - Conv1D weight is [in, out]; head h owns input rows h*64:(h+1)*64 of
    transformer.h[i].attn.c_proj (input is the 12 heads concatenated).
  - lm_head.weight is [vocab, d_model]; the unembedding direction for token t
    is lm_head.weight[t] (GPT-2 has no unembedding bias).
  - With output_hidden_states=True the LAST entry is already after ln_f;
    earlier entries are block outputs (pre-ln_f).
  - The backend (TransformerBridge) prepends BOS <|endoftext|> (50256); this
    script does the same so positions match exactly.

Usage:
  python scripts/reference_dla_lens.py                 # live backend on 127.0.0.1:8000
  python scripts/reference_dla_lens.py --dla-json FILE --lens-json FILE
      # compare against saved API responses instead of a live backend

Exit code non-zero on any mismatch. Tolerances: every DLA component <= 2e-3
absolute; reconciliation sum vs real D <= 2e-3; lens D per stage <= 2e-3.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CLEAN = "John and Mary went to the shop. John gave a book to"
CAND_A = " Mary"
CAND_B = " John"
BOS_ID = 50256
TOL_COMPONENT = 2e-3
TOL_RECONCILIATION = 2e-3
TOL_LENS_D = 2e-3
BASE_URL = "http://127.0.0.1:8000"


def encode(tok: GPT2TokenizerFast, text: str) -> torch.Tensor:
    ids = [BOS_ID] + tok(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dla-json", dest="dla_json", default=None, help="saved /api/analysis/dla response")
    parser.add_argument("--lens-json", dest="lens_json", default=None, help="saved /api/analysis/logit_lens response")
    parser.add_argument("--base-url", dest="base_url", default=BASE_URL)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    tid_a = tok(CAND_A, add_special_tokens=False)["input_ids"]
    tid_b = tok(CAND_B, add_special_tokens=False)["input_ids"]
    assert len(tid_a) == 1 and len(tid_b) == 1, "candidates must be single tokens"
    tid_a, tid_b = tid_a[0], tid_b[0]

    input_ids = encode(tok, CLEAN)
    out = model(input_ids, output_hidden_states=True)
    n_layers = len(model.transformer.h)

    pos = input_ids.shape[1] - 1
    real_D = float(out.logits[0, pos, tid_a] - out.logits[0, pos, tid_b])

    # Block outputs (residual after each block) via forward hooks. Note: in
    # transformers 5.x output_hidden_states does NOT contain block 11's
    # pre-ln_f output (the last entry is already after ln_f), so hooks are the
    # way to get every stage's residual.
    block_out: dict[int, torch.Tensor] = {}
    wte_out: dict[str, torch.Tensor] = {}

    def make_block_hook(_l):
        def hook(module, args, output):  # noqa: ANN001
            block_out[_l] = output.detach().clone()
        return hook

    def wte_hook(module, args, output):  # noqa: ANN001
        wte_out["t"] = output.detach().clone()

    handles = [model.transformer.h[l].register_forward_hook(make_block_hook(l)) for l in range(n_layers)]
    handles.append(model.transformer.wte.register_forward_hook(wte_hook))
    out = model(input_ids)
    for h in handles:
        h.remove()
    pos_ids = torch.arange(input_ids.shape[1])
    pos_embed_vec = model.transformer.wpe(pos_ids)[pos]  # [d_model] (wpe(pos_ids) is [seq, d])
    emb_stage_resid = wte_out["t"][0, pos] + pos_embed_vec  # embeddings stage residual

    # final LayerNorm exactly as the model does it
    ln_f = model.transformer.ln_f
    eps = float(model.config.layer_norm_epsilon)
    r = block_out[n_layers - 1][0, pos]  # residual entering ln_f (block 11 output)
    s = torch.sqrt(r.var(unbiased=False) + eps)
    w = model.lm_head.weight[tid_a] - model.lm_head.weight[tid_b]
    direction = (ln_f.weight * w) / s

    # capture per-head c_proj INPUT (pre-hook) and MLP outputs (forward hooks)
    cproj_in: dict[int, torch.Tensor] = {}
    mlp_out: dict[int, torch.Tensor] = {}
    pre_handles, fwd_handles = [], []

    def make_pre_hook(layer):
        def hook(module, args):  # noqa: ANN001
            cproj_in[layer] = args[0].detach().clone()
        return hook

    def make_fwd_hook(layer):
        def hook(module, args, output):  # noqa: ANN001
            mlp_out[layer] = output.detach().clone()
        return hook

    for l in range(n_layers):
        pre_handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(make_pre_hook(l)))
        fwd_handles.append(model.transformer.h[l].mlp.register_forward_hook(make_fwd_hook(l)))
    model(input_ids)
    for h in pre_handles + fwd_handles:
        h.remove()

    # ---- DLA components (reference) ----
    components: dict[str, float] = {}
    embed = model.transformer.wte(input_ids)[0, pos]
    pos_embed = model.transformer.wpe(torch.tensor([pos]))[0]

    def dla(c: torch.Tensor) -> float:
        return float(((c - c.mean()) @ direction).item())

    components["embed"] = dla(embed)
    components["pos_embed"] = dla(pos_embed)
    for l in range(n_layers):
        cp = model.transformer.h[l].attn.c_proj
        for h in range(model.config.n_head):
            head_out = cproj_in[l][0, pos, h * 64 : (h + 1) * 64] @ cp.weight[h * 64 : (h + 1) * 64, :]
            components[f"head {l}.{h}"] = dla(head_out)
        components[f"attn_bias.{l}"] = dla(cp.bias)
        components[f"mlp.{l}"] = dla(mlp_out[l][0, pos])
    components["ln_final_bias"] = float((ln_f.bias @ w).item())

    recon_sum = sum(components.values())
    recon_err = abs(recon_sum - real_D)

    # ---- logit lens (reference) ----
    def lens_logits(resid: torch.Tensor) -> torch.Tensor:
        mean = resid.mean()
        sd = torch.sqrt(resid.var(unbiased=False) + eps)
        return ((resid - mean) / sd * ln_f.weight + ln_f.bias) @ model.lm_head.weight.T

    lens_stages: dict[str, dict] = {}
    stage_resids: list[tuple[str, torch.Tensor]] = [("embeddings", emb_stage_resid)]
    stage_resids += [(f"block {i}", block_out[i][0, pos]) for i in range(n_layers)]
    probs_full = torch.softmax(out.logits[0, pos], dim=-1)
    for name, resid in stage_resids:
        logits = lens_logits(resid)
        lens_stages[name] = {
            "D": float(logits[tid_a] - logits[tid_b]),
            "prob_a": float(torch.softmax(logits, dim=-1)[tid_a]),
            "prob_b": float(torch.softmax(logits, dim=-1)[tid_b]),
            "top1": tok.decode([int(torch.argmax(logits))]),
        }
    lens_final_D = lens_stages[f"block {n_layers - 1}"]["D"]

    print("reference (raw HF):")
    print(f"  real D                = {real_D:.6f}")
    print(f"  reconciliation sum    = {recon_sum:.6f}  |err| = {recon_err:.2e} (tol {TOL_RECONCILIATION})")
    print(f"  lens block 11 D       = {lens_final_D:.6f}")
    print(f"  lens block 11 top1    = {lens_stages[f'block {n_layers - 1}']['top1']!r}")

    failures: list[str] = []
    if recon_err > TOL_RECONCILIATION:
        failures.append(f"reference self-check failed: reconciliation |sum - D| = {recon_err:.2e}")

    # ---- fetch backend results ----
    def api_post(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            args.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)

    if args.dla_json:
        with open(args.dla_json, encoding="utf-8") as f:
            api_dla = json.load(f)
    else:
        api_dla = api_post("/api/analysis/dla", {"task_id": "ioi_john_mary", "pair_id": None, "run": "clean"})

    if args.lens_json:
        with open(args.lens_json, encoding="utf-8") as f:
            api_lens = json.load(f)
    else:
        api_lens = api_post(
            "/api/analysis/logit_lens",
            {"text": CLEAN, "candidate_a": CAND_A, "candidate_b": CAND_B, "top_k": 5},
        )

    # ---- compare DLA ----
    api_components = {c["name"]: float(c["value"]) for c in api_dla["components"]}
    missing = (set(api_components) ^ set(components)) if len(api_components) != len(components) else set()
    max_dla_diff = 0.0
    for name, ref in components.items():
        if name not in api_components:
            failures.append(f"DLA component missing in API response: {name}")
            continue
        diff = abs(api_components[name] - ref)
        max_dla_diff = max(max_dla_diff, diff)
        if diff > TOL_COMPONENT:
            failures.append(f"DLA {name}: api {api_components[name]:.6f} vs reference {ref:.6f} (|diff| {diff:.2e})")
    api_recon = abs(api_dla["reconciliation"]["sum_of_terms"] - api_dla["reconciliation"]["model_D"])
    if api_recon > TOL_RECONCILIATION:
        failures.append(f"API reconciliation error {api_recon:.2e} > tol")
    if not api_dla["reconciliation"]["ok"]:
        failures.append("API reconciliation.ok is False")

    # ---- compare lens ----
    max_lens_diff = 0.0
    api_stages = {s["stage"]: s for s in api_lens["stages"]}
    for name, ref in lens_stages.items():
        if name not in api_stages:
            failures.append(f"lens stage missing in API response: {name}")
            continue
        diff = abs(api_stages[name]["D"] - ref["D"])
        max_lens_diff = max(max_lens_diff, diff)
        if diff > TOL_LENS_D:
            failures.append(f"lens {name} D: api {api_stages[name]['D']:.6f} vs reference {ref['D']:.6f} (|diff| {diff:.2e})")
        for key in ("prob_a", "prob_b"):
            d2 = abs(api_stages[name][key] - ref[key])
            if d2 > TOL_LENS_D:
                failures.append(f"lens {name} {key}: api {api_stages[name][key]:.6f} vs reference {ref[key]:.6f}")
    api_final = api_lens["final_matches_model"]
    api_lensD_vs_realD = abs(api_stages[f"block {n_layers - 1}"]["D"] - real_D)
    if api_lensD_vs_realD > 1e-3:
        failures.append(f"API lens block-11 D vs real model D: {api_lensD_vs_realD:.2e} > 1e-3")
    if not api_final["top1_agrees"] or lens_stages[f"block {n_layers - 1}"]["top1"] != api_stages[f"block {n_layers - 1}"]["top"][0]["token"]:
        failures.append("top-1 token disagreement at block 11")

    print("comparison:")
    print(f"  max |api - reference| DLA component = {max_dla_diff:.3e} (tol {TOL_COMPONENT})")
    print(f"  max |api - reference| lens stage D  = {max_lens_diff:.3e} (tol {TOL_LENS_D})")
    print(f"  API lens block-11 D vs real model D = {api_lensD_vs_realD:.3e} (tol 1e-3)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: all reference comparisons within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
