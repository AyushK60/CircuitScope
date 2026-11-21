"""Independent reference implementation of attribution patching, written SEPARATELY from the backend implementation.

Uses raw Hugging Face transformers + PyTorch only — no TransformerLens, no
backend imports.

Reimplementation of AtP (denoising direction, metric D):
  - one clean pass captures per-head z (forward PRE-hook on
    transformer.h[L].attn.c_proj input, head h owns input rows
    h*64:(h+1)*64 — GPT-2 Conv1D) and the MLP output (forward hook on
    transformer.h[L].mlp);
  - one corrupted pass with grad: a forward hook on transformer.wte returns
    output.detach().requires_grad_(True) so a graph exists (all parameters
    frozen); pre-hooks on every c_proj input and hooks on every mlp output
    call retain_grad(); D = logits[0,-1,a] - logits[0,-1,b]; D.backward();
  - AtP_D(c) = sum over positions/dims of (a_clean - a_corr) * dD/da |_corr
  - Rec_AtP = AtP_D / (D_clean - D_corr).

Gotchas: BOS token id 50256 is prepended to match
the bridge; forward hooks (NOT output_hidden_states); Conv1D weight layout.

Compares Rec_AtP for ALL 156 components on the first 3 usable prompts of a
study with the backend's per_prompt.atp values, tolerance 2e-3 absolute.
Exits non-zero on mismatch.

Usage:
  python scripts/reference_atp.py                # live backend on 127.0.0.1:8000
      (runs a backend study with n_prompts=4, seed=13 — a few minutes)
  python scripts/reference_atp.py --study-json FILE
      # compare against a saved study-results JSON instead (must contain
      # prompt_set.prompts and per_prompt[i].atp)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

BOS_ID = 50256
TOL = 2e-3
N_COMPONENTS = 156
BASE_URL = "http://127.0.0.1:8000"


def encode(tok, text: str) -> torch.Tensor:
    ids = [BOS_ID] + tok(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-json", dest="study_json", default=None)
    parser.add_argument("--base-url", dest="base_url", default=BASE_URL)
    args = parser.parse_args()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    # ---- fetch backend study results (prompts + per_prompt.atp) ----
    def api_call(path: str, payload: dict | None = None) -> dict:
        if payload is None:
            with urllib.request.urlopen(args.base_url + path, timeout=1800) as resp:
                return json.load(resp)
        req = urllib.request.Request(
            args.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return json.load(resp)

    if args.study_json:
        with open(args.study_json, encoding="utf-8") as f:
            study = json.load(f)
    else:
        print("running a backend study (n_prompts=4, seed=13) for the comparison...")
        sub = api_call("/api/study", {"n_prompts": 4, "seed": 13})
        job_id = sub["job_id"]
        for _ in range(3600):
            job = api_call(f"/api/experiments/{job_id}")
            if job["status"] in ("complete", "failed", "cancelled"):
                break
            time.sleep(0.5)
        if job["status"] != "complete":
            raise RuntimeError(f"backend study did not complete: {job.get('error')}")
        study = job["results"]

    prompts = study["prompt_set"]["prompts"]
    per_prompt = study["per_prompt"]
    usable = [
        (p, pp)
        for p, pp in zip(prompts, per_prompt)
        if pp.get("atp") is not None and p["corruptions"]["swap"]
    ][:3]
    if len(usable) < 3:
        raise RuntimeError(f"only {len(usable)} usable prompts with AtP values")

    failures: list[str] = []
    max_diff = 0.0
    for p, pp in usable:
        cand_a, cand_b = p["candidate_a"], p["candidate_b"]
        tid_a = tok(cand_a, add_special_tokens=False)["input_ids"][0]
        tid_b = tok(cand_b, add_special_tokens=False)["input_ids"][0]
        enc_clean = encode(tok, p["clean_text"])
        enc_corr = encode(tok, p["corruptions"]["swap"])

        # ---- clean capture (no grad) ----
        with torch.no_grad():
            clean_z: dict[int, torch.Tensor] = {}
            clean_mlp: dict[int, torch.Tensor] = {}
            handles = []
            for l in range(n_layers):
                def pre_hook(module, aargs, _l=l):  # noqa: ANN001
                    clean_z[_l] = aargs[0].detach().clone()
                    return None

                handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre_hook))

                def mlp_hook(module, aargs, output, _l=l):  # noqa: ANN001
                    clean_mlp[_l] = output.detach().clone()
                    return output

                handles.append(model.transformer.h[l].mlp.register_forward_hook(mlp_hook))
            out_clean = model(enc_clean)
            for h in handles:
                h.remove()
            d_clean = float(out_clean.logits[0, -1, tid_a] - out_clean.logits[0, -1, tid_b])

        # ---- corrupted pass with grad ----
        stored_z: dict[int, torch.Tensor] = {}
        stored_mlp: dict[int, torch.Tensor] = {}
        handles = []
        handles.append(
            model.transformer.wte.register_forward_hook(
                lambda module, aargs, output: output.detach().requires_grad_(True)
            )
        )
        for l in range(n_layers):
            def z_hook(module, aargs, _l=l):  # noqa: ANN001
                aargs[0].retain_grad()
                stored_z[_l] = aargs[0]
                return None

            handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(z_hook))

            def m_hook(module, aargs, output, _l=l):  # noqa: ANN001
                output.retain_grad()
                stored_mlp[_l] = output
                return output

            handles.append(model.transformer.h[l].mlp.register_forward_hook(m_hook))
        try:
            with torch.enable_grad():
                out_corr = model(enc_corr)
                missing = [l for l in range(n_layers) if l not in stored_z or l not in stored_mlp]
                if missing:
                    raise RuntimeError(f"reference AtP hooks did not fire for layers {missing}")
                d = out_corr.logits[0, -1, tid_a] - out_corr.logits[0, -1, tid_b]
                d.backward()
        finally:
            for h in handles:
                h.remove()
        d_corr = float(out_corr.logits[0, -1, tid_a].detach() - out_corr.logits[0, -1, tid_b].detach())
        denominator = d_clean - d_corr
        if denominator == 0.0:
            raise RuntimeError("zero denominator in reference AtP")

        ref_rec = [0.0] * N_COMPONENTS
        for l in range(n_layers):
            z_corr = stored_z[l].detach()
            g = stored_z[l].grad
            for h in range(n_heads):
                sl = slice(h * d_head, (h + 1) * d_head)
                a_clean = clean_z[l][0, :, sl]
                a_corr = z_corr[0, :, sl]
                ref_rec[l * n_heads + h] = float(
                    (((a_clean - a_corr) * g[0, :, sl]).sum()).item() / denominator
                )
            m_clean = clean_mlp[l][0, :, :]
            m_corr = stored_mlp[l].detach()[0, :, :]
            ref_rec[n_layers * n_heads + l] = float(
                (((m_clean - m_corr) * stored_mlp[l].grad[0, :, :]).sum()).item() / denominator
            )

        api_rec = pp["atp"]
        diffs = []
        for i in range(N_COMPONENTS):
            diff = abs(api_rec[i] - ref_rec[i])
            diffs.append(diff)
            max_diff = max(max_diff, diff)
            if diff > TOL:
                failures.append(
                    f"prompt {p['id']} component {i}: api {api_rec[i]:.6f} vs ref {ref_rec[i]:.6f} (|diff| {diff:.2e})"
                )
        print(
            f"prompt {p['id']}: D_clean={d_clean:.4f} D_corr={d_corr:.4f} "
            f"max |api-ref| Rec_AtP = {max(diffs):.3e}"
        )

    print(f"comparison: max |api - reference| Rec_AtP over 3 prompts x 156 components = {max_diff:.3e} (tol {TOL})")
    if failures:
        print("\nFAILURES:")
        for f in failures[:20]:
            print(f"  - {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1
    print("\nOK: Rec_AtP within tolerance on all compared components.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
