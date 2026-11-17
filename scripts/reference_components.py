"""Independent reference implementation of the Stage 3 exact per-component
effects (Task 7), written SEPARATELY from the backend implementation.

Uses raw Hugging Face transformers + PyTorch + stdlib only — no
TransformerLens, no backend imports.

For the first 3 usable prompts of a completed study it recomputes
independently:
  - exact denoising Rec_D and Rec_P under `swap` for ALL 156 components
    (144 heads + 12 MLPs), patching via forward hooks:
      head h  = forward PRE-hook on transformer.h[L].attn.c_proj replacing the
                input slice h*64:(h+1)*64 at all positions (GPT-2 Conv1D
                layout, as in scripts/reference_sweep.py);
      MLP L   = forward hook on transformer.h[L].mlp replacing its output at
                all positions;
  - zero ablation Abl_D for ALL 156 components (replace the activation with
    zeros in the CLEAN run; note the o-projection bias stays, matching the
    backend's hook_z replacement semantics);
  - mean ablation Abl_D for 20 components (18 heads + 2 MLPs) using the
    leave-one-out family mean over the other prompts of the same template,
    computed independently here from the study's prompt records.

Gotchas carried over from Stage 2: BOS 50256 prepended; forward hooks (NOT
output_hidden_states); Conv1D weight layout.

Compares with the backend's per_prompt values (tolerance 5e-3 on recoveries
and on Abl_D). Exits non-zero on mismatch.

Usage:
  python scripts/reference_components.py --api-json FILE
      # compare against a saved study-results JSON (GET
      # /api/experiments/{job_id}/export?fmt=json of a completed study;
      # must contain prompt_set.prompts and per_prompt)
  python scripts/reference_components.py
      # live backend on 127.0.0.1:8000: uses the most recent completed study
      # found in /api/experiments (config.job_type == "study")
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

BOS_ID = 50256
TOL = 5e-3
N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64
N_COMPONENTS = 156
N_MEAN_COMPONENTS = 20  # 18 heads + 2 MLPs
BASE_URL = "http://127.0.0.1:8000"


def encode(tok, text: str) -> torch.Tensor:
    ids = [BOS_ID] + tok(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-json", dest="api_json", default=None)
    parser.add_argument("--base-url", dest="base_url", default=BASE_URL)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    def api_call(path: str) -> dict:
        with urllib.request.urlopen(args.base_url + path, timeout=1800) as resp:
            return json.load(resp)

    if args.api_json:
        with open(args.api_json, encoding="utf-8") as f:
            study = json.load(f)
    else:
        listing = api_call("/api/experiments")
        study_id = next(
            (s["id"] for s in reversed(listing["saved"]) if s.get("config", {}).get("job_type") == "study"),
            None,
        )
        if study_id is None:
            raise RuntimeError("no completed study found in /api/experiments")
        print(f"using study {study_id} from the live backend")
        study = api_call(f"/api/experiments/{study_id}")["results"]

    prompts = study["prompt_set"]["prompts"]
    per_prompt = study["per_prompt"]
    usable = [
        (p, pp)
        for p, pp in zip(prompts, per_prompt)
        if pp["rec_d"]["swap"] is not None and not pp["baselines"]["swap"]["excluded"]
    ][:3]
    if len(usable) < 3:
        raise RuntimeError(f"only {len(usable)} usable prompts")

    tid_io = {p["candidate_a"]: tok(p["candidate_a"], add_special_tokens=False)["input_ids"][0] for p, _ in usable}
    tid_s = {p["candidate_b"]: tok(p["candidate_b"], add_special_tokens=False)["input_ids"][0] for p, _ in usable}

    def d_p_of(logits, tid_a, tid_b):
        last = logits[0, -1]
        probs = torch.softmax(last, dim=-1)
        return float(last[tid_a] - last[tid_b]), float(probs[tid_a])

    def capture_clean(enc):
        """Clean per-head z (c_proj input) and MLP outputs for all layers."""
        clean_z: dict[int, torch.Tensor] = {}
        clean_mlp: dict[int, torch.Tensor] = {}
        handles = []
        for l in range(N_LAYERS):
            def pre_hook(module, aargs, _l=l):  # noqa: ANN001
                clean_z[_l] = aargs[0].detach().clone()
                return None

            handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre_hook))

            def mlp_hook(module, aargs, output, _l=l):  # noqa: ANN001
                out = output[0] if isinstance(output, tuple) else output
                clean_mlp[_l] = out.detach().clone()
                return output

            handles.append(model.transformer.h[l].mlp.register_forward_hook(mlp_hook))
        try:
            logits = model(enc).logits
        finally:
            for h in handles:
                h.remove()
        return logits, clean_z, clean_mlp

    def run_patched(enc, hook_kind, l, idx, positions, values):
        """One run with the component replaced by ``values`` (already sliced
        to the same shape as the component's activation)."""
        handles = []

        def head_hook(module, aargs):  # noqa: ANN001
            x = aargs[0].clone()
            for p in positions:
                x[0, p, idx * D_HEAD : (idx + 1) * D_HEAD] = values[p]
            return (x,) + tuple(aargs[1:])

        def mlp_hook(module, aargs, output):  # noqa: ANN001
            out = output[0] if isinstance(output, tuple) else output
            out = out.clone()
            for p in positions:
                out[0, p, :] = values[p]
            if isinstance(output, tuple):
                return (out,) + tuple(output[1:])
            return out

        if hook_kind == "head":
            handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(head_hook))
        else:
            handles.append(model.transformer.h[l].mlp.register_forward_hook(mlp_hook))
        try:
            return model(enc).logits
        finally:
            for h in handles:
                h.remove()

    failures: list[str] = []
    max_diff = 0.0
    checked = 0

    for p, pp in usable:
        tid_a, tid_b = tid_io[p["candidate_a"]], tid_s[p["candidate_b"]]
        enc_clean = encode(tok, p["clean_text"])
        enc_swap = encode(tok, p["corruptions"]["swap"])
        seq = enc_clean.shape[1]
        positions = list(range(seq))

        logits_clean, cz, cm = capture_clean(enc_clean)
        d_clean, p_clean = d_p_of(logits_clean, tid_a, tid_b)
        logits_swap = model(enc_swap).logits
        d_corr, p_corr = d_p_of(logits_swap, tid_a, tid_b)
        denom_d, denom_p = d_clean - d_corr, p_clean - p_corr
        if denom_d == 0.0 or denom_p == 0.0:
            raise RuntimeError(f"zero denominator for {p['id']}")

        # ---- exact denoising Rec_D / Rec_P under swap, all 156 ----
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                sl = slice(h * D_HEAD, (h + 1) * D_HEAD)
                values = cz[l][0, :, sl]  # [seq, 64] clean slice
                logits = run_patched(enc_swap, "head", l, h, positions, values)
                d_p, p_p = d_p_of(logits, tid_a, tid_b)
                idx = l * N_HEADS + h
                for api_v, ref_v, label in (
                    (pp["rec_d"]["swap"][idx], (d_p - d_corr) / denom_d, "Rec_D"),
                    (pp["rec_p"]["swap"][idx], (p_p - p_corr) / denom_p, "Rec_P"),
                ):
                    diff = abs(api_v - ref_v)
                    max_diff = max(max_diff, diff)
                    checked += 1
                    if diff > TOL:
                        failures.append(f"{p['id']} {label} head {l}.{h}: api {api_v:.6f} vs ref {ref_v:.6f}")
            # MLP l
            values = cm[l][0, :, :]
            logits = run_patched(enc_swap, "mlp", l, None, positions, values)
            d_p, p_p = d_p_of(logits, tid_a, tid_b)
            idx = N_LAYERS * N_HEADS + l
            for api_v, ref_v, label in (
                (pp["rec_d"]["swap"][idx], (d_p - d_corr) / denom_d, "Rec_D"),
                (pp["rec_p"]["swap"][idx], (p_p - p_corr) / denom_p, "Rec_P"),
            ):
                diff = abs(api_v - ref_v)
                max_diff = max(max_diff, diff)
                checked += 1
                if diff > TOL:
                    failures.append(f"{p['id']} {label} mlp.{l}: api {api_v:.6f} vs ref {ref_v:.6f}")

        # ---- zero ablation Abl_D, all 156 ----
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                values = torch.zeros(seq, D_HEAD)
                logits = run_patched(enc_clean, "head", l, h, positions, values)
                d_abl, _ = d_p_of(logits, tid_a, tid_b)
                idx = l * N_HEADS + h
                diff = abs(pp["abl_zero"][idx] - (d_clean - d_abl))
                max_diff = max(max_diff, diff)
                checked += 1
                if diff > TOL:
                    failures.append(f"{p['id']} Abl_D zero head {l}.{h}: api {pp['abl_zero'][idx]:.6f} vs ref {d_clean - d_abl:.6f}")
            values = torch.zeros(seq, model.config.n_embd)
            logits = run_patched(enc_clean, "mlp", l, None, positions, values)
            d_abl, _ = d_p_of(logits, tid_a, tid_b)
            idx = N_LAYERS * N_HEADS + l
            diff = abs(pp["abl_zero"][idx] - (d_clean - d_abl))
            max_diff = max(max_diff, diff)
            checked += 1
            if diff > TOL:
                failures.append(f"{p['id']} Abl_D zero mlp.{l}: api {pp['abl_zero'][idx]:.6f} vs ref {d_clean - d_abl:.6f}")

        # ---- mean ablation Abl_D for 20 components ----
        # leave-one-out family mean computed independently from the study's
        # prompt records (other prompts of the same template, same token count).
        family = [
            q
            for q in prompts
            if q["template_id"] == p["template_id"] and q["id"] != p["id"]
        ]
        if pp["abl_mean"] is None or not family:
            print(f"prompt {p['id']}: mean ablation skipped (no family) — matches backend: {pp['abl_mean'] is None}")
            continue
        fam_z: dict[int, list[torch.Tensor]] = {l: [] for l in range(N_LAYERS)}
        fam_mlp: dict[int, list[torch.Tensor]] = {l: [] for l in range(N_LAYERS)}
        for q in family:
            enc_q = encode(tok, q["clean_text"])
            if enc_q.shape[1] != seq:
                failures.append(f"{p['id']}: family prompt {q['id']} not token-aligned")
                continue
            _, qz, qm = capture_clean(enc_q)
            for l in range(N_LAYERS):
                fam_z[l].append(qz[l])
                fam_mlp[l].append(qm[l])
        mean_comp_indices = (
            [(l, h, "head") for l in range(2) for h in range(N_HEADS)]
            + [(0, None, "mlp"), (1, None, "mlp")]
        )[:N_MEAN_COMPONENTS]
        for (l, h, kind) in mean_comp_indices:
            idx = l * N_HEADS + h if kind == "head" else N_LAYERS * N_HEADS + l
            if pp["abl_mean"] is None or idx >= len(pp["abl_mean"]) or pp["abl_mean"][idx] is None:
                continue
            if kind == "head":
                sl = slice(h * D_HEAD, (h + 1) * D_HEAD)
                mean_vals = torch.stack([z[0, :, sl] for z in fam_z[l]], dim=0).mean(dim=0)
                logits = run_patched(enc_clean, "head", l, h, positions, mean_vals)
            else:
                mean_vals = torch.stack([m[0, :, :] for m in fam_mlp[l]], dim=0).mean(dim=0)
                logits = run_patched(enc_clean, "mlp", l, None, positions, mean_vals)
            d_abl, _ = d_p_of(logits, tid_a, tid_b)
            diff = abs(pp["abl_mean"][idx] - (d_clean - d_abl))
            max_diff = max(max_diff, diff)
            checked += 1
            if diff > TOL:
                failures.append(f"{p['id']} Abl_D mean {kind} {l if kind=='mlp' else f'{l}.{h}'}: api {pp['abl_mean'][idx]:.6f} vs ref {d_clean - d_abl:.6f}")
        print(f"prompt {p['id']}: D_clean={d_clean:.4f} D_corr={d_corr:.4f} compared (cumulative checks={checked})")

    print(f"comparison: max |api - reference| over {checked} checks = {max_diff:.3e} (tol {TOL})")
    if failures:
        print("\nFAILURES:")
        for f in failures[:20]:
            print(f"  - {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1
    print("\nOK: all compared values within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
