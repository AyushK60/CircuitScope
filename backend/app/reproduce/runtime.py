"""The raw-HuggingFace runtime emitted into every reproduction script.

This is deliberately a plain string: generated scripts are TEXT that the server
never executes. The only thing substituted into the text is one ``json.loads``
literal (see ``generate.py``), so user strings never become code.

The runtime uses raw ``transformers`` + ``torch`` forward hooks only, never
TransformerLens and never backend imports. It is the independent code path for
the numbers the app displays.
"""

from __future__ import annotations

BODY = r'''
import argparse
import json
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DATA = json.loads(__DATA_LITERAL__)
KIND = __KIND_LITERAL__

BOS_ID = 50256
N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64
ROWS = []

# Set in main() from the command line before any runner is called.
args_mode = "sample"
args_n_prompts = None


def add(prefix, name, stored, recomputed, tol):
    """Record one comparison. ``stored`` and ``recomputed`` must be numbers."""
    stored = float(stored)
    recomputed = float(recomputed)
    diff = abs(stored - recomputed)
    ok = diff <= float(tol)
    ROWS.append((prefix + name, stored, recomputed, diff, ok, float(tol)))


def emit():
    print("%-50s %14s %14s %11s  %s" % ("quantity", "stored", "recomputed", "abs diff", "ok"))
    print("-" * 104)
    for name, stored, recomputed, diff, ok, tol in ROWS:
        print("%-50s %14.6f %14.6f %11.3e  %s" % (name, stored, recomputed, diff, "ok" if ok else "FAIL"))
    bad = [r for r in ROWS if not r[4]]
    print("-" * 104)
    if bad:
        print("FAIL: %d of %d compared values are outside tolerance" % (len(bad), len(ROWS)))
        for name, stored, recomputed, diff, ok, tol in bad[:20]:
            print("  - %s: stored %.6f vs recomputed %.6f (|diff| %.3e > %.3e)" %
                  (name, stored, recomputed, diff, tol))
        return 1
    print("PASS: all %d compared values are within tolerance" % len(ROWS))
    return 0


def check_versions(data):
    stored = data.get("versions") or {}
    local_torch = torch.__version__
    try:
        import transformers
        local_transformers = transformers.__version__
    except Exception:
        local_transformers = "unknown"
    if stored.get("torch") and stored["torch"] != local_torch:
        print("WARNING: local torch %s differs from stored %s (values may drift)" % (local_torch, stored["torch"]))
    if stored.get("transformers") and stored["transformers"] != local_transformers:
        print("WARNING: local transformers %s differs from stored %s (values may drift)" %
              (local_transformers, stored["transformers"]))


def load_model(revision):
    kwargs = {}
    rev = str(revision or "")
    if len(rev) == 40 and all(c in "0123456789abcdef" for c in rev.lower()):
        kwargs["revision"] = rev
        print("loading gpt2 at revision %s" % rev)
    else:
        print("loading gpt2 (stored revision is not a 40-hex hash: %r)" % (rev,))
    tok = GPT2TokenizerFast.from_pretrained("gpt2", **kwargs)
    model = GPT2LMHeadModel.from_pretrained("gpt2", **kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model


def encode(tok, text):
    ids = [BOS_ID] + tok(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids])


def single_id(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise SystemExit("candidate %r is not a single token" % (text,))
    return int(ids[0])


def d_of(logits, a, b):
    return float(logits[0, -1, a] - logits[0, -1, b])


def unwrap(output):
    return output[0] if isinstance(output, tuple) else output


def rewrap(output, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    return new


# ---------------------------------------------------------------------------
# direct logit attribution
# ---------------------------------------------------------------------------

def run_dla(data, tok, model, prefix=""):
    text = data["text"]
    a = single_id(tok, data["candidate_a"])
    b = single_id(tok, data["candidate_b"])
    input_ids = encode(tok, text)
    n_layers = len(model.transformer.h)
    pos = input_ids.shape[1] - 1

    block_out = {}
    wte_out = {}
    cproj_in = {}
    mlp_out = {}
    handles = []

    def make_block(l):
        def hook(module, args, output):
            block_out[l] = unwrap(output).detach().clone()
            return output
        return hook

    def wte_hook(module, args, output):
        wte_out["t"] = output.detach().clone()
        return output

    def make_pre(l):
        def hook(module, args):
            cproj_in[l] = args[0].detach().clone()
        return hook

    def make_mlp(l):
        def hook(module, args, output):
            mlp_out[l] = unwrap(output).detach().clone()
            return output
        return hook

    for l in range(n_layers):
        handles.append(model.transformer.h[l].register_forward_hook(make_block(l)))
        handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(make_pre(l)))
        handles.append(model.transformer.h[l].mlp.register_forward_hook(make_mlp(l)))
    handles.append(model.transformer.wte.register_forward_hook(wte_hook))
    try:
        out = model(input_ids)
    finally:
        for h in handles:
            h.remove()

    ln_f = model.transformer.ln_f
    eps = float(model.config.layer_norm_epsilon)
    r = block_out[n_layers - 1][0, pos]
    s = torch.sqrt(r.var(unbiased=False) + eps)
    w = model.lm_head.weight[a] - model.lm_head.weight[b]
    direction = (ln_f.weight * w) / s

    def dla(c):
        return float(((c - c.mean()) @ direction).item())

    components = {}
    components["embed"] = dla(model.transformer.wte(input_ids)[0, pos])
    components["pos_embed"] = dla(model.transformer.wpe(torch.tensor([pos]))[0])
    for l in range(n_layers):
        cp = model.transformer.h[l].attn.c_proj
        for h in range(model.config.n_head):
            head_out = cproj_in[l][0, pos, h * D_HEAD:(h + 1) * D_HEAD] @ cp.weight[h * D_HEAD:(h + 1) * D_HEAD, :]
            components["head %d.%d" % (l, h)] = dla(head_out)
        components["attn_bias.%d" % l] = dla(cp.bias)
        components["mlp.%d" % l] = dla(mlp_out[l][0, pos])
    components["ln_final_bias"] = float((ln_f.bias @ w).item())

    real_d = d_of(out.logits, a, b)
    recon_sum = sum(components.values())
    recon_err = abs(recon_sum - real_d)
    tol = float(data["tolerances"]["dla_term"])
    for name, value in data["stored"]["components"].items():
        if name not in components:
            add(prefix, "DLA " + name, value, float("nan"), tol)
            continue
        add(prefix, "DLA " + name, value, components[name], tol)
    add(prefix, "DLA reconciliation |sum - D|", data["stored"]["reconciliation"]["abs_error"], recon_err, tol)
    add(prefix, "DLA model D", data["stored"]["D"], real_d, float(data["tolerances"]["d"]))


# ---------------------------------------------------------------------------
# logit lens
# ---------------------------------------------------------------------------

def lens_logits(resid, ln_f, eps, lm_head):
    mean = resid.mean()
    sd = torch.sqrt(resid.var(unbiased=False) + eps)
    return ((resid - mean) / sd * ln_f.weight + ln_f.bias) @ lm_head.weight.T


def run_lens(data, tok, model, prefix=""):
    input_ids = encode(tok, data["text"])
    n_layers = len(model.transformer.h)
    pos = input_ids.shape[1] - 1
    block_out = {}
    wte_out = {}
    handles = []

    def make_block(l):
        def hook(module, args, output):
            block_out[l] = unwrap(output).detach().clone()
            return output
        return hook

    def wte_hook(module, args, output):
        wte_out["t"] = output.detach().clone()
        return output

    for l in range(n_layers):
        handles.append(model.transformer.h[l].register_forward_hook(make_block(l)))
    handles.append(model.transformer.wte.register_forward_hook(wte_hook))
    try:
        out = model(input_ids)
    finally:
        for h in handles:
            h.remove()

    ln_f = model.transformer.ln_f
    eps = float(model.config.layer_norm_epsilon)
    pos_embed = model.transformer.wpe(torch.arange(input_ids.shape[1]))[pos]
    residuals = [("embeddings", wte_out["t"][0, pos] + pos_embed)]
    residuals += [("block %d" % i, block_out[i][0, pos]) for i in range(n_layers)]
    stage = {}
    for name, resid in residuals:
        logits = lens_logits(resid, ln_f, eps, model.lm_head)
        stage[name] = {
            "D": None,
            "top1": int(torch.argmax(logits)),
            "logits": logits,
        }
        if data.get("candidate_a") is not None and data.get("candidate_b") is not None:
            a = single_id(tok, data["candidate_a"])
            b = single_id(tok, data["candidate_b"])
            stage[name]["D"] = float(logits[a] - logits[b])

    tol_d = float(data["tolerances"]["lens_d"])
    tol_identity = float(data["tolerances"]["lens_identity"])
    for stored in data["stored"]["stages"]:
        name = stored["stage"]
        if name not in stage:
            add(prefix, "lens " + name + " D", stored.get("D") or 0.0, float("nan"), tol_d)
            continue
        if stored.get("D") is not None:
            add(prefix, "lens " + name + " D", stored["D"], stage[name]["D"], tol_d)
        if stored.get("top1_id") is not None:
            add(prefix, "lens " + name + " top-1 id", stored["top1_id"], stage[name]["top1"], 0.0)

    real = out.logits[0, pos]
    identity = float((lens_logits(residuals[-1][1], ln_f, eps, model.lm_head) - real).abs().max())
    stored_final = data["stored"].get("final_matches_model") or {}
    # The identity is expected to be zero up to float32 error, so the honest
    # comparison is against the declared bound, not against a fabricated value.
    add(prefix, "lens final identity (expected 0, max abs logit diff)", 0.0, identity, tol_identity)
    if stored_final.get("max_abs_logit_diff_full") is not None:
        add(prefix, "lens final identity (stored vs recomputed)",
            stored_final["max_abs_logit_diff_full"], identity, tol_d)
    if data.get("candidate_a") is not None and data.get("candidate_b") is not None:
        a = single_id(tok, data["candidate_a"])
        b = single_id(tok, data["candidate_b"])
        add(prefix, "lens final D", data["stored"].get("final_D", d_of(out.logits, a, b)),
            d_of(out.logits, a, b), tol_d)


# ---------------------------------------------------------------------------
# patching primitives
# ---------------------------------------------------------------------------

def parse_hook(name):
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
    raise SystemExit("unsupported hook in reproduction script: %r" % (name,))


def capture_hook(model, enc, kind, layer):
    store = {}
    handle = None

    def fwd(module, args, output):
        store["x"] = unwrap(output).detach().clone()
        return output

    def pre(module, args):
        store["x"] = args[0].detach().clone()

    if kind == "z":
        handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(pre)
    elif kind == "attn_out":
        handle = model.transformer.h[layer].attn.register_forward_hook(fwd)
    elif kind == "mlp_out":
        handle = model.transformer.h[layer].mlp.register_forward_hook(fwd)
    else:
        handle = model.transformer.h[layer].register_forward_hook(fwd)
    try:
        model(enc)
    finally:
        handle.remove()
    return store["x"]


def run_patched(model, enc, kind, layer, src, positions, heads):
    def hook(module, args, output=None):
        if kind == "z":
            x = args[0].clone()
            for p in positions:
                for h in heads:
                    x[0, p, h * D_HEAD:(h + 1) * D_HEAD] = src[0, p, h * D_HEAD:(h + 1) * D_HEAD]
            return (x,) + tuple(args[1:])
        out = unwrap(output).clone()
        for p in positions:
            out[0, p, :] = src[0, p, :]
        return rewrap(output, out)

    if kind == "z":
        handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook)
    elif kind == "attn_out":
        handle = model.transformer.h[layer].attn.register_forward_hook(hook)
    elif kind == "mlp_out":
        handle = model.transformer.h[layer].mlp.register_forward_hook(hook)
    else:
        handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        return model(enc).logits
    finally:
        handle.remove()


def run_intervention(data, tok, model, prefix=""):
    clean = encode(tok, data["clean_text"])
    corr = encode(tok, data["corrupted_text"])
    a = single_id(tok, data["candidate_a"])
    b = single_id(tok, data["candidate_b"])
    kind, layer = parse_hook(data["hook"])
    if kind == "z":
        src = capture_hook(model, clean, kind, layer)
    else:
        src = capture_hook(model, clean, kind, layer)
    logits_clean = model(clean).logits
    logits_corr = model(corr).logits
    logits_patch = run_patched(model, corr, kind, layer, src, data["positions"], data["heads"] or [])
    d_clean = d_of(logits_clean, a, b)
    d_corr = d_of(logits_corr, a, b)
    d_patch = d_of(logits_patch, a, b)
    delta = d_patch - d_corr
    rec = None if (d_clean - d_corr) == 0.0 else delta / (d_clean - d_corr)
    m = data["stored"]["metrics"]
    tol = float(data["tolerances"]["d"])
    add(prefix, "D_clean", m["D_clean"], d_clean, tol)
    add(prefix, "D_corrupted", m["D_corrupted"], d_corr, tol)
    add(prefix, "D_patched", m["D_patched"], d_patch, tol)
    add(prefix, "Delta", m["Delta"], delta, tol)
    if m.get("Recovery") is not None and rec is not None:
        add(prefix, "Recovery", m["Recovery"], rec, tol)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def run_sweep(data, tok, model, prefix=""):
    clean = encode(tok, data["clean_text"])
    corr = encode(tok, data["corrupted_text"])
    a = single_id(tok, data["candidate_a"])
    b = single_id(tok, data["candidate_b"])
    mode = data["mode"]
    tol = float(data["tolerances"]["recovery"])
    d_clean = d_of(model(clean).logits, a, b)
    d_corr = d_of(model(corr).logits, a, b)
    denom = d_clean - d_corr
    add(prefix, "baseline D_clean", data["stored"]["baselines"]["D_clean"], d_clean, tol)
    add(prefix, "baseline D_corrupted", data["stored"]["baselines"]["D_corrupted"], d_corr, tol)

    # capture the clean activation for every layer once
    captured = {}
    handles = []
    if mode == "layer_position":
        for l in range(N_LAYERS):
            def make(l):
                def hook(module, args, output):
                    captured[l] = unwrap(output).detach().clone()
                    return output
                return hook
            handles.append(model.transformer.h[l].register_forward_hook(make(l)))
    else:
        for l in range(N_LAYERS):
            def make(l):
                def hook(module, args):
                    captured[l] = args[0].detach().clone()
                return hook
            handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(make(l)))
    try:
        model(clean)
    finally:
        for h in handles:
            h.remove()

    cells = None if args_mode == "full" else data["cells_sample"]
    if cells is None:
        n_cols = len(data["stored"]["recovery"][0])
        cells = [[l, c] for l in range(N_LAYERS) for c in range(n_cols)]

    for layer, col in cells:
        if mode == "layer_position":
            logits = run_patched(model, corr, "block_out", layer, captured[layer], [col], [])
            q = "block %d position %d" % (layer, col)
        else:
            logits = run_patched(model, corr, "z", layer, captured[layer], list(range(clean.shape[1])), [col])
            q = "block %d head %d" % (layer, col)
        dp = d_of(logits, a, b)
        rec = (dp - d_corr) / denom if denom != 0.0 else float("nan")
        add(prefix, q + " D_patched", data["stored"]["d_patched"][layer][col], dp, tol)
        add(prefix, q + " recovery", data["stored"]["recovery"][layer][col], rec, tol)


# ---------------------------------------------------------------------------
# study subset: exact denoising + attribution patching under swap
# ---------------------------------------------------------------------------

def capture_clean_components(model, enc):
    z = {}
    mlp = {}
    handles = []
    for l in range(N_LAYERS):
        def pre(module, args, _l=l):
            z[_l] = args[0].detach().clone()
        def mh(module, args, output, _l=l):
            mlp[_l] = unwrap(output).detach().clone()
            return output
        handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre))
        handles.append(model.transformer.h[l].mlp.register_forward_hook(mh))
    try:
        logits = model(enc).logits
    finally:
        for h in handles:
            h.remove()
    return logits, z, mlp


def component_vectors(data, tok, model, prefix=""):
    tol_rec = float(data["tolerances"]["recovery"])
    tol_atp = float(data["tolerances"]["atp"])
    n_prompts = data.get("n_prompts_default", 2)
    if args_n_prompts is not None:
        n_prompts = args_n_prompts
    prompts = data["prompts"]
    if args_mode != "full":
        prompts = prompts[:n_prompts]
    for p in prompts:
        a = single_id(tok, p["candidate_a"])
        b = single_id(tok, p["candidate_b"])
        enc_clean = encode(tok, p["clean_text"])
        enc_corr = encode(tok, p["corrupted_text"])
        seq = enc_clean.shape[1]
        positions = list(range(seq))
        logits_clean, z, mlp = capture_clean_components(model, enc_clean)
        d_clean = d_of(logits_clean, a, b)
        logits_corr = model(enc_corr).logits
        d_corr = d_of(logits_corr, a, b)
        denom = d_clean - d_corr
        if denom == 0.0:
            add(prefix, p["id"] + " denominator", 1.0, 0.0, 0.0)
            continue

        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                lg = run_patched(model, enc_corr, "z", l, z[l], positions, [h])
                dp = d_of(lg, a, b)
                idx = l * N_HEADS + h
                add(prefix, "%s head %d.%d Rec_D" % (p["id"], l, h),
                    p["rec_d"][idx], (dp - d_corr) / denom, tol_rec)
            lg = run_patched(model, enc_corr, "mlp_out", l, mlp[l], positions, [])
            dp = d_of(lg, a, b)
            idx = N_LAYERS * N_HEADS + l
            add(prefix, "%s mlp.%d Rec_D" % (p["id"], l),
                p["rec_d"][idx], (dp - d_corr) / denom, tol_rec)

        # AtP: one corrupted pass with a graph
        stored_z = {}
        stored_mlp = {}
        handles = []
        handles.append(model.transformer.wte.register_forward_hook(
            lambda m, a_, out: out.detach().requires_grad_(True)))
        for l in range(N_LAYERS):
            def z_hook(module, a_, _l=l):
                a_[0].retain_grad()
                stored_z[_l] = a_[0]
            def m_hook(module, a_, output, _l=l):
                tensor = unwrap(output)
                tensor.retain_grad()
                stored_mlp[_l] = tensor
                return output
            handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(z_hook))
            handles.append(model.transformer.h[l].mlp.register_forward_hook(m_hook))
        try:
            with torch.enable_grad():
                out = model(enc_corr)
                dgraph = out.logits[0, -1, a] - out.logits[0, -1, b]
                dgraph.backward()
        finally:
            for h in handles:
                h.remove()
        rec_atp = [0.0] * 156
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                sl = slice(h * D_HEAD, (h + 1) * D_HEAD)
                a_clean = z[l][0, :, sl]
                a_corr = stored_z[l].detach()[0, :, sl]
                g = stored_z[l].grad[0, :, sl]
                rec_atp[l * N_HEADS + h] = float((((a_clean - a_corr) * g).sum()).item() / denom)
            idx = N_LAYERS * N_HEADS + l
            m_clean = mlp[l][0, :, :]
            m_corr = stored_mlp[l].detach()[0, :, :]
            g = stored_mlp[l].grad[0, :, :]
            rec_atp[idx] = float((((m_clean - m_corr) * g).sum()).item() / denom)
        for i, stored in enumerate(p["atp"]):
            if stored is None:
                continue
            add(prefix, "%s component %d Rec_AtP" % (p["id"], i), stored, rec_atp[i], tol_atp)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce numbers shown by CircuitScope.")
    parser.add_argument("--mode", choices=["sample", "full"], default=DATA.get("mode_default", "sample"))
    parser.add_argument("--n-prompts", dest="n_prompts", type=int, default=None,
                        help="study_subset only: how many prompts to check (default 2)")
    global args_mode, args_n_prompts
    args = parser.parse_args()
    args_mode = args.mode
    args_n_prompts = args.n_prompts

    check_versions(DATA)
    tok, model = load_model(DATA.get("model_revision"))

    if KIND in ("dla",):
        run_dla(DATA, tok, model)
    elif KIND in ("lens",):
        run_lens(DATA, tok, model)
    elif KIND == "intervention":
        run_intervention(DATA, tok, model)
    elif KIND == "sweep":
        run_sweep(DATA, tok, model)
    elif KIND == "study_subset":
        component_vectors(DATA, tok, model)
    elif KIND == "report":
        run_dla(DATA["dla"], tok, model, "dla: ")
        run_lens(DATA["lens"], tok, model, "lens: ")
        run_sweep(DATA["position_sweep"], tok, model, "positions: ")
        run_sweep(DATA["head_sweep"], tok, model, "heads: ")
    else:
        print("unknown kind: %r" % (KIND,))
        return 2
    return emit()


if __name__ == "__main__":
    sys.exit(main())
'''
