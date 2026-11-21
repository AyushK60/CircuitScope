"""Attribution patching (AtP), denoising direction, metric D.

Definition (per component c with activation a, evaluated at the corrupted run):

    AtP_D(c) = sum over positions and dims of (a_clean - a_corr) * dD/da |_corr

then the recovery-like normalisation is

    Rec_AtP = AtP_D / (D_clean - D_corr)

For a head, ``a`` is that head's hook_z slice [seq, 64]; for an MLP it is the
mlp.hook_out slice [seq, 768]. One forward pass on the corrupted prompt plus
one backward pass give all 156 components.

Recipe (parameters are frozen, so activations need an explicit graph):
inside ``torch.enable_grad()``, run the corrupted prompt with
``bridge.run_with_hooks``; a hook on ``hook_embed`` returns
``x.detach().requires_grad_(True)`` so a graph exists; hooks on every
``blocks.{L}.attn.hook_z`` and ``blocks.{L}.mlp.hook_out`` call
``tensor.retain_grad()`` and store the reference (NOT detached — detaching
would cut the indirect paths through later layers). Then ``D.backward()`` and
read ``.grad`` from the stored tensors. Gradients are read INSIDE the try
block (before cleanup); the ``finally`` drops all stored references and
``run_with_hooks`` removes its hooks, so no state leaks into later requests.

KNOWN LIMITATION (stated openly, per the spec): AtP is a FIRST-ORDER
approximation and is known to be unreliable when attention saturates
(Syed et al. 2023). Disagreement with exact patching is expected for some
components and is a result, not a bug.
"""

from __future__ import annotations

import torch

from ..config import MAX_LAYERS
from ..model_runtime.loader import ADAPTER
from .components import COMPONENT_NAMES, N_COMPONENTS, token_ids

MAX_HEADS = 12

LIMITATION_NOTE = (
    "Attribution patching is a first-order (linear) approximation of the exact "
    "patching effect and is known to be unreliable when attention saturates; "
    "disagreement with exact patching is expected for some components and is a "
    "result, not a bug."
)


def compute_atp(
    corrupted_text: str,
    candidate_a: str,
    candidate_b: str,
    logits_clean: torch.Tensor,
    clean_cache: dict,
) -> dict:
    """AtP for all 156 components (heads + MLPs), denoising direction, metric D.

    ``logits_clean``/``clean_cache`` come from one clean capture pass (see
    components.run_clean). The corrupted pass runs once with grad enabled;
    ``D_clean - D_corr`` comes from the same real runs used everywhere else.
    """
    tok_a, tok_b = token_ids(candidate_a, candidate_b)
    bridge = ADAPTER.ensure_loaded()
    d_clean = float(logits_clean[0, -1, tok_a] - logits_clean[0, -1, tok_b])
    toks = bridge.to_tokens(corrupted_text)

    stored: dict[str, torch.Tensor] = {}

    def embed_hook(t: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        # Leaf with a graph: everything downstream of the embeddings becomes
        # part of the autograd graph even though all parameters are frozen.
        return t.detach().requires_grad_(True)

    def make_hook(name: str):
        def hook_fn(t: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
            # Do NOT detach: indirect paths through later layers must survive.
            t.retain_grad()
            stored[name] = t
            return t

        return hook_fn

    fwd_hooks: list[tuple[str, object]] = [("hook_embed", embed_hook)]
    for l in range(MAX_LAYERS):
        for name in (f"blocks.{l}.attn.hook_z", f"blocks.{l}.mlp.hook_out"):
            fwd_hooks.append((name, make_hook(name)))

    atp_d = [0.0] * N_COMPONENTS
    d_corr = float("nan")
    try:
        with torch.enable_grad():
            logits = bridge.run_with_hooks(toks, fwd_hooks=fwd_hooks)
            missing = [name for name, _fn in fwd_hooks if name != "hook_embed" and name not in stored]
            if missing:
                raise RuntimeError(f"AtP hooks did not fire: {missing[:4]}...")
            d = logits[0, -1, tok_a] - logits[0, -1, tok_b]
            d.backward()
            # Read gradients NOW (tensors still hold .grad); the finally below
            # then drops every reference so no graph survives the call.
            for l in range(MAX_LAYERS):
                name_z = f"blocks.{l}.attn.hook_z"
                z_t = stored[name_z]
                if z_t.grad is None:
                    raise RuntimeError(f"no gradient captured for {name_z}")
                for h in range(MAX_HEADS):
                    a_clean = clean_cache[name_z][0, :, h, :]  # [seq, 64]
                    a_corr = z_t.detach()[0, :, h, :]
                    g = z_t.grad[0, :, h, :]
                    atp_d[l * MAX_HEADS + h] = float(((a_clean - a_corr) * g).sum().item())
                name_m = f"blocks.{l}.mlp.hook_out"
                m_t = stored[name_m]
                if m_t.grad is None:
                    raise RuntimeError(f"no gradient captured for {name_m}")
                atp_d[144 + l] = float(
                    ((clean_cache[name_m][0, :, :] - m_t.detach()[0, :, :]) * m_t.grad[0, :, :]).sum().item()
                )
            d_corr = float(logits[0, -1, tok_a].detach() - logits[0, -1, tok_b].detach())
    finally:
        # Drop all references to the grad graph; run_with_hooks has already
        # removed the hooks. A later normal request must build no graph
        # (tested in backend/tests/test_atp.py).
        stored.clear()

    denominator = d_clean - d_corr
    rec_atp: list[float | None] = [
        (v / denominator) if denominator != 0.0 else None for v in atp_d
    ]
    return {
        "component_names": list(COMPONENT_NAMES),
        "atp_d": atp_d,
        "rec_atp": rec_atp,
        "D_clean": d_clean,
        "D_corr": d_corr,
        "denominator": denominator,
        "note": LIMITATION_NOTE,
    }
