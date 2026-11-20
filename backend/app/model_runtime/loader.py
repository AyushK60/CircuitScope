"""Model adapter: lazy, thread-safe singleton around TransformerBridge.

Loads GPT-2 Small via TransformerLens 4.x TransformerBridge on first use and
exposes verified metadata, tokenization, inference, capture and patch
operations. CPU-only torch build is used in this environment (torch 2.15.0+cpu);
device selection degrades gracefully and records what actually runs.
"""

from __future__ import annotations

import threading

import torch

from ..config import MODEL_ID

_LOAD_LOCK = threading.Lock()


class ModelNotLoadedError(RuntimeError):
    pass


class ModelAdapter:
    def __init__(self) -> None:
        self._bridge: object | None = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._bridge is not None

    def ensure_loaded(self, device: str | None = None):
        with _LOAD_LOCK:
            if self._bridge is not None:
                return self._bridge
            from transformer_lens.model_bridge import TransformerBridge

            target = device or ("cuda" if torch.cuda.is_available() else "cpu")
            bridge = TransformerBridge.boot_transformers(MODEL_ID, device=target)
            bridge.eval()
            # set_grad_enabled is thread-local; request/job threads would still
            # build autograd graphs. Freezing parameters makes inference grad-free
            # on every thread.
            for p in bridge.parameters():
                p.requires_grad_(False)
            torch.set_grad_enabled(False)
            self._bridge = bridge
            self._device = str(next(bridge.parameters()).device)
            return bridge

    def bridge(self):
        if self._bridge is None:
            raise ModelNotLoadedError("model not loaded")
        return self._bridge

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> dict:
        bridge = self.ensure_loaded()
        cfg = bridge.cfg
        # Real HF commit hash of the loaded weights (40 hex), via provenance's
        # resolution order (bridge config → tokenizer → local cache). Lazy
        # import: provenance imports this module's ADAPTER singleton.
        from ..provenance import _model_revision

        return {
            "model_id": MODEL_ID,
            "revision": getattr(bridge.tokenizer, "name_or_path", MODEL_ID),
            "revision_hash": _model_revision(),
            "device": self._device,
            "dtype": str(next(bridge.parameters()).dtype),
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "d_head": cfg.d_head,
            "d_mlp": cfg.d_mlp,
            "n_ctx": cfg.n_ctx,
            "d_vocab": cfg.d_vocab,
            "n_params_approx": round((cfg.n_params or 124439808) / 1e6, 1),
            "tokenizer_class": type(bridge.tokenizer).__name__,
        }

    # -- tokenization ------------------------------------------------------

    def tokenize(self, text: str) -> dict:
        bridge = self.ensure_loaded()
        str_toks = bridge.to_str_tokens(text)
        ids = bridge.to_tokens(text)[0].tolist()
        return {
            "tokens": str_toks,
            "token_ids": ids,
            "n_tokens": len(str_toks),
            "single_token_candidates": {
                t: bridge.to_single_token(t)
                for t in (" Mary", " John")
                if bridge.to_single_token(t) is not None
            },
        }

    def token_id_for_str(self, s: str) -> int:
        bridge = self.ensure_loaded()
        try:
            tid = bridge.to_single_token(s)
        except AssertionError as e:
            raise ValueError(f"candidate {s!r} is not a single token") from e
        if tid is None:
            raise ValueError(f"candidate {s!r} is not a single token")
        return int(tid)


# Module-level singleton used by the API and worker.
ADAPTER = ModelAdapter()
