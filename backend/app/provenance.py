"""Reproducibility provenance for experiment records.

collect() returns a dict describing exactly what ran: model identity + real
Hugging Face revision, device/dtype, library versions, git state, UTC timestamp.
Everything except timestamp_utc and git_commit is cached (git and metadata
lookups are slow, but a long-running server must not report a stale HEAD, so
git_commit is computed fresh on every call). Model loading is never triggered
here — callers run collect() after the model is already loaded.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version as _dist_version
from pathlib import Path

from .config import MODEL_ID
from .model_runtime.loader import ADAPTER

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _version(dist_name: str) -> str:
    try:
        return _dist_version(dist_name)
    except Exception:
        return "unknown"


def _model_revision() -> str:
    """Real HF commit hash of the loaded weights, or 'unknown'.

    Order: (1) the loaded HF model config's _commit_hash via the bridge,
    (2) the tokenizer's _commit_hash, (3) the local HF cache snapshot dir for
    MODEL_ID (what from_pretrained(MODEL_ID) resolves to without a revision).
    """
    try:
        if ADAPTER.is_loaded:
            bridge = ADAPTER.bridge()
            for path in ("hf_model.config", "original_model.config", "tokenizer"):
                obj: object = bridge
                for attr in path.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is None:
                    continue
                commit = getattr(obj, "_commit_hash", None)
                if commit:
                    return str(commit)
    except Exception:
        pass
    # Cache-path fallback: models--<id>/snapshots/<hash> (single snapshot here).
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_dir = Path(HF_HUB_CACHE) / f"models--{MODEL_ID}"
        refs = repo_dir / "refs" / "main"
        if refs.exists():
            rev = refs.read_text().strip()
            if rev:
                return rev
        snaps = repo_dir / "snapshots"
        if snaps.exists():
            for entry in snaps.iterdir():
                if entry.is_dir() and len(entry.name) == 40:
                    return entry.name
    except Exception:
        pass
    return "unknown"


def _git_commit() -> str:
    """Short HEAD hash, '+dirty' suffix when the tree is not clean."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            timeout=3,
            check=False,
            capture_output=True,
            text=True,
        )
        if head.returncode != 0:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            timeout=3,
            check=False,
            capture_output=True,
            text=True,
        )
        commit = head.stdout.strip()
        if not commit:
            return "unknown"
        if status.returncode == 0 and status.stdout.strip():
            commit += "+dirty"
        return commit
    except Exception:
        return "unknown"


_STATIC_CACHE: dict | None = None


def _static() -> dict:
    """Static provenance, cached only once the model is loaded.

    Caching before load would freeze the placeholder device/dtype defaults for
    the whole process (wrong on a CUDA machine), so an unloaded call is never
    cached.
    """
    global _STATIC_CACHE
    if _STATIC_CACHE is not None:
        return _STATIC_CACHE
    device = "cpu"
    dtype = "float32"
    try:
        if ADAPTER.is_loaded:
            bridge = ADAPTER.bridge()
            param = next(bridge.parameters())
            device = str(param.device)
            dtype = str(param.dtype)
    except Exception:
        pass
    static = {
        "model_id": MODEL_ID,
        "model_revision": _model_revision(),
        "device": device,
        "dtype": dtype,
        "versions": {
            "torch": _version("torch"),
            "transformer_lens": _version("transformer_lens"),
            "transformers": _version("transformers"),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
    }
    if ADAPTER.is_loaded:
        _STATIC_CACHE = static
    return static


def _clear_static_cache() -> None:
    global _STATIC_CACHE
    _STATIC_CACHE = None


def collect() -> dict:
    """Provenance for one record; safe to call after the model is loaded.

    git_commit is computed per call so a long-running server reports the
    commit it is actually running on, not the one from process start.
    """
    out = dict(_static())
    out["git_commit"] = _git_commit()
    out["timestamp_utc"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return out
