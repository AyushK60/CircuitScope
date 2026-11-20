"""Make sure the GPT-2 weights the app uses are in the local Hugging Face cache.

Default mode: check the cache; if the weights are already cached, report and
exit 0 without downloading or loading anything. If not, load the model through
the same path the app uses (backend.app.model_runtime.loader -> TransformerLens
TransformerBridge), which downloads into the HF cache with visible progress.

--check: report only (exit 0 cached / 1 not cached). Never downloads, never
loads the model.

Usage (from the repo root, in the project venv):
    .venv\\Scripts\\python scripts\\download_model.py
    .venv\\Scripts\\python scripts\\download_model.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _hf_cache_dir() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / "models--gpt2"


def model_cached() -> tuple[bool, str]:
    """(cached?, human-readable detail). Local disk inspection only; no network.

    Cached means: a 40-hex snapshot directory exists and contains the config,
    the tokenizer files and the weights the app's from_pretrained("gpt2") needs.
    """
    repo_dir = _hf_cache_dir()
    if not repo_dir.exists():
        return False, f"no HF cache directory at {repo_dir}"
    snaps = repo_dir / "snapshots"
    if not snaps.exists():
        return False, f"no snapshots directory in {repo_dir}"
    for entry in sorted(snaps.iterdir()):
        if not (entry.is_dir() and len(entry.name) == 40):
            continue
        files = {f.name for f in entry.iterdir()}
        has_config = "config.json" in files
        has_weights = any(
            name in files or (entry / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin")
        ) or any(n.startswith("pytorch_model-") and n.endswith(".bin") for n in files)
        has_tokenizer = bool(files & {"tokenizer.json", "vocab.json", "tokenizer_config.json"})
        if has_config and has_weights and has_tokenizer:
            return True, f"snapshot {entry.name} ({len(files)} files) in {repo_dir}"
        return False, (
            f"snapshot {entry.name} is incomplete "
            f"(config={has_config}, weights={has_weights}, tokenizer={has_tokenizer})"
        )
    return False, f"no 40-hex snapshot directory under {snaps}"


def load_through_app_path() -> int:
    """Load GPT-2 exactly like the app does, populating the HF cache."""
    print("Loading GPT-2 through the app's path (TransformerLens TransformerBridge)...")
    print("First-time downloads print their own progress bars.")
    sys.path.insert(0, str(REPO_ROOT))
    from backend.app.model_runtime.loader import ADAPTER
    from backend.app.provenance import _model_revision

    bridge = ADAPTER.ensure_loaded()
    revision = _model_revision()
    device = str(next(bridge.parameters()).device)
    print(f"Model loaded: gpt2, revision {revision}, device {device}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the weights are already cached (exit 0/1); never downloads",
    )
    args = parser.parse_args()

    cached, detail = model_cached()
    if args.check:
        if cached:
            print(f"OK: GPT-2 weights are cached ({detail})")
            return 0
        print(f"NOT CACHED: {detail}")
        print("Run scripts/setup.bat (or: .venv\\Scripts\\python scripts\\download_model.py)")
        return 1

    if cached:
        print(f"OK: GPT-2 weights are already cached ({detail}); nothing to download.")
        return 0
    print(f"GPT-2 weights not cached yet ({detail}).")
    return load_through_app_path()


if __name__ == "__main__":
    sys.exit(main())
