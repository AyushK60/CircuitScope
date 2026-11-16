"""Stage 0 benchmark: how expensive are the Stage 2 sweeps on this machine?

READ-ONLY on all app data: writes only its own result file
``outputs/bench_sweep.json``. It never touches the database.

Measured (warm up once, then median/min over N repetitions, time.perf_counter):
  (a) one clean forward pass, run_with_cache filtered to a single hook
  (b) one clean forward pass, run_with_cache capturing all 12 blocks.{i}.hook_out
  (c) one patched forward pass via engine.run_patched, blocks.5.hook_out, pos 9
  (d) one patched forward pass via engine.run_patched, blocks.5.attn.hook_z,
      pos 9, one head
  (e) batched throughput: plain forward over batches of 8 and 32 copies of the
      corrupted tokens; per-sequence time reported

Extrapolated (formula printed, median timing used):
  layer x position sweep  : 12 x 14  =  168 patched runs (hook_out)
  layer x head @ pos 9    : 12 x 12  =  144 patched runs (hook_z)
  layer x head x position : 12 x 12 x 14 = 2016 patched runs (hook_z)
  ... and each multiplied by the planned 50-prompt prompt set.

Usage: python scripts/bench_sweep.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from backend.app.model_runtime import engine  # noqa: E402
from backend.app.model_runtime.loader import ADAPTER  # noqa: E402

CLEAN = "John and Mary went to the shop. John gave a book to"
CORRUPTED = "John and Mary went to the shop. Mary gave a book to"
HOOK_RESID = "blocks.5.hook_out"
HOOK_Z = "blocks.5.attn.hook_z"
POSITION = 9
N_LAYERS = 12
N_HEADS = 12
SEQ = 14  # BOS + 13 tokens
REPS = 12  # timed repetitions per measurement (>= 10)
WARMUP = 2
PROMPTS = 50  # planned prompt-set size

OUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "bench_sweep.json"


def time_it(fn, reps: int = REPS, warmup: int = WARMUP) -> dict:
    """Warm up, then time `reps` calls; returns median/min seconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": round(statistics.median(samples), 6),
        "min_s": round(min(samples), 6),
        "reps": reps,
    }


def main() -> dict:
    torch.set_grad_enabled(False)
    print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} "
          f"| torch threads={torch.get_num_threads()}")

    print("loading model (first use, ~30-60s when cached) ...")
    t0 = time.perf_counter()
    bridge = ADAPTER.ensure_loaded()
    print(f"loaded in {time.perf_counter() - t0:.1f}s on {ADAPTER._device}")

    toks_clean = bridge.to_tokens(CLEAN)
    toks_corr = bridge.to_tokens(CORRUPTED)
    assert toks_clean.shape[1] == SEQ == toks_corr.shape[1]

    # One clean capture of both target hooks, reused by every patched run.
    _, clean_cache = bridge.run_with_cache(
        toks_clean, names_filter=lambda n: n in (HOOK_RESID, HOOK_Z)
    )
    assert HOOK_RESID in clean_cache and HOOK_Z in clean_cache

    results: dict = {"environment": {}, "measured": {}, "extrapolations": {}}
    results["environment"] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "device": ADAPTER._device,
    }

    print(f"\nmeasuring ({REPS} timed reps each, after {WARMUP} warmups)...")

    # (a) clean forward + single-hook capture
    a = time_it(lambda: bridge.run_with_cache(toks_clean, names_filter=lambda n: n == HOOK_RESID))
    print(f"(a) clean forward, 1 hook captured          median {a['median_s']*1000:8.2f} ms  min {a['min_s']*1000:7.2f} ms")
    results["measured"]["a_clean_forward_one_hook"] = a

    # (b) clean forward + all 12 blocks.{i}.hook_out
    all_hook_out = {f"blocks.{i}.hook_out" for i in range(N_LAYERS)}
    b = time_it(lambda: bridge.run_with_cache(toks_clean, names_filter=lambda n: n in all_hook_out))
    print(f"(b) clean forward, 12 hook_out captured     median {b['median_s']*1000:8.2f} ms  min {b['min_s']*1000:7.2f} ms")
    results["measured"]["b_clean_forward_12_hook_out"] = b

    # (c) patched forward, residual hook, one position
    c = time_it(lambda: engine.run_patched(CORRUPTED, clean_cache, HOOK_RESID, [POSITION]))
    print(f"(c) patched, hook_out @pos {POSITION} (1 pos)    median {c['median_s']*1000:8.2f} ms  min {c['min_s']*1000:7.2f} ms")
    results["measured"]["c_patched_hook_out_1pos"] = c

    # (d) patched forward, hook_z, one head at one position
    d = time_it(lambda: engine.run_patched(CORRUPTED, clean_cache, HOOK_Z, [POSITION], [0]))
    print(f"(d) patched, hook_z @pos {POSITION}, 1 head     median {d['median_s']*1000:8.2f} ms  min {d['min_s']*1000:7.2f} ms")
    results["measured"]["d_patched_hook_z_1head_1pos"] = d

    # (e) batched plain forward throughput
    t_hookout = c["median_s"]
    t_hookz = d["median_s"]
    per_seq = {}
    for batch in (8, 32):
        batch_toks = toks_corr.expand(batch, -1).contiguous()

        def run_batch():
            bridge(batch_toks)

        e = time_it(run_batch)
        per_seq[batch] = {
            "batch_forward_median_s": e["median_s"],
            "per_sequence_median_s": round(e["median_s"] / batch, 6),
        }
        print(f"(e) batch {batch:2d} plain forward              median {e['median_s']*1000:8.2f} ms "
              f"-> per-seq {e['median_s']/batch*1000:7.2f} ms")
    results["measured"]["e_batched_forward"] = per_seq

    # Extrapolations (median timing; formulas visible in output).
    n_hookout_1prompt = N_LAYERS * SEQ      # 168
    n_hookz_pos9_1prompt = N_LAYERS * N_HEADS  # 144
    n_hookz_full_1prompt = N_LAYERS * N_HEADS * SEQ  # 2016
    ext = {
        "layer_x_position_hook_out_1prompt": {
            "runs": n_hookout_1prompt,
            "seconds": round(n_hookout_1prompt * t_hookout, 1),
        },
        "layer_x_head_pos9_hook_z_1prompt": {
            "runs": n_hookz_pos9_1prompt,
            "seconds": round(n_hookz_pos9_1prompt * t_hookz, 1),
        },
        "layer_x_head_x_position_hook_z_1prompt": {
            "runs": n_hookz_full_1prompt,
            "seconds": round(n_hookz_full_1prompt * t_hookz, 1),
        },
    }
    print("\nextrapolations (median per-run time x runs):")
    print(f"  layer x position (hook_out): 12 x {SEQ} = {n_hookout_1prompt} runs "
          f"= {ext['layer_x_position_hook_out_1prompt']['seconds']:.0f} s per prompt")
    print(f"  layer x head @ pos {POSITION} (hook_z): 12 x {N_HEADS} = {n_hookz_pos9_1prompt} runs "
          f"= {ext['layer_x_head_pos9_hook_z_1prompt']['seconds']:.0f} s per prompt")
    print(f"  layer x head x position (hook_z): 12 x {N_HEADS} x {SEQ} = {n_hookz_full_1prompt} runs "
          f"= {ext['layer_x_head_x_position_hook_z_1prompt']['seconds']:.0f} s per prompt")
    for name, e in list(ext.items()):
        ext[name + "_x50_prompts"] = {
            "runs": e["runs"] * PROMPTS,
            "seconds": round(e["seconds"] * PROMPTS, 1),
            "minutes": round(e["seconds"] * PROMPTS / 60, 1),
        }
        print(f"  x{PROMPTS} prompts: {name} = {ext[name + '_x50_prompts']['minutes']:.0f} min")
    results["extrapolations"] = ext

    # Could batching replace per-run patching?
    fast_per_seq = per_seq[32]["per_sequence_median_s"]
    verdict = (
        f"Plain-forward batching achieves {fast_per_seq*1000:.1f} ms/sequence at batch 32, "
        f"but a patched sweep needs a DIFFERENT hook patch per run (per-layer/per-head/per-position "
        f"splice), which batching cannot express without engine changes: engine.run_patched issues "
        f"one run_with_hooks call per patch config and takes a single corrupted_text string, not a "
        f"batch, and the patch hook writes src[0, p] per sequence index 0 only. So: batching plain "
        f"forwards does NOT replace per-run patching as-is; batched patched runs would require "
        f"engine changes (batched corrupted input + per-sequence patch source/positions). "
        f"NOT made in this stage."
    )
    results["batching_verdict"] = verdict
    print("\n" + verdict)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
