# ARCHITECTURE

```
Browser (React 19 + TS, Vite, port 5173)
  Landing / Model Explorer / Playground / Learn
        │  fetch (typed client: frontend/src/api/client.ts)
        ▼
FastAPI (127.0.0.1:8000)  backend/app/
  api/routes.py        → /api/health, /models, /predict, /inspect,
                         /experiments (+status/cancel/list/export), /hooks
  schemas (pydantic)   → input bounds: 512 chars, top_k ≤ 20, layer ≤ 11
  experiments/jobs.py  → validate → serial worker thread → JobRecord lifecycle
  experiments/metrics.py → D, Delta, Recovery (unclamped, guarded)
  model_runtime/loader.py → thread-safe TransformerBridge singleton
  model_runtime/engine.py → predict / capture / attention_pattern / run_patched
  storage/db.py        → SQLite (outputs/experiments.db): id, config, results JSON
        │
        ▼
PyTorch 2.15.0+cpu · TransformerLens 4.3.0 TransformerBridge · GPT-2 Small (124M)
  hook points: 637 registered; verified-capturable subset allowlisted in config.py
```

## Intervention semantics (the core science)

`run_patched(corrupted_text, clean_cache, hook, positions, heads)`:

1. Clone the clean-run tensor (`clean_cache[hook].clone()`) — baselines immutable.
2. Register a forward hook on the exact hook point that writes
   `z[0, pos, :] = src[0, pos, :]` (per-head for 4-D `hook_z`).
3. Run the corrupted prompt through the real model; hooks auto-reset afterward.
4. Return measured logits → metrics.

Equivalence verified against an independent implementation
(`scripts/reference_patch.py`: raw HF transformers + PyTorch forward hooks, no
TransformerLens) to within 5e-3: D_clean 2.5128, D_corrupted −4.1768,
D_patched 1.9426, Recovery 0.9148.

## Frontend truthfulness contract

- The hero schematic is labeled "ARCHITECTURAL ILLUSTRATION"; live views are
  labeled with hook IDs, shapes, device, and value scales.
- The Explorer's block count/head count/shape labels come from `/api/health`
  metadata, never hardcoded.
- Playground metrics cards, run columns, and exports all render
  `ExperimentResult` fields verbatim; no-effect and failed runs have distinct
  honest states.
- GPT-2 block order rendered: pre-LN (x → LN1 → attn → + → LN2 → MLP → +),
  matching the running model, with residual paths drawn as violet dashed edges.
