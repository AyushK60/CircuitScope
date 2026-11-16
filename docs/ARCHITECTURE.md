# ARCHITECTURE

```
Browser (React 19 + TS, Vite, port 5173)
  Landing / Model Explorer / Playground / Learn
  pages/playground/    → split panels (presets, url, TargetStrip, MetricCard,
                         GuidedWizard, ExperimentForm, ResultsPanel,
                         ContinuationPanel, QuestionPanel, HistoryPanel,
                         useIoiPreset hook); Playground.tsx composes them
  components/architecture/ → canvas levels (ModelLevel, BlockLevel,
                         AttentionLevel, shared parts, layout constants);
                         ArchitectureCanvas.tsx composes them
        │  fetch (typed client: frontend/src/api/client.ts)
        ▼
FastAPI (127.0.0.1:8000)  backend/app/
  api/routes.py        → /api/health, /models, /predict, /inspect,
                         /experiments (+status/cancel/list/export), /hooks,
                         /api/tasks (+ per-task validation)
  schemas (pydantic)   → input bounds: 512 chars, top_k ≤ 20, layer ≤ 11
  tasks/registry.py    → task definition files (examples/tasks/*.json);
                         malformed files recorded, never fatal; per-pair
                         validation against the real tokenizer runs lazily
                         on request and is cached
  experiments/jobs.py  → job-type registry (JOB_TYPES: validate/run fns) →
                         serial worker thread → JobRecord lifecycle with
                         type/progress/progress_message; JobCancelled is
                         distinct from ExperimentValidationError
  experiments/metrics.py → D, Delta, Recovery (unclamped, guarded)
  model_runtime/loader.py → thread-safe TransformerBridge singleton
  model_runtime/engine.py → predict / capture / attention_pattern / run_patched
  storage/db.py        → SQLite (outputs/experiments.db): id, config, results JSON
        │
        ▼
PyTorch 2.15.0+cpu · TransformerLens 4.3.0 TransformerBridge · GPT-2 Small (124M)
  hook points: 637 registered; verified-capturable subset allowlisted in config.py
```

## Task registry (v2, Stage 1)

Each task (e.g. IOI) is a data file in `examples/tasks/*.json` — prompts,
candidate tokens, positions of interest, recommended hooks, source citation.
Adding a task needs no code change. `GET /api/tasks` lists them (with any
load errors); `GET /api/tasks/{id}` returns the task plus validation performed
against the real tokenizer (single-token candidates, token alignment, position
ranges, hook allowlist). The Playground loads the IOI default pair at mount
and falls back to its hardcoded preset if the registry is unavailable.

## Job engine (v2, Stage 1)

`ExperimentRunner.submit(cfg, job_type="intervention")` remains the one heavy
job at a time. Registered job types provide `validate_fn(cfg)` and
`run_fn(config, ctx)`; `ctx.report(progress, message)` clamps to [0, 1] and
never decreases, `ctx.check_cancel()` raises `JobCancelled` (distinct from
`ExperimentValidationError`, which maps to "failed"). Results are persisted to
SQLite **before** a job is marked complete, so "complete" always implies the
record is retrievable. The intervention pipeline reports progress at fixed
milestones (0.05 / 0.2 / 0.5 / 0.9 / 1.0); live job status JSON carries
`type`, `progress`, `progress_message`.

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
