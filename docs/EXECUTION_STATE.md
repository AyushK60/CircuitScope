# EXECUTION STATE — resumable checkpoint

Last updated: 2026-10-30 (v2 upgrade). See FINAL_ACCEPTANCE_REPORT.md for the
honest gate-by-gate status and docs/DECISIONS.md for design decisions.

## v2 upgrade (this session)

- **/api/flow** — one real forward pass recorded: exact tokens, per-block
  residual-stream L2 norms (per token + mean), and all 12 layers' attention
  patterns. Powers the playback and data views.
- **"Sumi & Ruri" visual system** — Japanese traditional color palette
  (墨/藍鉄 base, 瑠璃 primary, 浅葱 measured signal, 朱 interventions, 山吹
  patched, 若竹 clean, 猩紅 corrupted), mincho display headings, washi-texture
  background, dark color-scheme for native controls.
- **Computation Flow playback** (Explorer ▸ Computation Flow tab) — guided
  replay of 16 recorded stages (tokenize, embed, blocks 0–11, final LN,
  logits) with play/pause/step/jump and a completed state. Every heatmap, norm
  and score is measured data from /api/flow; timing label states that stage
  pacing is presentation, not GPU kernel timing.
- **Connected data views** — model-level residual-norm-across-blocks chart
  (measured, clickable), per-token norm chart at block level, head-level
  attention heatmap (existing) — all from the same captured run.
- **Responsive fixes** — labels no longer overlap inline controls; SVG
  diagrams are viewBox-scalable (320px verified, overflow ≤ 2px).

## Extension release (this session)

- **/api/continue** — deterministic greedy GPT-2 continuation (1–8 tokens,
  measured per-step candidates; pytest-verified deterministic).
- **/api/predict candidates** — optional measured candidate comparison (D,
  probabilities) on any prompt; multi-token candidates rejected honestly.
- **Neural Network Lab** (#/lab) — a genuine 2-8-8-1 MLP trained live in the
  browser (real gradient descent, two-moons): loss curve, decision boundary,
  per-node activations, probe inspection. Clearly labeled as an educational
  simulation, not GPT-2.
- **Playground**: editable candidate A/B with live measured comparison card;
  four-step guided experiment wizard; greedy continuation panel; persisted
  experiment history panel (load past results).
- **Explorer**: intervention overlay (hook + position marked on the canvas
  from URL params, synced with Playground), all-heads attention gallery
  (small multiples of every measured pattern in a block).
- Tests: 35 pytest · 7 Vitest · 13 E2E · production build — all passing.

## What works right now

- **Backend** (`backend/`): GPT-2 Small loads via TransformerLens 4.3.0
  `TransformerBridge` (CPU, torch 2.15.0+cpu). Real prediction, activation
  capture, attention patterns, and clean/corrupted/patched experiments.
  FastAPI on 127.0.0.1:8000, SQLite persistence, serial one-job worker.
- **Frontend** (`frontend/`): React 19 + TypeScript + Vite. Landing (live
  prediction demo), Model Explorer (3 levels + inspector), Intervention
  Playground (real 3-run experiments, metrics, exports), Learn page.
- **Tests**: 30 pytest (engine, API, flow capture, reference agreement), 6
  Vitest, 8 Playwright E2E — all passing.

## Launch procedure (Windows, from E:\CircuitScope)

```bat
:: terminal 1 — backend (model loads lazily on first request, ~30–60 s once cached)
.venv\Scripts\python -m backend.app.main

:: terminal 2 — frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**. Backend API docs at http://127.0.0.1:8000/docs.

Model weights: cached in `%USERPROFILE%\.cache\huggingface\hub\models--gpt2`
(downloaded once, ~500 MB).

## Test commands

```bat
.venv\Scripts\python -m pytest backend\tests -q          :: scientific + API tests
.venv\Scripts\python scripts\reference_patch.py          :: independent HF reference
cd frontend && npx vitest run                            :: unit tests
cd frontend && npx playwright test                       :: E2E (needs both servers up)
```

## Known defects / limitations (honest list)

1. Torch build is CPU-only (2.15.0+cpu). GPU works only after installing a
   CUDA-enabled torch wheel; the RTX 5050 (Blackwell) needs a cu128+ build.
   Backend auto-selects `cuda` when available.
2. `blocks.N.attn.hook_result` exists in hook_dict but is NOT captured by
   `run_with_cache` in TransformerLens 4.3.0 — excluded from the allowlist.
3. ~~Experiment history browse UI~~ — resolved in the HORIZON release: the
   Playground history panel lists persisted experiments, with reopen,
   delete-with-confirmation, and two-experiment comparison.
4. Flow playback stage pacing is fixed (1.6 s/stage) — deliberately not
   presented as measured latency; only elapsed_seconds is a real measurement.
5. No PNG chart export (JSON + CSV only).
6. Playwright suite requires both servers running (no auto-webServer config).

## Next steps if resuming

Remaining (optional or not yet done). History browse, rename/delete and the
history comparison are complete; they are no longer listed here.

- Residual-stream/MLP activation magnitude views beyond norms (e.g., top
  contributing dimensions) via /api/inspect.
- CUDA torch install + GPU verification run (M7 gate: GPU/OOM fallback not tested).
- Fresh clean-checkout install and startup on Windows (M7 gate; not separately verified).
- Optional: zero-ablation intervention type (backend design ready via heads/positions extension).
- Optional P2 items from README section 9 (another small model, more intervention types, PNG export).
