# EXECUTION STATE — resumable checkpoint

Last updated: 2026-10-30. All milestone gates M0–M7 have been executed; see
FINAL_ACCEPTANCE_REPORT.md for the honest gate-by-gate status.

## What works right now

- **Backend** (`backend/`): GPT-2 Small loads via TransformerLens 4.3.0
  `TransformerBridge` (CPU, torch 2.15.0+cpu). Real prediction, activation
  capture, attention patterns, and clean/corrupted/patched experiments.
  FastAPI on 127.0.0.1:8000, SQLite persistence, serial one-job worker.
- **Frontend** (`frontend/`): React 19 + TypeScript + Vite. Landing (live
  prediction demo), Model Explorer (3 levels + inspector), Intervention
  Playground (real 3-run experiments, metrics, exports), Learn page.
- **Tests**: 26 pytest (engine, API, reference agreement), 6 Vitest,
  6 Playwright E2E — all passing.

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
3. Experiment history in the UI shows current-session jobs; SQLite persists
   results but there is no history browse screen yet (API + export exist).
4. Explorer attention heatmap is the only measured activation view; MLP/
   residual activation views are API-supported but not yet visualized.
5. No PNG chart export (JSON + CSV only).
6. Playwright suite requires both servers running (no auto-webServer config).

## Next steps if resuming

- History browse UI + rename/delete with confirmation.
- Residual-stream/MLP activation views in Explorer (data exists via /api/inspect).
- CUDA torch install + GPU verification run.
- Optional: zero-ablation intervention type (backend design ready via heads/positions extension).
