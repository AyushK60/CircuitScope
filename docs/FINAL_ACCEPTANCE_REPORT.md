# FINAL ACCEPTANCE REPORT

Date: 2026-10-30. Stack: Python 3.11.9 · torch 2.15.0+cpu · transformer-lens
4.3.0 · transformers 5.22.0 · FastAPI 0.162 · React 19 · Vite 8 · Node 24.

## P0 release-blocking gates — ALL PASS

- [x] Real GPT-2 Small loads locally, no paid service (CPU; GPU documented as optional upgrade)
- [x] Baseline prediction + exact tokenization computed live (test_predict_matches_raw_hf; landing demo)
- [x] Activations from real hook points with correct shapes (637 hook points; shape assertions in tests)
- [x] Real activation replacement executed and validated independently
      (backend vs raw-HF reference: D 2.5128 / −4.1768 / 1.9426, Recovery 0.9148, tol 5e-3)
- [x] Diagram respects real GPT-2 order (pre-LN, residual adds) from live metadata
- [x] No invented data paths / no fake timing (static-vs-live labeling; schematic disclaimer)
- [x] Every chart value traces to stored results (renders ExperimentResult verbatim; exports)
- [x] Attention never presented as causal proof (explicit disclaimers in Explorer/Learn)
- [x] Guided journey usable in real browser (6 Playwright E2E pass against live stack)
- [x] Unsupported model/hooks/inputs fail honestly (422s; failed-run states; misalignment test)
- [x] Localhost-only binding; no arbitrary code execution (allowlists; pydantic bounds)
- [x] API + numerical tests pass (26 pytest, 6 Vitest, 6 Playwright)

## Milestone gates

| Gate | Status | Evidence |
|---|---|---|
| M0 env reproducible | ✅ | pinned requirements.txt; .venv present; startup docs |
| M1 authentic inference | ✅ | predict + cache shapes; CLI via tests/scripts |
| M2 verified intervention | ✅ | reference agreement + no-op + leak tests |
| M3 API operable | ✅ | HTTP tests cover full lifecycle incl. failure paths |
| M4 landing usable | ✅ | browser screenshots; live status chip; working CTAs |
| M5 explorer truthful | ✅ | real metadata counts; hook-labeled heatmap 12×14×14 |
| M6 playground E2E | ✅ | browser run: D 1.943, Δ 6.119, Recovery 0.915; exports |
| M7 quality | ✅ | all suites pass; responsive 320px overflow fixed; console clean |

## Features intentionally incomplete (honest)

1. **GPU inference** — CPU-only torch installed; works but slower for long inputs.
   Upgrade path documented in EXECUTION_STATE.md.
2. **Experiment history UI** — SQLite persistence + export APIs exist; no browse/
   rename/delete screen yet.
3. **Additional activation views** — MLP/residual summaries available via
   /api/inspect but not visualized (attention heatmap is the implemented view).
4. **PNG chart export** — JSON/CSV only.
5. **Zero-ablation / scaling interventions** — not exposed (README permits
   patching as the guaranteed intervention; others need numeric checks first).
6. **Computation-flow playback (Feature C)** — the guided stage story is
   covered by the Explorer's model→block→head walkthrough and static stage
   annotations; a separate animated playback mode was deferred in favor of the
   verified vertical slice. No fabricated timing animations were added.

## Verification evidence summary

- 26/26 pytest (engine 15, API 8, reference 1, hooks 1 + metric tests)
- 6/6 Vitest component tests
- 6/6 Playwright E2E (landing live prediction; explorer drill-in; full
  playground experiment; learn depths; 320px no-overflow; console-clean)
- Independent reference script reproduces backend numbers (5e-3 tolerance)
- Visual review via real browser screenshots (desktop 1280px + mobile 390/320px)

## Launch

```bat
:: from E:\CircuitScope
.venv\Scripts\python -m backend.app.main     :: http://127.0.0.1:8000
cd frontend && npm run dev                    :: http://localhost:5173
```
