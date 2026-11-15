# CircuitScope — Implementation Plan

Short working plan derived from `README.md`. Detailed resumable state lives in
`docs/EXECUTION_STATE.md`; every deviation from the README's preferred stack is
recorded in `docs/DECISIONS.md`.

## Milestones

| ID | Milestone | Gate | Status |
| --- | --- | --- | --- |
| M0 | Environment audit + repo foundation | Reproducible env, pinned deps, startup docs | done |
| M1 | Real local GPT-2 execution | Authentic prediction + one real activation, CLI smoke test | done |
| M2 | Verified activation intervention | Patch matches independent reference; no-op control passes | done |
| M3 | Backend API + persistence | Full experiment operable via documented HTTP API | done |
| M4 | Product shell + Discover landing | Nontechnical visitor can start a real experiment | done |
| M5 | Neural Network Explorer | Visual topology reconciles with real model metadata | done |
| M6 | Intervention Playground + Compare | End-to-end journey preset → patch → export | done |
| M7 | Quality, tests, handoff | pytest + Vitest + Playwright pass; honest limitations | done, with open gaps (GPU inference not run; clean-checkout install on Windows not separately verified) |

Current status (final release audit, 2026-10-30): all 22 acceptance items PASS.
See `docs/CIRCUITSCOPE_FINAL_RELEASE_AUDIT.md`. Remaining work is limited to the
documented limitations in `docs/EXECUTION_STATE.md`.

## Vertical-slice order (build priority)

1. **M0–M2 first, headless**: prove the science (real logits, real hook capture,
   real patch, reproducible numbers) with pytest before any browser work.
2. **M3**: wrap the verified engine in FastAPI so the frontend consumes only
   real measurements.
3. **M4–M6**: React app consuming the real API — landing, explorer, playground.
4. **M7**: browser E2E, accessibility, docs, demo guide.

## Non-negotiables

- No fabricated tensor values, predictions, or experiment outcomes anywhere.
- Every chart value traces to a stored experiment result.
- Model/hook metadata comes from the loaded model, never hardcoded.
- Honest reporting of no-effect results and unmet gates.
