# ATLAS PROGRESS — transformation log

## Current pass: PASS 5 (complete) — release candidate

## Gates

| Gate | Status | Evidence |
|---|---|---|
| A — Model correctness | ✅ | 30 pytest incl. reference agreement (D 2.5128/−4.1768/1.9426, Recovery 0.9148 vs raw-HF reference); intervention verified again live in browser (D 1.943, Δ 6.119, Recovery 0.915) |
| B — Architecture explorer | ✅ | ArchitectureCanvas: whole-model view with 12 blocks in verified order (defect fixed), residual bus, pan/zoom/fit/prev-next/jump/keyboard; block → attention-anatomy → head drill-in; real heatmaps + inspector |
| C — Landing page | ✅ | Hero with substantial model schematic + 5 differentiated sections (01–05); live status; measured demo; reference-result panel |
| D — Playground | ✅ | Target strip connects hook/position to the real pipeline; 3 real runs; metric cards; comparisons; error states (E2E) |
| E — Visual verification | ✅ | 8 screenshots in docs/screenshots/ (desktop + 320px) |
| F — Delivery | ✅ | 30 pytest · 6 Vitest · 9 Playwright · production build · launch commands below |

## Completed in this transformation

1. PASS 1: baseline audit; **confirmed and fixed Failure F** — Final-LN/Unembed/Logits
   nodes were mis-placed into block rows 3–5 by a counter bug in the old diagram
   (browser-measured wrong order documented in ATLAS_BASELINE.md). Replaced by
   ArchitectureCanvas with explicit stage coordinates.
2. PASS 2: "Sumi & Persimmon" warm-mineral palette (replaces navy/teal);
   ArchitectureCanvas (pan/zoom camera, wheel zoom, drag pan, fit/reset,
   prev/next block, jump select, keyboard nav, hover inspector with edge/hook
   meanings, measured norm/logit/token overlays from /api/flow).
3. PASS 3: landing rebuild (hero model, live demo, flow strip, verified
   reference results, 144-head matrix, lab CTA); playground target strip +
   explorer→playground hook/position handoff (#/playground?hook=…&position=…);
   Learn contextual diagrams; router query-param support.
4. PASS 4: E2E rewritten for the new journey (+handoff test); mobile overflow
   fixes (grid min-width, ref-row wrap, code wrap); 8-screen screenshot audit;
   stale-Vite-cache incident resolved (legend gradient) by dev-server restart.

## Tests at release

30 pytest / 6 Vitest / 9 Playwright E2E — all passing. Production build passes.

## Known defects / deferred

1. Torch is CPU-only (2.15.0+cpu) — GPU path untested (documented upgrade path).
2. Experiment history browse UI absent (SQLite + exports exist).
3. No PNG chart export (JSON/CSV only).
4. Explorer canvas: attention-anatomy arrow routing is functional but could be
   more elegant at very small viewport widths (still readable, no overlap).
5. Transient E2E click races observed interactively during the session around
   the App's 10s health re-render; the Playwright suite itself is stable.

## Launch

```bat
:: from E:\CircuitScope
.venv\Scripts\python -m backend.app.main     :: http://127.0.0.1:8000
cd frontend && npm run dev                    :: http://localhost:5173
```

Screenshots: docs/screenshots/01…08*.png
