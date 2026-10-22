# ATLAS BASELINE — 2026-10-30

## State at takeover

- Git: clean tree at `f73434a` ("v2 docs"), history intact (6 commits).
- Backend: http://127.0.0.1:8000 — healthy, model loaded (CPU).
- Frontend: http://localhost:5173 — Vite dev server up.
- Tests at baseline: **30 pytest / 6 Vitest / 8 Playwright E2E — all passing.**
- Production build: passing.

## Preserved assets (verified working)

- Real GPT-2 Small inference via TransformerLens 4.3.0 TransformerBridge (backend/app/model_runtime/).
- Verified activation patching with independent reference agreement
  (D 2.5128 / −4.1768 / 1.9426, Recovery 0.9148; scripts/reference_patch.py).
- /api/flow recorded-stage capture (residual norms + all attention patterns).
- FastAPI contracts, SQLite persistence, serial job worker, exports.
- All existing test suites.

## Confirmed defects

1. **Architecture ordering defect (Failure F) — CONFIRMED REAL.**
   In `frontend/src/components/ArchDiagram.tsx` (FullModelDiagram), the final
   three pipeline nodes (Final LayerNorm, Unembed, Logits) are placed with a
   `row++` counter that does not account for the 12 block rows, so they
   visually collide/interleave with Blocks 0–2. Browser-measured visual order:
   `Input, Tokenizer, Embeddings, Block 0, Final LayerNorm, Block 1, Unembed,
   Block 2, Logits, Block 3…` — wrong. The backend computation itself is
   correct and untouched. Fix: rebuilt canvas in PASS 2 with explicit stage
   coordinates (verified by new test).
2. Visual identity: navy/teal generic dashboard look (Failure A) — addressed by
   the new "Sumi & Persimmon" warm-mineral palette (PASS 2).
3. Whole-model diagram occupies little space; no pan/zoom, no attention
   anatomy level (Failures B/C) — addressed by the ArchitectureCanvas.

## Plan

PASS 1 preserve (this file) → PASS 2 centerpiece (palette + ArchitectureCanvas
with semantic zoom scales 1–5) → PASS 3 landing/playground/learn → PASS 4
tests + screenshot audit → PASS 5 polish. Progress: docs/ATLAS_PROGRESS.md.
