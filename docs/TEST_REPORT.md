# TEST REPORT — 2026-10-30

## Backend (pytest) — 26 passed

```
.venv\Scripts\python -m pytest backend\tests -q
..........  [engine: metadata, tokenization, HF cross-check, reproducibility,
             capture shapes, unsupported hooks, no-op patch, real patch effect,
             cache immutability, hook leak check, invalid positions, metric math,
             end-to-end experiment, config validation, allowlist]
..........  [api: health, models, predict, bounds, attention inspect,
             invalid hook, experiment lifecycle, invalid hook 422,
             misaligned prompts fail honestly, persistence]
....        [reference agreement vs scripts/reference_patch.py (tol 5e-3);
             hook allowlist membership]
```

## Independent reference

```
.venv\Scripts\python scripts\reference_patch.py
D_clean 2.5128 · D_corrupted -4.1768 · D_patched 1.9426 · Recovery 0.9148
```
Matches backend within 5e-3 (raw HF implementation, no TransformerLens).

## Frontend unit (Vitest) — 6 passed

`npx vitest run` — TopPredictions (measured values + skeleton), TokenRow
(exact tokens + ids + highlight), FullModelDiagram (12 real block nodes from
metadata), AttentionHeatmap (matrix cells + hook provenance).

## Browser E2E (Playwright) — 6 passed

Requires backend :8000 + `npm run dev` :5173 running.

1. landing live status + real prediction (66.2% candidate visible)
2. explorer drill-in to head 4 heatmap with `blocks.5.attn.hook_pattern`
   and 12 × 14 × 14 shape
3. full playground experiment: metrics, three run columns, exports
4. learn page, three depths
5. 320px viewport: no horizontal overflow
6. zero console errors across all routes

## Known gaps

- GPU path untested (CPU-only torch installed).
- No PNG export test (feature absent).
