# TEST REPORT — 2026-10-30 (v2)

## Backend (pytest) — 30 passed

```
.venv\Scripts\python -m pytest backend\tests -q
```
Includes 4 new flow-capture tests: block/shape structure (12 layers × 12 heads
× seq²), attention rows sum to 1 (rounded transport), residual norms accumulate
block 0 → 11, /api/flow endpoint + input bounds.

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

## Browser E2E (Playwright) — 8 passed

Requires backend :8000 + `npm run dev` :5173 running.

1. landing live status + real prediction (66.2% candidate visible)
2. explorer drill-in to head 4 heatmap with `blocks.5.attn.hook_pattern`
   and 12 × 14 × 14 shape
3. full playground experiment: metrics, three run columns, exports
4. learn page, three depths
5. 320px viewport: no horizontal overflow on landing AND explorer
6. flow playback: real block stages, recorded hook data, honest timing label,
   step/jump controls, completed-run state
7. anatomy data view: measured residual norms for all 12 blocks
8. zero console errors across all routes

## Visual screenshot audit (browser, real captures)

- Landing 1280px: sumi/ruri identity, live status chip, live prediction demo ✓
- Explorer anatomy (block level): pre-LN diagram + residual-stream label,
  verified order table, measured per-token norms, head grid ✓
- Explorer flow: playback at block stage with measured heatmap, timing note ✓
- Playground 1280px: three real runs compared, metric cards, labels fixed ✓
- Explorer 320px: no horizontal overflow after SVG viewBox fix ✓
- Learn page ✓

## Known gaps

- GPU path untested (CPU-only torch installed).
- No PNG export test (feature absent).
