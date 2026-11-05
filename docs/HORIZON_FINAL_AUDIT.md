# HORIZON FINAL AUDIT — independent release verification, 2026-10-30

Independent auditor pass over the HORIZON expansion. Documentation claims were
not trusted: every item below was verified by code inspection, executed test
suites, live API calls, an independent numerical cross-check, and real browser
interaction at 1440×900, 390px and 320px.

## Acceptance matrix

| Feature | Verdict | Evidence |
|---|---|---|
| Predict & Continue (custom text, real prediction) | PASS | `POST /api/predict` on "Ramu was hungry. He went outside to buy": exact tokens incl. multi-token name ("Ram"+"u"), measured top-5 (" some" 19.4%, " a" 18.2%, " food" 6.3%); landing demo runs the same endpoint |
| Continuation generation | PASS | `POST /api/continue` greedy: "…buy some food. He saw" — each step's prefix grows by the previous chosen token (autoregressive, recomputed, not hardcoded); a second prompt ("Once upon a time…") produced "…that the universe was expanding" — no stale reuse. Sampling mode with temperature/top-k/seed: same seed reproduces exactly (pytest) |
| Question & Candidate Explorer | PASS | Independent raw-HF cross-check (BOS-convention match): " the father" total −6.5119 vs API −6.51; " his son" −8.7297 vs −8.73. Multi-token = SUM of conditional log-probs (pytest: total == Σ per-token; mean == total/n); length warning; explicit not-a-QA disclaimer |
| Guided Tour | PASS after fix | Example ① configures real inputs and runs the real experiment; step guidance updated to warn when the clean/corrupted pair is degenerate (see defects) |
| NN Lab visual gate | PASS | Browser-rendered topology: 24 neurons in 4→6→6→5→3, 105 weighted connections (aria-label + DOM count; 106th `path` is the arrow marker). All 24 activations match an independent Python reimplementation of the documented spec (`scripts/reference_feedforward.py`): layer-1 [0.1219, 0, 0.5425, 0, 1.4282, 0.1134] rendered as 0.122/0.00/0.542…; logits/probs match to the displayed precision. Input change (in·0 → 2.0) recalculated correctly against the reference ([0.0, 0.0, 0.328, 0.956, 1.421, 0.118]; probs 20.9/39.8/39.3%) |
| NN Lab interactions | PASS | neuron selection shows z/bias/ReLU/incoming weights (z=0.5425, bias 0.0119); edge click selects target neuron; zoom/fit controls; slider-driven live recompute; reset; schematic replay labeled; accessible activation table; 390/320px zero overflow |
| GPT-2 Explorer separation & accuracy | PASS | Educational lab explicitly labeled "separate from GPT-2 Small"; explorer stage order verified (blocks strictly before Final LN → Unembed → Logits); hook IDs `blocks.{L}.attn.hook_pattern` match backend identifiers |
| Attention correctness | PASS after fix | Stale-data defect found and fixed (see defects); hook label now syncs with the navigated layer (block 3 → `blocks.3.attn.hook_pattern`); causal-mask cells rendered neutral; head \|Δ\| comparison verified ("vs head 9") |
| Intervention overlay & engine | PASS | Overlay marks exactly `blocks.7.attn.hook_out @ pos 6` from URL params; 3-run experiment measured D 1.943 / Δ 6.119 / Recovery 0.915 (matches the raw-HF reference 1.9426/0.9148); no-op and baseline-immutability tests pass |
| Experiment history | PASS after fix | **P0 fixed**: completed experiments are now persisted under their job ID (were never saved — see defects); live run → history row → Load restores identical stored results (Δ 6.119, Recovery 0.915); delete with two-click confirmation verified (API + UI); compare shows real per-experiment Δ; different experiment types honestly labeled |
| Regression suites | PASS | 42 pytest · 13 Vitest · 13 Playwright E2E · production build — all executed this session |

## Defects discovered and repaired (second pass verified)

1. **P0 — experiments were never persisted.** `db.save_experiment` had no
   caller: the runner kept results only in memory, so history showed stale
   rows and "Load" returned no results. Fixed: the runner persists under the
   job's stable ID (`INSERT OR REPLACE`); pytest now asserts a live run lands
   in SQLite with its full results (Recovery ≈ 0.9148).
2. **P0 — experiment-ID mismatch.** Even where saving existed, rows were keyed
   by a fresh UUID, so lookups by job ID failed. Fixed by passing `exp_id`.
3. **P1 — stale attention data.** Navigating to a different block kept the
   previous layer's heatmap. Fixed: the displayed pattern derives from the
   flow capture (which holds every layer of the same run), so block, hook
   label, and data are always synchronized.
4. **P1 — degenerate clean/corrupted pairs accepted.** Identical prompts
   produced a meaningless patch (Δ 0.000, Recovery n/a) in the guided tour.
   Fixed: backend rejects identical pairs with actionable guidance; the tour
   warns on prediction-focused examples; pytest covers it.
5. **P2 — honest type labeling** in history for candidate-comparison records
   (Δ comparison correctly reports n/a for them).

## Verification runs (both passes)

```bat
.venv\Scripts\python -m pytest backend\tests -q      :: 42 passed (2nd pass)
cd frontend && npx vitest run                         :: 13 passed
cd frontend && npx playwright test                    :: 13 passed
cd frontend && npm run build                          :: clean
python scripts\reference_feedforward.py "0.5,-1,1.2,0"   :: matches browser
```

## Screenshots (docs/screenshots/)

01–08 (previous audit) · 09 token selection · 10 NN101 · 11 NN Lab ·
12 intervention overlay · 13 attention gallery · 14 NN Lab neuron inspector ·
15 Question Explorer (multi-token) · 16 attention compare ·
17 NN Lab audit (input 2.0) · 18/20 history compare · 19 history loaded ·
21 block-3 head-0 synced heatmap.

## Remaining limitations (documented, non-blocking)

1. CPU-only torch; GPU path untested.
2. Sampling continuation re-forwards per token (bounded ≤ 8 tokens).
3. Question scoring is teacher-forced under a fixed template with BOS
   convention — verified consistent, but not a QA engine.
4. History compare is metric-aware only for intervention experiments; other
   types honestly show n/a.
5. Attention-anatomy arrow routing is functional rather than beautiful.

## Environment

Windows 11 · Python 3.11.9 (venv) · torch 2.15.0+cpu · transformer-lens 4.3.0 ·
transformers 5.22.0 · FastAPI 0.162 · React 19 · Vite 8 · Node 24 ·
GPT-2 Small (124.4M params, CPU).

## Startup

```bat
:: from E:\CircuitScope
.venv\Scripts\python -m backend.app.main     :: http://127.0.0.1:8000
cd frontend && npm run dev                    :: http://localhost:5173
```

## Final release assessment

**PASS.** All Horizon features are implemented, the four defects found by this
audit (two P0, two P1) are repaired and re-verified, numerical results are
traceable to the real model and independently cross-checked, and the complete
browser journey works at desktop and mobile widths.
