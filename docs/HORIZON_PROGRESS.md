# HORIZON PROGRESS

## M0 — baseline ✅

Git clean at `726b91d`. 35 pytest · 13 E2E passing. Servers verified.

## Milestone checklist

| M | Scope | Status |
|---|---|---|
| M0 | Baseline + protection | ✅ 35 pytest / 13 E2E at `726b91d` |
| M1 | Predict & Continue: sampling (temperature/top-k/seed, reproducible), single-token stepping, reset, honest warning, landing CTA | ✅ |
| M2 | Guided tour (3 curated in-Playground examples + attention link) + Question & Candidate Explorer (multi-token conditional log-prob scoring; total/mean/first-token metrics; length warning) | ✅ |
| M3 | NN Lab rebuilt: 4-6-6-5-3, ReLU + stable softmax, 105 connections, neuron/edge inspection (z, bias, weights), pan/zoom, schematic replay, accessible activation table, GPT-2 contrast | ✅ |
| M4 | Attention: causal-mask dimming + \|Δ\| head comparison | ✅ |
| M5 | Intervention overlay | ✅ (re-verified: badge + banner via URL params) |
| M6 | History: delete with confirm, type display, two-experiment Δ/Recovery comparison | ✅ |
| M7 | Nav reordered (Explorer · NN Lab · Playground · Learn), docs, verification | ✅ |

## Verification at close

- Backend: **40 pytest** (continuation determinism/sampling/bounds; question
  scoring single + multi-token; deletion; all prior suites).
- Frontend unit: **13 Vitest** (incl. 6 feedforward-math tests: architecture
  4-6-6-5-3, 105 connections, independent hand-calculation of layer-1
  pre-activations, stable softmax, seed determinism).
- Browser E2E: **13 Playwright** — incl. NN Lab live-compute assertions and
  sampling-mode parameters.
- Browser walkthrough: NN Lab neuron inspector (z=0.5425, ReLU verified),
  Question Explorer multi-token (−6.51 vs −8.73 total log-P, 2 tok each,
  mean/first-token distinguished), sampling continuation with seed 1234,
  attention \|Δ\| vs head 9, zero console errors, 390px zero overflow on all
  five pages.
- Screenshots: 14-nnlab-forward.png, 15-question-explorer.png,
  16-attention-compare.png (docs/screenshots/).

## Known limitations

- Sampling uses full re-forward per token (CPU); bounded to 8 tokens.
- Question scoring is teacher-forced under a fixed template — not a QA engine.
- History compare covers intervention experiments (the only persisted type).


## Startup

```bat
.venv\Scripts\python -m backend.app.main    :: 127.0.0.1:8000
cd frontend && npm run dev                  :: localhost:5173
```
