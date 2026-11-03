# HORIZON TEST REPORT — 2026-10-30

## Backend (pytest) — 40 passed

New suites:
- `test_continuation.py` (10): greedy determinism, step growth, bounds (n/mode/
  temperature/top-k), seeded sampling reproducibility + seed sensitivity,
  candidate comparison vs reference D 2.5128, honest multi-token rejection,
  history persistence.
- Prior suites unchanged and passing (engine, API, flow, reference agreement).

## Frontend unit (Vitest) — 13 passed

- `feedforward.test.ts` (6): architecture is exactly 4-6-6-5-3; weight-matrix
  dimensions per layer; exactly 105 connections; layer-1 pre-activations match
  an independent hand calculation (tol 1e-10); ReLU applied correctly; stable
  softmax sums to 1 and is shift-invariant; seed determinism + input
  dependence.
- Component tests unchanged and passing.

## Browser E2E (Playwright) — 13 passed

Includes: NN Lab renders the 105-connection network and computes live (slider
input → output update; neuron inspection shows pre-activation z and ReLU
note); continuation generates real text and exposes sampling parameters;
custom candidates update the live comparison; intervention overlay from URL
params; mobile 320px no-overflow; zero console errors.

## Browser walkthrough evidence (manual, this session)

- NN Lab neuron inspector: layer 1 index 2 → z 0.5425, bias 0.0119, activation
  0.5425 (ReLU), incoming weights listed (screenshots/14).
- Question Explorer multi-token: " the father" total log-P −6.51 (mean −3.26,
  first-P 0.24%) vs " his son" −8.73 — three distinct metrics rendered, no
  correctness claim (screenshots/15).
- Sampling continuation: "sampling · temperature 0.9 · top_k 20 · seed 1234".
- Attention compare: legend shows "|Δ| vs head 9" (screenshots/16).
- 390px: zero horizontal overflow on all five pages.

## Builds

`npm run build` clean; backend pytest clean; no new dependencies added.
