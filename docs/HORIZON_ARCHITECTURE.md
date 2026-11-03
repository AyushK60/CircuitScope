# HORIZON ARCHITECTURE — extension design notes

## New backend capabilities (backend/app/)

| Endpoint | What it does | Integrity notes |
|---|---|---|
| `POST /api/continue` | Autoregressive continuation, 1–8 tokens. `mode=greedy` (argmax) or `mode=sample` (temperature softmax + top-k + seeded `torch.Generator`). Each step re-runs the model on the growing prefix and records the measured top-5 distribution. | Seed reproducibility pytest-verified; bounds enforced by pydantic (n 1–8, temp 0.05–2, top-k 0–50); no caching of activations |
| `POST /api/predict` (extended) | Optional `candidate_a`/`candidate_b` → measured D, probabilities at the final position | Multi-token candidates rejected via `to_single_token` |
| `POST /api/question` | Question & Candidate Explorer scoring: builds a documented prompt template (`{context}\nQuestion: {question}\nAnswer:`), teacher-forces each candidate, reports total/mean/per-token log-probs + first-token probability | Multi-token = SUM of conditional log-probs (never first-token-only); length warning when candidate lengths differ; explicitly not a QA engine |
| `DELETE /api/experiments/{id}` | Removes one persisted experiment | 404 when absent |

## Frontend structure

- `src/lib/feedforward.ts` — pure, unit-tested math for the NN Lab
  (4-6-6-5-3, ReLU, stable softmax, deterministic seed 1234; 105 connections).
- `src/pages/NeuralLab.tsx` — the lab: pan/zoom SVG network, slider inputs,
  neuron/edge inspection (z, bias, incoming weights), layer-by-layer schematic
  replay, accessible activation table, GPT-2 contrast panel. Clearly labeled
  as an educational simulation.
- `src/pages/Playground.tsx` — now hosts: editable candidate comparison (live),
  guided tour (curated examples + 4-step walkthrough), Predict & Continue
  (mode/temperature/top-k/seed + single-token stepping), Question & Candidate
  Explorer, experiment history (load / delete-with-confirm / two-way Δ compare).
- `src/components/ArchitectureCanvas.tsx` — intervention overlay (URL-param
  synchronized hook + token-position marking).
- `src/components/AttentionHeatmap.tsx` — causal-mask dimming (key-after-query
  cells rendered as neutral) and |Δ| head comparison.

## Shared context flows

Predict → live comparison (Playground) → drill into the same block via the
canvas → overlay the intervention target (URL params) → run the 3-pass
experiment → history persists and reloads it. The tour steps operate the same
controls; nothing is simulated except the clearly-labeled NN Lab.
