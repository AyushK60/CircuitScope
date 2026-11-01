# HORIZON PROGRESS

## M0 — baseline ✅

Git clean at `726b91d`. 35 pytest · 13 E2E passing. Servers verified.

## Milestone checklist

| M | Scope | Status |
|---|---|---|
| M0 | Baseline + protection | ✅ |
| M1 | Predict & Continue (sampling mode, stepping UI, landing CTA) | in progress |
| M2 | Guided Tour (5 curated examples) + Question & Candidate Explorer (multi-token scoring) | pending |
| M3 | NN Lab rebuild: 4-6-6-5-3 feedforward, ReLU, softmax, neuron/edge inspection, playback, pan/zoom | in progress |
| M4 | Attention: head comparison + causal-mask rendering | pending |
| M5 | Intervention overlay | ✅ (previous release; re-verify) |
| M6 | History: delete w/ confirm, type display, compare | pending |
| M7 | Nav ordering, docs, full verification | pending |

## Startup

```bat
.venv\Scripts\python -m backend.app.main    :: 127.0.0.1:8000
cd frontend && npm run dev                  :: localhost:5173
```
