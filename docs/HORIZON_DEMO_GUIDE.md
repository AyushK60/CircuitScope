# HORIZON DEMO GUIDE — presenting CircuitScope

## Startup (Windows, from E:\CircuitScope)

```bat
:: terminal 1 — backend (model loads on first request; cached afterwards)
.venv\Scripts\python -m backend.app.main

:: terminal 2 — frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**.

## The five-minute walkthrough

1. **Landing** — point out the live status chip ("GPT-2 ready") and the live
   prediction in section 01: the tokenization and probabilities are computed by
   the real model as you watch.
2. **Predict & Continue** (Intervention Playground) — enter a custom prompt
   such as *"Ramu was hungry. He went outside to buy"*. Show the measured
   top candidates. Press **Generate 5 tokens** (greedy), then **+ 1 token** to
   step. Switch the mode to *sample* to show temperature / top-k / seed. Read
   the honest warning: generations are not guaranteed factual.
3. **Question & Candidate Explorer** — use the father/son preset and press
   **Score candidates**. Explain: these are *continuation log-probabilities
   under a fixed template*, not answers; multi-token candidates use summed
   conditional log-probs. Low absolute probabilities are real — GPT-2 Small
   finds neither continuation likely.
4. **Model Explorer** — click a block → open attention anatomy → pick a head:
   the heatmap is a real captured pattern; click a query token to see its
   measured top attended weights; use "compare with" for a |Δ| map; dark-gray
   cells are causally masked positions.
5. **Intervention** — back in the Playground, run the John/Mary patch
   (guided-tour example ③). Show the target strip, the three runs, and
   D / Δ / Recovery. Open the experiment's overlay link to see the marked
   block on the architecture canvas.
6. **History** — load a previous experiment's stored results; demonstrate
   delete-with-confirmation and the Δ comparison.
7. **Neural Network Lab** — the classic 4-6-6-5-3 dense network. Move an input
   slider: every neuron updates live. Click a neuron to inspect its bias,
   pre-activation and incoming weights. State clearly: *this is an educational
   simulation, separate from GPT-2* — then open the Model Explorer to contrast
   attention + residual stream with dense layers.

## Short explanations for questions you will get

- **Next-token prediction:** GPT-2 scores every vocabulary token for what comes
  next; the bars are softmax probabilities of those real logits at one position —
  not confidence in a whole sentence.
- **Tokenization:** text is split into subword pieces (byte-level BPE); a
  leading space changes the token. The app always shows the exact tokens.
- **Attention:** each head computes where every token looks back at earlier
  tokens (rows sum to 1). It is a measurement, not proof of causation.
- **Activation patching:** copy one internal activation from a clean run into a
  corrupted run, re-execute, and measure the logit change (D, Δ, Recovery).
  Independently reproduced by `scripts/reference_patch.py` within 5e-3.
- **Feedforward lab vs transformer:** dense layers replace values layer by
  layer; transformers add every block's contribution into one shared residual
  stream and use attention instead of fixed connectivity.
- **Question comparisons:** they measure which candidate continuation the model
  finds more likely under a template. They do not establish factual correctness,
  and differing token lengths make raw totals harder to interpret (use mean).

## Where the evidence lives

- `docs/HORIZON_TEST_REPORT.md` — test counts and numerical spot-checks.
- `docs/ATLAS_FINAL_AUDIT.md` — architecture-accuracy gates.
- `docs/SCIENTIFIC_VALIDATION.md` — reference agreement and controls.
- `docs/screenshots/` — captured UI evidence.
