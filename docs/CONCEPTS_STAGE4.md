# Stage 4 concepts — how the Circuit Report is written

The Report page turns measurements into sentences. This note explains the
machinery, its guarantees, and its limits. Companion to `docs/ARCHITECTURE.md`
and the Stage 4 spec (`docs/v2/STAGE_4.md`).

## 1. The claim ledger

Every generated sentence is a claim object:

```
{ id, level, text, claim_type, evidence }
```

- `level` — one of `curious`, `student`, `expert`. Each claim exists at all
  three levels; the frontend renders the selected one.
- `claim_type` — one of:
  - **measured**: a number or comparison taken from the data
    (`answer.gap`, `words.early`, `study.atp`, ...);
  - **interpretation**: a hedged reading of the data (`words.procedure`);
  - **caveat**: a fixed limitation (`limits.*`, the lens heuristic note).
- `evidence` — the data paths used, e.g. `["sweep.recovery[9][6]",
  "procedure: ..."]`. Paths resolve against the section's compact `data`
  object; the frontend's "where does this come from?" disclosure re-resolves
  and displays them next to their values. Fixed statements that are not data
  use the documented `fixed:<identifier>` convention and are labelled as such
  in the UI.

Guarantee: no free text is written at runtime. `story.py` picks facts by
declared deterministic procedures and calls `fill_template`, which raises if a
placeholder is missing — a template can never silently render with a gap.

## 2. Where the procedures are declared

The facts are chosen by simple, reproducible procedures, stated in the
evidence list of every claim that uses them:

| Claim | Procedure |
| --- | --- |
| `lens.biggest_jump` | largest increase between consecutive lens stages; ties → earliest stage |
| `words.early` / `words.late` | argmax recovery over block rows 0–3 / 8–11, **excluding the identity cell** (last block × final position, which is 1 by construction); ties → lower (layer, position) |
| `parts.recovery_top.*` | top-3 heads by recovery value; ties → index order |
| `parts.dla_top.*` | top-3 heads by DLA value (descending); ties → index order |
| `paper.*` | top-10 heads by recovery (ties → index), counted against the reference groups; missing groups report the best rank of their heads in the full 144 |
| `study.overlap_*` | top-10 overlap recomputed from stored `per_prompt` data (prompt-averaged Rec_D[swap] vs AtP/DLA, excluded pairs dropped) — the one new computation, declared as an extra check |

The clock and RNG are never read. Same stored data → same sentences.

## 3. Thresholds live in one place

`backend/app/explain/thresholds.py` declares every boundary; the qualitative
word is a pure function of the number:

- Recovery-style buckets: ≥ 0.8 "restores most of" · [0.4, 0.8) "restores a
  large part of" · [0.1, 0.4) "restores some of" · (−0.1, 0.1) "has little
  effect on" · ≤ −0.1 "pushes the other way". Never stronger than the bucket.
- DLA values are raw logit units, so they get their own declared scale:
  |v| ≥ 1.0 strong, ≥ 0.3 weak, else small.
- Live trust-check tolerances: final-cell recovery 1e-3; last-block unchanged
  1e-4; baseline contrast gate 1e-3 (matching `sweeps.py`).
- Curious readability limits: mean sentence ≤ 18 words, longest sentence
  ≤ 28 words, Flesch reading ease ≥ 60.

All boundaries are unit-tested at their exact values
(`backend/tests/test_explain_thresholds.py`).

## 4. Banned words (principle 4)

No generated claim may contain, at any level, case-insensitively: *proves,
proof, understands, knows, thinks, reasons, decides, causes* (whole word) and
the phrases *responsible for*, *is the reason*, *explains why*. Causal-sounding
sentences must name the intervention ("in this test, copying X changed Y by
Z"). The list lives in `backend/app/explain/banned_words.py` and is enforced by
`lint.py` at generation time — a report job that lints badly fails instead of
shipping.

At the `curious` level a banned-idiom list (same module) also applies.

## 5. Readability limits and their rationale

A visitor who may not read English as a first language should be able to read
the curious report in ~2 minutes. The declared limits:

- mean sentence length ≤ 18 words, longest sentence ≤ 28 words;
- Flesch reading ease ≥ 60, with a documented syllable heuristic: a word's
  syllables = number of vowel groups (`[aeiouy]+`), min 1, minus one if the
  word ends in a silent 'e' (excluding `-le`, `-ee`, `-ye`, `-ie`, `-ue`,
  `-oe` endings). Digits count as one-syllable words. The heuristic is
  deliberately simple; the limits leave margin so ordinary short English
  passes comfortably (the real generated curious report measures Flesch ≈ 86).
- no idiom from the banned list; no technical term unless wrapped as a
  glossary term (`{{term:id|display}}`), which the UI renders as a
  dotted-underline tooltip.

The curious lint runs over every generated claim at report time, over the
exported Markdown's claim bullets, and (in tests) over the glossary's curious
definitions.

## 6. Missing data is stated, not hidden

Whenever a section's input is absent (no study attached, empty sweep, missing
reference file), the section renders a `not_run` caveat: "Not run yet: …
attaching a completed study gives this part of the report." Values are never
filled from memory or from other runs. The how_sure section without a study is
the canonical example.

## 7. What the trust panel checks live

Each section's `trust.checks` is computed from the stored data at report time
(never hard-coded):

- **prediction_matches_lens_final** (answer): the prediction's D equals the
  lens final-stage D within 1e-3.
- **lens_final_identity** (where_formed): the lens final stage reproduces the
  model's real logits (tolerance 1e-3, top-1 agreement).
- **final_cell_recovery** / **last_block_other_positions** (which_words): the
  sweep invariants from Stage 2, recomputed from the stored grid.
- **baseline_contrast** (both sweeps): |D_clean − D_corrupted| ≥ 1e-3.
- **dla_reconciliation** (which_parts): |Σ terms + β·w − D| ≤ 2e-3.
- **reference_loaded** (vs_paper): the known-heads file loaded with ≥ 1 group.
- **study_exclusions_recount** (how_sure): exclusion counts recomputed from
  stored per-prompt baselines match the stored summary (mismatch == 0).

A failed check is displayed in red with its number and the word FAILED —
colour is never the only signal.

## 8. Honest disagreement with the paper

Our ranking basis (single-head denoising recovery over the whole head output)
differs from the paper's direct-effect path patching, so rankings can differ
without contradiction. The `vs_paper` section states the group counts, names
groups we did NOT find near the top (with their best rank of 144), separates
the paper's fuzzy heads, and always carries the method-difference caveat. When
the leading group is not the name movers, an explicit disagreement claim is
generated from the measured counts.
