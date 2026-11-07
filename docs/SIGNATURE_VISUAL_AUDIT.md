# SIGNATURE VISUAL AUDIT — design polish pass, 2026-10-30

Baseline: commit `14ade61`, all suites green (42 pytest / 13 Vitest / 13 E2E),
servers running. Baseline screenshots captured to `outputs/baseline/` before
any change; final states to `docs/screenshots/22–24`.

## Initial deficiencies found by browser inspection (1440×900)

1. **Landing hero extended to the viewport edge** while every other section is
   contained at 1180px — visually inconsistent, and the right schematic card
   hugged the screen edge at 1440.
2. **Hero CTA row wrapped awkwardly** — the third button sat alone on a second
   line.
3. **NN Lab inputs panel wasted half its width** — four sliders stacked
   vertically in a full-width panel.
4. **Native control accents clashed** — range inputs and checkboxes rendered
   lavender/purple against the warm mineral palette.
5. **NN Lab neuron value labels were 6.5px** — legible but tight.
6. **Explorer inspector scrolled away** while panning the tall canvas.
7. **Hero schematic strokes were dim** against the panel background.

## Improvements made (CSS + component-level, no logic changes)

1. Hero contained to the shared 1180px grid (`hero-contained`), schematic
   strokes brightened, drop shadow added; all three hero CTAs now fit one row
   (`white-space: nowrap`, tightened padding).
2. NN Lab inputs re-laid out as a responsive 4-column grid with tabular-numeric
   value readouts; neuron label size increased.
3. Explorer inspector is sticky on desktop (`top: 74px`) so measurements stay
   visible while panning; stacks normally below 900px.
4. `accent-color: var(--accent)` applied to range/checkbox inputs — persimmon
   controls across the app.
5. Preserved: all tokens, warnings, measured-data labeling, honest states.

## Verification

- Build clean; **42 pytest · 13 Vitest · 13 Playwright E2E** all passing after
  the changes (second run of the full suite in this pass).
- Zero horizontal overflow at **320 / 390 / 768 / 1280 / 1920px** on all five
  pages (measured, not assumed).
- Sticky inspector verified (position: sticky, top 74px at scroll).
- Mobile NN Lab verified: input grid collapses to one column, network keeps its
  structure with fit-to-view zoom (screenshot 24).
- No console errors (E2E journey test).

## Screenshots

Baseline: `outputs/baseline/b-*.png` (before).
Final: `docs/screenshots/22-signature-landing.png`,
`23-signature-lab.png`, `24-signature-lab-mobile.png`, plus 01–21 from the
earlier audits.

## Intentionally unchanged

- The scientific engine, API contracts, metrics, and test expectations.
- The GPT-2 Explorer's accurate topology and measured-data labeling.
- The heatmap colormaps and their numerical meaning.
- Page structure and navigation routes (only CSS/landing/NN-Lab presentation
  details were touched).

## Remaining limitations

- The two-moons training demo from an earlier iteration was replaced by the
  specified 4-6-6-5-3 forward-pass lab (documented in HORIZON docs).
- Attention-anatomy arrow routing remains serviceable rather than elegant.
- Hero schematic is intentionally a labeled static illustration (no fake data).
