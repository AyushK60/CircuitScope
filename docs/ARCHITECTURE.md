# ARCHITECTURE

```
Browser (React 19 + TS, Vite, port 5173)
  Landing / Model Explorer / Playground / Learn / Sweeps / Study
  pages/playground/    → split panels (presets, url, TargetStrip, MetricCard,
                         GuidedWizard, ExperimentForm, ResultsPanel,
                         ContinuationPanel, QuestionPanel, HistoryPanel,
                         useIoiPreset hook); Playground.tsx composes them
  pages/Sweeps.tsx     → Sweeps page: patching sweep, DLA, logit lens
  pages/Study.tsx      → Study page: prompt-set study (RQ1/RQ2 tables,
                         top-heads with known-group badges, overlap matrices,
                         caveats); pages/study/ holds Rq1Table, OverlapMatrix,
                         TopHeads (+ heads.ts helpers)
  components/SweepHeatmap.tsx → diverging Recovery heatmap + numbers table
  components/architecture/ → canvas levels (ModelLevel, BlockLevel,
                         AttentionLevel, shared parts, layout constants);
                         ArchitectureCanvas.tsx composes them
        │  fetch (typed client: frontend/src/api/client.ts)
        ▼
FastAPI (127.0.0.1:8000)  backend/app/
  api/routes.py        → /api/health, /models, /predict, /inspect,
                         /experiments (+status/cancel/list/export), /hooks,
                         /api/tasks (+ per-task validation),
                         /api/sweeps,
                         /api/study,
                         /api/analysis/dla, /api/analysis/logit_lens
  schemas (pydantic)   → input bounds: 512 chars, top_k ≤ 20, layer ≤ 11
  tasks/registry.py    → task definition files (examples/tasks/*.json);
                         malformed files recorded, never fatal; per-pair
                         validation against the real tokenizer runs lazily
                         on request and is cached
  experiments/jobs.py  → job-type registry (JOB_TYPES: validate/run fns) →
                         serial worker thread → JobRecord lifecycle with
                         type/progress/progress_message; JobCancelled is
                         distinct from ExperimentValidationError
  experiments/sweeps.py → "sweep" job type (layer_position,
                         layer_head); one clean cache reused for every cell
  experiments/study.py → "study" job type: prompt set → per-prompt
                         clean cache, 3 corrupted baselines, exact denoising
                         (3 corruptions × 2 metrics × 156 components), AtP
                         (swap/D), DLA (clean run), zero/mean ablation
                         (two-pass, leave-one-out template-family means);
                         pre-registered statistics + exclusions
  experiments/metrics.py → D, Delta, Recovery (unclamped, guarded)
  analysis/promptsets.py → deterministic IOI prompt-set generator
                         (tokenizer-filtered single-token words; three
                         templates; swap/third/resample corruptions; token
                         alignment asserted, misaligned draws rejected)
  analysis/components.py → exact per-component effects: Rec_D/Rec_P
                         (denoising) and Abl_D (zero/mean ablation) for all
                         156 components
  analysis/atp.py      → attribution patching (gradient × activation
                         difference; explicit grad recipe because parameters
                         are frozen; first-order limitation stated)
  analysis/stats.py    → hand-rolled statistics (numpy only):
                         rankdata/spearman/bootstrap CI/permutation null/
                         top-k overlap
  analysis/dla.py      → direct logit attribution (reconciliation-
                         checked decomposition of the final residual)
  analysis/lens.py     → logit lens (final LN + unembed per stage)
  model_runtime/loader.py → thread-safe TransformerBridge singleton
  model_runtime/engine.py → predict / capture / attention_pattern /
                         run_patched / patched_logits (logits-only helper
                         shared with run_patched; outputs unchanged) /
                         ablated_logits + interpolated_patch_logits
                        
  storage/db.py        → SQLite (outputs/experiments.db): id, config, results JSON
        │
        ▼
PyTorch 2.15.0+cpu · TransformerLens 4.3.0 TransformerBridge · GPT-2 Small (124M)
  hook points: 637 registered; verified-capturable subset allowlisted in config.py
```

## Task registry

Each task (e.g. IOI) is a data file in `examples/tasks/*.json` — prompts,
candidate tokens, positions of interest, recommended hooks, source citation.
Adding a task needs no code change. `GET /api/tasks` lists them (with any
load errors); `GET /api/tasks/{id}` returns the task plus validation performed
against the real tokenizer (single-token candidates, token alignment, position
ranges, hook allowlist). The Playground loads the IOI default pair at mount
and falls back to its hardcoded preset if the registry is unavailable.

## Job engine

`ExperimentRunner.submit(cfg, job_type="intervention")` remains the one heavy
job at a time. Registered job types provide `validate_fn(cfg)` and
`run_fn(config, ctx)`; `ctx.report(progress, message)` clamps to [0, 1] and
never decreases, `ctx.check_cancel()` raises `JobCancelled` (distinct from
`ExperimentValidationError`, which maps to "failed"). Results are persisted to
SQLite **before** a job is marked complete, so "complete" always implies the
record is retrievable. The intervention pipeline reports progress at fixed
milestones (0.05 / 0.2 / 0.5 / 0.9 / 1.0); live job status JSON carries
`type`, `progress`, `progress_message`.

## Sweeps, DLA, logit lens

- **Sweeps** (`POST /api/sweeps` → job type `"sweep"`, poll
  `GET /api/experiments/{id}`): denoising only — clean activations are copied
  into the corrupted run, stated in every result as
  `direction: "clean -> corrupted (denoising)"`. Modes: `layer_position`
  (12 × seq grid of `blocks.{L}.hook_out` at one position) and `layer_head`
  (12 × 12 grid of `blocks.{L}.attn.hook_z` for head h at all positions). The
  clean cache is captured once and reused for every cell; the corrupted
  baseline runs once. `|D_clean − D_corrupted| < 1e-3` rejects the sweep;
  `< 0.5` adds an instability warning. Persisted configs carry
  `job_type: "sweep"`; the Playground history panel filters non-intervention
  records out (`config.job_type` absent or `"intervention"` = intervention).
- **DLA** (`POST /api/analysis/dla`): decomposes the final residual
  (`blocks.11.hook_out[0,-1]`) into embed, pos_embed, 144 heads (via
  `hook_z` @ `W_O`), 12 attn biases, 12 MLPs, plus the final-LN bias term;
  each term is `((c − mean(c))/s · γ) · w` with `s` and `w = W_U[:,a] − W_U[:,b]`
  from the full residual and the real model's LN epsilon. The response's
  reconciliation block reports `|sum_of_terms − model_D|` (identity must hold
  within 2e-3; `ok:false` is never hidden). Labelled a partial view:
  direct effect only, no effects mediated by later layers.
- **Logit lens** (`POST /api/analysis/logit_lens`): applies the final LN
  (biased variance, model epsilon) + `W_U` to the residual at the final
  position after the embeddings and after every block; reports D, P(A), P(B)
  and top-k tokens per stage. The block-11 stage must reproduce the model's
  real logits (`ok:false` is never hidden). Labelled a heuristic view.

Independent cross-checks (raw HF transformers, no backend imports):
`scripts/reference_dla_lens.py` and `scripts/reference_sweep.py`.

## Prompt sets, components, AtP, study

- **Prompt sets** (`analysis/promptsets.py`): deterministic given `(n, seed)`
  via `random.Random(seed)`. Word pools are filtered through the REAL
  tokenizer on first use (names must be single-token with AND without the
  leading space, because the first subject stands sentence-initially). Three
  templates; template × S2-role balanced as n allows. Every prompt carries
  `swap`/`third`/`resample` corruptions, all asserted token-aligned with the
  clean prompt; misaligned draws are rejected and counted, never accepted.
- **Exact per-component effects** (`analysis/components.py`): one clean cache
  (all `attn.hook_z` + `mlp.hook_out`) per prompt; 156 patched runs per
  (prompt, corruption) via `engine.patched_logits` give `D_patched` and
  `P_patched` (both metrics from the same runs) → `Rec_D`, `Rec_P`. Ablation
  runs in the CLEAN direction via `engine.ablated_logits` (zeros, or a
  leave-one-out mean over the other prompts of the same template; two-pass
  design). `Abl_D = D_clean − D_ablated` in raw logit units.
  Consistency gate: per-head `Rec_D[swap]` for the default IOI pair equals the
  `layer_head` sweep within 1e-4 (tested).
- **Attribution patching** (`analysis/atp.py`): one grad-enabled forward +
  backward on the corrupted prompt (hook on `hook_embed` creates the graph
  because parameters are frozen; `retain_grad` on every `hook_z`/`mlp.hook_out`,
  NOT detached). `AtP_D(c) = Σ (a_clean − a_corr) · dD/da |_corr`;
  `Rec_AtP = AtP_D / (D_clean − D_corr)`. Gradients are read inside the try
  block; the `finally` drops the graph so nothing leaks into later requests
  (tested). Linearity gate: `engine.interpolated_patch_logits` (alpha
  interpolation) central-difference matches `AtP_D` (Pearson > 0.999, tested).
  Known limitation, stated everywhere it is shown: AtP is first-order and
  unreliable under saturated attention.
- **Statistics** (`analysis/stats.py`): hand-rolled numpy (no scipy) —
  average-rank Spearman (constant input → `None`), percentile bootstrap CI,
  permutation null, top-k overlap + random expectation k·k/144.
- **Study job** (`experiments/study.py`, `POST /api/study`): pipeline per
  prompt — clean cache, 3 corrupted baselines, exact denoising for 3
  corruptions × both metrics, AtP (swap/D), DLA on the clean run, then a
  second pass for zero/mean ablation. Pre-registered analysis (PAPER_NOTES
  §4b, committed before any full run): exclusion rule
  `|D_clean − D_corr| < 0.5` per (prompt, corruption), model accuracy =
  fraction with `D_clean > 0` (headline numbers repeated on that subset),
  RQ1 Spearman + bootstrap CI + permutation null, RQ2 pairwise Spearman +
  top-k overlap with bootstrap CIs, known-heads anchor (descriptive only).
  Persisted with `job_type: "study"` before `complete`; cancelled/failed
  studies leave no partial results.
- **Reference checks** (raw HF, no backend imports):
  `scripts/reference_atp.py` (Rec_AtP on 3 prompts, tol 2e-3) and
  `scripts/reference_components.py` (Rec_D/Rec_P + zero ablation on all 156
  components and mean ablation on 20, from a study export, tol 5e-3).
  `scripts/make_figures.py` renders the report figures as hand-written SVG
  (no matplotlib) from a completed study into git-ignored `outputs/paper/`.

## Intervention semantics (the core science)

`run_patched(corrupted_text, clean_cache, hook, positions, heads)`:

1. Clone the clean-run tensor (`clean_cache[hook].clone()`) — baselines immutable.
2. Register a forward hook on the exact hook point that writes
   `z[0, pos, :] = src[0, pos, :]` (per-head for 4-D `hook_z`).
3. Run the corrupted prompt through the real model; hooks auto-reset afterward.
4. Return measured logits → metrics.

Equivalence verified against an independent implementation
(`scripts/reference_patch.py`: raw HF transformers + PyTorch forward hooks, no
TransformerLens) to within 5e-3: D_clean 2.5128, D_corrupted −4.1768,
D_patched 1.9426, Recovery 0.9148.

## Frontend truthfulness contract

- The hero schematic is labeled "ARCHITECTURAL ILLUSTRATION"; live views are
  labeled with hook IDs, shapes, device, and value scales.
- The Explorer's block count/head count/shape labels come from `/api/health`
  metadata, never hardcoded.
- Playground metrics cards, run columns, and exports all render
  `ExperimentResult` fields verbatim; no-effect and failed runs have distinct
  honest states.
- GPT-2 block order rendered: pre-LN (x → LN1 → attn → + → LN2 → MLP → +),
  matching the running model, with residual paths drawn as violet dashed edges.

## Explain package, report job, glossary

New backend package `backend/app/explain/` (all pure text machinery; no model
access, no LLM):

- `thresholds.py` — every qualitative boundary in one place: recovery buckets
  (0.8 / 0.4 / 0.1 / −0.1), DLA magnitude buckets (1.0 / 0.3 raw logit units),
  the live trust-check tolerances (final-cell recovery 1e-3, last-block
  unchanged 1e-4, baseline contrast 1e-3) and the curious readability limits
  (mean ≤ 18 words, longest ≤ 28, Flesch ≥ 60). Formatting helpers
  `fmt_num` / `fmt_pct` / `ordinal` render "n/a" for undefined values.
- `templates.py` — every generated sentence as data: one template per claim
  with its `claim_type`, its declared placeholders and its text at all three
  levels (curious / student / expert). Placeholders use `$name` (filled by a
  strict regex substitution; missing ones raise); glossary terms use
  `{{term:id|display}}` markers, validated against the glossary at import.
- `story.py` — pure functions from compact stored result dicts to claim
  objects `{id, level, text, claim_type, evidence}`. Facts are picked by
  declared deterministic procedures (largest lens jump; argmax recovery over
  early/late block rows excluding the identity cell; top-3/top-10 by value
  with index tie-breaks; group counts over the top 10). The compaction helpers
  (`compact_sweep`, `compact_dla`, `compact_lens`, `compact_study`,
  `compact_prediction`, `compact_reference`) build the same dicts stored in
  `sections[].data`, so every evidence path resolves.
- `lint.py` — the claim lint used by tests AND at runtime: banned
  words/phrases at all levels (principle 4), glossary-marker validation, and
  the curious-only readability limits + banned-idiom list. A report/export
  that lints badly fails honestly (500/failed job) instead of shipping.
- `glossary.py` — loader for the shared `examples/glossary.json`.

New job type `"report"` (`backend/app/experiments/report.py`), registered from
`routes.py` like sweep/study. Steps: prediction (one forward pass, top token +
candidate probabilities) → logit lens → layer×position sweep → layer×head
sweep → DLA (all reusing the sweep and study implementations; sub-sweeps run under a
rescaled `_PhaseCtx` so progress stays monotonic) → optional study load
(validated at submit time → 422 for unknown / non-study / incomplete) →
section assembly → lint. Results: `sections[]` (id, title, question, claims,
compact `data`, `evidence_tier`, analogy, `trust`), `lint` (ok + curious
readability), `provenance`, `generated_at` (provenance timestamp only).

Trust panel data (Task 5) is computed live from stored data per section:
DLA reconciliation, lens-final identity, sweep invariants (final-cell
recovery == 1 within 1e-3; non-final positions of the last block unchanged
within 1e-4), baseline contrast gate + warnings, study exclusion recount,
prediction-vs-lens cross-check. Failed checks carry `ok: false` and the UI
renders them in red with the number.

Endpoints (additive):
- `POST /api/report` `{task_id, pair_id?, study_job_id?}` → `{job_id, status}`;
  poll `GET /api/experiments/{job_id}` (config persisted with
  `"job_type": "report"`, so the Playground history filter hides it).
- `GET /api/report/{job_id}/export?fmt=md&level=curious|student|expert` →
  Markdown (claim bullets, key-number tables, trust panels, analogy notes,
  provenance footer); linted; 404 for non-report jobs.
- `POST /api/explain` `{job_id, level}` → claims for completed `sweep`/`study`
  jobs; the server reads the stored results, the client never supplies data.

Shared glossary: `examples/glossary.json` (20 terms × 3 levels + analogies
with mandatory `breaks_when`), loaded by `backend/app/explain/glossary.py`
and `frontend/src/lib/glossary.ts`. `<Term>` renders dotted-underline
keyboard-focusable buttons with an accessible popover (Escape closes,
`aria-describedby`). `GlossaryList` is reachable from the Report and Learn
pages. Frontend: `frontend/src/pages/Report.tsx` (route `report`, nav label
"Report" after "Study"), `pages/report/` components (claims + evidence
disclosure, trust panel, analogy box, lens sparkline, "In plain words" box
reused on Sweeps and Study). Reading level persists in localStorage
(`lib/level.ts`, try/catch). No new dependencies.

## Reproduce, verify, agreement, tour

New pieces (all additive; existing response shapes unchanged):

- **`backend/app/reproduce/`** (`generate.py`, `runtime.py`). Generates standalone
  Python scripts as text. The server never runs them. Each script embeds ONE JSON
  literal (loaded with `json.loads`) holding prompts, candidates, hooks, positions,
  stored numbers, tolerances and provenance. It imports only `torch`,
  `transformers` and the standard library (checked by an AST test). Kinds:
  `intervention`, `sweep`, `study_subset`, `report`, `dla`, `lens`.
  Route: `POST /api/reproduce` (job kinds need a completed job of the matching
  type; `dla` and `lens` are computed at request time).
- **`backend/app/experiments/verify.py`**: job type `verify`. A registry of
  reference checks (`dla_lens`, `sweep`, `components`, `atp`, `patch`; `feedforward`
  is manual). The server computes the API-side numbers in process into
  `outputs/verify_tmp/<job_id>/`, runs the reference script with a fixed argv
  (`sys.executable`, script path, offline args; no shell, no user strings), parses
  `max |api - reference|`, and removes the temp directory in a `finally` (with a
  short retry for Windows file locks). Routes: `POST /api/verify`,
  `GET /api/verify/status` (latest result per check; `stale` when the script's
  SHA-256 changed).
- **`backend/app/analysis/agreement.py`** and **`backend/app/experiments/agreement.py`**:
  job type `agreement` (about 8 s). Exact `Rec_D` (reusing `analysis/components.py`),
  AtP (`analysis/atp.py`) and DLA (`analysis/dla.py`) for all 156 components, with
  statistics from `analysis/stats.py`. Route: `POST /api/agreement`.
  The consistency gate (exact per-head values vs the `layer_head` sweep,
  1e-4) is tested and shown live in the report.
- **`backend/app/explain/`**: `story_agreement_methods` claims at three levels, with
  thresholds in `thresholds.py` (Pearson 0.9 / 0.6; group difference 0.2). The
  report job runs the agreement step and adds the `methods_agree` section after
  `which_parts`. The Markdown export includes it.
- **Frontend**: routes `verify` (Verify page, `pages/Verify.tsx`,
  `pages/verify/VerificationCard.tsx`) and `tour` (Tour page, `pages/Tour.tsx`,
  scene logic in `pages/tour/scenes.ts`). The Sweeps page gained the agreement
  section (`components/AgreementScatter.tsx`). `components/ReproduceButton.tsx` is
  on the Sweeps results, DLA and lens results, Study results and the Report header.
  The download filename is read from `Content-Disposition`; the backend exposes
  that header through CORS (`expose_headers`).

Data flow for a reproduction download: UI button -> `api.reproduce` ->
`POST /api/reproduce` -> generator reads the stored job (or recomputes `dla`/`lens`)
-> script text -> Blob download. Nothing in this path executes user input or
the generated script.

Verification flow: UI "Run this check" -> `POST /api/verify` (one heavy job at a
time; a second heavy job gets 422) -> temp inputs -> reference script subprocess
(timeout and cancel terminate, then kill) -> result record persisted before
`complete` (check id, ok, exit code, max diff, tolerance, elapsed, script SHA-256,
git commit, model revision, versions, stdout tail, time).
