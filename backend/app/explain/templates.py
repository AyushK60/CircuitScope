"""Template registry for generated claims.

All generated text lives here as data - never scattered f-strings at runtime.
Each template declares:
  - its claim_type ("measured" | "interpretation" | "caveat"),
  - the value placeholders it needs,
  - its text at all three levels (curious / student / expert).

Two syntaxes inside template text:
  - ``$name``          - a value placeholder, filled from a values dict by
                         ``fill_template`` (a tiny regex substitution; Python's
                         str.format cannot be used because the glossary markers
                         below would be parsed as format fields).
  - ``{{term:id|display}}`` - a glossary-term marker, returned to the frontend
                         intact so it can render a tooltip; ``lint`` strips it
                         before counting words.

At import time every ``{{term:...}}`` marker is validated against the shared
glossary (examples/glossary.json): an unknown term id is an import error, not a
runtime surprise. ``fill_template`` raises if a placeholder is missing or
unknown values are left unfilled, so a template can never silently render
with a gap.
"""

from __future__ import annotations

import re

from .glossary import has_term

PLACEHOLDER_RE = re.compile(r"\$([a-z][a-z0-9_]*)")
TERM_MARKER_RE = re.compile(r"\{\{term:([a-z_]+)\|[^{}]*\}\}")

CLAIM_TYPES = ("measured", "interpretation", "caveat")
LEVELS = ("curious", "student", "expert")


def _check_markers() -> None:
    """Import-time honesty check: every glossary marker must exist."""
    problems = []
    for key, spec in TEMPLATES.items():
        for level, text in spec["levels"].items():
            for m in TERM_MARKER_RE.finditer(text):
                if not has_term(m.group(1)):
                    problems.append(f"{key}[{level}]: unknown glossary term '{m.group(1)}'")
    if problems:
        raise ValueError("template/glossary mismatch: " + "; ".join(problems))


# claim id -> {"claim_type", "placeholders", "levels": {level: text}}
TEMPLATES: dict[str, dict] = {
    # ---- generic "not run yet" (principle 5: missing data is stated) ---------
    "not_run": {
        "claim_type": "caveat",
        "placeholders": ["what", "how"],
        "levels": {
            "curious": "Not run yet: $what. $how gives this part of the report.",
            "student": "Not run yet: $what. $how gives this part of the report; nothing is filled in from other runs or from memory.",
            "expert": "Data absent: $what. Producer: $how. The engine renders 'not run yet' instead of substituting values from any other run.",
        },
    },
    # ---- answer -------------------------------------------------------------
    "answer.top_token": {
        "claim_type": "measured",
        "placeholders": ["top_token", "top_pct"],
        "levels": {
            "curious": 'The model\'s top next token is "$top_token" - about a $top_pct chance.',
            "student": 'The model\'s top next {{term:token|token}} is "$top_token", with a probability of $top_pct.',
            "expert": 'Top next token: "$top_token" (probability $top_pct at the final position).',
        },
    },
    "answer.candidates": {
        "claim_type": "measured",
        "placeholders": ["cand_a", "pa_pct", "cand_b", "pb_pct"],
        "levels": {
            "curious": 'The two candidates: "$cand_a" gets a $pa_pct chance; "$cand_b" gets $pb_pct.',
            "student": 'Between the two candidates, "$cand_a" gets probability $pa_pct and "$cand_b" gets $pb_pct.',
            "expert": 'Candidate probabilities: "$cand_a" $pa_pct, "$cand_b" $pb_pct (full-vocabulary softmax).',
        },
    },
    "answer.gap": {
        "claim_type": "measured",
        "placeholders": ["d", "cand_a"],
        "levels": {
            "curious": 'The {{term:score_gap|score gap}} is $d. A positive gap means the model prefers "$cand_a".',
            "student": 'The {{term:score_gap|score gap}} D = logit("$cand_a") − logit(the other candidate) is $d; positive means "$cand_a" is ahead.',
            "expert": 'Score gap D = $d (logit difference at the final position; sign convention: positive favours "$cand_a").',
        },
    },
    # ---- where_formed (logit lens) -------------------------------------------
    "lens.biggest_jump": {
        "claim_type": "measured",
        "placeholders": ["d_after", "d_before", "jump", "stage"],
        "levels": {
            "curious": "The score gap grows most at the output of $stage: from $d_before to $d_after.",
            "student": "The largest single increase of the score gap happens across $stage: from $d_before to $d_after (a difference of $jump).",
            "expert": "Largest stage-to-stage increase of D on the lens: across $stage, $d_before → $d_after (Δ = $jump).",
        },
    },
    "lens.final": {
        "claim_type": "measured",
        "placeholders": ["final_d", "tol"],
        "levels": {
            "curious": "At the end of the run the score gap is $final_d.",
            "student": "After the last block the score gap reaches $final_d, and the lens readout reproduces the model's real logits exactly.",
            "expert": "Final lens D = $final_d; the final stage equals the real final logits (checked live, tolerance $tol).",
        },
    },
    "lens.heuristic": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "The lens is a rough readout of the model's notes. Early blocks are not trained to be read this way.",
            "student": "The logit lens applies the final readout to intermediate notes. Early blocks were never trained to be read this way, so early values are a heuristic view.",
            "expert": "Logit lens applies ln_final + W_U to blocks.{i}.hook_out. Early-stage readouts are untrained; treat the trace as coarse, not as the model's intermediate decisions.",
        },
    },
    # ---- which_words (layer × position sweep) ---------------------------------
    "words.early": {
        "claim_type": "measured",
        "placeholders": ["token", "predicate", "rec"],
        "levels": {
            "curious": 'In early blocks (0 to 3) the strongest effect sits on "$token". Patching that word alone $predicate ({{term:recovery|recovery}} $rec).',
            "student": 'Among blocks 0–3, the largest {{term:recovery|recovery}} cell sits on "$token": patching that word alone $predicate (recovery $rec).',
            "expert": 'Blocks 0–3 argmax(recovery) at "$token": full-residual patch at that position $predicate (recovery $rec).',
        },
    },
    "words.late": {
        "claim_type": "measured",
        "placeholders": ["token", "predicate", "rec"],
        "levels": {
            "curious": 'In late blocks (8 to 11) the strongest effect sits on "$token". Patching that word alone $predicate (recovery $rec).',
            "student": 'Among blocks 8–11, the largest {{term:recovery|recovery}} cell sits on "$token": patching that word alone $predicate (recovery $rec).',
            "expert": 'Blocks 8–11 argmax(recovery) at "$token": full-residual patch at that position $predicate (recovery $rec).',
        },
    },
    "words.procedure": {
        "claim_type": "interpretation",
        "placeholders": [],
        "levels": {
            "curious": "How the words were picked: the run copies each block's output at each word into the corrupted run, one at a time. It records what comes back. The report names the strongest cell among early blocks and among late blocks.",
            "student": "Procedure: for every (block, word) cell the clean block-output activation is copied into the corrupted run (denoising) and the recovery is recorded; ties are broken by the lower block and earlier word. The claims above cite the strongest cell in blocks 0–3 and in blocks 8–11.",
            "expert": "Procedure: grid sweep over blocks.{l}.hook_out at each position (denoising), Recovery = (D_patched − D_corr)/(D_clean − D_corr); reported cells are argmax over rows 0–3 and rows 8–11 with ties broken by (layer, position) index order.",
        },
    },
    # ---- which_parts (head sweep + DLA) ----------------------------------------
    "parts.methods": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "This section shows two different tests. The patching test copies one head's outputs from the clean run and asks what comes back. The direct test asks how much the head pushes the final scores by itself. The two tests can disagree.",
            "student": "Two tests appear here, answering different questions. Patching asks: does copying this head's outputs restore the answer? DLA asks: how much does the head push the final scores straight into the readout? They can disagree because patching includes effects through later blocks and DLA does not.",
            "expert": "Method note: denoising patch of blocks.{l}.attn.hook_z (all positions) measures sufficiency of the whole head output; DLA measures the direct write into ln_final only. The two rankings need not agree.",
        },
    },
    "parts.recovery_top": {
        "claim_type": "measured",
        "placeholders": ["label", "predicate", "rank", "rec"],
        "levels": {
            "curious": "Head $label: patching it alone $predicate (recovery $rec).",
            "student": "Head $label: copying its outputs into the corrupted run $predicate (recovery $rec, rank $rank of 144 heads).",
            "expert": "Head $label: full denoising patch of blocks.{l}.attn.hook_z at all positions $predicate (recovery $rec; rank $rank of 144).",
        },
    },
    "parts.dla_top": {
        "claim_type": "measured",
        "placeholders": ["label", "val", "rank", "n_heads"],
        "levels": {
            "curious": "In the direct test, head $label pushes the score gap by $val (rank $rank of $n_heads).",
            "student": "Direct test: head $label contributes $val to the score gap straight into the final readout (rank $rank of $n_heads heads).",
            "expert": "DLA: head $label contributes $val logit units directly (rank $rank of $n_heads; effects through later blocks excluded).",
        },
    },
    "parts.dla_negative": {
        "claim_type": "measured",
        "placeholders": ["label", "val"],
        "levels": {
            "curious": "The strongest direct push the other way comes from head $label ($val).",
            "student": "The strongest direct push against the answer comes from head $label: $val logit units.",
            "expert": "Most negative DLA among heads: $label at $val logit units (a negative name-mover-style direct write).",
        },
    },
    "parts.reconciliation": {
        "claim_type": "measured",
        "placeholders": ["err", "tol"],
        "levels": {
            "curious": "The direct pushes add up to the model's real score gap within $err (tolerance $tol).",
            "student": "The per-component direct pushes sum to the model's real score gap within $err (tolerance $tol) - a check that the arithmetic is exact, not that any head is 'the' cause.",
            "expert": "DLA reconciliation: |Σ terms + β·w − D| = $err ≤ $tol (telescoping identity of the linear read).",
        },
    },
    # ---- vs_paper ---------------------------------------------------------------
    "paper.overall": {
        "claim_type": "measured",
        "placeholders": ["n_in_groups", "chance"],
        "levels": {
            "curious": "$n_in_groups of our top 10 heads are on the published circuit list. A random list of 10 would hold about $chance.",
            "student": "$n_in_groups of our top 10 heads (by recovery) belong to at least one group from the published IOI circuit list. A random set of 10 heads would contain about $chance such heads.",
            "expert": "Group membership of top-10 heads (by denoising recovery): $n_in_groups of 10 match ≥1 Wang et al. 2022 group (expected under random selection: $chance).",
        },
    },
    "paper.group": {
        "claim_type": "measured",
        "placeholders": ["count", "group_label"],
        "levels": {
            "curious": "Heads from the $group_label group of the paper, among our top 10: $count.",
            "student": "Heads from the published $group_label group among our top 10: $count.",
            "expert": "Top-10 overlap with $group_label: $count head(s).",
        },
    },
    "paper.group_missing": {
        "claim_type": "measured",
        "placeholders": ["group_label", "best_rank"],
        "levels": {
            "curious": "We did not find any $group_label head in our top 10. The best one ranks $best_rank of 144.",
            "student": "No $group_label head appears in our top 10; the best-placed one ranks $best_rank of 144 heads.",
            "expert": "$group_label: no head in top 10 by recovery; best group rank $best_rank/144.",
        },
    },
    "paper.disagreement": {
        "claim_type": "measured",
        "placeholders": ["top_label", "top_count", "nm_count", "nm_best_rank"],
        "levels": {
            "curious": "Our top 10 is led by $top_label heads ($top_count of 10). The name mover heads from the paper appear $nm_count times, with the best at rank $nm_best_rank.",
            "student": "In our measurement the top group is $top_label ($top_count of 10), while the paper's main name mover heads appear $nm_count times in our top 10 (best rank $nm_best_rank of 144).",
            "expert": "Ranking divergence vs paper: top group by top-10 membership is $top_label ($top_count/10); name movers $nm_count/10, best rank $nm_best_rank/144.",
        },
    },
    "paper.method_difference": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "Our test copies a whole head's output from a clean run and asks what the corrupted run gains. The paper used a different test, which asked what each head writes straight into the answer. The two tests rank heads differently.",
            "student": "Method difference: our sweep copies whole head outputs from a clean run (denoising) and asks what restores the answer; the paper identified its main heads by direct-effect path patching, which asks what each head writes directly. The two methods rank heads differently, so a difference here is a difference of method, not a contradiction.",
            "expert": "Method note: our ranking basis is single-head denoising recovery (whole hook_z patch, all positions); Wang et al. identify name movers via direct-effect path patching. The metrics are not expected to produce identical rankings.",
        },
    },
    "paper.fuzzy": {
        "claim_type": "measured",
        "placeholders": ["n_fuzzy"],
        "levels": {
            "curious": 'The paper also lists three "fuzzy" heads. Our top 10 contains $n_fuzzy of them.',
            "student": 'The published list also names three "fuzzy" heads (0.10, 5.8, 5.9), reported separately from the core groups; our top 10 contains $n_fuzzy of them.',
            "expert": 'Fuzzy heads (0.10, 5.8, 5.9 per the reference file): $n_fuzzy of 3 appear in our top 10.',
        },
    },
    "paper.reference": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "The head groups come from a reference list checked against the paper. The list is an anchor for comparison, not our measurement.",
            "student": "The group lists come from a reference file transcribed from Wang et al. 2022 and checked against the paper. They are an anchor for comparison; our own measurements are the recovery grid.",
            "expert": "Reference: examples/reference/ioi_wang2022_heads.json (verified_against_paper: true). Descriptive anchor only; no result depends on it.",
        },
    },
    # ---- how_sure (study) ---------------------------------------------------------
    "study.size": {
        "claim_type": "measured",
        "placeholders": ["n_prompts"],
        "levels": {
            "curious": "The study used $n_prompts sentences from a fixed {{term:prompt_set|prompt set}}.",
            "student": "The attached study ran on a generated {{term:prompt_set|prompt set}} of $n_prompts sentences (seeded, reproducible).",
            "expert": "Study: n_prompts = $n_prompts (seeded deterministic generation; pre-registered analysis).",
        },
    },
    "study.accuracy": {
        "claim_type": "measured",
        "placeholders": ["acc"],
        "levels": {
            "curious": "The model itself picked the expected token on $acc of them.",
            "student": "Model accuracy: on $acc of the prompts the model itself preferred the expected candidate (D_clean > 0).",
            "expert": "Model accuracy (fraction of prompts with D_clean > 0): $acc.",
        },
    },
    "study.exclusions": {
        "claim_type": "measured",
        "placeholders": ["n_excluded"],
        "levels": {
            "curious": "$n_excluded prompt pairs were dropped: the corrupted sentence barely moved the score gap.",
            "student": "$n_excluded (prompt, corruption) pairs were excluded under the pre-registered rule: |D_clean − D_corr| < 0.5, i.e. the corruption barely moved the score gap.",
            "expert": "Exclusions (pre-registered rule |D_clean − D_corr| < 0.5): $n_excluded pairs dropped across corruption conditions.",
        },
    },
    "study.atp": {
        "claim_type": "measured",
        "placeholders": ["rho", "lo", "hi"],
        "levels": {
            "curious": "The fast method ({{term:attribution_patching|attribution patching}}) tracks the exact test. It matches the exact test's head order: $rho on a −1 to 1 scale. The middle 95% of resamples: $lo to $hi.",
            "student": "{{term:attribution_patching|Attribution patching}} vs exact patching: mean signed Spearman {{term:rank_correlation|rank correlation}} over heads is $rho (95% bootstrap interval $lo to $hi over prompts).",
            "expert": "RQ1 (pre-registered): exact Rec_D[swap] vs AtP over 144 heads, signed Spearman pooled over prompts: mean $rho, bootstrap 95% CI [$lo, $hi].",
        },
    },
    "study.dla": {
        "claim_type": "measured",
        "placeholders": ["rho", "lo", "hi"],
        "levels": {
            "curious": "The direct test gets a {{term:rank_correlation|rank correlation}} of $rho against the exact test. The middle 95% of resamples: $lo to $hi.",
            "student": "DLA vs exact patching: mean signed Spearman rank correlation over heads is $rho (95% bootstrap interval $lo to $hi). DLA is a partial view, so a lower value is expected, not a fault.",
            "expert": "RQ1: exact Rec_D[swap] vs DLA (direct effects only), signed Spearman over 144 heads: mean $rho, bootstrap 95% CI [$lo, $hi].",
        },
    },
    "study.overlap_atp": {
        "claim_type": "measured",
        "placeholders": ["k_atp", "expected"],
        "levels": {
            "curious": "Extra check, recomputed from the stored per-prompt data: the top 10 lists of the exact test and the fast test share $k_atp of 10 heads. Two random lists would share about $expected.",
            "student": "Extra check (recomputed from stored per-prompt data, not from the pre-registered summary): the top-10 head lists of exact patching and attribution patching share $k_atp of 10; two random lists would share about $expected.",
            "expert": "Extra check (recomputed from per_prompt, prompt-averaged Rec_D[swap] vs AtP): top-10 overlap $k_atp/10; random expectation $expected.",
        },
    },
    "study.overlap_dla": {
        "claim_type": "measured",
        "placeholders": ["k_dla"],
        "levels": {
            "curious": "The same extra check for the direct test. Its top 10 list shares $k_dla of 10 heads with the exact test's list.",
            "student": "The same extra check for DLA: its top-10 list shares $k_dla of 10 heads with the exact patching top-10 list (recomputed from stored per-prompt data).",
            "expert": "Extra check (recomputed): prompt-averaged Rec_D[swap] vs DLA top-10 overlap $k_dla/10.",
        },
    },
    "study.stable": {
        "claim_type": "measured",
        "placeholders": ["k_third", "k_resample"],
        "levels": {
            "curious": "The top 10 list is close across corruption types. Overlap with the swap run: $k_third of 10 for the third-word change, $k_resample of 10 for the resample change.",
            "student": "Ranking stability across corruption methods (pre-registered top-10 overlaps, swap_D row of the stored matrix): third-word corruption $k_third of 10, resample corruption $k_resample of 10.",
            "expert": "Top-10 stability (stored rq2.topk.k10, swap_D row): swap vs third_D $k_third/10; swap vs resample_D $k_resample/10.",
        },
    },
    "study.sensitivity": {
        "claim_type": "measured",
        "placeholders": ["k_p", "k_zero"],
        "levels": {
            "curious": "But the top 10 list does not stay the same under every test. With chance scores and whole-sentence swaps it shares only $k_p of 10 heads with the swap run. With the {{term:ablation|cover-up test}} it shares $k_zero of 10.",
            "student": "The ranking is not invariant to the test used (pre-registered top-10 overlaps, swap_D row): with the probability metric and resample corruption the overlap is $k_p of 10; with zero ablation it is $k_zero of 10. Metric and intervention type both change the ranking.",
            "expert": "Top-10 overlap with swap_D (stored rq2.topk.k10): resample_P $k_p/10; abl_zero_D $k_zero/10. Rankings are not invariant to the metric (D vs P) or to the intervention (denoising vs ablation).",
        },
    },
    "study.not_run": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "No study is attached, so there is no multi-prompt evidence here. Run a study on the Study page, then generate the report again with that study selected.",
            "student": "No study is attached to this report, so the multi-prompt numbers are absent (this page never fills them from memory or from other runs). Run a study on the Study page, then generate the report again with that study selected.",
            "expert": "study_job_id was not provided: the how_sure section has no data. The report engine renders 'not run yet' rather than substituting values from any other run.",
        },
    },
    # ---- agreement (exact vs AtP vs DLA) ---------------------------------------------
    "agree.atp_overall": {
        "claim_type": "measured",
        "placeholders": ["rho", "phrase", "overlap", "expected"],
        "levels": {
            "curious": "The fast {{term:attribution_patching|attribution patching}} $phrase on the 144 heads (Pearson $rho on a −1 to 1 scale). Its top 10 list shares $overlap of 10 heads with the exact test; chance is about $expected.",
            "student": "{{term:attribution_patching|Attribution patching}} vs exact patching over the 144 head values: Pearson correlation $rho (it $phrase). The two top-10 lists share $overlap of 10 heads; two random lists would share about $expected.",
            "expert": "Rec_AtP vs exact Rec_D, 144 heads: Pearson $rho; top-10 overlap $overlap/10 (random expectation $expected). Spearman and layer-group values are in the statistics table.",
        },
    },
    "agree.dla_overall": {
        "claim_type": "measured",
        "placeholders": ["rho", "phrase", "overlap", "expected"],
        "levels": {
            "curious": "The {{term:direct_contribution|direct test}} $phrase on the 144 heads (Pearson $rho). Its top 10 list shares $overlap of 10 heads with the exact test; chance is about $expected.",
            "student": "Direct logit attribution vs exact patching over the 144 head values: Pearson correlation $rho (it $phrase). The two top-10 lists share $overlap of 10 heads; two random lists would share about $expected.",
            "expert": "DLA vs exact Rec_D, 144 heads: Pearson $rho; top-10 overlap $overlap/10 (random expectation $expected). DLA reads only direct effects, so this is not expected to be perfect.",
        },
    },
    "agree.groups": {
        "claim_type": "measured",
        "placeholders": ["method_label", "better", "worse", "rho_better", "rho_worse"],
        "levels": {
            "curious": "For $method_label, agreement is stronger in $better than in $worse: Pearson $rho_better against $rho_worse.",
            "student": "By layer group, agreement of $method_label with the exact test is stronger for $better ($rho_better) than for $worse ($rho_worse).",
            "expert": "Pearson by layer group for $method_label: $better $rho_better vs $worse $rho_worse (groups differ by more than the declared threshold).",
        },
    },
    "agree.methods": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "Each test asks a different question. Exact {{term:patching|patching}} copies one part at a time and reruns the model. Attribution patching uses a straight-line guess. The direct test reads the final scores. They can agree on the big parts and still disagree in the details.",
            "student": "Method note: exact patching reruns the model with one component's clean activation; {{term:attribution_patching|attribution patching}} takes a first-order linear step from the corrupted run; {{term:direct_contribution|DLA}} reads only the final write into the score. They answer different questions, so agreement is evidence about the scores, not about one shared mechanism.",
            "expert": "Metrics are not interchangeable: exact Rec_D is a rerun effect; Rec_AtP is a gradient approximation at the corrupted point; DLA is a linear read of the final residual. Correlation across heads is expected to be positive but need not be 1, especially where attention changes nonlinearly.",
        },
    },
    # ---- limits (fixed caveats) -----------------------------------------------------
    "limits.one_prompt": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "This report reads one example sentence. It shows what happened here, not a rule for every sentence.",
            "student": "The live sections measure one example pair of sentences. They are an illustration of the method, not a generalisation about all prompts.",
            "expert": "Single-prompt sections use the registry's default IOI pair; no sampling variability is captured there (see the study section for multi-prompt evidence).",
        },
    },
    "limits.sufficiency": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "Copying activations tests whether a part is enough to bring the answer back. It does not test whether the model needs that part.",
            "student": "Denoising patching is sufficiency-style evidence: it tests whether a component's clean output is enough to restore the answer. Necessity needs ablation, which the study measures separately.",
            "expert": "Direction fixed: clean → corrupted (denoising, sufficiency). Necessity-style evidence (zero/mean ablation) is only in the study section.",
        },
    },
    "limits.dla_partial": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "The direct test ignores effects that pass through later blocks, so it is a partial view.",
            "student": "Direct logit attribution ignores effects mediated by later blocks; a small DLA value does not mean a component is unimportant.",
            "expert": "DLA is a linear read of the final residual only; indirect paths through later blocks are excluded by construction.",
        },
    },
    "limits.model": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "All of this was measured on GPT-2 Small, a tiny model, with one fixed patching direction.",
            "student": "Everything here is measured on GPT-2 Small (124M parameters) with the fixed clean → corrupted (denoising) direction; other models or directions may differ.",
            "expert": "Model: GPT-2 Small (TransformerBridge, CPU); direction: denoising only. No claims transfer to other models.",
        },
    },
    "limits.other_prompts": {
        "claim_type": "caveat",
        "placeholders": [],
        "levels": {
            "curious": "Results can shift with other sentences or other words. A different kind of change can also move them.",
            "student": "Patching results depend on the corruption method and the metric (a known sensitivity); other sentences or corruptions can produce different rankings.",
            "expert": "Known sensitivity: corruption choice and metric change patching rankings (Zhang & Nanda 2024); the study's RQ2 quantifies this on the prompt set.",
        },
    },
}


def template_keys() -> list[str]:
    return sorted(TEMPLATES)


def fill_template(key: str, level: str, values: dict) -> str:
    """Fill one template at one level. Missing placeholders raise (never
    silently render a gap); unknown glossary terms raise at import time."""
    if key not in TEMPLATES:
        raise KeyError(f"unknown template: {key!r}")
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level!r}")
    tpl = TEMPLATES[key]["levels"][level]

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in values:
            raise KeyError(f"template {key}[{level}] placeholder ${name} missing from values")
        v = values[name]
        return str(v)

    out = PLACEHOLDER_RE.sub(sub, tpl)
    leftover = PLACEHOLDER_RE.search(out)
    if leftover:  # e.g. a literal "$word" the values dict did not intend
        raise KeyError(f"template {key}[{level}] has unfilled placeholder: {leftover.group(0)}")
    return out


def template_claim_type(key: str) -> str:
    return TEMPLATES[key]["claim_type"]


_check_markers()
