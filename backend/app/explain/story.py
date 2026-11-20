"""Story functions (Stage 4 Task 1): pure functions from stored result dicts
to lists of claim objects.

Contract for every claim object: {id, level, text, claim_type, evidence} —
one entry per level (curious/student/expert) for each template instance.

Rules (spec):
  - facts are picked by simple deterministic procedures (largest jump, top-k by
    value, counts of known heads in the top-10), and the procedure is stated in
    the evidence list; ties are sorted by index; the clock and RNG are never
    read;
  - qualitative words always come from thresholds bucket functions applied to
    the SAME value cited by the claim;
  - missing data renders a "Not run yet" claim instead of being hidden.

Evidence paths are "<step>.<path into the section data>" (e.g.
"sweep.recovery[9][6]", "dla.reconciliation.abs_error") so the frontend can
resolve every cited number from sections[].data; fixed statements that are not
data use the documented convention "fixed:<identifier>".
"""

from __future__ import annotations

import numpy as np

from ..analysis.stats import topk_overlap
from .templates import LEVELS, fill_template, template_claim_type
from .thresholds import (
    AGREEMENT_GROUP_DIFF,
    agreement_phrase,
    fmt_num,
    fmt_pct,
    ordinal,
    recovery_predicate,
)

N_HEADS = 144
TOP_K = 10
EARLY_BLOCKS = (0, 1, 2, 3)
LATE_BLOCKS = (8, 9, 10, 11)

GROUP_LABELS = {
    "name_movers": "name mover",
    "negative_name_movers": "negative name mover",
    "s_inhibition": "S-inhibition",
    "induction": "induction",
    "duplicate_token": "duplicate token",
    "previous_token": "previous token",
    "backup_name_movers": "backup name mover",
}

PROC_TOP_RECOVERY = "procedure: top-3 heads by recovery value, ties broken by index (layer, head)"
PROC_TOP_DLA = "procedure: top-3 heads by DLA value (descending), ties broken by index"
PROC_EARLY_LATE = (
    "procedure: argmax of recovery over rows 0-3 (early) and rows 8-11 (late), "
    "excluding the trivial identity cell (last block, final position), "
    "ties broken by (layer, position) index order"
)
PROC_TOP10 = "procedure: top-10 heads by recovery value, ties broken by index"


def _mk(key: str, values: dict, evidence: list[str], suffix: str = "") -> list[dict]:
    """One template instance -> one claim per level, same id/evidence."""
    cid = f"{key}.{suffix}" if suffix else key
    ct = template_claim_type(key)
    out = []
    for level in LEVELS:
        out.append(
            {
                "id": cid,
                "level": level,
                "text": fill_template(key, level, values),
                "claim_type": ct,
                "evidence": list(evidence),
            }
        )
    return out


def _not_run(what: str, how: str, evidence: list[str] | None = None, suffix: str = "") -> list[dict]:
    return _mk(
        "not_run",
        {"what": what, "how": how},
        evidence or [f"fixed:not_run:{what}"],
        suffix=suffix,
    )


# ---------------------------------------------------------------------------
# compaction helpers: stored job results -> the compact section data dicts
# (these same dicts are stored in sections[].data so evidence paths resolve)
# ---------------------------------------------------------------------------


def compact_prediction(candidate_a: str, candidate_b: str, top: list[dict], comparison: dict) -> dict:
    return {
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "top_token": top[0]["token"] if top else None,
        "top_prob": top[0]["prob"] if top else None,
        "top3": top,
        "prob_a": comparison.get("prob_a"),
        "prob_b": comparison.get("prob_b"),
        "D": comparison.get("D"),
        "sign_convention": comparison.get(
            "sign_convention", "positive D means candidate A scores above candidate B"
        ),
    }


def compact_lens(lens_result: dict) -> dict:
    stages = [
        {
            "stage": s["stage"],
            "layer": s["layer"],
            "D": s["D"],
            "prob_a": s["prob_a"],
            "prob_b": s["prob_b"],
            "top1": s["top"][0]["token"] if s.get("top") else None,
            "top1_prob": s["top"][0]["prob"] if s.get("top") else None,
        }
        for s in lens_result["stages"]
    ]
    # deterministic: largest increase between consecutive stages; ties -> earliest
    max_jump = None
    for i in range(1, len(stages)):
        d0, d1 = stages[i - 1]["D"], stages[i]["D"]
        if d0 is None or d1 is None:
            continue
        diff = d1 - d0
        if max_jump is None or diff > max_jump["jump"]:
            max_jump = {"stage": stages[i]["stage"], "D_before": d0, "D_after": d1, "jump": diff}
    return {
        "stages": stages,
        "max_jump": max_jump,
        "final_d": stages[-1]["D"] if stages else None,
        "final_check": lens_result.get("final_matches_model"),
        "note": lens_result.get("note", ""),
    }


def compact_sweep(sweep_result: dict) -> dict:
    return {
        "mode": sweep_result["mode"],
        "direction": sweep_result["direction"],
        "baselines": sweep_result["baselines"],
        "tokens": sweep_result["col_labels"],
        "recovery": sweep_result["recovery"],
        "d_patched": sweep_result["d_patched"],
        "n_runs": sweep_result["n_runs"],
    }


def compact_dla(dla_result: dict) -> dict:
    heads = [c for c in dla_result["components"] if c["kind"] == "head"]
    ordered = sorted(range(len(heads)), key=lambda i: (-heads[i]["value"], i))
    top = [
        {
            "label": heads[i]["name"].replace("head ", ""),
            "layer": heads[i]["layer"],
            "index": heads[i]["head"],
            "value": heads[i]["value"],
            "rank": r + 1,
        }
        for r, i in enumerate(ordered)
    ]
    return {
        "D": dla_result["D"],
        "reconciliation": dla_result["reconciliation"],
        "top": top,
        "most_negative": top[-1] if top else None,
        "notes": dla_result["notes"],
        # All 171 terms, stored additively so the report reproduce script can
        # recompute every one of them (Stage 5 Task 1).
        "components": [
            {
                "name": c["name"],
                "kind": c["kind"],
                "layer": c.get("layer"),
                "head": c.get("head"),
                "value": c["value"],
            }
            for c in dla_result["components"]
        ],
    }


def compact_study(study_result: dict) -> dict:
    """Compact the stored study result for the how_sure section. The ONLY new
    computation is the top-10 overlap recomputed from stored per_prompt data
    (declared in the spec as an extra check); everything else is read from the
    stored pre-registered summary."""
    per = study_result.get("per_prompt") or []
    summary = study_result.get("summary") or {}
    rq1 = (summary.get("rq1") or {}).get("all_prompts") or {}

    def _signed(method: str) -> dict | None:
        block = ((rq1.get("methods") or {}).get(method) or {}).get("heads") or {}
        s = block.get("signed") or {}
        if s.get("mean") is None:
            return None
        return {"mean": s["mean"], "ci_low": s.get("ci_low"), "ci_high": s.get("ci_high")}

    def _vec(v):
        if v is None or any(x is None for x in v):
            return None
        return [float(x) for x in v]

    def _overlap(other_key: str) -> int | None:
        exact, other = [], []
        for row in per:
            if (row.get("baselines") or {}).get("swap", {}).get("excluded"):
                continue
            e = _vec((row.get("rec_d") or {}).get("swap"))
            o = _vec(row.get(other_key))
            if e is None or o is None:
                continue
            exact.append(e[:N_HEADS])
            other.append(o[:N_HEADS])
        if not exact or not other:
            return None
        return topk_overlap(
            np.mean(np.asarray(exact), axis=0), np.mean(np.asarray(other), axis=0), TOP_K
        )

    overlap = None
    k_atp = _overlap("atp")
    k_dla = _overlap("dla")
    if k_atp is not None or k_dla is not None:
        overlap = {
            "atp": k_atp,
            "dla": k_dla,
            "expected_random_k10": round(TOP_K * TOP_K / N_HEADS, 4),
        }

    stability: list[dict] = []
    rq2 = summary.get("rq2") or {}
    names = rq2.get("condition_names") or []
    k10 = ((rq2.get("topk") or {}).get("k10") or {}).get("matrix") or []
    if names and k10:
        for cond in ("third_D", "resample_D", "swap_P", "resample_P", "abl_zero_D", "abl_mean_D"):
            if cond in names:
                j = names.index(cond)
                v = k10[0][j] if k10[0] is not None and j < len(k10[0]) else None
                stability.append({"condition": cond, "k10_overlap": v})

    return {
        "params": study_result.get("params"),
        "n_prompts": study_result.get("params", {}).get("n_prompts")
        if study_result.get("params")
        else None,
        "model_accuracy": study_result.get("model_accuracy"),
        "exclusions": study_result.get("exclusions"),
        "exclusion_rule": study_result.get("exclusion_rule"),
        "rq1_atp_heads": _signed("atp"),
        "rq1_dla_heads": _signed("dla"),
        "beats_control": rq1.get("beats_control"),
        "overlap": overlap,
        "stability": stability,
        "expected_random_k10": round(TOP_K * TOP_K / N_HEADS, 4),
    }


def compact_agreement(agreement_result: dict) -> dict:
    """Compact one agreement result for the report section / explain endpoint.

    The 144 exact head values are kept because the report's live consistency
    check compares them with the layer x head sweep already in the report.
    """
    vectors = agreement_result.get("vectors") or {}
    exact = vectors.get("exact") or []
    return {
        "task_id": agreement_result.get("task_id"),
        "pair_id": agreement_result.get("pair_id"),
        "scatter": agreement_result.get("scatter") or [],
        "exact_heads": [float(v) for v in exact[:N_HEADS]],
        "stats": agreement_result.get("stats") or {},
        "top10": agreement_result.get("top10") or {},
        "baselines": agreement_result.get("baselines") or {},
        "dla_reconciliation": agreement_result.get("dla_reconciliation") or {},
        "notes": agreement_result.get("notes") or [],
        "method_notes": agreement_result.get("method_notes") or {},
    }


def compact_reference(reference: dict | None) -> dict | None:
    if not reference:
        return None
    return {
        "groups": [
            {"group": g["group"], "heads": list(g.get("heads", []))}
            for g in reference.get("groups", [])
        ],
        "fuzzy_heads": list(reference.get("fuzzy_heads", [])),
        "verified_against_paper": bool(reference.get("verified_against_paper")),
        "note": reference.get("note", ""),
        "citation": (reference.get("source") or {}).get("citation", ""),
    }


def _rank_grid(sweep: dict) -> list[tuple[float, int, int]]:
    cells = []
    for l, row in enumerate(sweep["recovery"]):
        for h, v in enumerate(row):
            if v is not None:
                cells.append((v, l, h))
    cells.sort(key=lambda t: (-t[0], t[1], t[2]))
    return cells


def top10_by_recovery(sweep: dict) -> list[dict]:
    """Flatten a layer_head sweep grid to the top-10 heads by recovery
    (ties broken by index). Rows = layers, cols = heads."""
    out = []
    for rank, (v, l, h) in enumerate(_rank_grid(sweep)[:TOP_K], start=1):
        out.append({"rank": rank, "label": f"{l}.{h}", "layer": l, "index": h, "recovery": v})
    return out


def _rank_of(sweep: dict) -> dict[str, int]:
    return {f"{l}.{h}": i + 1 for i, (v, l, h) in enumerate(_rank_grid(sweep))}


# ---------------------------------------------------------------------------
# story functions: section data dict -> claims
# ---------------------------------------------------------------------------


def story_prediction(pred: dict | None) -> list[dict]:
    if not pred or pred.get("D") is None or not pred.get("top_token"):
        return _not_run("the model's answer for this prompt", "one forward pass on the clean prompt", suffix="answer")
    claims = []
    claims += _mk(
        "answer.top_token",
        {"top_token": pred["top_token"], "top_pct": fmt_pct(pred["top_prob"])},
        ["prediction.top_token", "prediction.top_prob"],
    )
    claims += _mk(
        "answer.candidates",
        {
            "cand_a": pred["candidate_a"],
            "pa_pct": fmt_pct(pred["prob_a"]),
            "cand_b": pred["candidate_b"],
            "pb_pct": fmt_pct(pred["prob_b"]),
        },
        ["prediction.candidate_a", "prediction.prob_a", "prediction.candidate_b", "prediction.prob_b"],
    )
    claims += _mk(
        "answer.gap",
        {"d": fmt_num(pred["D"], 2), "cand_a": pred["candidate_a"]},
        [
            "prediction.D",
            "prediction.candidate_a",
            "prediction.candidate_b",
            "prediction.sign_convention",
        ],
    )
    return claims


def story_lens(lens: dict | None) -> list[dict]:
    if not lens or not lens.get("stages") or lens.get("max_jump") is None:
        return _not_run("where the answer forms", "the logit lens over the clean prompt", suffix="lens")
    claims = []
    mj = lens["max_jump"]
    claims += _mk(
        "lens.biggest_jump",
        {
            "stage": mj["stage"],
            "d_before": fmt_num(mj["D_before"], 2),
            "d_after": fmt_num(mj["D_after"], 2),
            "jump": fmt_num(mj["jump"], 2),
        },
        [
            "lens.max_jump.stage",
            "lens.max_jump.D_before",
            "lens.max_jump.D_after",
            "lens.max_jump.jump",
            "procedure: largest increase between consecutive lens stages, ties by earliest stage",
        ],
    )
    fc = lens.get("final_check") or {}
    claims += _mk(
        "lens.final",
        {"final_d": fmt_num(lens.get("final_d"), 2), "tol": fmt_num(fc.get("tolerance"), 3)},
        ["lens.final_d", "lens.final_check.tolerance", "lens.final_check.max_abs_logit_diff_candidates"],
    )
    claims += _mk("lens.heuristic", {}, ["lens.note"])
    return claims


def _best_cell(sweep: dict, blocks, exclude: set | None = None) -> dict | None:
    rec = sweep.get("recovery") or []
    tokens = sweep.get("tokens") or []
    exclude = exclude or set()
    best = None
    for l in blocks:
        if l >= len(rec):
            continue
        row = rec[l]
        for p, v in enumerate(row):
            if v is None or (l, p) in exclude:
                continue
            if best is None or v > best["recovery"]:
                best = {
                    "layer": l,
                    "position": p,
                    "token": tokens[p] if p < len(tokens) else None,
                    "recovery": v,
                }
    return best


def story_sweep_position(data: dict | None) -> list[dict]:
    """data: {"sweep": compact layer_position sweep} (early/late are computed
    here by the declared argmax procedure)."""
    sweep = (data or {}).get("sweep") or (data or {})
    if not sweep or sweep.get("mode") != "layer_position" or not sweep.get("recovery"):
        return _not_run("which words matter", "the layer x position patching sweep", suffix="words")
    claims = []
    # The last block's final position is the identity cell (recovery == 1 by
    # construction); it is excluded from the argmax and the procedure says so.
    identity = (len(sweep["recovery"]) - 1, len(sweep["recovery"][-1]) - 1)
    for key, blocks, tpl in (("early", EARLY_BLOCKS, "words.early"), ("late", LATE_BLOCKS, "words.late")):
        best = _best_cell(sweep, blocks, exclude={identity})
        if best is None or best["token"] is None:
            claims += _not_run(
                f"the strongest {key}-block effect",
                "a layer x position sweep with valid recovery values",
                suffix=key,
            )
            continue
        claims += _mk(
            tpl,
            {
                "token": best["token"],
                "predicate": recovery_predicate(best["recovery"]),
                "rec": fmt_num(best["recovery"], 2),
            },
            [
                f"sweep.recovery[{best['layer']}][{best['position']}]",
                f"procedure: argmax of recovery over rows {blocks[0]}-{blocks[-1]}, "
                "excluding the identity cell (last block, final position), ties by (layer, position)",
            ],
        )
    claims += _mk("words.procedure", {}, ["fixed:words.procedure", PROC_EARLY_LATE])
    return claims


def story_sweep_heads(data: dict | None) -> list[dict]:
    """data: {"sweep": compact layer_head sweep, "dla": compact DLA or None}."""
    data = data or {}
    sweep = data.get("sweep")
    dla = data.get("dla")
    claims = []
    claims += _mk("parts.methods", {}, ["fixed:parts.methods"])
    if sweep and sweep.get("mode") == "layer_head" and sweep.get("recovery"):
        for entry in top10_by_recovery(sweep)[:3]:
            claims += _mk(
                "parts.recovery_top",
                {
                    "label": entry["label"],
                    "predicate": recovery_predicate(entry["recovery"]),
                    "rec": fmt_num(entry["recovery"], 2),
                    "rank": ordinal(entry["rank"]),
                },
                [f"sweep.recovery[{entry['layer']}][{entry['index']}]", PROC_TOP_RECOVERY],
                suffix=str(entry["rank"]),
            )
    else:
        claims += _not_run(
            "the top heads by the patching test", "the layer x head patching sweep", suffix="recovery"
        )

    if dla and dla.get("top"):
        n_heads = len(dla["top"])
        for entry in dla["top"][:3]:
            claims += _mk(
                "parts.dla_top",
                {
                    "label": entry["label"],
                    "val": fmt_num(entry["value"], 2),
                    "rank": ordinal(entry["rank"]),
                    "n_heads": n_heads,
                },
                [
                    f"dla.top[{entry['rank'] - 1}].value",
                    f"dla.top[{entry['rank'] - 1}].label",
                    PROC_TOP_DLA,
                ],
                suffix=str(entry["rank"]),
            )
        neg = dla.get("most_negative")
        top3_labels = {e["label"] for e in dla["top"][:3]}
        if neg and neg["label"] not in top3_labels:
            claims += _mk(
                "parts.dla_negative",
                {"label": neg["label"], "val": fmt_num(neg["value"], 2)},
                ["dla.most_negative.label", "dla.most_negative.value", PROC_TOP_DLA],
            )
        rec_ = dla.get("reconciliation") or {}
        if rec_.get("abs_error") is not None:
            claims += _mk(
                "parts.reconciliation",
                {"err": fmt_num(rec_["abs_error"], 4), "tol": fmt_num(rec_.get("tolerance"), 3)},
                ["dla.reconciliation.abs_error", "dla.reconciliation.tolerance"],
            )
    else:
        claims += _not_run(
            "the direct contributions", "direct logit attribution on the clean prompt", suffix="dla"
        )
    return claims


def story_agreement_with_paper(data: dict | None) -> list[dict]:
    """data: {"sweep": compact layer_head sweep, "reference": compact known-heads}."""
    data = data or {}
    sweep = data.get("sweep")
    reference = data.get("reference")
    if not sweep or sweep.get("mode") != "layer_head" or not sweep.get("recovery"):
        return _not_run(
            "the comparison with the published circuit", "the layer x head patching sweep", suffix="paper"
        )
    if not reference or not reference.get("groups"):
        return _not_run(
            "the comparison with the published circuit", "the known-heads reference file", suffix="paper"
        )

    claims = []
    top10 = top10_by_recovery(sweep)
    order: list[str] = []
    membership: dict[str, set[str]] = {}
    for g in reference["groups"]:
        order.append(g["group"])
        membership[g["group"]] = set(g.get("heads", []))
    for entry in top10:
        entry["groups"] = [gid for gid in order if entry["label"] in membership.get(gid, set())]

    group_counts = {gid: sum(1 for e in top10 if gid in e["groups"]) for gid in order}
    rank_of = _rank_of(sweep)

    n_in_groups = sum(1 for e in top10 if e["groups"])
    n_known = len({h for g in reference["groups"] for h in g.get("heads", [])})
    chance = fmt_num(TOP_K * n_known / N_HEADS, 1)
    claims += _mk("paper.overall", {"n_in_groups": n_in_groups, "chance": chance}, ["top10", PROC_TOP10, "reference.groups"])
    for gid in order:
        if group_counts[gid] > 0:
            claims += _mk(
                "paper.group",
                {"count": group_counts[gid], "group_label": GROUP_LABELS.get(gid, gid)},
                ["top10", "reference.groups", PROC_TOP10],
                suffix=gid,
            )
        else:
            ranks = [rank_of[h] for h in membership[gid] if h in rank_of]
            if ranks:
                claims += _mk(
                    "paper.group_missing",
                    {
                        "group_label": GROUP_LABELS.get(gid, gid),
                        "best_rank": ordinal(min(ranks)),
                    },
                    ["top10", "reference.groups", PROC_TOP10],
                    suffix=gid,
                )

    # honest disagreement: compare the leading group with the paper's main path
    if order:
        top_gid = max(order, key=lambda g: (group_counts[g], -order.index(g)))
        if top_gid != "name_movers":
            nm_ranks = [rank_of[h] for h in membership.get("name_movers", set()) if h in rank_of]
            claims += _mk(
                "paper.disagreement",
                {
                    "top_label": GROUP_LABELS.get(top_gid, top_gid),
                    "top_count": group_counts[top_gid],
                    "nm_count": group_counts.get("name_movers", 0),
                    "nm_best_rank": ordinal(min(nm_ranks)) if nm_ranks else "n/a",
                },
                ["top10", "reference.groups", PROC_TOP10],
            )
    claims += _mk("paper.method_difference", {}, ["fixed:paper.method_difference"])
    fuzzy_in_top10 = [f for f in reference.get("fuzzy_heads", []) if any(e["label"] == f for e in top10)]
    claims += _mk(
        "paper.fuzzy",
        {"n_fuzzy": len(fuzzy_in_top10)},
        ["top10", "reference.fuzzy_heads", PROC_TOP10],
    )
    claims += _mk("paper.reference", {}, ["reference.note", "reference.verified_against_paper"])
    return claims


def story_agreement_methods(data: dict | None) -> list[dict]:
    """data: compact agreement result (stats, top10, notes).

    The qualitative phrase comes from ``agreement_phrase`` at the declared
    Pearson boundaries; the layer-group sentence is emitted only when both
    group values exist and differ by more than AGREEMENT_GROUP_DIFF.
    """
    data = data or {}
    stats = data.get("stats") or {}
    atp = stats.get("atp") or {}
    dla = stats.get("dla") or {}
    if atp.get("pearson_heads") is None or dla.get("pearson_heads") is None:
        return _not_run(
            "whether exact patching, attribution patching and direct attribution agree",
            "the agreement measurement on one task pair",
            suffix="agreement",
        )

    claims: list[dict] = []
    for method, block in (("atp", atp), ("dla", dla)):
        key = f"agree.{method}_overall"
        claims += _mk(
            key,
            {
                "rho": fmt_num(block.get("pearson_heads"), 3),
                "phrase": agreement_phrase(block["pearson_heads"], "the exact test"),
                "overlap": block.get("top10_overlap", "n/a"),
                "expected": fmt_num(block.get("expected_random_top10"), 1),
            },
            [
                f"stats.{method}.pearson_heads",
                f"stats.{method}.top10_overlap",
                f"stats.{method}.expected_random_top10",
                "fixed:agreement.phrase",
            ],
        )
        groups = block.get("pearson_groups") or {}
        early = (groups.get("blocks_0_5") or {}).get("pearson")
        late = (groups.get("blocks_6_11") or {}).get("pearson")
        if early is not None and late is not None and abs(early - late) > AGREEMENT_GROUP_DIFF:
            if early >= late:
                better, worse, rho_b, rho_w = "blocks 0 to 5", "blocks 6 to 11", early, late
            else:
                better, worse, rho_b, rho_w = "blocks 6 to 11", "blocks 0 to 5", late, early
            claims += _mk(
                "agree.groups",
                {
                    "method_label": {"atp": "the fast method", "dla": "the direct test"}.get(method, method),
                    "better": better,
                    "worse": worse,
                    "rho_better": fmt_num(rho_b, 3),
                    "rho_worse": fmt_num(rho_w, 3),
                },
                [
                    f"stats.{method}.pearson_groups.blocks_0_5.pearson",
                    f"stats.{method}.pearson_groups.blocks_6_11.pearson",
                ],
                suffix=method,
            )
    claims += _mk("agree.methods", {}, ["fixed:agree.methods"])
    return claims


def story_study(study: dict | None) -> list[dict]:
    if not study or study.get("n_prompts") is None:
        return _not_run(
            "the multi-prompt evidence",
            "attaching a completed study to the report",
            suffix="study",
        )
    claims = []
    claims += _mk("study.size", {"n_prompts": study["n_prompts"]}, ["study.n_prompts"])
    acc = study.get("model_accuracy")
    if acc is not None:
        claims += _mk("study.accuracy", {"acc": fmt_pct(acc)}, ["study.model_accuracy"])
    ex = study.get("exclusions") or {}
    n_excluded = sum(int(ex.get(k) or 0) for k in ("swap", "third", "resample"))
    claims += _mk(
        "study.exclusions", {"n_excluded": n_excluded}, ["study.exclusions", "study.exclusion_rule"]
    )

    atp = study.get("rq1_atp_heads")
    if atp and atp.get("mean") is not None:
        claims += _mk(
            "study.atp",
            {
                "rho": fmt_num(atp["mean"], 3),
                "lo": fmt_num(atp.get("ci_low"), 3),
                "hi": fmt_num(atp.get("ci_high"), 3),
            },
            ["study.rq1_atp_heads.mean", "study.rq1_atp_heads.ci_low", "study.rq1_atp_heads.ci_high"],
        )
    dla = study.get("rq1_dla_heads")
    if dla and dla.get("mean") is not None:
        claims += _mk(
            "study.dla",
            {
                "rho": fmt_num(dla["mean"], 3),
                "lo": fmt_num(dla.get("ci_low"), 3),
                "hi": fmt_num(dla.get("ci_high"), 3),
            },
            ["study.rq1_dla_heads.mean", "study.rq1_dla_heads.ci_low", "study.rq1_dla_heads.ci_high"],
        )

    overlap = study.get("overlap")
    if overlap:
        expected = fmt_num(overlap.get("expected_random_k10", TOP_K * TOP_K / N_HEADS), 1)
        if overlap.get("atp") is not None:
            claims += _mk(
                "study.overlap_atp",
                {"k_atp": overlap["atp"], "expected": expected},
                ["study.overlap.atp", "study.overlap.expected_random_k10"],
            )
        if overlap.get("dla") is not None:
            claims += _mk(
                "study.overlap_dla",
                {"k_dla": overlap["dla"], "expected": expected},
                ["study.overlap.dla", "study.overlap.expected_random_k10"],
            )

    stab = {s["condition"]: s.get("k10_overlap") for s in study.get("stability", [])}
    if stab.get("third_D") is not None and stab.get("resample_D") is not None:
        claims += _mk(
            "study.stable",
            {"k_third": stab["third_D"], "k_resample": stab["resample_D"]},
            [
                "study.stability",
                "procedure: stored rq2.topk.k10 matrix, swap_D row vs third_D and resample_D",
            ],
        )
    if stab.get("resample_P") is not None and stab.get("abl_zero_D") is not None:
        claims += _mk(
            "study.sensitivity",
            {"k_p": stab["resample_P"], "k_zero": stab["abl_zero_D"]},
            [
                "study.stability",
                "procedure: stored rq2.topk.k10 matrix, swap_D row vs resample_P and abl_zero_D",
            ],
        )
    return claims


def story_limits() -> list[dict]:
    keys = [
        "limits.one_prompt",
        "limits.sufficiency",
        "limits.dla_partial",
        "limits.model",
        "limits.other_prompts",
    ]
    claims: list[dict] = []
    for k in keys:
        claims += _mk(k, {}, [f"fixed:{k}"])
    return claims
