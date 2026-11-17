"""Stage 3 Task 6: the "study" job type — the pre-registered Stage 3 study.

Pipeline per prompt (Task 6 spec):
  clean cache -> 3 corrupted baselines -> exact denoising for 3 corruptions
  (both metrics, all 156 components) -> AtP (swap/D) -> DLA (Stage 2) on the
  clean run -> second pass for zero/mean ablation.

The analysis follows the pre-registered plan in docs/PAPER_NOTES.md section 4b
(written and committed BEFORE this study first ran):
  - exclusion rule: per (prompt, corruption) pair excluded iff
    |D_clean - D_corr| < 0.5 (counts reported per condition);
  - model accuracy = fraction of prompts with D_clean > 0;
  - headline numbers repeated on the D_clean > 0 subset (sensitivity check);
  - RQ1: per-prompt Spearman(Rec_D[swap], AtP / DLA) over heads / MLPs / all
    156, signed (primary) and absolute (secondary), bootstrap 95% CI over
    prompts (percentile, 2000 resamples, numpy.random.default_rng(seed)),
    permutation null (1000 permutations, same seed);
  - RQ2: 3 corruptions x 2 metrics + zero/mean ablation, prompt-averaged
    144-head vectors, pairwise Spearman + top-k overlap (k=5, 10) with
    bootstrap CIs over prompts, random expectation k*k/144;
  - known-heads anchor: mean rank by Rec_D[swap] per group (descriptive only).
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from ..analysis import components as comps
from ..analysis import promptsets
from ..analysis import stats as st
from ..analysis.atp import compute_atp
from ..analysis.dla import compute_dla
from ..config import MAX_LAYERS
from ..experiments.jobs import ExperimentValidationError, JobContext, register_job_type
from ..model_runtime.loader import ADAPTER
from ..provenance import collect as collect_provenance

EXCLUSION_THRESHOLD = 0.5  # |D_clean - D_corr| below this -> excluded
N_BOOT = 2000
N_PERM = 1000
CORRUPTIONS = ("swap", "third", "resample")
CONDITIONS = [
    "swap_D", "swap_P", "third_D", "third_P",
    "resample_D", "resample_P", "abl_zero_D", "abl_mean_D",
]
GROUP_SLICES = {"heads": slice(0, 144), "mlps": slice(144, 156), "all": slice(0, comps.N_COMPONENTS)}

REFERENCE_HEADS_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "examples" / "reference" / "ioi_wang2022_heads.json"
)


def validate_study_config(cfg: dict) -> dict:
    n = cfg.get("n_prompts", 40)
    seed = cfg.get("seed", 0)
    if not isinstance(n, int) or isinstance(n, bool) or not 4 <= n <= 100:
        raise ExperimentValidationError("n_prompts must be an int in [4, 100]")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ExperimentValidationError("seed must be a non-negative int")
    return {"n_prompts": n, "seed": seed, "job_type": "study"}  # persisted for history filtering


def _load_known_heads() -> dict:
    try:
        with open(REFERENCE_HEADS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # the anchor is descriptive only; never fail a study
        return {"error": f"could not load {REFERENCE_HEADS_PATH.name}: {type(e).__name__}: {e}", "groups": []}


def _r6(values) -> list:
    return [None if v is None else round(float(v), 6) for v in values]


def _clean_vec(vec) -> list[float] | None:
    """A vector usable for numpy stats, or None if it is absent or contains
    any undefined (None) entry (e.g. a zero denominator)."""
    if vec is None or any(v is None for v in vec):
        return None
    return [float(v) for v in vec]


def _mean_ci(values: list[float | None], rng: np.random.Generator) -> dict:
    """Bootstrap 95% CI of the mean over prompts; None (undefined Spearman)
    values are dropped and counted."""
    valid = [v for v in values if v is not None]
    n_invalid = len(values) - len(valid)
    if not valid:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_invalid": n_invalid}
    arr = np.asarray(valid, dtype=float)
    ci = st.bootstrap_ci(arr, np.mean, N_BOOT, rng)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n": int(arr.shape[0]),
        "n_invalid": n_invalid,
    }


def _rq1_block(rows: list[dict], rng: np.random.Generator) -> dict:
    """RQ1 statistics for one subset of prompt rows (already filtered)."""
    groups = ("heads", "mlps", "all")
    out: dict = {"n_prompts": len(rows), "prompt_ids": [r["id"] for r in rows], "methods": {}}
    for method, key in (("atp", "atp"), ("dla", "dla")):
        out["methods"][method] = {}
        for group in groups:
            s = GROUP_SLICES[group]
            signed, absolute = [], []
            for row in rows:
                exact = _clean_vec(row["rec_d"]["swap"])
                cheap = _clean_vec(row[key])
                if exact is None or cheap is None:
                    continue
                ex = np.asarray(exact[s], dtype=float)
                ch = np.asarray(cheap[s], dtype=float)
                signed.append(st.spearman(ex, ch))
                absolute.append(st.spearman(np.abs(ex), np.abs(ch)))
            out["methods"][method][group] = {
                "signed": _mean_ci(signed, rng),
                "abs": _mean_ci(absolute, rng),
                "n_components": {"heads": 144, "mlps": 12, "all": 156}[group],
                "low_resolution": group == "mlps",  # n=12 is low resolution
            }
    # Null control: correlation of the exact scores with a random permutation
    # of the component labels (same rng, seeded by the study seed).
    nulls: dict[str, list[float]] = {g: [] for g in groups}
    for row in rows:
        exact = _clean_vec(row["rec_d"]["swap"])
        if exact is None:
            continue
        for group in groups:
            s = GROUP_SLICES[group]
            arr = np.asarray(exact[s], dtype=float)
            res = st.permutation_null(arr, arr, N_PERM, rng)
            if res["p2_5"] is not None:
                nulls[group].extend([res["p2_5"], res["p97_5"]])
    out["null"] = {}
    for group in groups:
        vals = np.asarray(nulls[group], dtype=float)
        out["null"][group] = {
            "p2_5": float(np.percentile(vals, 2.5)) if vals.size else None,
            "p97_5": float(np.percentile(vals, 97.5)) if vals.size else None,
            "note": "null percentiles pooled over per-prompt nulls (1000 label permutations each)",
        }
    beats: dict = {}
    for method in out["methods"]:
        beats[method] = {}
        for group in groups:
            lo = out["methods"][method][group]["signed"]["ci_low"]
            null_hi = out["null"][group]["p97_5"]
            beats[method][group] = bool(lo is not None and null_hi is not None and lo > null_hi)
    out["beats_control"] = beats
    out["beats_control_rule"] = "CI lower bound (signed) > null 97.5th percentile"
    return out


def _bootstrap_condition_stat(cond_items: list[list[tuple[str, list[float]]]], rng: np.random.Generator) -> dict:
    """Pairwise Spearman + top-k overlap between the 8 conditions'
    prompt-averaged 144-head vectors, with percentile bootstrap CIs over
    prompts. ``cond_items[i]`` is a list of (prompt_id, 144-vector) in prompt
    order; each pairwise comparison uses the paired common prompt set."""
    n_cond = len(CONDITIONS)

    def _avg(pairs: list[tuple[str, list[float]]], idx: np.ndarray | None) -> np.ndarray:
        sel = pairs if idx is None else [pairs[i] for i in idx]
        return np.asarray([v for _, v in sel], dtype=float).mean(axis=0)

    mat = [[None] * n_cond for _ in range(n_cond)]
    ci_lo = [[None] * n_cond for _ in range(n_cond)]
    ci_hi = [[None] * n_cond for _ in range(n_cond)]
    ov5 = [[None] * n_cond for _ in range(n_cond)]
    ov10 = [[None] * n_cond for _ in range(n_cond)]
    ov5_lo = [[None] * n_cond for _ in range(n_cond)]
    ov5_hi = [[None] * n_cond for _ in range(n_cond)]
    ov10_lo = [[None] * n_cond for _ in range(n_cond)]
    ov10_hi = [[None] * n_cond for _ in range(n_cond)]

    for i in range(n_cond):
        if cond_items[i]:
            mat[i][i], ci_lo[i][i], ci_hi[i][i] = 1.0, 1.0, 1.0
            ov5[i][i], ov10[i][i] = 5, 10
            ov5_lo[i][i], ov5_hi[i][i] = 5.0, 5.0
            ov10_lo[i][i], ov10_hi[i][i] = 10.0, 10.0
    for a in range(n_cond):
        for b in range(a + 1, n_cond):
            if not cond_items[a] or not cond_items[b]:
                continue
            # Paired comparison on the common prompt set (both lists stay in
            # prompt order, so rows align).
            common = {pid for pid, _ in cond_items[a]} & {pid for pid, _ in cond_items[b]}
            va = [p for p in cond_items[a] if p[0] in common]
            vb = [p for p in cond_items[b] if p[0] in common]
            if not va or not vb:
                continue
            avg_a, avg_b = _avg(va, None), _avg(vb, None)
            rho = st.spearman(avg_a, avg_b)
            mat[a][b] = mat[b][a] = rho
            ov5[a][b] = ov5[b][a] = st.topk_overlap(avg_a, avg_b, 5)
            ov10[a][b] = ov10[b][a] = st.topk_overlap(avg_a, avg_b, 10)
            n = min(len(va), len(vb))
            rhos, o5s, o10s = [], [], []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                aa, bb = _avg(va, idx), _avg(vb, idx)
                r = st.spearman(aa, bb)
                if r is not None:
                    rhos.append(r)
                o5s.append(st.topk_overlap(aa, bb, 5))
                o10s.append(st.topk_overlap(aa, bb, 10))
            if rhos:
                ci_lo[a][b] = ci_lo[b][a] = float(np.percentile(rhos, 2.5))
                ci_hi[a][b] = ci_hi[b][a] = float(np.percentile(rhos, 97.5))
            ov5_lo[a][b] = ov5_lo[b][a] = float(np.percentile(o5s, 2.5))
            ov5_hi[a][b] = ov5_hi[b][a] = float(np.percentile(o5s, 97.5))
            ov10_lo[a][b] = ov10_lo[b][a] = float(np.percentile(o10s, 2.5))
            ov10_hi[a][b] = ov10_hi[b][a] = float(np.percentile(o10s, 97.5))

    avg_vectors = {
        CONDITIONS[i]: _r6(_avg(cond_items[i], None).tolist()) if cond_items[i] else None
        for i in range(n_cond)
    }
    return {
        "condition_names": CONDITIONS,
        "n_per_condition": {CONDITIONS[i]: len(cond_items[i]) for i in range(n_cond)},
        "spearman": {
            "matrix": mat,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "note": "pairwise Spearman between prompt-averaged 144-head score vectors "
            "(paired common prompt set per pair); bootstrap 95% CI over prompts "
            "(resample prompts, re-average, recompute)",
        },
        "topk": {
            "k5": {"matrix": ov5, "ci_low": ov5_lo, "ci_high": ov5_hi},
            "k10": {"matrix": ov10, "ci_low": ov10_lo, "ci_high": ov10_hi},
            "expected_random": {
                "k5": st.expected_random_overlap(5, 144),
                "k10": st.expected_random_overlap(10, 144),
                "note": "expected |A ∩ B| for two independent random k-subsets of 144 heads",
            },
        },
        "prompt_averaged_vectors": avg_vectors,
    }


def _anchors(avg_swap: np.ndarray | None) -> dict:
    known = _load_known_heads()
    out: dict = {
        "known_heads_reference": known,
        "ranking_basis": "prompt-averaged Rec_D[swap]; rank 1 = highest score",
        "descriptive_only": True,
    }
    if avg_swap is None:
        out["mean_ranks_by_group"] = None
        return out
    order = np.argsort(-avg_swap, kind="stable")
    ranks = np.empty(144, dtype=float)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    by_group: dict = {}
    for g in known.get("groups", []):
        idxs: list[int] = []
        for spec in g.get("heads", []):
            try:
                l_s, h_s = spec.split(".")
                l, h = int(l_s), int(h_s)
            except ValueError:
                continue
            if 0 <= l < MAX_LAYERS and 0 <= h < 12:
                idxs.append(l * 12 + h)
        by_group[g["group"]] = {
            "heads": g.get("heads", []),
            "ranks": [float(ranks[i]) for i in idxs],
            "mean_rank": float(np.mean([ranks[i] for i in idxs])) if idxs else None,
        }
    out["mean_ranks_by_group"] = by_group
    return out


def run_study(cfg: dict, ctx: JobContext) -> dict:
    norm = validate_study_config(cfg)
    n, seed = norm["n_prompts"], norm["seed"]
    t0 = time.perf_counter()
    ADAPTER.ensure_loaded()

    ctx.report(0.01, "generating prompt set")
    ps = promptsets.generate(n, seed)
    prompts = ps["prompts"]

    rows: list[dict] = []
    # Two-pass design: pass 1 caches clean activations per prompt (cloned into
    # the leave-one-out family); pass 2 computes mean ablations.
    family_caches: dict[int, list[dict]] = {t: [] for t in (1, 2, 3)}
    family_rows: dict[int, list[dict]] = {t: [] for t in (1, 2, 3)}

    # ---- pass 1
    for i, p in enumerate(prompts):
        ctx.check_cancel()
        cand_a, cand_b = p["candidate_a"], p["candidate_b"]
        logits_clean, cache = comps.run_clean(p["clean_text"])
        tok_a, tok_b = comps.token_ids(cand_a, cand_b)
        d_clean, p_clean = comps.metrics_from_logits(logits_clean, tok_a, tok_b)
        # Clones for the leave-one-out family (the engine never mutates a
        # cache, but the family outlives this iteration).
        family_caches[p["template_id"]].append({k: v.clone() for k, v in cache.items()})

        row: dict = {
            "id": p["id"],
            "template_id": p["template_id"],
            "s2_role": p["s2_role"],
            "D_clean": d_clean,
            "P_clean": p_clean,
            "baselines": {},
            "rec_d": {},
            "rec_p": {},
            "atp": None,
            "dla": None,
            "abl_zero": None,
            "abl_mean": None,
        }
        for k_i, kind in enumerate(CORRUPTIONS):
            ctx.check_cancel()
            text = p["corruptions"].get(kind)
            if text is None:
                row["baselines"][kind] = {"missing": True, "excluded": True}
                row["rec_d"][kind] = None
                row["rec_p"][kind] = None
                continue
            eff = comps.denoise_effects(
                text, cand_a, cand_b, logits_clean, cache, check_cancel=ctx.check_cancel
            )
            excluded = (
                abs(eff["D_clean"] - eff["D_corr"]) < EXCLUSION_THRESHOLD
                or eff["denominator_d"] == 0.0
            )
            row["baselines"][kind] = {
                "D_corr": round(eff["D_corr"], 6),
                "P_corr": round(eff["P_corr"], 6),
                "abs_D_diff": round(abs(eff["D_clean"] - eff["D_corr"]), 6),
                "excluded": bool(excluded),
            }
            row["rec_d"][kind] = eff["rec_d"]
            row["rec_p"][kind] = eff["rec_p"]
            ctx.report(
                min(0.88, (i + 0.15 + 0.15 * (k_i + 1) / 3) / n),
                f"prompt {i + 1}/{n}: exact denoising ({kind})",
            )

        ctx.check_cancel()
        swap_text = p["corruptions"].get("swap")
        if swap_text is not None:
            atp = compute_atp(swap_text, cand_a, cand_b, logits_clean, cache)
            row["atp"] = atp["rec_atp"]
        ctx.report(min(0.89, (i + 0.92) / n), f"prompt {i + 1}/{n}: AtP")

        ctx.check_cancel()
        dla = compute_dla(p["clean_text"], cand_a, cand_b)
        dla_vec: list[float | None] = [None] * comps.N_COMPONENTS
        for c in dla["components"]:
            if c["kind"] == "head":
                dla_vec[c["layer"] * 12 + c["head"]] = c["value"]
            elif c["kind"] == "mlp":
                dla_vec[144 + c["layer"]] = c["value"]
        row["dla"] = dla_vec
        ctx.report(min(0.895, (i + 0.94) / n), f"prompt {i + 1}/{n}: DLA")

        ctx.check_cancel()
        zabl = comps.zero_ablation_effects(
            p["clean_text"], cand_a, cand_b, logits_clean, cache, check_cancel=ctx.check_cancel
        )
        row["abl_zero"] = zabl["abl_d"]
        ctx.report(min(0.9, (i + 1) / n * 0.9), f"prompt {i + 1}/{n}: zero ablation")

        rows.append(row)
        family_rows[p["template_id"]].append(row)

    # ---- pass 2: mean ablation (needs the family caches from pass 1)
    n_mean_skipped = 0
    for i, p in enumerate(prompts):
        ctx.check_cancel()
        row = rows[i]
        cand_a, cand_b = p["candidate_a"], p["candidate_b"]
        fam_idx = [
            j for j, fr in enumerate(family_rows[p["template_id"]]) if fr is not row
        ]
        family = [family_caches[p["template_id"]][j] for j in fam_idx]
        if not family:
            row["abl_mean"] = None
            n_mean_skipped += 1
            continue
        logits_clean, cache = comps.run_clean(p["clean_text"])
        mabl = comps.mean_ablation_effects(
            p["clean_text"], cand_a, cand_b, logits_clean, cache, family,
            check_cancel=ctx.check_cancel,
        )
        row["abl_mean"] = mabl["abl_d"]
        ctx.report(0.9 + 0.08 * (i + 1) / n, f"prompt {i + 1}/{n}: mean ablation")

    # ---- pre-registered statistics
    ctx.report(0.985, "computing pre-registered statistics")
    rng = np.random.default_rng(seed)

    accuracy = sum(1 for r in rows if r["D_clean"] > 0) / len(rows)

    rq1_all = _rq1_block([r for r in rows if _clean_vec(r["rec_d"]["swap"]) is not None], rng)
    subset = [r for r in rows if r["D_clean"] > 0 and _clean_vec(r["rec_d"]["swap"]) is not None]
    rq1_subset = _rq1_block(subset, rng)

    def _cond_items(kind: str, field: str) -> list[tuple[str, list[float]]]:
        items = []
        for r in rows:
            if r["baselines"][kind]["excluded"]:
                continue
            v = _clean_vec((r[field].get(kind) or [None] * comps.N_COMPONENTS)[:144])
            if v is not None:
                items.append((r["id"], v))
        return items

    cond_items: list[list[tuple[str, list[float]]]] = []
    for name in CONDITIONS:
        if name == "abl_zero_D":
            cond_items.append(
                [
                    (r["id"], v)
                    for r in rows
                    if r["abl_zero"] is not None
                    and (v := _clean_vec(r["abl_zero"][:144])) is not None
                ]
            )
        elif name == "abl_mean_D":
            cond_items.append(
                [
                    (r["id"], v)
                    for r in rows
                    if r["abl_mean"] is not None
                    and (v := _clean_vec(r["abl_mean"][:144])) is not None
                ]
            )
        else:
            kind, metric = name.split("_")
            cond_items.append(_cond_items(kind, "rec_d" if metric == "D" else "rec_p"))
    rq2 = _bootstrap_condition_stat(cond_items, rng)

    swap_items = _cond_items("swap", "rec_d")
    avg_swap = (
        np.asarray([v for _, v in swap_items], dtype=float).mean(axis=0) if swap_items else None
    )
    anchors = _anchors(avg_swap)

    exclusions = {
        kind: sum(1 for r in rows if r["baselines"][kind]["excluded"]) for kind in CORRUPTIONS
    }
    exclusions["abl_zero"] = 0
    exclusions["abl_mean"] = n_mean_skipped

    elapsed = time.perf_counter() - t0
    return {
        "job_type": "study",
        "params": {"n_prompts": n, "seed": seed},
        "direction": "denoising = clean -> corrupted (sufficiency); ablation on the clean run (necessity)",
        "metrics": {
            "D": "logit(IO) - logit(S) at the final position",
            "P": "softmax probability of the IO candidate over the full vocab",
            "Rec_D": "(D_patched - D_corr) / (D_clean - D_corr)",
            "Rec_P": "(P_patched - P_corr) / (P_clean - P_corr)",
            "Abl_D": "D_clean - D_ablated (raw logit units)",
        },
        "exclusion_rule": f"|D_clean - D_corr| < {EXCLUSION_THRESHOLD} per (prompt, corruption) pair",
        "prompt_set": {
            "prompts": prompts,
            "rejected_draws": ps["rejected_draws"],
            "template_counts": ps["template_counts"],
            "s2_role_counts": ps["s2_role_counts"],
        },
        "model_accuracy": accuracy,
        "exclusions": exclusions,
        "per_prompt": [
            {
                "id": r["id"],
                "template_id": r["template_id"],
                "s2_role": r["s2_role"],
                "D_clean": round(r["D_clean"], 6),
                "P_clean": round(r["P_clean"], 6),
                "baselines": r["baselines"],
                "rec_d": {k: (_r6(v) if v is not None else None) for k, v in r["rec_d"].items()},
                "rec_p": {k: (_r6(v) if v is not None else None) for k, v in r["rec_p"].items()},
                "atp": _r6(r["atp"]) if r["atp"] is not None else None,
                "dla": _r6(r["dla"]) if r["dla"] is not None else None,
                "abl_zero": _r6(r["abl_zero"]) if r["abl_zero"] is not None else None,
                "abl_mean": _r6(r["abl_mean"]) if r["abl_mean"] is not None else None,
            }
            for r in rows
        ],
        "baselines_per_prompt": [
            {
                "id": r["id"],
                "D_clean": round(r["D_clean"], 6),
                "P_clean": round(r["P_clean"], 6),
                **{f"{k}_D_corr": r["baselines"][k].get("D_corr") for k in CORRUPTIONS},
                **{f"{k}_excluded": r["baselines"][k].get("excluded") for k in CORRUPTIONS},
            }
            for r in rows
        ],
        "summary": {
            "rq1": {"all_prompts": rq1_all, "subset_D_clean_gt0": rq1_subset},
            "rq2": rq2,
            "anchors": anchors,
        },
        "provenance": {"elapsed_seconds": round(elapsed, 4), **collect_provenance()},
        "elapsed_seconds": round(elapsed, 4),
    }


def register() -> None:
    """Register the "study" job type (called from routes.py, like sweeps.py)."""
    register_job_type("study", validate_study_config, run_study)
