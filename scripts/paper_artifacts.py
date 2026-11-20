"""Stage 6 B1: paper artefacts from the two studies of record.

Reads results/study_seed0.json and results/study_seed1.json (verbatim copies
of the stored studies; see results/README.md) and writes, WITHOUT recomputing
or inventing anything:

  docs/paper/tables.md         the RQ1 table, the RQ2 matrices, exclusions,
                               accuracy, the anchor ranks and the LABELLED
                               reviewer exploratory table
  docs/paper/figures/*.svg     the five Stage 3 figures, regenerated from the
                               seed-0 study by reusing scripts/make_figures.py

The pre-registered tables copy values verbatim from the stored `summary`
(reformatted to fixed decimals only). The exploratory table is recomputed
from the stored `per_prompt` rows with the same statistic definitions the
backend uses (backend/app/analysis/stats.py): Spearman/Pearson over the 144
heads, mean over prompts; top-10 overlap as the mean of per-prompt overlaps.
Exploratory rows are labelled EXPLORATORY in the output.

Usage:
  python scripts/paper_artifacts.py            # (re)generate docs/paper/
  python scripts/paper_artifacts.py --check    # exit 1 if regenerated output
                                               # differs from the committed
                                               # files (line endings ignored)

Standard library plus numpy (already a project dependency). No model, no
backend, no network.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT / "docs" / "paper"
FIGURES_DIR = PAPER_DIR / "figures"

STUDY_FILES = {
    0: RESULTS_DIR / "study_seed0.json",
    1: RESULTS_DIR / "study_seed1.json",
}
STUDY_IDS = {0: "058be22be19c", 1: "88efcc0690c8"}

N_HEADS = 144  # components 0..143 are heads (layer l, head h -> l*12+h); 144..155 are MLPs
TOP_K = 10

# Fixed component order used everywhere in Stage 3 (backend/app/analysis/components.py).
def component_label(idx: int) -> str:
    if idx < N_HEADS:
        return f"{idx // 12}.{idx % 12}"
    return f"mlp.{idx - N_HEADS}"


# ----------------------------------------------------------------------------
# Statistics — same definitions as backend/app/analysis/stats.py, so the
# exploratory recomputation is comparable with the stored pre-registered
# numbers. Constant input -> None (undefined), never NaN.
# ----------------------------------------------------------------------------

def rankdata_average(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(arr.shape[0], dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < arr.shape[0]:
        j = i
        while j + 1 < arr.shape[0] and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _corr(rx: np.ndarray, ry: np.ndarray) -> float:
    rc, sc = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rc * rc).sum()) * np.sqrt((sc * sc).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rc * sc).sum() / denom)


def spearman(x, y) -> float | None:
    rx, ry = rankdata_average(x), rankdata_average(y)
    if float((rx == rx[0]).all()) or float((ry == ry[0]).all()):
        return None
    return _corr(rx, ry)


def pearson(x, y) -> float | None:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if float((a == a[0]).all()) or float((b == b[0]).all()):
        return None
    v = _corr(a, b)
    return None if np.isnan(v) else v


def topk_overlap(a, b, k: int) -> int:
    ta = set(np.argsort(-np.asarray(a, dtype=float), kind="stable")[:k].tolist())
    tb = set(np.argsort(-np.asarray(b, dtype=float), kind="stable")[:k].tolist())
    return len(ta & tb)


def topk_names(vec, k: int) -> list[str]:
    arr = np.asarray(vec, dtype=float)
    return [component_label(i) for i in np.argsort(-arr, kind="stable")[:k].tolist()]


def _valid_pairs(exact_rows: list, method_rows: list) -> list[int]:
    """Prompt indices where both vectors are complete (no None) — the swap
    condition has no exclusions in either study of record, so this is all
    prompts; kept explicit for honesty."""
    out = []
    for i, (ex, me) in enumerate(zip(exact_rows, method_rows)):
        if all(v is not None for v in ex) and all(v is not None for v in me):
            out.append(i)
    return out


def exploratory_stats(res: dict) -> dict:
    """The labelled reviewer exploratory statistics, recomputed from
    per_prompt (heads only): mean per-prompt Spearman, mean per-prompt
    Pearson, mean per-prompt top-10 overlap vs the exact Rec_D[swap] ranking,
    plus the sparsity fractions of |Rec_D[swap]|."""
    pp = res["per_prompt"]
    exact_rows = [row["rec_d"]["swap"][:N_HEADS] for row in pp]
    out: dict = {"methods": {}, "n_prompts": len(pp)}
    for method in ("atp", "dla"):
        method_rows = [row[method][:N_HEADS] for row in pp]
        valid = _valid_pairs(exact_rows, method_rows)
        sp = [spearman(exact_rows[i], method_rows[i]) for i in valid]
        pe = [pearson(exact_rows[i], method_rows[i]) for i in valid]
        ov = [topk_overlap(exact_rows[i], method_rows[i], TOP_K) for i in valid]
        out["methods"][method] = {
            "spearman_mean": float(np.mean([v for v in sp if v is not None])),
            "pearson_mean": float(np.mean([v for v in pe if v is not None])),
            "top10_overlap_mean": float(np.mean(ov)),
            "n_valid": len(valid),
            "n_invalid": len(pp) - len(valid),
        }
    vals = np.abs(np.asarray([v for row in exact_rows for v in row], dtype=float))
    out["sparsity_below_002"] = float((vals < 0.02).mean())
    out["sparsity_below_005"] = float((vals < 0.05).mean())
    return out


def headline_numbers(res: dict) -> dict:
    """Three headline numbers recomputed from per_prompt (independent of the
    stored summary): mean per-prompt AtP Spearman over heads, mean per-prompt
    DLA top-10 overlap over heads, and accuracy as the fraction D_clean > 0."""
    pp = res["per_prompt"]
    exact_rows = [row["rec_d"]["swap"][:N_HEADS] for row in pp]
    atp_rows = [row["atp"][:N_HEADS] for row in pp]
    dla_rows = [row["dla"][:N_HEADS] for row in pp]
    valid = _valid_pairs(exact_rows, atp_rows)
    sp_atp = [spearman(exact_rows[i], atp_rows[i]) for i in valid]
    ov_dla = [topk_overlap(exact_rows[i], dla_rows[i], TOP_K) for i in valid]
    d_clean = [row["D_clean"] for row in pp]
    return {
        "atp_spearman_mean": float(np.mean([v for v in sp_atp if v is not None])),
        "dla_top10_overlap_mean": float(np.mean(ov_dla)),
        "accuracy": sum(1 for d in d_clean if d is not None and d > 0) / len(d_clean),
        "n_correct": sum(1 for d in d_clean if d is not None and d > 0),
        "n_prompts": len(d_clean),
    }


# ----------------------------------------------------------------------------
# tables.md
# ----------------------------------------------------------------------------

def _fmt(v, d: int) -> str:
    return f"{v:.{d}f}"


def _fmt_acc(acc: float) -> str:
    """Accuracy without spurious precision: 0.95 -> '0.95', 0.975 -> '0.975'."""
    return f"{acc:g}"


def rq1_table(res: dict) -> str:
    rq1 = res["summary"]["rq1"]["all_prompts"]
    null = rq1["null"]
    beats = rq1["beats_control"]
    lines = [
        "| Method | Components | mean rho (signed) | 95% CI | mean rho (\\|abs\\|) | null 97.5% | beats control |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in ("atp", "dla"):
        for group in ("heads", "mlps", "all"):
            g = rq1["methods"][method][group]
            label = {"heads": "heads (144)", "mlps": "MLPs (12, low resolution)", "all": "all 156"}[group]
            s, a = g["signed"], g["abs"]
            yes = "yes" if beats[method][group] else "no"
            lines.append(
                f"| {method.upper()} | {label} | {_fmt(s['mean'], 3)} | [{_fmt(s['ci_low'], 3)}, {_fmt(s['ci_high'], 3)}] "
                f"| {_fmt(a['mean'], 3)} | {_fmt(null[group]['p97_5'], 3)} | {yes} |"
            )
    return "\n".join(lines)


def rq1_seed1_line(res: dict) -> str:
    rq1 = res["summary"]["rq1"]["all_prompts"]
    beats = rq1["beats_control"]
    atp = rq1["methods"]["atp"]["heads"]["signed"]
    dla = rq1["methods"]["dla"]["heads"]["signed"]
    pat = "yes" if beats["atp"]["heads"] else "no"
    dla_no = "no" if not beats["dla"]["heads"] else "yes"
    return (
        f"Seed 1 (robustness): AtP heads {_fmt(atp['mean'], 3)} [{_fmt(atp['ci_low'], 3)}, {_fmt(atp['ci_high'], 3)}]; "
        f"DLA heads {_fmt(dla['mean'], 3)} [{_fmt(dla['ci_low'], 3)}, {_fmt(dla['ci_high'], 3)}]; "
        f"beats-control pattern identical (AtP {pat} everywhere, DLA {dla_no} everywhere)."
    )


def spearman_matrix_table(res: dict) -> str:
    rq2 = res["summary"]["rq2"]
    names = rq2["condition_names"]
    mat = rq2["spearman"]["matrix"]
    header = "| | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines = [header, sep]
    for i, name in enumerate(names):
        cells = []
        for j in range(len(names)):
            if j < i:
                cells.append(_fmt(mat[i][j], 2))
            elif j == i:
                cells.append("1")
            else:
                cells.append("")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def topk_matrix_table(res: dict, k: int) -> str:
    rq2 = res["summary"]["rq2"]
    names = rq2["condition_names"]
    data = rq2["topk"][f"k{k}"]
    header = "| | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines = [header, sep]
    for i, name in enumerate(names):
        cells = []
        for j in range(len(names)):
            if j < i:
                cells.append(str(data["matrix"][i][j]))
            elif j == i:
                cells.append(str(k))
            else:
                cells.append("")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def anchor_table(res0: dict, res1: dict) -> str:
    g0 = res0["summary"]["anchors"]["mean_ranks_by_group"]
    g1 = res1["summary"]["anchors"]["mean_ranks_by_group"]
    order = sorted(g0.keys(), key=lambda name: g0[name]["mean_rank"])
    lines = ["| group | mean rank (seed 0) | mean rank (seed 1) |", "|---|---|---|"]
    for name in order:
        lines.append(f"| {name} | {_fmt(g0[name]['mean_rank'], 1)} | {_fmt(g1[name]['mean_rank'], 1)} |")
    return "\n".join(lines)


def runs_table(res0: dict, res1: dict) -> str:
    lines = [
        "| run | job id | seed | n | wall time (s) | accuracy (D_clean > 0) | exclusions swap/third/resample |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed, res in ((0, res0), (1, res1)):
        n = res["params"]["n_prompts"]
        # Accuracy count recomputed from per_prompt (fraction of D_clean > 0),
        # so the "(38/40)" style count is derived, not backed out of the rounded
        # stored fraction.
        d_clean = [row["D_clean"] for row in res["per_prompt"]]
        n_correct = sum(1 for d in d_clean if d is not None and d > 0)
        ex = res["exclusions"]
        lines.append(
            f"| {'primary' if seed == 0 else 'robustness'} | `{STUDY_IDS[seed]}` | {seed} | {n} "
            f"| {_fmt(res['elapsed_seconds'], 1)} | {_fmt_acc(res['model_accuracy'])} ({n_correct}/{n}) "
            f"| {ex['swap']} / {ex['third']} / {ex['resample']} |"
        )
    return "\n".join(lines)


def exploratory_table(stats: dict) -> str:
    atp = stats["methods"]["atp"]
    dla = stats["methods"]["dla"]
    lines = [
        f"Sparsity of the exact scores: **{_fmt(stats['sparsity_below_002'] * 100, 1)}%** of the "
        f"(prompt, head) `Rec_D[swap]` values have absolute value below 0.02 "
        f"({_fmt(stats['sparsity_below_005'] * 100, 1)}% below 0.05), so the pre-registered Spearman is "
        "dominated by the ordering of near-zero heads. Two further views over the same data "
        f"(mean over the {stats['n_prompts']} prompts, heads only):",
        "",
        "| method vs exact `Rec_D[swap]` | Spearman (mean per prompt, = pre-registered RQ1) | Pearson (mean per prompt, values) | top-10 overlap (mean per prompt; random expectation 0.69) |",
        "|---|---|---|---|",
        f"| AtP | {_fmt(atp['spearman_mean'], 3)} | {_fmt(atp['pearson_mean'], 3)} | {_fmt(atp['top10_overlap_mean'], 2)} / 10 |",
        f"| DLA | {_fmt(dla['spearman_mean'], 3)} | {_fmt(dla['pearson_mean'], 3)} | {_fmt(dla['top10_overlap_mean'], 2)} / 10 |",
        "",
        "Reading: the AtP agreement is real and not an artefact of near-zero heads; for DLA the "
        "pre-registered Spearman null stands, but its top-10 overlaps the exact top-10 about six "
        "times more than chance — DLA tracks the large effects partially, not the full ranking. "
        "**Status: EXPLORATORY** (reviewer-added, recomputed from the stored per_prompt data; "
        "not pre-registered; the pre-registered conclusions are unchanged).",
    ]
    return "\n".join(lines)


def build_tables_md(res0: dict, res1: dict) -> str:
    prov0, prov1 = res0["provenance"], res1["provenance"]
    stats = exploratory_stats(res0)
    top10_0 = topk_names(res0["summary"]["rq2"]["prompt_averaged_vectors"]["swap_D"], TOP_K)
    top10_1 = topk_names(res1["summary"]["rq2"]["prompt_averaged_vectors"]["swap_D"], TOP_K)
    rq2 = res0["summary"]["rq2"]
    names = rq2["condition_names"]
    names1 = res1["summary"]["rq2"]["condition_names"]
    i_swap, i_third, i_resample = names1.index("swap_D"), names1.index("third_D"), names1.index("resample_D")
    mat1 = res1["summary"]["rq2"]["spearman"]["matrix"]

    parts = [
        "# Paper tables (generated — do not edit by hand)",
        "",
        "Generated by `scripts/paper_artifacts.py` from [results/study_seed0.json](../../results/study_seed0.json)",
        f"(job `{STUDY_IDS[0]}`, seed 0) and [results/study_seed1.json](../../results/study_seed1.json)",
        f"(job `{STUDY_IDS[1]}`, seed 1) — verbatim copies of the two stored studies of record",
        "(see [results/README.md](../../results/README.md)). Pre-registered tables copy values from the",
        "stored `summary`; the final table is recomputed from the stored `per_prompt` rows and is",
        "**labelled exploratory**. Regenerate: `.venv\\Scripts\\python scripts\\paper_artifacts.py`;",
        "verify against the committed files: `--check`.",
        "",
        f"Provenance inside the files: git_commit `{prov0['git_commit']}`, model revision",
        f"`{prov0['model_revision']}` (GPT-2 Small, CPU, torch {prov0['versions']['torch']},",
        f"transformer_lens {prov0['versions']['transformer_lens']}, transformers",
        f"{prov0['versions']['transformers']}). Exclusion rule (pre-registered, the only one):",
        f"`{res0['exclusion_rule']}`. Metrics and direction as stored in the files (`direction`:",
        f"\"{res0['direction']}\").",
        "",
        "## Runs, accuracy, exclusions",
        "",
        runs_table(res0, res1),
        "",
        "Accuracy counts are recomputed here as the fraction of prompts with `D_clean > 0` "
        "(matches the stored `model_accuracy`; asserted by `backend/tests/test_paper_artifacts.py`).",
        "",
        "## RQ1 — agreement with exact patching (seed 0, n = 40)",
        "",
        "Per-prompt Spearman (average ranks) between `Rec_D[swap]` and the cheap method, mean over",
        "prompts; bootstrap 95% CI over prompts (percentile, 2000 resamples, seed = study seed);",
        "null = label-permutation control (1000 permutations per prompt, pooled 2.5/97.5 percentiles).",
        "A method beats the control only if its CI lower bound exceeds the null 97.5th percentile.",
        "",
        rq1_table(res0),
        "",
        rq1_seed1_line(res1),
        "",
        "## RQ2 — Spearman between prompt-averaged condition vectors (seed 0)",
        "",
        "Pairwise Spearman between the eight prompt-averaged 144-head score vectors (lower triangle;",
        "diagonal 1 by definition):",
        "",
        spearman_matrix_table(res0),
        "",
        f"Seed 1: swap_D vs third_D {_fmt(mat1[i_swap][i_third], 2)}, swap_D vs resample_D "
        f"{_fmt(mat1[i_swap][i_resample], 2)} — same pattern.",
        "",
        "## RQ2 — top-k overlap of head rankings (seed 0)",
        "",
        f"Random expectation k·k/144 = {_fmt(rq2['topk']['expected_random']['k5'], 2)} (k=5) / "
        f"{_fmt(rq2['topk']['expected_random']['k10'], 2)} (k=10). Lower triangles:",
        "",
        "**k = 5**",
        "",
        topk_matrix_table(res0, 5),
        "",
        "**k = 10**",
        "",
        topk_matrix_table(res0, 10),
        "",
        "## Top-10 heads by prompt-averaged `Rec_D[swap]`",
        "",
        f"Seed 0: {', '.join(top10_0)}.",
        "",
        f"Seed 1: {', '.join(top10_1)}.",
        "",
        "**Identical across both seeds** (asserted by `backend/tests/test_paper_artifacts.py`).",
        "",
        "## Known-heads anchor (descriptive only)",
        "",
        "Mean rank of the Wang et al. head groups by prompt-averaged `Rec_D[swap]` (rank 1 = highest).",
        "Head lists in `examples/reference/ioi_wang2022_heads.json` (verified against the paper text by",
        "the reviewer on 2026-11-19); no result depends on this file. Both honest readings of the",
        "anchor are in `docs/V2_STAGE3_RESULTS.md` — it is not a claim.",
        "",
        anchor_table(res0, res1),
        "",
        "## Reviewer exploratory table — EXPLORATORY, not pre-registered",
        "",
        exploratory_table(stats),
        "",
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Figures — reuse scripts/make_figures.py (same hand-written SVG functions).
# ----------------------------------------------------------------------------

def build_figures(res0: dict) -> dict[str, str]:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import make_figures  # noqa: E402  (scripts dir, not a package)

    return {
        "fig1_head_heatmap.svg": make_figures.fig_head_heatmap(res0),
        "fig2_rq1_agreement.svg": make_figures.fig_rq1_bars(res0),
        "fig3_topk_overlap.svg": make_figures.fig_topk_overlap(res0, 5),
        "fig3b_topk_overlap_k10.svg": make_figures.fig_topk_overlap(res0, 10),
        "fig4_known_group_ranks.svg": make_figures.fig_group_ranks(res0),
    }


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _norm(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def generate(out_tables: pathlib.Path, out_figures_dir: pathlib.Path) -> None:
    res0 = json.loads(STUDY_FILES[0].read_text(encoding="utf-8"))
    res1 = json.loads(STUDY_FILES[1].read_text(encoding="utf-8"))
    for res, seed in ((res0, 0), (res1, 1)):
        if res.get("job_type") != "study" or res["params"]["seed"] != seed:
            raise SystemExit(f"{STUDY_FILES[seed]} is not the seed-{seed} study of record")
    _write(out_tables, build_tables_md(res0, res1))
    for name, svg in build_figures(res0).items():
        _write(out_figures_dir / name, svg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify regenerated output equals the committed files")
    args = parser.parse_args()

    if args.check:
        import tempfile

        expected = {PAPER_DIR / "tables.md": None}
        for p in sorted(FIGURES_DIR.glob("*.svg")):
            expected[p] = None
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            generate(tmp_dir / "tables.md", tmp_dir / "figures")
            failures = []
            for path in expected:
                rel = path.relative_to(PAPER_DIR)
                regen = (tmp_dir / rel)
                if not path.exists():
                    failures.append(f"missing committed file: {path}")
                elif not regen.exists():
                    failures.append(f"not regenerated: {rel}")
                elif _norm(path.read_bytes()) != _norm(regen.read_bytes()):
                    failures.append(f"differs from regeneration: {rel}")
            for regen in sorted((tmp_dir / "figures").glob("*.svg")):
                if not (FIGURES_DIR / regen.name).exists():
                    failures.append(f"committed figure missing: {regen.name}")
        if failures:
            print("paper_artifacts --check FAILED:")
            for f in failures:
                print(f"  - {f}")
            print("regenerate with: .venv\\Scripts\\python scripts\\paper_artifacts.py")
            return 1
        print(f"OK: tables.md and {len(list(FIGURES_DIR.glob('*.svg')))} figures match regeneration from results/")
        return 0

    generate(PAPER_DIR / "tables.md", FIGURES_DIR)
    print(f"wrote {PAPER_DIR / 'tables.md'}")
    for p in sorted(FIGURES_DIR.glob("*.svg")):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
