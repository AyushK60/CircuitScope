"""Stage 3 Task 8: figures from a completed study — hand-written SVG, no
matplotlib, no new dependencies (stdlib only).

Reads a completed study (by --job-id from a live backend, or --json file) and
writes to outputs/paper/ (git-ignored):
  - stage3_summary.json  (the study summary + params, for the report)
  - fig1_head_heatmap.svg      12x12 head heatmap of prompt-averaged Rec_D[swap]
  - fig2_rq1_agreement.svg     agreement bars (Spearman, CI, null band) for RQ1
  - fig3_topk_overlap.svg      top-5/top-10 overlap matrices across conditions
  - fig4_known_group_ranks.svg ranks of the known Wang et al. 2022 head groups

All figures are readable in grayscale, carry axis labels, n, seed, and a
one-line caption stating direction and metric. Every number comes verbatim
from the stored study results — nothing is recomputed or invented here.

Usage:
  python scripts/make_figures.py --job-id <id>     # live backend on :8000
  python scripts/make_figures.py --json <path>     # saved study results JSON
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = pathlib.Path("outputs/paper")

# grayscale ramp (light -> dark), readable in print
INK = "#1c1c1c"
MID = "#9a9a9a"
LIGHT = "#ececec"


def _shade(v: float, vmax: float) -> str:
    """Grayscale fill for a value in [-vmax, vmax]: diverging around mid-grey."""
    if v is None or vmax <= 0:
        return LIGHT
    t = max(-1.0, min(1.0, v / vmax))
    # 0 -> light, extremes -> dark
    g = round(236 - abs(t) * 200)
    return f"#{g:02x}{g:02x}{g:02x}"


def _text_color(v: float, vmax: float) -> str:
    return "#f4f1ea" if v is not None and vmax > 0 and abs(v / vmax) > 0.55 else INK


def _fmt(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def fig_head_heatmap(res: dict) -> str:
    """Figure 1: 12x12 heatmap of prompt-averaged Rec_D[swap] (heads)."""
    vec = res["summary"]["rq2"]["prompt_averaged_vectors"].get("swap_D")
    cell, label_w, label_h = 44, 56, 44
    w = label_w + 12 * cell + 20
    h = label_h + 12 * cell + 90
    vmax = max(abs(v) for v in vec) if vec else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" role="img" '
        f'aria-label="Heatmap of prompt-averaged Rec_D swap across 144 heads">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{label_w}" y="18" font-family="monospace" font-size="13" fill="{INK}">'
        f"Prompt-averaged Rec_D[swap] per head — denoising (clean -&gt; corrupted), metric D "
        f"(n={res['params']['n_prompts']} prompts, seed {res['params']['seed']})</text>",
    ]
    # column labels (head index) and row labels (layer)
    for hh in range(12):
        x = label_w + hh * cell
        parts.append(
            f'<text x="{x + cell / 2}" y="{label_h - 6}" text-anchor="middle" font-family="monospace" font-size="11" fill="{INK}">{hh}</text>'
        )
    for l in range(12):
        y = label_h + l * cell
        parts.append(
            f'<text x="{label_w - 8}" y="{y + cell / 2 + 4}" text-anchor="end" font-family="monospace" font-size="11" fill="{INK}">{l}</text>'
        )
        for hh in range(12):
            v = vec[l * 12 + hh]
            parts.append(
                f'<rect x="{label_w + hh * cell}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                f'fill="{_shade(v, vmax)}" stroke="#bbb" stroke-width="0.5"/>'
            )
            parts.append(
                f'<text x="{label_w + hh * cell + cell / 2 - 1}" y="{y + cell / 2 + 4}" text-anchor="middle" '
                f'font-family="monospace" font-size="9" fill="{_text_color(v, vmax)}">{_fmt(v, 1)}</text>'
            )
    parts.append(
        f'<text x="{label_w}" y="{h - 52}" font-family="monospace" font-size="12" fill="{INK}">'
        f"rows: layer L · columns: head H · values: Rec_D = (D_patched - D_corr)/(D_clean - D_corr)</text>"
    )
    parts.append(
        f'<text x="{label_w}" y="{h - 32}" font-family="monospace" font-size="11" fill="{MID}">'
        f"grayscale: darker = |Rec_D| larger (max |value| {_fmt(vmax)}) · n={res['params']['n_prompts']}, "
        f"seed {res['params']['seed']} · exclusion rule: |D_clean - D_corr| &lt; 0.5 per (prompt, corruption)</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def fig_rq1_bars(res: dict) -> str:
    """Figure 2: RQ1 agreement — mean Spearman with bootstrap CI and the null band."""
    rq1 = res["summary"]["rq1"]["all_prompts"]
    rows = []
    for method in ("atp", "dla"):
        for group in ("heads", "mlps", "all"):
            g = rq1["methods"][method][group]["signed"]
            rows.append((f"{method.upper()} · {group}", g["mean"], g["ci_low"], g["ci_high"], rq1["null"][group]))
    bar_h, gap, left, plot_w = 34, 26, 240, 520
    w = left + plot_w + 120
    h = 90 + len(rows) * (bar_h + gap)
    lim = 1.0
    def x_of(v):
        return left + plot_w * (1 + v / lim) / 2  # -1..1 -> left..left+plot_w
    zero_x = x_of(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" role="img" '
        f'aria-label="RQ1 agreement bars with bootstrap CI and null band">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="16" y="26" font-family="monospace" font-size="14" fill="{INK}">'
        f"RQ1: per-prompt Spearman(Rec_D[swap], method) — mean with bootstrap 95% CI over prompts</text>",
        f'<text x="16" y="46" font-family="monospace" font-size="11" fill="{MID}">'
        f"denoising, metric D · n={res['params']['n_prompts']} prompts, seed {res['params']['seed']} · "
        f"shaded band: null 2.5-97.5 percentiles (label permutation)</text>",
    ]
    # axis ticks
    for tv in (-1.0, -0.5, 0.0, 0.5, 1.0):
        parts.append(
            f'<text x="{x_of(tv)}" y="{70}" text-anchor="middle" font-family="monospace" font-size="11" fill="{MID}">{tv}</text>'
        )
        parts.append(f'<line x1="{x_of(tv)}" y1="76" x2="{x_of(tv)}" y2="{h - 46}" stroke="#ddd" stroke-width="1"/>')
    y = 90
    for label, mean, lo, hi, null in rows:
        band_x1, band_x2 = x_of(null["p2_5"]), x_of(null["p97_5"])
        parts.append(
            f'<rect x="{band_x1}" y="{y}" width="{band_x2 - band_x1}" height="{bar_h}" fill="{LIGHT}" stroke="#ccc" stroke-width="0.5"/>'
        )
        parts.append(f'<line x1="{zero_x}" y1="{y}" x2="{zero_x}" y2="{y + bar_h}" stroke="#888" stroke-width="1"/>')
        # bar from 0 to mean
        bx1, bx2 = min(zero_x, x_of(mean)), max(zero_x, x_of(mean))
        parts.append(f'<rect x="{bx1}" y="{y + 8}" width="{bx2 - bx1}" height="{bar_h - 16}" fill="{INK}"/>')
        # CI whisker
        parts.append(
            f'<line x1="{x_of(lo)}" y1="{y + bar_h / 2}" x2="{x_of(hi)}" y2="{y + bar_h / 2}" stroke="{INK}" stroke-width="2"/>'
        )
        for v in (lo, hi):
            parts.append(
                f'<line x1="{x_of(v)}" y1="{y + bar_h / 2 - 5}" x2="{x_of(v)}" y2="{y + bar_h / 2 + 5}" stroke="{INK}" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{left - 10}" y="{y + bar_h / 2 + 4}" text-anchor="end" font-family="monospace" font-size="12" fill="{INK}">{label}</text>'
        )
        parts.append(
            f'<text x="{left + plot_w + 12}" y="{y + bar_h / 2 + 4}" font-family="monospace" font-size="11" fill="{INK}">'
            f"{_fmt(mean, 3)} [{_fmt(lo, 2)}, {_fmt(hi, 2)}]</text>"
        )
        y += bar_h + gap
    parts.append("</svg>")
    return "\n".join(parts)


def fig_topk_overlap(res: dict, k: int) -> str:
    """Figure 3: top-k overlap matrix across conditions."""
    rq2 = res["summary"]["rq2"]
    data = rq2["topk"][f"k{k}"]
    names = rq2["condition_names"]
    n = len(names)
    cell = 52
    label_w = 120
    w = label_w + n * cell + 30
    h = 70 + n * cell + 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" role="img" '
        f'aria-label="Top-{k} head overlap matrix across conditions">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="16" y="26" font-family="monospace" font-size="14" fill="{INK}">'
        f"Top-{k} head overlap |A ∩ B| between conditions (144 heads, signed scores)</text>",
        f'<text x="16" y="46" font-family="monospace" font-size="11" fill="{MID}">'
        f"denoising + ablation conditions · random expectation k·k/144 = "
        f"{rq2['topk']['expected_random'][f'k{k}']:.2f} · n={res['params']['n_prompts']}, seed {res['params']['seed']}</text>",
    ]
    for j, name in enumerate(names):
        parts.append(
            f'<text x="{label_w + j * cell + cell / 2}" y="{64}" text-anchor="middle" font-family="monospace" '
            f'font-size="10" fill="{INK}" transform="rotate(45 {label_w + j * cell + cell / 2} 64)">{name}</text>'
        )
    for i, name in enumerate(names):
        y = 70 + i * cell
        parts.append(
            f'<text x="{label_w - 8}" y="{y + cell / 2 + 4}" text-anchor="end" font-family="monospace" font-size="11" fill="{INK}">{name}</text>'
        )
        for j in range(n):
            v = data["matrix"][i][j]
            parts.append(
                f'<rect x="{label_w + j * cell}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                f'fill="{_shade(v, k)}" stroke="#bbb" stroke-width="0.5"/>'
            )
            parts.append(
                f'<text x="{label_w + j * cell + cell / 2 - 1}" y="{y + cell / 2 + 4}" text-anchor="middle" '
                f'font-family="monospace" font-size="11" fill="{_text_color(v, k)}">{_fmt(v, 0)}</text>'
            )
    parts.append(
        f'<text x="16" y="{h - 24}" font-family="monospace" font-size="11" fill="{MID}">'
        f"grayscale: darker = larger overlap · conditions: 3 corruptions x 2 metrics + zero/mean ablation</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def fig_group_ranks(res: dict) -> str:
    """Figure 4: mean rank of known Wang et al. 2022 head groups (descriptive)."""
    anchors = res["summary"]["anchors"]
    by_group = anchors.get("mean_ranks_by_group") or {}
    groups = [(name, g["mean_rank"], len(g["heads"])) for name, g in by_group.items() if g.get("mean_rank") is not None]
    groups.sort(key=lambda t: t[1])
    bar_h, gap, left, plot_w = 34, 24, 240, 420
    w = left + plot_w + 120
    h = 90 + max(1, len(groups)) * (bar_h + gap)
    max_rank = max((r for _, r, _ in groups), default=144)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" role="img" '
        f'aria-label="Mean rank of known head groups by Rec_D swap">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="16" y="26" font-family="monospace" font-size="14" fill="{INK}">'
        f"Known-heads anchor: mean rank by prompt-averaged Rec_D[swap] (rank 1 = highest)</text>",
        f'<text x="16" y="46" font-family="monospace" font-size="11" fill="{MID}">'
        f"descriptive only · head lists transcribed from memory (Wang et al. 2022), reviewer must verify · "
        f"n={res['params']['n_prompts']}, seed {res['params']['seed']}</text>",
    ]
    y = 90
    for name, rank, n_heads in groups:
        bw = plot_w * rank / max_rank
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{bw}" height="{bar_h - 12}" fill="{INK}"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + bar_h / 2 + 4}" text-anchor="end" font-family="monospace" font-size="12" fill="{INK}">{name}</text>'
        )
        parts.append(
            f'<text x="{left + plot_w + 12}" y="{y + bar_h / 2 + 4}" font-family="monospace" font-size="11" fill="{INK}">'
            f"{rank:.1f} ({n_heads} heads)</text>"
        )
        y += bar_h + gap
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", dest="job_id", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--base-url", dest="base_url", default=BASE_URL)
    args = parser.parse_args()

    if args.json_path:
        with open(args.json_path, encoding="utf-8") as f:
            res = json.load(f)
    else:
        if not args.job_id:
            raise SystemExit("provide --job-id or --json")
        with urllib.request.urlopen(f"{args.base_url}/api/experiments/{args.job_id}", timeout=600) as resp:
            job = json.load(resp)
        if job.get("status") != "complete" or not job.get("results"):
            raise SystemExit(f"job {args.job_id} is not complete")
        res = job["results"]
    if res.get("job_type") != "study":
        raise SystemExit("not a study result")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "params": res["params"],
        "model_accuracy": res["model_accuracy"],
        "exclusions": res["exclusions"],
        "exclusion_rule": res["exclusion_rule"],
        "elapsed_seconds": res["elapsed_seconds"],
        "prompt_set_counts": {
            "template_counts": res["prompt_set"]["template_counts"],
            "s2_role_counts": res["prompt_set"]["s2_role_counts"],
            "rejected_draws": res["prompt_set"]["rejected_draws"],
        },
        "summary": res["summary"],
    }
    (OUT_DIR / "stage3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures = {
        "fig1_head_heatmap.svg": fig_head_heatmap(res),
        "fig2_rq1_agreement.svg": fig_rq1_bars(res),
        "fig3_topk_overlap.svg": fig_topk_overlap(res, 5),
        "fig3b_topk_overlap_k10.svg": fig_topk_overlap(res, 10),
        "fig4_known_group_ranks.svg": fig_group_ranks(res),
    }
    for name, svg in figures.items():
        (OUT_DIR / name).write_text(svg, encoding="utf-8")
        print(f"wrote {OUT_DIR / name}")
    print(f"wrote {OUT_DIR / 'stage3_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
