"""Stage 6 B5: print the numbers to speak in a walkthrough/demo video.

Reads ONE stored report from the owner's local database (read-only, the job
id is a SQL parameter, the DB is opened mode=ro) and prints every number the
2-minute walkthrough speaks, in speaking order. NEVER type these numbers
from memory or from another doc: run this script and paste its output.

Usage:
  python scripts\\walkthrough_numbers.py <report_job_id> [--db PATH]

Defaults: --db outputs\\experiments.db (git-ignored, owner's data; the
script only reads). Exit codes: 0 ok; 1 usage/not-found/not-a-report;
2 report not complete.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_DB = REPO_ROOT / "outputs" / "experiments.db"


def pct(v) -> str:
    return f"{100 * float(v):.1f}%"


def num(v, d: int) -> str:
    return f"{float(v):.{d}f}"


def signed(v, d: int) -> str:
    f = float(v)
    return f"+{f:.{d}f}" if f >= 0 else f"{f:.{d}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_job_id")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    db_path = pathlib.Path(args.db)
    if not db_path.exists():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 1
    # mode=ro: this tool can never create or write the owner's database.
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT status, results, config FROM experiments WHERE id = ?",
            (args.report_job_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        print(f"no experiment with id {args.report_job_id!r} in {db_path}", file=sys.stderr)
        return 1
    status, results_text, config_text = row
    if status != "complete" or results_text is None:
        print(f"job {args.report_job_id} is not complete (status: {status})", file=sys.stderr)
        return 2
    res = json.loads(results_text)
    cfg = json.loads(config_text)
    if res.get("job_type") != "report":
        print(f"job {args.report_job_id} is a {res.get('job_type')!r} run, not a report", file=sys.stderr)
        return 1

    prov = res.get("provenance", {})
    study_id = res.get("study_job_id")
    print(f"Report {args.report_job_id} (stored {prov.get('timestamp_utc', '?')}, "
          f"git_commit {prov.get('git_commit', '?')}, model {prov.get('model_id', '?')} "
          f"revision {prov.get('model_revision', '?')})")
    print(f"task {res.get('task_id')}, pair {res.get('pair_id')}"
          + (f", over study {study_id}" if study_id else ", single pair (no study)"))
    print(f"open in the app as:  #/report?job={args.report_job_id}")
    print()
    print("NOTE: values below are verbatim from the stored report. Percentages/D/recoveries are")
    print("formatted here; the stored raw values are what the app displays.")

    sections = {s["id"]: s.get("data", {}) for s in res.get("sections", [])}

    pred = sections.get("answer", {}).get("prediction", {})
    if pred:
        print()
        print("PREDICTION (first thing to say)")
        print(f"  top next token          : \"{pred.get('top_token')}\"  ({pct(pred.get('top_prob'))})")
        print(f"  candidate A \"{pred.get('candidate_a')}\" : {pct(pred.get('prob_a'))}")
        print(f"  candidate B \"{pred.get('candidate_b')}\"   : {pct(pred.get('prob_b'))}")
        print(f"  D = logit(A) - logit(B) : {num(pred.get('D'), 5)}  (positive = A above B)")

    lens = sections.get("where_formed", {}).get("lens", {})
    if lens:
        jump = lens.get("max_jump", {})
        fc = lens.get("final_check", {})
        print()
        print("WHERE THE BELIEF FORMS (logit lens — heuristic view, not causal evidence)")
        if jump:
            print(f"  biggest stage jump      : {jump.get('stage')}  "
                  f"(stage D {num(jump.get('D_before'), 2)} -> {num(jump.get('D_after'), 2)})")
        if fc:
            print(f"  final-stage D           : {num(lens.get('final_d'), 4)} "
                  f"(reproduces the real logits within {num(fc.get('max_abs_logit_diff_candidates'), 8)}, tol {fc.get('tolerance')})")

    words = sections.get("which_words", {})
    if words:
        print()
        print("WHERE PATCHING MOVES THE ANSWER (layer x position sweep, denoising, metric D)")
        early, late = words.get("early", {}), words.get("late", {})
        if early:
            print(f"  early-block best        : layer {early.get('layer')}, position {early.get('position')} "
                  f"(\"{early.get('token')}\"), recovery {num(early.get('recovery'), 2)}")
        if late:
            print(f"  late-block best         : layer {late.get('layer')}, position {late.get('position')} "
                  f"(\"{late.get('token')}\"), recovery {num(late.get('recovery'), 2)}")
        print("  (last block x final position = identity cell: recovery 1 by construction, excluded)")

    parts = sections.get("which_parts", {})
    if parts:
        print()
        print("WHICH HEADS MATTER (layer x head sweep + DLA on the clean prompt)")
        recs = parts.get("top_recovery", [])
        if recs:
            print("  top by patch recovery   : "
                  + ", ".join(f"{r['label']} ({num(r['recovery'], 3)})" for r in recs))
        dlas = parts.get("top_dla", [])
        if dlas:
            print("  top by DLA              : "
                  + ", ".join(f"{r['label']} ({signed(r['value'], 2)})" for r in dlas))

    study = sections.get("how_sure", {}).get("study", {})
    if study:
        print()
        print("STUDY NUMBERS (stored study section of the report)")
        params = study.get("params", {})
        print(f"  n = {params.get('n_prompts')}, seed {params.get('seed')}; accuracy {study.get('model_accuracy')}; "
              f"exclusions swap/third/resample "
              f"{study.get('exclusions', {}).get('swap')}/{study.get('exclusions', {}).get('third')}/"
              f"{study.get('exclusions', {}).get('resample')}")
        atp = study.get("rq1_atp_heads", {})
        dla = study.get("rq1_dla_heads", {})
        beats = study.get("beats_control", {})
        if atp:
            print(f"  AtP vs exact (heads)    : mean Spearman {num(atp.get('mean'), 3)} "
                  f"[{num(atp.get('ci_low'), 3)}, {num(atp.get('ci_high'), 3)}], "
                  f"beats null control: {'YES' if beats.get('atp', {}).get('heads') else 'NO'}")
        if dla:
            print(f"  DLA vs exact (heads)    : mean Spearman {num(dla.get('mean'), 3)} "
                  f"[{num(dla.get('ci_low'), 3)}, {num(dla.get('ci_high'), 3)}], "
                  f"beats null control: {'YES' if beats.get('dla', {}).get('heads') else 'NO — the honest null'}")
        overlap = study.get("overlap", {})
        if overlap:
            print(f"  top-10 overlap w/ exact : AtP {overlap.get('atp')}/10, DLA {overlap.get('dla')}/10 "
                  f"(random expectation {study.get('expected_random_k10')})")
        print(f"  exclusion rule          : {study.get('exclusion_rule')}")

    ref = sections.get("vs_paper", {}).get("reference", {})
    if ref:
        print()
        print("ANCHOR (descriptive only — Wang et al. head groups; not a claim)")
        groups = ref.get("groups", [])
        print("  head groups used        : "
              + "; ".join(f"{g['group']} ({', '.join(g['heads'])})" for g in groups))

    print()
    print("Reminder from the walkthrough doc: attention/lens views are correlational/heuristic;")
    print("only patching/ablation results are causal about the measured intervention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
