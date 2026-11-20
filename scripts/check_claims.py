"""Stage 6 B2: check the claims ledger against the stored results.

Loads docs/paper/claims.json, resolves each claim's evidence — a JSON path
into a results/*.json file, or a generated-table row in docs/paper/tables.md —
and fails (exit 1) if any value differs from expected_value by more than
tolerance. Standard library only (the vector checks use pure-Python exact
summation, not numpy).

Evidence modes
  scalar (default)        resolve evidence.path in the JSON file; numbers are
                          compared with |value - expected| <= tolerance;
                          booleans must match exactly.
  check: top_k_names      resolve path to a vector; compute the top-k
                          component names (fixed Stage 3 order: heads l.h
                          first, then mlp.l) and require exact equality with
                          the expected list.
  check: vector_mean_vs_rows  resolve path to a stored prompt-averaged vector
                          and recompute its mean from the stored per-prompt
                          rows (exact summation); max |stored - recomputed|
                          must be <= tolerance.
  table row (file .md)    locate the named section in the generated Markdown,
                          require the named row line to contain expected_value
                          and the section to carry its EXPLORATORY label.

Usage:
  python scripts/check_claims.py                     # check only
  python scripts/check_claims.py --write-ledger      # check + regenerate docs/paper/CLAIMS_LEDGER.md
  python scripts/check_claims.py --claims FILE --repo-root DIR   # for tests

Exit codes: 0 all claims pass; 1 one or more claims fail or evidence is
unresolvable. The checker never edits results files.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

LEDGER_PATH = REPO_ROOT / "docs" / "paper" / "CLAIMS_LEDGER.md"


def load_json(path: pathlib.Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(doc, dotted: str):
    """Resolve 'summary.rq1.spearman.matrix.0.2' — numeric segments index lists."""
    cur = doc
    for seg in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            if seg not in cur:
                raise KeyError(f"path segment {seg!r} not found")
            cur = cur[seg]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {seg!r}")
    return cur


def component_label(idx: int) -> str:
    """Fixed Stage 3 component order: 0..143 heads (l*12+h), 144..155 MLPs."""
    if idx < 144:
        return f"{idx // 12}.{idx % 12}"
    return f"mlp.{idx - 144}"


def top_k_names(vec, k: int) -> list[str]:
    order = sorted(range(len(vec)), key=lambda i: (-float(vec[i]), i))[:k]
    return [component_label(i) for i in order]


def check_scalar(claim: dict, value) -> tuple[bool, str]:
    expected = claim["expected_value"]
    tol = claim.get("tolerance", 0)
    if isinstance(expected, bool) or isinstance(value, bool):
        ok = value is expected or value == expected and isinstance(expected, bool) == isinstance(value, bool)
        detail = f"stored {value!r} vs expected {expected!r} (boolean, exact)"
        return ok, detail
    if not isinstance(value, (int, float)):
        return False, f"stored value is {type(value).__name__}, expected a number"
    diff = abs(float(value) - float(expected))
    ok = diff <= tol
    detail = f"stored {float(value):.10g} vs expected {float(expected):.10g} | diff {diff:.3g} <= tol {tol:.3g}"
    return ok, detail


def check_top_k_names(claim: dict, vec) -> tuple[bool, str]:
    k = int(claim["evidence"].get("top_k", 10))
    names = top_k_names(vec, k)
    expected = claim["expected_value"]
    ok = names == list(expected)
    detail = f"stored top-{k} = {', '.join(names)} vs expected {', '.join(expected)}"
    return ok, detail


def check_vector_mean_vs_rows(claim: dict, doc) -> tuple[bool, str]:
    ev = claim["evidence"]
    stored = resolve_path(doc, ev["path"])
    rows = resolve_path(doc, ev["rows_path"])
    n_comp = int(ev["n_components"])
    n = len(rows)
    row_vecs = []
    for row in rows:
        vec = resolve_path(row, ev["row_vector_path"])[:n_comp]
        if any(v is None for v in vec):
            return False, f"a per-prompt vector contains None — invariant requires complete rows ({row.get('id')})"
        row_vecs.append([float(v) for v in vec])
    # exact mean per component over all rows (math.fsum: order-independent of
    # accumulation error at this scale)
    max_diff = 0.0
    for i in range(n_comp):
        mean_i = math.fsum(vec[i] for vec in row_vecs) / n
        max_diff = max(max_diff, abs(float(stored[i]) - mean_i))
    tol = float(claim.get("tolerance", 0))
    ok = max_diff <= tol
    detail = f"max |stored - recomputed mean| over {n_comp} components = {max_diff:.3g} <= tol {tol:.3g} ({n} rows)"
    return ok, detail


def check_table_row(claim: dict, tables_text: str) -> tuple[bool, str]:
    ev = claim["evidence"]
    section_name = ev["table"]
    row_name = ev["row"]
    expected = str(claim["expected_value"])
    lines = tables_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and section_name.lower() in line.lower():
            start = i
            break
    if start is None:
        return False, f"section {section_name!r} not found in tables.md"
    section_lines = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        section_lines.append(line)
    section = "\n".join(section_lines)
    row_line = next((ln for ln in section_lines if ln.startswith(f"| {row_name} |")), None)
    if row_line is None:
        return False, f"row {row_name!r} not found in section {section_name!r}"
    if expected not in row_line:
        return False, f"row {row_name!r} does not contain {expected!r}: {row_line}"
    if "EXPLORATORY" not in section.upper():
        return False, "the exploratory table is missing its EXPLORATORY label"
    return True, f"tables.md [{section_name} / {row_name}] contains {expected!r} and the EXPLORATORY label"


def check_claim(claim: dict, files: dict, tables_text: str | None) -> tuple[bool, str]:
    ev = claim["evidence"]
    fname = ev["file"]
    if fname.endswith(".md"):
        if tables_text is None:
            return False, f"tables file not found for {fname}"
        return check_table_row(claim, tables_text)
    if fname not in files:
        return False, f"results file not found: {fname}"
    doc = files[fname]
    mode = ev.get("check")
    try:
        if mode == "top_k_names":
            return check_top_k_names(claim, resolve_path(doc, ev["path"]))
        if mode == "vector_mean_vs_rows":
            return check_vector_mean_vs_rows(claim, doc)
        return check_scalar(claim, resolve_path(doc, ev["path"]))
    except KeyError as e:
        return False, f"evidence path unresolvable: {e}"
    except (IndexError, ValueError, TypeError) as e:
        return False, f"evidence unusable: {e}"


def check_all(claims_path: pathlib.Path, repo_root: pathlib.Path) -> tuple[list[dict], bool]:
    spec = load_json(claims_path)
    claims = spec["claims"]
    files: dict[str, object] = {}
    tables_text = None
    results = []
    all_ok = True
    for claim in claims:
        fname = claim["evidence"]["file"]
        if fname.endswith(".md"):
            path = repo_root / fname
            if tables_text is None:
                tables_text = path.read_text(encoding="utf-8") if path.exists() else None
        elif fname not in files:
            path = repo_root / fname
            files[fname] = load_json(path) if path.exists() else None
        ok, detail = check_claim(claim, files, tables_text)
        all_ok = all_ok and ok
        results.append({"claim": claim, "ok": ok, "detail": detail})
    return results, all_ok


def write_ledger(results: list[dict], claims_path: pathlib.Path) -> str:
    lines = [
        "# Claims ledger (generated — do not edit by hand)",
        "",
        f"Generated by `scripts/check_claims.py --write-ledger` from `{claims_path.name}` and the",
        "stored results files. Every claim's evidence is re-resolved at check time; the ledger is",
        "a human-readable snapshot of that check. `python scripts/check_claims.py` re-runs it",
        "(exit non-zero on any failure); CI runs it in the `backend-fast` job.",
        "",
        "Status meanings: **pre-registered** = analysis rule committed before either study ran",
        "(`docs/PAPER_NOTES.md` §4b); **exploratory** = reviewer-added, labelled as such;",
        "**descriptive** = a measurement or data-integrity fact, not a hypothesis test.",
        "",
    ]
    for r in results:
        c = r["claim"]
        ev = c["evidence"]
        ev_desc = ev["file"]
        if "path" in ev:
            ev_desc += f" :: {ev['path']}"
        elif "table" in ev:
            ev_desc += f" [{ev['table']} / {ev['row']}]"
        lines += [
            f"## {c['id']} — {r['ok'] and 'PASS' or 'FAIL'} — {c['status']}",
            "",
            c["statement"],
            "",
            f"- Evidence: `{ev_desc}`",
            f"- Check: {r['detail']}",
            f"- Expected: `{json.dumps(c['expected_value'])}` ± {c.get('tolerance', 0)}",
            f"- Caveat: {c['caveat']}",
            f"- Would be falsified by: {c['would_be_falsified_by']}",
            "",
        ]
    text = "\n".join(lines)
    with open(LEDGER_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", default=str(REPO_ROOT / "docs" / "paper" / "claims.json"))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--write-ledger", action="store_true", help="regenerate docs/paper/CLAIMS_LEDGER.md")
    args = parser.parse_args()

    claims_path = pathlib.Path(args.claims)
    repo_root = pathlib.Path(args.repo_root)
    results, all_ok = check_all(claims_path, repo_root)

    width = max(len(r["claim"]["id"]) for r in results)
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"{r['claim']['id']:<{width}}  {mark}  ({r['claim']['status']})  {r['detail']}")
    n_pass = sum(1 for r in results if r["ok"])
    print(f"{n_pass}/{len(results)} claims pass against the stored results.")

    if args.write_ledger:
        write_ledger(results, claims_path)
        print(f"wrote {LEDGER_PATH}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
