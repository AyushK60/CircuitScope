// Evidence path resolver for claim provenance (Stage 4 Task 6).
// Paths look like "sweep.recovery[9][6]" or "lens.max_jump.D_before" and are
// resolved against the section's compact `data` object. Fixed statements use
// the documented "fixed:..." / "procedure:..." conventions and are shown as
// notes, never as data.

export interface EvidenceItem {
  path: string;
  kind: "data" | "note";
  value?: string;
}

const IDX_RE = /\[([^\]]+)\]|[^.[]+/g;

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "n/a";
    return String(Math.abs(v) >= 1e6 ? v.toExponential(3) : Number(v.toFixed(6)));
  }
  if (typeof v === "string") return v;
  if (Array.isArray(v) && v.length <= 6) return `[${v.map(formatValue).join(", ")}]`;
  return "…";
}

export function resolveEvidence(data: Record<string, unknown> | undefined, path: string): EvidenceItem {
  if (path.startsWith("fixed:") || path.startsWith("procedure:")) {
    return { path, kind: "note" };
  }
  try {
    const key = path.slice(0, path.indexOf("."));
    let node: unknown = data?.[key];
    let rest = path.slice(path.indexOf(".") + 1);
    if (rest === "") {
      // path is a top-level key
      node = data?.[key];
      rest = "";
    }
    for (const m of rest.matchAll(IDX_RE)) {
      const tok = m[0];
      if (tok.startsWith("[")) {
        node = (node as unknown[])[Number(tok.slice(1, -1))];
      } else if (tok !== "") {
        node = (node as Record<string, unknown>)[tok];
      }
    }
    if (node === undefined) return { path, kind: "data", value: undefined };
    return { path, kind: "data", value: formatValue(node) };
  } catch {
    return { path, kind: "data", value: undefined };
  }
}
