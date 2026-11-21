import type { ReportSection } from "./Claims";

function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(6)));
  return String(v);
}

/** "Why trust this?" panel. A failed check is shown in red
 * with its number and labelled FAILED — colour is never the only signal. */
export default function TrustPanel({ trust }: { trust: ReportSection["trust"] }) {
  return (
    <details className="trust-panel">
      <summary className="inspector-meta">Why trust this?</summary>
      <div className="trust-body">
        {trust.measured && (
          <p>
            <strong>What was measured:</strong> {trust.measured}
          </p>
        )}
        {trust.against.length > 0 && (
          <div>
            <strong>Against what:</strong>
            <ul>
              {trust.against.map((a, i) => (
                <li key={i}>
                  {a.label} = {fmtVal(a.value)}
                  {a.note ? ` (${a.note})` : ""}
                </li>
              ))}
            </ul>
          </div>
        )}
        {trust.checks.length > 0 && (
          <div>
            <strong>Checks that ran live:</strong>
            <ul>
              {trust.checks.map((c) => (
                <li key={c.id} className={c.ok ? "check-ok" : "check-failed"}>
                  {c.ok ? "ok —" : "FAILED —"} {c.label}: {fmtVal(c.value)}
                  {c.tolerance !== null && c.tolerance !== undefined ? ` (limit ${fmtVal(c.tolerance)})` : ""}
                </li>
              ))}
            </ul>
          </div>
        )}
        {trust.warnings.length > 0 && (
          <div>
            <strong>Warnings:</strong>
            <ul>
              {trust.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        <div>
          <strong>What this does NOT show:</strong>
          <ul>
            {trust.not_shown.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      </div>
    </details>
  );
}
