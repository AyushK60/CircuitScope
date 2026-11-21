import type { VerifyCheckEntry } from "../../api/client";

function statusInfo(entry: VerifyCheckEntry): {
  word: string;
  icon: string;
  className: string;
} {
  if (entry.manual) return { word: "Manual check", icon: "✋", className: "manual" };
  if (entry.status === "never run") return { word: "Never run", icon: "–", className: "never" };
  if (entry.stale) return { word: "Stale result", icon: "↻", className: "stale" };
  if (entry.ok) return { word: "Pass", icon: "✓", className: "pass" };
  return { word: "Fail", icon: "✕", className: "fail" };
}

/**
 * One verification card. Status is stated in words AND an
 * icon, never by colour alone. The last result shows max difference against
 * tolerance, when it ran, and the commit; raw output is collapsed.
 */
export default function VerificationCard({
  entry,
  running,
  progress,
  progressMessage,
  onRun,
  onCancel,
}: {
  entry: VerifyCheckEntry;
  running: boolean;
  progress: number;
  progressMessage: string;
  onRun: () => void;
  onCancel: () => void;
}) {
  const info = statusInfo(entry);
  const result = entry.result;
  return (
    <section className={`panel verify-card ${info.className}`} aria-labelledby={`vc-${entry.check_id}`}>
      <div className="verify-head">
        <h3 id={`vc-${entry.check_id}`}>{entry.title}</h3>
        <span className={`verify-status ${info.className}`} role="status" aria-label={`Status: ${info.word}`}>
          <span aria-hidden className="verify-icon">
            {info.icon}
          </span>{" "}
          {info.word}
        </span>
      </div>
      <p className="inspector-meta">{entry.what}</p>
      <p className="inspector-meta">
        Script <code className="mono">{entry.script}</code> · typical runtime {entry.runtime} · tolerance{" "}
        {entry.tolerance} (absolute)
      </p>

      {result && (
        <div className="verify-result">
          <div className="table-scroll">
            <table className="kv-table">
              <tbody>
                <tr>
                  <td>Max |app − reference|</td>
                  <td className="mono">
                    {result.max_diff == null ? "n/a" : result.max_diff.toExponential(3)}
                  </td>
                </tr>
                <tr>
                  <td>Exit code</td>
                  <td className="mono">{result.exit_code == null ? "n/a" : result.exit_code}</td>
                </tr>
                <tr>
                  <td>When</td>
                  <td>{result.checked_at ?? "unknown"}</td>
                </tr>
                <tr>
                  <td>Verified at commit</td>
                  <td className="mono">{result.git_commit ?? "unknown"}</td>
                </tr>
                <tr>
                  <td>Elapsed</td>
                  <td className="mono">{result.elapsed_seconds.toFixed(1)} s</td>
                </tr>
                {result.error && (
                  <tr>
                    <td>Error</td>
                    <td>{result.error}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <details>
            <summary>Reference script output (last 60 lines)</summary>
            <pre className="verify-stdout">{result.stdout_tail || "(no output captured)"}</pre>
          </details>
        </div>
      )}

      {running ? (
        <div style={{ marginTop: "0.6rem" }}>
          <div className="inspector-meta" role="status" style={{ marginBottom: "0.25rem" }}>
            {Math.round(progress * 100)}% — {progressMessage || "running"}
          </div>
          <div style={{ background: "var(--bg-2)", borderRadius: 4, height: 10, overflow: "hidden" }}>
            <div
              style={{
                width: `${Math.round(progress * 100)}%`,
                height: "100%",
                background: "var(--sage)",
                transition: "width 0.3s",
              }}
            />
          </div>
          <button onClick={onCancel} style={{ marginTop: "0.5rem" }}>
            Cancel
          </button>
        </div>
      ) : (
        <div className="verify-actions">
          {entry.manual ? (
            <span className="inspector-meta">
              This one needs a browser: open the Neural Lab and use the feedforward checker there.
            </span>
          ) : (
            <button className="primary" onClick={onRun}>
              Run this check
            </button>
          )}
        </div>
      )}
    </section>
  );
}
