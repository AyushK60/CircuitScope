import { useState } from "react";
import type { VerifyCheckEntry } from "../../api/client";

function statusInfo(entry: VerifyCheckEntry): { word: string; stamp: string; className: string } {
  if (entry.manual) return { word: "Manual check", stamp: "By hand", className: "manual" };
  if (entry.status === "never run") return { word: "Never run", stamp: "Not run", className: "never" };
  if (entry.stale) return { word: "Stale result", stamp: "Stale", className: "stale" };
  if (entry.ok) return { word: "Pass", stamp: "Pass", className: "pass" };
  return { word: "Fail", stamp: "Fail", className: "fail" };
}

/** How far the largest gap sits inside the limit, in words. */
function marginText(maxDiff: number, tolerance: number): string {
  if (maxDiff > tolerance) return "over the limit";
  if (maxDiff <= 0) return "no gap at all";
  const x = tolerance / maxDiff;
  return `${x >= 10 ? Math.round(x) : x.toFixed(1)}× inside the limit`;
}

/**
 * One verification row. Status is stated in words (the stamp), never by colour alone. The row shows
 * the largest gap against the limit on a gauge; the table, commit and raw output open underneath, and
 * the row opens by itself when a run finishes.
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
  const [open, setOpen] = useState(false);
  const [wasRunning, setWasRunning] = useState(running);
  if (running !== wasRunning) {
    setWasRunning(running);
    if (wasRunning && !running) setOpen(true);
  }

  const diff = result?.max_diff ?? null;
  const over = diff !== null && diff > entry.tolerance;
  const fill = diff === null ? 0 : Math.min(1, diff / (entry.tolerance * 1.25));

  return (
    <section className={`vf-row verify-card ${info.className}`} aria-labelledby={`vc-${entry.check_id}`}>
      <details open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary>
          <span className={`vf-stamp ${info.className}`} role="status" aria-label={`Status: ${info.word}`}>
            {info.stamp}
          </span>
          <span className="vf-title">
            <h3 id={`vc-${entry.check_id}`}>{entry.title}</h3>
            <span className="vf-what">{entry.what}</span>
          </span>
          <span className="vf-gauge-col">
            {running ? (
              <span className="vf-run">
                <span role="status">
                  {Math.round(progress * 100)}% — {progressMessage || "running"}
                </span>
                <span className="vf-gauge">
                  <i style={{ width: `${Math.round(progress * 100)}%` }} />
                </span>
              </span>
            ) : diff !== null ? (
              <>
                <span className={`vf-big${over ? " over" : ""}${entry.stale ? " stale" : ""}`}>
                  {diff.toExponential(1)} <small>{marginText(diff, entry.tolerance)}</small>
                </span>
                <span className="vf-gauge" aria-hidden>
                  <i className={over ? "over" : entry.stale ? "stale" : ""} style={{ width: `${fill * 100}%` }} />
                  <u />
                </span>
                <span className="vf-gl">
                  <span>largest gap</span>
                  <span>limit {entry.tolerance}</span>
                </span>
              </>
            ) : (
              <>
                <span className="vf-gauge none" aria-hidden />
                <span className="vf-gl">
                  <span>{entry.manual ? "checked by hand" : "no result yet"}</span>
                  <span>{entry.manual ? "" : `limit ${entry.tolerance}`}</span>
                </span>
              </>
            )}
          </span>
          <span className="vf-actions">
            {running ? (
              <button onClick={onCancel}>Cancel</button>
            ) : entry.manual ? null : (
              <button
                className={entry.status === "never run" || entry.stale ? "primary" : undefined}
                onClick={onRun}
                aria-label={result ? "Run this check again" : "Run this check"}
              >
                {result ? "Run again" : "Run"}
              </button>
            )}
          </span>
        </summary>

        <div className="vf-body">
          <p className="inspector-meta">
            Script <code className="mono">{entry.script}</code> · typical runtime {entry.runtime} · tolerance{" "}
            {entry.tolerance} (absolute)
          </p>
          {entry.stale && (
            <p className="vf-stale">
              The reference script changed after this result was recorded, so run it again before relying on it.
            </p>
          )}
          {entry.manual && (
            <p className="inspector-meta">
              This one needs a browser: open the <a href="#/lab">Neural Lab</a> and use the feedforward checker there.
            </p>
          )}
          {result && (
            <div className="verify-result">
              <div className="table-scroll">
                <table className="kv-table">
                  <tbody>
                    <tr>
                      <td>Max |app − reference|</td>
                      <td className="mono">{result.max_diff == null ? "n/a" : result.max_diff.toExponential(3)}</td>
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
        </div>
      </details>
    </section>
  );
}
