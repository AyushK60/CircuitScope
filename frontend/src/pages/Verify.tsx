import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Health, type VerifyJobStatus, type VerifyStatus } from "../api/client";
import VerificationCard from "./verify/VerificationCard";

/**
 * Verify page: run the repository's independent reference
 * scripts against the app's own numbers. The server supplies every input; the
 * request carries only an allowlisted check id. A pass is evidence that two
 * implementations agree, not that the interpretation is right.
 */
export default function Verify({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [status, setStatus] = useState<VerifyStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeCheck, setActiveCheck] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<VerifyJobStatus | null>(null);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(() => {
    api
      .verifyStatus()
      .then(setStatus)
      .catch((e) => setError(String((e as Error).message ?? e)));
  }, []);

  useEffect(refresh, [refresh]);

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = window.setInterval(() => {
      api
        .verifyJob(jobId)
        .then((r) => {
          setJob(r);
          if (r.status === "failed") {
            setError(r.error ?? "verification failed");
            setActiveCheck(null);
          }
          if (r.status === "complete" || r.status === "failed" || r.status === "cancelled") {
            setActiveCheck(null);
            refresh();
          }
        })
        .catch((e) => setError(String((e as Error).message ?? e)));
    }, 700);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId, refresh]);

  const run = (checkId: string) => {
    setError(null);
    setJob(null);
    setJobId(null);
    setActiveCheck(checkId);
    api
      .startVerify(checkId)
      .then((r) => setJobId(r.job_id))
      .catch((e) => {
        setActiveCheck(null);
        setError(String((e as Error).message ?? e));
      });
  };

  const cancel = () => {
    if (jobId) api.cancelExperiment(jobId).catch(() => {});
  };

  const running = job?.status === "running" || job?.status === "queued";

  return (
    <main className="page">
      <h1>Verify</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "80ch" }}>
        Each card runs an independent reference script from <code className="mono">scripts/</code> on the
        same inputs the app used, then compares its numbers with the app&apos;s. The server computes the
        inputs itself; the request only names an allowlisted check.
      </p>
      <div className="state-banner info" role="note">
        <strong>What a pass means:</strong> two independent implementations agree on these numbers within
        the stated tolerance. It does not validate the scientific interpretation, the choice of prompt, or
        whether a component is part of a circuit.
      </div>

      {!ready && (
        <div className="state-banner warn" role="status">
          The model is still loading. Reference scripts need the cached GPT-2 weights.
        </div>
      )}
      {error && (
        <div className="state-banner error" role="alert">
          {error}
        </div>
      )}
      {status?.current_git_commit && (
        <p className="inspector-meta">
          Current commit <code className="mono">{status.current_git_commit}</code>. A card reads
          &quot;stale&quot; when its reference script changed after the stored result was recorded.
        </p>
      )}

      {!status && !error && (
        <div className="state-banner info" role="status">
          <span className="spinner" aria-hidden /> Loading verification status…
        </div>
      )}

      <div className="verify-grid">
        {status?.checks.map((entry) => (
          <VerificationCard
            key={entry.check_id}
            entry={entry}
            running={activeCheck === entry.check_id && running}
            progress={job?.progress ?? 0}
            progressMessage={job?.progress_message ?? ""}
            onRun={() => run(entry.check_id)}
            onCancel={cancel}
          />
        ))}
      </div>
    </main>
  );
}
