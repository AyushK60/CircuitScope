import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Health, type VerifyJobStatus, type VerifyStatus } from "../api/client";
import VerificationCard from "./verify/VerificationCard";

/**
 * Verify page: run the repository's independent reference scripts against the app's own numbers.
 * The server supplies every input; the request carries only an allowlisted check id. A pass is
 * evidence that two implementations agree, not that the interpretation is right.
 */
export default function Verify({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [status, setStatus] = useState<VerifyStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeCheck, setActiveCheck] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<VerifyJobStatus | null>(null);
  const [queue, setQueue] = useState<string[]>([]);
  const [queueTotal, setQueueTotal] = useState(0);
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

  const run = useCallback((checkId: string) => {
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
  }, []);

  // "Run all" works through a queue: the next check starts when the previous one has finished
  useEffect(() => {
    if (queue.length === 0 || activeCheck !== null || !ready) return;
    const [next, ...rest] = queue;
    setQueue(rest);
    run(next);
  }, [queue, activeCheck, ready, run]);

  const cancel = () => {
    setQueue([]);
    if (jobId) api.cancelExperiment(jobId).catch(() => {});
  };

  const running = job?.status === "running" || job?.status === "queued";
  const checks = status?.checks ?? [];
  const runnable = checks.filter((c) => !c.manual);
  const fresh = (c: (typeof checks)[number]) => c.status === "pass" && !c.stale && c.ok === true;
  const nPass = runnable.filter(fresh).length;
  const nStale = runnable.filter((c) => c.stale).length;
  const nFail = runnable.filter((c) => c.status === "fail" && !c.stale).length;
  const nNever = runnable.filter((c) => c.status === "never run").length;
  const nManual = checks.length - runnable.length;
  const total = Math.max(1, runnable.length);
  const seg = (n: number) => (n / total) * 100;
  const ringStyle = {
    background: `conic-gradient(var(--ok) 0 ${seg(nPass)}%, var(--brass) ${seg(nPass)}% ${seg(nPass + nStale)}%, var(--danger) ${seg(nPass + nStale)}% ${seg(nPass + nStale + nFail)}%, var(--line-0) ${seg(nPass + nStale + nFail)}% 100%)`,
  };

  const todo = runnable.filter((c) => !fresh(c)).map((c) => c.check_id);
  const queueActive = queue.length > 0 || (queueTotal > 0 && activeCheck !== null);
  const runAll = () => {
    setQueueTotal(todo.length);
    setQueue(todo);
  };
  const doneInQueue = queueTotal - queue.length - (activeCheck ? 1 : 0);

  return (
    <main className="page vf-page">
      <header className="vf-header">
        <div>
          <p className="ex-kick">Do the numbers hold up?</p>
          <h1 className="ex-title">Verify</h1>
          <p className="ex-sub">
            Each check runs an independent reference script from <code className="mono">scripts/</code> on the same
            inputs the app used, then compares its numbers with the app&apos;s. The server computes the inputs itself;
            the request only names an allowlisted check.
          </p>
        </div>
        {status && runnable.length > 0 && (
          <div className="vf-score">
            <div className="vf-ring" style={ringStyle} role="img" aria-label={`${nPass} of ${runnable.length} checks verified`}>
              <div>
                {nPass}/{runnable.length}
              </div>
            </div>
            <p>
              verified
              {status.current_git_commit && (
                <>
                  {" at "}
                  <code className="mono">{status.current_git_commit}</code>
                </>
              )}
              <br />
              {nStale} stale · {nNever} never run{nFail > 0 ? ` · ${nFail} failing` : ""}
              {nManual > 0 ? ` · ${nManual} manual` : ""}
            </p>
          </div>
        )}
      </header>

      <div className="vf-note" role="note">
        <strong>What a pass means:</strong> two independent implementations agree on these numbers within the stated
        tolerance. It does not validate the scientific interpretation, the choice of prompt, or whether a component is
        part of a circuit.
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
          Current commit <code className="mono">{status.current_git_commit}</code>. A check reads &quot;stale&quot; when
          its reference script changed after the stored result was recorded.
        </p>
      )}

      {!status && !error && (
        <div className="state-banner info" role="status">
          <span className="spinner" aria-hidden /> Loading verification status…
        </div>
      )}

      {status && runnable.length > 0 && (
        <div className="vf-bar">
          <button className="primary" onClick={runAll} disabled={!ready || running || queueActive || todo.length === 0}>
            Run all that can run
          </button>
          <span>
            {queueActive
              ? `Running ${Math.min(queueTotal, doneInQueue + 1)} of ${queueTotal}, one after another…`
              : todo.length === 0
                ? "Every check passes at this commit."
                : `${todo.length} to run. Checks go one after another; the slow ones take a few minutes each.`}
          </span>
        </div>
      )}

      <div className="vf-ledger">
        {checks.map((entry) => (
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
