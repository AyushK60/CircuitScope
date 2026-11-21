import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type Health,
  type ReportJobStatus,
  type ReportResult,
  type ReportSection,
  type VerifyStatus,
} from "../api/client";
import { ClaimText } from "./report/Claims";
import LensSparkline from "./report/LensSparkline";
import ReproduceButton from "../components/ReproduceButton";
import {
  AUTO_ADVANCE_MS,
  TOUR_SCENES,
  isLastScene,
  nextScene,
  pickSavedReport,
  prevScene,
  progressPercent,
  shouldAutoAdvance,
  type SavedReportCandidate,
  type TourScene,
} from "./tour/scenes";

/**
 * Guided tour: six scenes over one real Circuit Report.
 * Prefers a saved report whose provenance commit matches the current commit;
 * otherwise it generates a fresh one and says so. Reduced motion turns off
 * auto-advance entirely.
 */

function sectionOf(report: ReportResult, id: string): ReportSection | undefined {
  return report.sections.find((s) => s.id === id);
}

function BarPair({ section }: { section: ReportSection }) {
  const data = section.data as Record<string, unknown>;
  const pred = (data.prediction ?? {}) as Record<string, unknown>;
  const rows = [
    { label: String(pred.candidate_a ?? "A"), value: Number(pred.prob_a ?? NaN) },
    { label: String(pred.candidate_b ?? "B"), value: Number(pred.prob_b ?? NaN) },
  ];
  const max = Math.max(...rows.map((r) => (Number.isFinite(r.value) ? r.value : 0)), 1e-9);
  return (
    <div className="prob-bars">
      {rows.map((r) => (
        <div className="bar-row" key={r.label}>
          <span className="label">{r.label}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{ left: 0, width: `${Math.max(0, ((Number.isFinite(r.value) ? r.value : 0) / max) * 100)}%` }}
            />
          </span>
          <span className="value">
            {Number.isFinite(r.value) ? `${(r.value * 100).toFixed(1)}%` : "n/a"}
          </span>
        </div>
      ))}
    </div>
  );
}

function WordsVisual({ section }: { section: ReportSection }) {
  const data = section.data as Record<string, unknown>;
  const sweep = (data.sweep ?? {}) as { recovery?: number[][]; tokens?: string[] };
  if (!sweep.recovery || sweep.recovery.length === 0) {
    return <p className="inspector-meta">No position sweep in this report.</p>;
  }
  const nCols = Math.max(...sweep.recovery.map((r) => r.length));
  const best = Array.from({ length: nCols }, (_, c) =>
    Math.max(...sweep.recovery!.map((row) => (Number.isFinite(row[c]) ? row[c] : -Infinity))),
  );
  const maxAbs = Math.max(...best.map((v) => Math.abs(v)), 1e-9);
  return (
    <div className="tour-words" role="img" aria-label="Best restoration per token position from the position sweep">
      {best.map((v, i) => (
        <div className="tour-word" key={i}>
          <span className="tour-word-token">{sweep.tokens?.[i] ?? `#${i}`}</span>
          <span className="tour-word-track">
            <span
              className="tour-word-fill"
              style={{ width: `${Math.max(0, Math.abs(v) / maxAbs) * 100}%` }}
            />
          </span>
          <span className="mono tour-word-value">{Number.isFinite(v) ? v.toFixed(2) : "—"}</span>
        </div>
      ))}
      <p className="inspector-meta">Best restoration over all blocks at each token; measured values only.</p>
    </div>
  );
}

function PartsVisual({ section }: { section: ReportSection }) {
  const data = section.data as Record<string, unknown>;
  const rec = (data.top_recovery ?? []) as { rank: number; label: string; recovery: number }[];
  const dla = (data.top_dla ?? []) as { rank: number; label: string; value: number }[];
  return (
    <div className="grid-2">
      <div>
        <h3>Patching rank</h3>
        {rec.length === 0 ? (
          <p className="inspector-meta">No head sweep in this report.</p>
        ) : (
          <ol className="tour-list">
            {rec.map((e) => (
              <li key={`r-${e.label}`}>
                head {e.label} · recovery <span className="mono">{e.recovery.toFixed(3)}</span>
              </li>
            ))}
          </ol>
        )}
      </div>
      <div>
        <h3>Direct push (DLA)</h3>
        {dla.length === 0 ? (
          <p className="inspector-meta">No DLA ranking in this report.</p>
        ) : (
          <ol className="tour-list">
            {dla.map((e) => (
              <li key={`d-${e.label}`}>
                head {e.label} · DLA <span className="mono">{e.value.toFixed(3)}</span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}

function MethodsVisual({ section }: { section: ReportSection }) {
  const data = section.data as Record<string, unknown>;
  const stats = (data.stats ?? {}) as Record<
    string,
    { pearson_heads?: number | null; top10_overlap?: number; expected_random_top10?: number }
  >;
  if (!stats.atp && !stats.dla) {
    return <p className="inspector-meta">The agreement measurement was not part of this report.</p>;
  }
  const rows: { name: string; pearson: number | null; overlap: number | null; expected: number | null }[] = [
    {
      name: "Attribution patching",
      pearson: stats.atp?.pearson_heads ?? null,
      overlap: stats.atp?.top10_overlap ?? null,
      expected: stats.atp?.expected_random_top10 ?? null,
    },
    {
      name: "Direct logit attribution",
      pearson: stats.dla?.pearson_heads ?? null,
      overlap: stats.dla?.top10_overlap ?? null,
      expected: stats.dla?.expected_random_top10 ?? null,
    },
  ];
  const recon = (data.dla_reconciliation ?? {}) as { abs_error?: number; tolerance?: number; ok?: boolean };
  return (
    <div>
      <div className="table-scroll">
        <table className="kv-table">
          <thead>
            <tr>
              <th scope="col">Method vs exact</th>
              <th scope="col">Pearson (heads)</th>
              <th scope="col">Top-10 overlap</th>
              <th scope="col">Random expectation</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td className="mono">{r.pearson == null ? "n/a" : r.pearson.toFixed(3)}</td>
                <td className="mono">{r.overlap == null ? "n/a" : `${r.overlap}/10`}</td>
                <td className="mono">{r.expected == null ? "n/a" : r.expected.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {recon.abs_error != null && (
        <p className="inspector-meta">
          DLA reconciliation: abs error <span className="mono">{recon.abs_error.toExponential(2)}</span>{" "}
          against tolerance {recon.tolerance} — {recon.ok ? "ok" : "MISMATCH"}.
        </p>
      )}
    </div>
  );
}

function CheckVisual({
  report,
  jobId,
  verify,
}: {
  report: ReportResult;
  jobId: string | null;
  verify: VerifyStatus | null;
}) {
  const limits = sectionOf(report, "limits");
  const notShown = limits?.trust.not_shown ?? [];
  return (
    <div>
      <h3>Live verification status</h3>
      {verify ? (
        <ul className="tour-verify-list">
          {verify.checks.map((c) => {
            const word = c.manual ? "manual" : c.stale ? "stale" : c.status;
            const icon = c.manual ? "✋" : c.stale ? "↻" : c.ok ? "✓" : c.status === "fail" ? "✕" : "–";
            return (
              <li key={c.check_id}>
                <span aria-hidden>{icon}</span> <strong>{c.title}</strong>: {word}
                {c.result?.max_diff != null && (
                  <span className="inspector-meta">
                    {" "}
                    (max diff {c.result.max_diff.toExponential(2)}, tolerance {c.tolerance})
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="inspector-meta">Verification status is unavailable right now.</p>
      )}
      {jobId && (
        <ReproduceButton
          spec={{ kind: "report", job_id: jobId, mode: "sample" }}
          runtime="About 30–60 s for --mode sample on CPU"
          compact
        />
      )}
      <h3>What this does not show</h3>
      <ul className="caveat-list">
        {notShown.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
    </div>
  );
}

function SceneVisual({
  scene,
  report,
  jobId,
  verify,
}: {
  scene: TourScene;
  report: ReportResult;
  jobId: string | null;
  verify: VerifyStatus | null;
}) {
  const section = sectionOf(report, scene.sectionId);
  if (scene.visual === "check") return <CheckVisual report={report} jobId={jobId} verify={verify} />;
  if (!section) {
    return <p className="inspector-meta">This report has no &quot;{scene.sectionId}&quot; section.</p>;
  }
  if (scene.visual === "answer") return <BarPair section={section} />;
  if (scene.visual === "lens") {
    const lens = (section.data.lens ?? {}) as { stages?: { stage: string; D: number | null }[] };
    return <LensSparkline stages={lens.stages ?? []} />;
  }
  if (scene.visual === "words") return <WordsVisual section={section} />;
  if (scene.visual === "parts") return <PartsVisual section={section} />;
  return <MethodsVisual section={section} />;
}

export default function Tour({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [report, setReport] = useState<ReportResult | null>(null);
  const [reportJobId, setReportJobId] = useState<string | null>(null);
  const [source, setSource] = useState<
    { mode: "saved"; at: string | null } | { mode: "fresh" } | null
  >(null);
  const [verify, setVerify] = useState<VerifyStatus | null>(null);
  const [phase, setPhase] = useState<"loading" | "generating" | "ready" | "error">("loading");
  const [genProgress, setGenProgress] = useState(0);
  const [genMessage, setGenMessage] = useState("");
  const [error, setError] = useState<string | null>(null);

  const [scene, setScene] = useState(0);
  const [paused, setPaused] = useState(false);
  const [reducedMotion, setReducedMotion] = useState<boolean>(
    () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches === true,
  );
  const [done, setDone] = useState(false);
  const pollRef = useRef<number | null>(null);
  const sceneRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Initial value comes from the lazy state initialiser; this effect only
    // subscribes to changes (no synchronous setState here).
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const onChange = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, []);

  const loadVerify = useCallback(() => {
    api
      .verifyStatus()
      .then(setVerify)
      .catch(() => setVerify(null));
  }, []);

  const generateFresh = useCallback((taskId: string) => {
    setPhase("generating");
    setGenProgress(0);
    setGenMessage("starting a real report computation");
    api
      .startReport({ task_id: taskId, pair_id: null, study_job_id: null })
      .then((r) => {
        pollRef.current = window.setInterval(() => {
          api
            .reportStatus(r.job_id)
            .then((job: ReportJobStatus) => {
              setGenProgress(job.progress ?? 0);
              setGenMessage(job.progress_message || "computing");
              if (job.status === "complete" && job.results) {
                if (pollRef.current) clearInterval(pollRef.current);
                setReport(job.results);
                setReportJobId(r.job_id);
                setSource({ mode: "fresh" });
                setPhase("ready");
                loadVerify();
              } else if (job.status === "failed" || job.status === "cancelled") {
                if (pollRef.current) clearInterval(pollRef.current);
                setError(job.error ?? `report job ${job.status}`);
                setPhase("error");
              }
            })
            .catch((e) => {
              if (pollRef.current) clearInterval(pollRef.current);
              setError(String((e as Error).message ?? e));
              setPhase("error");
            });
        }, 500);
      })
      .catch((e) => {
        setError(String((e as Error).message ?? e));
        setPhase("error");
      });
  }, [loadVerify]);

  useEffect(() => {
    // No "started once" guard: under StrictMode the effect runs, is cleaned up,
    // and runs again. The first run's results are dropped by `cancelled`, so the
    // second run must do the work (a guard here left the page stuck loading).
    let cancelled = false;
    (async () => {
      try {
        const [taskList, vstatus] = await Promise.all([
          api.tasks(),
          api.verifyStatus().catch(() => null),
        ]);
        if (cancelled) return;
        setVerify(vstatus);
        const taskId = taskList.tasks[0]?.id;
        if (!taskId) throw new Error("no task definitions are available");
        const list = await api.listExperiments();
        const ids = list.saved
          .filter((s) => (s.config as Record<string, unknown> | null)?.job_type === "report")
          .map((s) => s.id)
          .reverse()
          .slice(0, 8);
        const candidates: SavedReportCandidate[] = [];
        const fullById = new Map<string, ReportResult>();
        for (const id of ids) {
          try {
            const full = await api.getSavedExperiment(id);
            const rr = full.results as unknown as ReportResult | null;
            if (!rr || !Array.isArray(rr.sections)) continue;
            fullById.set(id, rr);
            candidates.push({
              id,
              task_id: rr.task_id ?? null,
              generated_at: rr.generated_at ?? null,
              git_commit: String(rr.provenance?.git_commit ?? ""),
            });
          } catch {
            /* skip unreadable saved reports */
          }
        }
        const pick = pickSavedReport(candidates, vstatus?.current_git_commit ?? null, taskId);
        if (pick) {
          const rr = fullById.get(pick.id) ?? null;
          if (rr) {
            if (cancelled) return;
            setReport(rr);
            setReportJobId(pick.id);
            setSource({ mode: "saved", at: rr.generated_at });
            setPhase("ready");
            return;
          }
        }
        if (!cancelled) generateFresh(taskId);
      } catch (e) {
        if (!cancelled) {
          setError(String((e as Error).message ?? e));
          setPhase("error");
        }
      }
    })();
    return () => {
      cancelled = true;
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [generateFresh]);

  // Auto-advance: never when reduced motion is set, paused, finished or not ready.
  useEffect(() => {
    if (phase !== "ready" || done || !shouldAutoAdvance(reducedMotion, paused)) return;
    if (isLastScene(scene)) return;
    const t = window.setTimeout(() => setScene((s) => nextScene(s)), AUTO_ADVANCE_MS);
    return () => clearTimeout(t);
  }, [phase, done, reducedMotion, paused, scene]);

  useEffect(() => {
    if (phase !== "ready") return;
    sceneRef.current?.focus();
  }, [scene, phase]);

  const goNext = () => {
    if (isLastScene(scene)) setDone(true);
    else setScene((s) => nextScene(s));
  };
  const goPrev = () => {
    setDone(false);
    setScene((s) => prevScene(s));
  };

  if (phase === "error") {
    return (
      <main className="page tour-page">
        <h1>Tour</h1>
        <div className="state-banner error" role="alert">
          The tour needs a real report, and none could be produced: {error}
        </div>
        <p className="inspector-meta">
          The tour never shows placeholder numbers. Check that the backend is running and the model is
          loaded, then reload this page.
        </p>
      </main>
    );
  }

  if (phase !== "ready" || !report) {
    return (
      <main className="page tour-page">
        <h1>Tour</h1>
        {phase === "generating" ? (
          <section className="panel">
            <p role="status">
              No saved report matches this exact commit, so the tour is generating a fresh one. This is
              real computation on the model, not a recording.
            </p>
            <div className="inspector-meta" role="status" style={{ margin: "0.6rem 0 0.25rem" }}>
              {Math.round(genProgress * 100)}% — {genMessage}
            </div>
            <div style={{ background: "var(--bg-2)", borderRadius: 4, height: 10, overflow: "hidden" }}>
              <div
                style={{
                  width: `${Math.round(genProgress * 100)}%`,
                  height: "100%",
                  background: "var(--sage)",
                  transition: "width 0.3s",
                }}
              />
            </div>
          </section>
        ) : (
          <div className="state-banner info" role="status">
            <span className="spinner" aria-hidden /> Looking for a saved report at the current commit…
          </div>
        )}
        {!ready && (
          <div className="state-banner warn" role="status">
            The model is still loading. A fresh report needs the loaded GPT-2 weights.
          </div>
        )}
      </main>
    );
  }

  const current = TOUR_SCENES[scene];
  const section = sectionOf(report, current.sectionId);
  const claims = (section?.claims ?? []).filter((c) => c.level === "curious").slice(0, 2);

  return (
    <main
      className="page tour-page"
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") {
          e.preventDefault();
          goNext();
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          goPrev();
        }
      }}
    >
      <div className="tour-head">
        <h1>Tour</h1>
        <p className="inspector-meta">
          {source?.mode === "saved"
            ? `From a saved report made at ${source.at ?? "an earlier time"} — reused because its commit matches this one.`
            : "A fresh report was generated for this tour. This is real computation, not a recording."}
          {reducedMotion && " Reduced motion is on, so the tour only advances when you press Next."}
        </p>
      </div>

      <div className="tour-progress" role="progressbar" aria-valuemin={1} aria-valuemax={TOUR_SCENES.length} aria-valuenow={scene + 1} aria-label="Tour progress">
        {TOUR_SCENES.map((s, i) => (
          <button
            key={s.id}
            className={i === scene ? "tour-dot active" : i < scene ? "tour-dot done" : "tour-dot"}
            aria-label={`Scene ${i + 1}: ${s.title}`}
            aria-current={i === scene ? "step" : undefined}
            onClick={() => {
              setScene(i);
              setDone(false);
            }}
          />
        ))}
        <span className="inspector-meta">
          Scene {scene + 1} of {TOUR_SCENES.length} · {progressPercent(scene)}%
        </span>
      </div>

      <section
        className="panel tour-scene"
        aria-labelledby="tour-scene-title"
        tabIndex={-1}
        ref={sceneRef}
      >
        <h2 id="tour-scene-title">{current.title}</h2>
        <div className="tour-scene-body">
          <SceneVisual scene={current} report={report} jobId={reportJobId} verify={verify} />
        </div>
        <div className="tour-claims" aria-live="polite">
          {claims.map((c) => (
            <p key={c.id}>
              <ClaimText text={c.text} level="curious" />
            </p>
          ))}
          {claims.length === 0 && (
            <p className="inspector-meta">This scene has no short claim; the visual carries the measured values.</p>
          )}
        </div>
        <p className="inspector-meta">
          <a href={`#/report${reportJobId ? `?job=${reportJobId}` : ""}`}>See the full report →</a>
        </p>
      </section>

      {done && (
        <div className="state-banner ok" role="status">
          Tour complete. Six scenes, all from one real report. Press Previous to look again, or open the
          full report.
        </div>
      )}

      <div className="tour-controls">
        <button onClick={goPrev} disabled={scene === 0}>
          ← Previous
        </button>
        {!reducedMotion && (
          <button onClick={() => setPaused((p) => !p)} aria-pressed={paused}>
            {paused ? "Resume auto-play" : "Pause auto-play"}
          </button>
        )}
        <button className="primary" onClick={goNext}>
          {isLastScene(scene) ? "Finish" : "Next →"}
        </button>
      </div>
    </main>
  );
}
