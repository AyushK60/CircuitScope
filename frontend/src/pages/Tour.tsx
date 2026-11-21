import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type Health,
  type ReportJobStatus,
  type ReportResult,
  type VerifyStatus,
} from "../api/client";
import SceneVisual from "./tour/Visuals";
import { sceneCopy } from "./tour/copy";
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
} from "./tour/scenes";

/**
 * Guided tour: six scenes over one real Circuit Report.
 * Prefers a saved report whose provenance commit matches the current commit;
 * otherwise it generates a fresh one and says so. Reduced motion turns off
 * auto-advance entirely.
 */

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
  const [dir, setDir] = useState<"fwd" | "back">("fwd");
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
    const t = window.setTimeout(() => {
      setDir("fwd");
      setScene((s) => nextScene(s));
    }, AUTO_ADVANCE_MS);
    return () => clearTimeout(t);
  }, [phase, done, reducedMotion, paused, scene]);

  useEffect(() => {
    if (phase !== "ready") return;
    sceneRef.current?.focus();
  }, [scene, phase]);

  const goTo = (i: number) => {
    setDir(i >= scene ? "fwd" : "back");
    setScene(i);
    setDone(false);
  };
  const goNext = () => {
    if (isLastScene(scene)) setDone(true);
    else goTo(nextScene(scene));
  };
  const goPrev = () => goTo(prevScene(scene));

  const swipe = useRef<number | null>(null);

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
          <section className="pg-card">
            <p role="status">
              No saved report matches this exact commit, so the tour is generating a fresh one. This is
              real computation on the model, not a recording.
            </p>
            <div className="inspector-meta" role="status" style={{ margin: "0.6rem 0 0.25rem" }}>
              {Math.round(genProgress * 100)}% — {genMessage}
            </div>
            <div className="tr-gen">
              <div style={{ width: `${Math.round(genProgress * 100)}%` }} />
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
  const copy = sceneCopy(current.id, report);
  const playing = !reducedMotion && !paused && !done && !isLastScene(scene);

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
        <p className="tr-source">
          {source?.mode === "saved"
            ? `From a saved report made at ${source.at ?? "an earlier time"}, reused because its commit matches this one.`
            : "A fresh report was generated for this tour. This is real computation, not a recording."}
          {reducedMotion && " Reduced motion is on, so the tour only advances when you press Next."}
        </p>
      </div>

      <div
        className="tr-cinema"
        onPointerDown={(e) => {
          swipe.current = (e.target as HTMLElement).closest("a,button,input,summary") ? null : e.clientX;
        }}
        onPointerUp={(e) => {
          if (swipe.current == null) return;
          const dx = e.clientX - swipe.current;
          swipe.current = null;
          if (dx < -70) goNext();
          else if (dx > 70) goPrev();
        }}
      >
        <div className="tr-glow" aria-hidden />
        <section
          key={scene}
          className={`tr-card ${dir}`}
          aria-labelledby="tour-scene-title"
          tabIndex={-1}
          ref={sceneRef}
        >
          <h2 id="tour-scene-title" className="tr-title">{current.title}</h2>
          <div className="tr-grid">
            <div className="tr-copy">
              <p className="tr-headline" aria-live="polite">{copy.headline}</p>
              <p className="tr-support">{copy.hint}</p>
              <a className="tr-link" href={`#/report${reportJobId ? `?job=${reportJobId}` : ""}`}>
                See the full report →
              </a>
            </div>
            <div className="tr-visual">
              <SceneVisual scene={current} report={report} jobId={reportJobId} verify={verify} />
            </div>
          </div>
        </section>

        <div className="tr-foot">
          <div className="tr-progress" role="progressbar" aria-valuemin={1} aria-valuemax={TOUR_SCENES.length} aria-valuenow={scene + 1} aria-label="Tour progress">
            <div className="tr-segs">
              {TOUR_SCENES.map((s, i) => (
                <button
                  key={s.id}
                  type="button"
                  className={i === scene ? "tr-seg active" : i < scene ? "tr-seg done" : "tr-seg"}
                  aria-label={`Scene ${i + 1}: ${s.title}`}
                  aria-current={i === scene ? "step" : undefined}
                  title={s.title}
                  onClick={() => goTo(i)}
                >
                  <span className="bar">
                    {i === scene && (
                      <i
                        key={`${scene}-${playing}`}
                        className={`tr-fill ${playing ? "run" : "still"}`}
                        style={{ animationDuration: `${AUTO_ADVANCE_MS}ms` }}
                      />
                    )}
                  </span>
                </button>
              ))}
            </div>
            <span className="tr-count">
              Scene {scene + 1} of {TOUR_SCENES.length} · {progressPercent(scene)}%
            </span>
          </div>
          <div className="tr-buttons">
            <button type="button" onClick={goPrev} disabled={scene === 0}>
              ← Previous
            </button>
            {!reducedMotion && (
              <button type="button" onClick={() => setPaused((p) => !p)} aria-pressed={paused}>
                {paused ? "Resume auto-play" : "Pause auto-play"}
              </button>
            )}
            <button type="button" className="primary" onClick={goNext}>
              {isLastScene(scene) ? "Finish" : "Next →"}
            </button>
          </div>
        </div>
      </div>

      {done && (
        <div className="state-banner ok" role="status" style={{ marginTop: "1rem" }}>
          Tour complete. Six scenes, all from one real report. Press Previous to look again, or open the
          full report.
        </div>
      )}
    </main>
  );
}
