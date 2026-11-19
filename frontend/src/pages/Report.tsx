import { useEffect, useRef, useState } from "react";
import {
  api,
  type Health,
  type ReportJobStatus,
  type ReportResult,
  type SavedExperiment,
  type TaskSummary,
} from "../api/client";
import type { ReadingLevel } from "../lib/glossary";
import { loadLevel, saveLevel } from "../lib/level";
import SweepHeatmap from "../components/SweepHeatmap";
import GlossaryList from "../components/GlossaryList";
import AgreementScatter from "../components/AgreementScatter";
import ReproduceButton from "../components/ReproduceButton";
import { ClaimsList, type ReportSection } from "./report/Claims";
import TrustPanel from "./report/TrustPanel";
import AnalogyBox from "./report/AnalogyBox";
import LensSparkline from "./report/LensSparkline";

/**
 * Circuit Report page (Stage 4 Task 6): one page, ~2 minutes of reading,
 * every sentence generated from measured values by fixed templates and
 * lint-checked on the server. Reading level is remembered in localStorage
 * (inside try/catch) and applied to the rendered claims.
 */

const LEVELS: { id: ReadingLevel; label: string }[] = [
  { id: "curious", label: "Curious" },
  { id: "student", label: "Student" },
  { id: "expert", label: "Expert" },
];

interface LensStageData {
  stage: string;
  D: number | null;
}

/** A small probability bar pair for the answer section (simple bars). */
function ProbBars({ rows }: { rows: { label: string; value: number | null }[] }) {
  const max = Math.max(...rows.map((r) => r.value ?? 0), 1e-9);
  return (
    <div className="prob-bars">
      {rows.map((r) => (
        <div className="bar-row" key={r.label}>
          <span className="label">{r.label}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{ left: 0, width: `${Math.max(0, ((r.value ?? 0) / max) * 100)}%` }}
            />
          </span>
          <span className="value">{r.value == null ? "n/a" : `${(r.value * 100).toFixed(1)}%`}</span>
        </div>
      ))}
    </div>
  );
}

function TopBars({ rows }: { rows: { label: string; value: number | null }[] }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value ?? 0)), 1e-9);
  return (
    <div className="prob-bars">
      {rows.map((r) => (
        <div className="bar-row" key={r.label}>
          <span className="label">{r.label}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{
                left: (r.value ?? 0) >= 0 ? "50%" : `${50 - (Math.abs(r.value ?? 0) / max) * 50}%`,
                width: `${(Math.abs(r.value ?? 0) / max) * 50}%`,
                background: "var(--sage)",
              }}
            />
            <span
              style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--line-1)" }}
            />
          </span>
          <span className="value">{r.value == null ? "n/a" : r.value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function SectionVisual({ section }: { section: ReportSection }) {
  const data = section.data as Record<string, unknown>;
  if (section.id === "answer") {
    const pred = (data.prediction ?? {}) as Record<string, unknown>;
    return (
      <ProbBars
        rows={[
          { label: String(pred.candidate_a ?? ""), value: (pred.prob_a as number) ?? null },
          { label: String(pred.candidate_b ?? ""), value: (pred.prob_b as number) ?? null },
        ]}
      />
    );
  }
  if (section.id === "where_formed") {
    const lens = (data.lens ?? {}) as { stages?: LensStageData[] };
    return <LensSparkline stages={lens.stages ?? []} />;
  }
  if (section.id === "which_words" || section.id === "which_parts" || section.id === "vs_paper") {
    const sweep = (data.sweep ?? null) as
      | {
          recovery?: number[][];
          tokens?: string[];
          mode?: string;
          direction?: string;
        }
      | null;
    if (!sweep?.recovery || sweep.recovery.length === 0) return null;
    const isHeadGrid = sweep.mode === "layer_head";
    const rowLabels = sweep.recovery.map((_, i) => `block ${i}`);
    const colLabels = isHeadGrid
      ? (sweep.tokens ?? Array.from({ length: sweep.recovery[0]?.length ?? 0 }, (_, h) => `head ${h}`))
      : (sweep.tokens ?? []);
    return (
      <SweepHeatmap
        rowLabels={rowLabels}
        colLabels={colLabels}
        recovery={sweep.recovery}
        dPatched={sweep.recovery.map((row) => row.map(() => Number.NaN))}
        direction={sweep.direction}
      />
    );
  }
  if (section.id === "how_sure") {
    const study = (data.study ?? null) as Record<string, unknown> | null;
    if (!study) return null;
    const atp = (study.rq1_atp_heads ?? null) as { mean?: number } | null;
    const dla = (study.rq1_dla_heads ?? null) as { mean?: number } | null;
    const rows = [];
    if (atp?.mean != null) rows.push({ label: "AtP vs exact (rho)", value: atp.mean });
    if (dla?.mean != null) rows.push({ label: "DLA vs exact (rho)", value: dla.mean });
    const overlap = (study.overlap ?? null) as Record<string, number> | null;
    if (overlap?.atp != null) rows.push({ label: "top-10 overlap exact/AtP", value: overlap.atp });
    if (overlap?.dla != null) rows.push({ label: "top-10 overlap exact/DLA", value: overlap.dla });
    return rows.length > 0 ? <TopBars rows={rows} /> : null;
  }
  if (section.id === "methods_agree") {
    const scatter = (data.scatter ?? []) as {
      label: string;
      kind: "head" | "mlp";
      layer: number;
      head: number | null;
      exact: number;
      atp: number;
      dla: number;
    }[];
    if (scatter.length === 0) return null;
    return (
      <div className="grid-2" style={{ alignItems: "start" }}>
        <AgreementScatter
          title="Attribution patching vs exact patching"
          xLabel="Exact recovery"
          yLabel="AtP recovery"
          points={scatter.map((p) => ({ ...p, x: p.exact, y: p.atp }))}
        />
        <AgreementScatter
          title="Direct logit attribution vs exact patching"
          xLabel="Exact recovery"
          yLabel="DLA value"
          points={scatter.map((p) => ({ ...p, x: p.exact, y: p.dla }))}
        />
      </div>
    );
  }
  return null;
}

function speak(text: string) {
  try {
    const u = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(u);
  } catch {
    /* unsupported; button is hidden anyway */
  }
}

function SectionCard({ section, level }: { section: ReportSection; level: ReadingLevel }) {
  const claimTexts = section.claims
    .filter((c) => c.level === level)
    .map((c) => c.text.replace(/\{\{term:[a-z_]+\|([^{}]*)\}\}/g, "$1"))
    .join(" ");
  const canListen = typeof window !== "undefined" && "speechSynthesis" in window;
  return (
    <section className="panel report-section" id={`report-${section.id}`} aria-labelledby={`h-${section.id}`}>
      <h2 id={`h-${section.id}`}>{section.title}</h2>
      <p className="inspector-meta">
        {section.question} · evidence: {section.evidence_tier}
        {canListen && claimTexts && (
          <>
            {" · "}
            <button className="ghost" onClick={() => speak(claimTexts)} aria-label={`Listen to ${section.title}`}>
              Listen
            </button>
          </>
        )}
      </p>
      <ClaimsList claims={section.claims} data={section.data} level={level} />
      {section.analogy && <AnalogyBox analogy={section.analogy} />}
      <SectionVisual section={section} />
      <TrustPanel trust={section.trust} />
      <p className="inspector-meta">
        <a href="#/verify">How was this checked? →</a>
      </p>
    </section>
  );
}

export default function Report({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [taskId, setTaskId] = useState("");
  const [studies, setStudies] = useState<SavedExperiment[]>([]);
  const [studyJobId, setStudyJobId] = useState("");
  const [level, setLevel] = useState<ReadingLevel>(loadLevel);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<ReportJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    api
      .tasks()
      .then((r) => {
        setTasks(r.tasks);
        if (r.tasks.length > 0) setTaskId((cur) => cur || r.tasks[0].id);
      })
      .catch((e) => setError(String(e.message ?? e)));
    api
      .listExperiments()
      .then((r) =>
        setStudies(r.saved.filter((s) => (s.config as Record<string, unknown> | null)?.job_type === "study")),
      )
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = window.setInterval(() => {
      api
        .reportStatus(jobId)
        .then((r) => {
          setJob(r);
          if (r.status === "failed") setError(r.error ?? "report failed");
        })
        .catch((e) => setError(String(e.message ?? e)));
    }, 400);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId]);

  // A tour (or a shared link) can point at a stored report: #/report?job=<id>.
  useEffect(() => {
    const query = window.location.hash.split("?")[1];
    if (!query) return;
    const id = new URLSearchParams(query).get("job");
    if (!id) return;
    api
      .reportStatus(id)
      .then((r) => {
        setJob(r);
        setJobId(id);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const running = job?.status === "running" || job?.status === "queued";
  const result: ReportResult | null = job?.status === "complete" ? job.results : null;

  const changeLevel = (lv: ReadingLevel) => {
    setLevel(lv);
    saveLevel(lv);
  };

  const generate = () => {
    setError(null);
    setJob(null);
    setJobId(null);
    api
      .startReport({ task_id: taskId, pair_id: null, study_job_id: studyJobId || null })
      .then((r) => setJobId(r.job_id))
      .catch((e) => setError(String(e.message ?? e)));
  };

  const cancel = () => {
    if (jobId) api.cancelExperiment(jobId).catch(() => {});
  };

  return (
    <main className="page report-page">
      <h1>Report</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "80ch" }}>
        The Circuit Report: one page about one example, written from measured values by fixed templates. Every
        sentence shows where its numbers come from; a "Why trust this?" panel lists the live checks and what the
        report does not show.
      </p>

      <section className="panel">
        <h2>Generate</h2>
        <div className="lab-input-grid" style={{ margin: "0.8rem 0" }}>
          <div className="lab-input">
            <label htmlFor="report-task">Task</label>
            <select id="report-task" value={taskId} onChange={(e) => setTaskId(e.target.value)} disabled={running}>
              {tasks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>
          <div className="lab-input">
            <label htmlFor="report-study">Include a saved study (optional)</label>
            <select
              id="report-study"
              value={studyJobId}
              onChange={(e) => setStudyJobId(e.target.value)}
              disabled={running}
            >
              <option value="">None — single prompt only</option>
              {studies.map((s) => (
                <option key={s.id} value={s.id}>
                  study {s.id} ({String((s.config as Record<string, unknown>).n_prompts ?? "?")} prompts)
                </option>
              ))}
            </select>
          </div>
          <div className="lab-input" role="radiogroup" aria-label="Reading level">
            <label>Reading level</label>
            <div className="depth-toggle" style={{ margin: 0 }}>
              {LEVELS.map((lv) => (
                <button
                  key={lv.id}
                  role="radio"
                  aria-checked={level === lv.id}
                  className={level === lv.id ? "active" : ""}
                  onClick={() => changeLevel(lv.id)}
                  disabled={running}
                >
                  {lv.label}
                </button>
              ))}
            </div>
          </div>
          <div className="lab-input" style={{ gap: "0.5rem" }}>
            <button className="primary" onClick={generate} disabled={!ready || !taskId || running}>
              Generate
            </button>
            <button onClick={cancel} disabled={!running}>
              Cancel
            </button>
          </div>
        </div>
        <p className="inspector-meta">
          A full report takes about 20 seconds on CPU (two patching sweeps, a logit lens, direct logit attribution
          and one prediction pass). No language model writes this text — it is template output from measured values.
        </p>

        {running && job && (
          <div style={{ margin: "0.6rem 0" }}>
            <div className="inspector-meta" style={{ marginBottom: "0.25rem" }} role="status">
              {Math.round((job.progress ?? 0) * 100)}% — {job.progress_message || "running"}
            </div>
            <div style={{ background: "var(--bg-2)", borderRadius: 4, height: 10, overflow: "hidden" }}>
              <div
                style={{
                  width: `${Math.round((job.progress ?? 0) * 100)}%`,
                  height: "100%",
                  background: "var(--sage)",
                  transition: "width 0.3s",
                }}
              />
            </div>
          </div>
        )}
        {job?.status === "cancelled" && (
          <div className="state-banner warn" role="status" style={{ marginTop: "0.5rem" }}>
            Report cancelled. No results were saved.
          </div>
        )}
        {error && (
          <div className="state-banner error" role="alert" style={{ marginTop: "0.5rem" }}>
            {error}
          </div>
        )}
        {result && jobId && (
          <p className="inspector-meta" style={{ marginTop: "0.6rem" }}>
            Curious-level readability: {result.lint.curious_readability.flesch_reading_ease} Flesch ·{" "}
            {result.lint.curious_readability.mean_sentence_words} words/sentence average.{" "}
            <a href={api.reportExportUrl(jobId, level)} download>
              Download Markdown
            </a>
          </p>
        )}
        {result && jobId && (
          <ReproduceButton
            spec={{ kind: "report", job_id: jobId, mode: "sample" }}
            runtime="The combined script runs DLA, the lens and both sweeps; about 30–60 s for --mode sample."
          />
        )}
      </section>

      {!result && !running && (
        <div className="state-banner info" role="status">
          No report yet — choose a task and press Generate. Optionally attach a completed study for the
          multi-prompt "How sure can we be?" section.
        </div>
      )}

      {result && (
        <>
          {result.sections.map((s) => (
            <SectionCard key={s.id} section={s} level={level} />
          ))}
          <GlossaryList level={level} />
        </>
      )}
    </main>
  );
}
