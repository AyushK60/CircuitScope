import { useEffect, useRef, useState, type ReactNode } from "react";
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
import type { ReportSection } from "./report/Claims";
import ArticleClaims from "./report/ArticleClaims";
import InShort from "./report/InShort";
import { PushBars } from "./report/MiniCharts";
import TrustPanel from "./report/TrustPanel";
import AnalogyBox from "./report/AnalogyBox";
import LensSparkline from "./report/LensSparkline";
import { useActiveSection } from "./report/useActiveSection";

/**
 * Circuit Report: one article about one example, ~2 minutes of reading. Every sentence is generated
 * from measured values by fixed templates and lint-checked on the server. The reading level is
 * remembered in localStorage (inside try/catch) and applied to the rendered claims.
 */

const LEVELS: { id: ReadingLevel; label: string }[] = [
  { id: "curious", label: "Curious" },
  { id: "student", label: "Student" },
  { id: "expert", label: "Expert" },
];

const SHORT_TITLES: Record<string, string> = {
  answer: "The answer",
  where_formed: "Where it forms",
  which_words: "Which words",
  which_parts: "Which parts",
  methods_agree: "Methods agree",
  vs_paper: "The paper",
  how_sure: "How sure",
  limits: "Limits",
};

interface LensStageData {
  stage: string;
  D: number | null;
}

function ProbBars({ rows }: { rows: { label: string; value: number | null }[] }) {
  const max = Math.max(...rows.map((r) => r.value ?? 0), 1e-9);
  return (
    <div className="prob-bars">
      {rows.map((r, i) => (
        <div className="bar-row" key={r.label}>
          <span className="label">{r.label}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{
                left: 0,
                width: `${Math.max(0, ((r.value ?? 0) / max) * 100)}%`,
                background: i === 0 ? "var(--measure)" : "var(--danger)",
              }}
            />
          </span>
          <span className="value">{r.value == null ? "n/a" : `${(r.value * 100).toFixed(1)}%`}</span>
        </div>
      ))}
    </div>
  );
}

function TopBars({ rows }: { rows: { label: string; value: number | null }[] }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value ?? 0), 1e-9));
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
                background: "var(--measure)",
              }}
            />
            <span style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--line-1)" }} />
          </span>
          <span className="value">{r.value == null ? "n/a" : r.value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

/** The figure(s) for a section, each with a caption. */
function sectionFigures(section: ReportSection): { node: ReactNode; caption: string }[] {
  const data = section.data as Record<string, unknown>;
  const out: { node: ReactNode; caption: string }[] = [];

  if (section.id === "answer") {
    const pred = (data.prediction ?? {}) as Record<string, unknown>;
    out.push({
      caption: "Probability of each candidate next word, measured on this run.",
      node: (
        <ProbBars
          rows={[
            { label: String(pred.candidate_a ?? ""), value: (pred.prob_a as number) ?? null },
            { label: String(pred.candidate_b ?? ""), value: (pred.prob_b as number) ?? null },
          ]}
        />
      ),
    });
  }
  if (section.id === "where_formed") {
    const lens = (data.lens ?? {}) as { stages?: LensStageData[] };
    out.push({
      caption: "Score gap D after each block (logit lens). A heuristic view of the model's notes.",
      node: <LensSparkline stages={lens.stages ?? []} />,
    });
  }
  if (section.id === "which_words" || section.id === "which_parts" || section.id === "vs_paper") {
    const sweep = (data.sweep ?? null) as { recovery?: number[][]; tokens?: string[]; mode?: string; direction?: string } | null;
    if (sweep?.recovery && sweep.recovery.length > 0) {
      const isHeadGrid = sweep.mode === "layer_head";
      const rowLabels = sweep.recovery.map((_, i) => `block ${i}`);
      const colLabels = isHeadGrid
        ? (sweep.tokens ?? Array.from({ length: sweep.recovery[0]?.length ?? 0 }, (_, h) => `head ${h}`))
        : (sweep.tokens ?? []);
      out.push({
        caption: isHeadGrid
          ? "Share of the answer restored when one attention head is patched in (blocks down, heads across)."
          : "Share of the answer restored when one block's output at one word is patched in.",
        node: (
          <SweepHeatmap
            rowLabels={rowLabels}
            colLabels={colLabels}
            recovery={sweep.recovery}
            dPatched={sweep.recovery.map((row) => row.map(() => Number.NaN))}
            direction={sweep.direction}
          />
        ),
      });
    }
  }
  if (section.id === "which_parts") {
    const dla = (data.dla ?? {}) as { top?: { label: string; value: number }[]; most_negative?: { label: string; value: number } };
    const rows = [...(dla.top ?? []).slice(0, 8), ...(dla.most_negative ? [dla.most_negative] : [])];
    if (rows.length > 0) {
      out.push({
        caption: "Direct pushes: the heads that add most to the score gap (blue), and the strongest that pull back (red).",
        node: <PushBars rows={rows.map((r) => ({ label: r.label, value: r.value }))} />,
      });
    }
  }
  if (section.id === "how_sure") {
    const study = (data.study ?? null) as Record<string, unknown> | null;
    if (study) {
      const atp = (study.rq1_atp_heads ?? null) as { mean?: number } | null;
      const dla = (study.rq1_dla_heads ?? null) as { mean?: number } | null;
      const rows = [];
      if (atp?.mean != null) rows.push({ label: "AtP vs exact (rho)", value: atp.mean });
      if (dla?.mean != null) rows.push({ label: "DLA vs exact (rho)", value: dla.mean });
      const overlap = (study.overlap ?? null) as Record<string, number> | null;
      if (overlap?.atp != null) rows.push({ label: "top-10 overlap exact/AtP", value: overlap.atp });
      if (overlap?.dla != null) rows.push({ label: "top-10 overlap exact/DLA", value: overlap.dla });
      if (rows.length > 0) out.push({ caption: "Agreement measured across the prompts of the attached study.", node: <TopBars rows={rows} /> });
    }
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
    if (scatter.length > 0) {
      out.push({
        caption: "One dot per component. The closer to the dashed line, the more the two methods agree.",
        node: (
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
        ),
      });
    }
  }
  return out;
}

function speak(text: string) {
  try {
    const u = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(u);
  } catch {
    /* unsupported; button is hidden anyway */
  }
}

function SectionCard({
  section,
  level,
  index,
  firstFigure,
}: {
  section: ReportSection;
  level: ReadingLevel;
  index: number;
  firstFigure: number;
}) {
  const claimTexts = section.claims
    .filter((c) => c.level === level)
    .map((c) => c.text.replace(/\{\{term:[a-z_]+\|([^{}]*)\}\}/g, "$1"))
    .join(" ");
  const canListen = typeof window !== "undefined" && "speechSynthesis" in window;
  const figures = sectionFigures(section);
  return (
    <section className="rp-section report-section" id={`report-${section.id}`} aria-labelledby={`h-${section.id}`}>
      <div className="rp-head">
        <span className="rp-num">
          {index + 1} · {SHORT_TITLES[section.id]?.toUpperCase() ?? section.id.toUpperCase()}
        </span>
        <h2 id={`h-${section.id}`}>{section.title}</h2>
        <p className="rp-q">
          {section.question} <span className="rp-tier">evidence: {section.evidence_tier}</span>
          {canListen && claimTexts && (
            <>
              {" "}
              <button className="ghost rp-listen" onClick={() => speak(claimTexts)} aria-label={`Listen to ${section.title}`}>
                Listen
              </button>
            </>
          )}
        </p>
      </div>
      <ArticleClaims section={section} level={level} />
      {figures.map((f, i) => (
        <figure className="rp-fig" key={i}>
          <div className="rp-fig-body">{f.node}</div>
          <figcaption>
            <b>Figure {firstFigure + i}.</b> {f.caption}
          </figcaption>
        </figure>
      ))}
      {section.analogy && <AnalogyBox analogy={section.analogy} />}
      <TrustPanel trust={section.trust} />
      <p className="rp-check">
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
  const [showSettings, setShowSettings] = useState(false);

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
      .then((r) => setStudies(r.saved.filter((s) => (s.config as Record<string, unknown> | null)?.job_type === "study")))
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
  const activeId = useActiveSection(result ? result.sections.map((s) => `report-${s.id}`) : []);

  const changeLevel = (lv: ReadingLevel) => {
    setLevel(lv);
    saveLevel(lv);
  };

  const generate = () => {
    setError(null);
    setJob(null);
    setJobId(null);
    setShowSettings(false);
    api
      .startReport({ task_id: taskId, pair_id: null, study_job_id: studyJobId || null })
      .then((r) => setJobId(r.job_id))
      .catch((e) => setError(String(e.message ?? e)));
  };

  const cancel = () => {
    if (jobId) api.cancelExperiment(jobId).catch(() => {});
  };

  const answerData = result?.sections.find((s) => s.id === "answer")?.data as
    | { prediction?: { top_token?: string } }
    | undefined;
  const topWord = answerData?.prediction?.top_token?.trim();
  const taskName = result ? tasks.find((t) => t.id === result.task_id)?.name : undefined;
  const settingsOpen = !result || showSettings;
  // figure numbers run through the whole article: each section starts after the last one's figures
  const firstFigure: number[] = [];
  if (result) {
    let next = 1;
    for (const s of result.sections) {
      firstFigure.push(next);
      next += sectionFigures(s).length;
    }
  }

  return (
    <main className="page rp-page report-page">
      <div className="rp-layout">
        <aside className="rp-toc" aria-label="In this report">
          {result && (
            <>
              <small>In this report</small>
              {result.sections.map((s, i) => (
                <button
                  key={s.id}
                  className={activeId === `report-${s.id}` ? "on" : ""}
                  onClick={() => document.getElementById(`report-${s.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" })}
                >
                  {i + 1} {SHORT_TITLES[s.id] ?? s.title}
                </button>
              ))}
            </>
          )}
        </aside>

        <article className="rp-article">
          <p className="ex-kick">Circuit report</p>
          <h1 className="ex-title rp-title">{topWord ? `Why GPT-2 finishes the sentence with “${topWord}”` : "Report"}</h1>
          {result ? (
            <p className="rp-byline">
              {taskName && <span>{taskName}</span>}
              <span>generated {new Date(result.generated_at).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}</span>
              <span className="badge">TEMPLATE TEXT · NO AI WRITER</span>
            </p>
          ) : (
            <p className="ex-sub">
              The Circuit Report: one article about one example, written from measured values by fixed templates.
              Every sentence shows where its numbers come from, and a "Why trust this?" panel lists the live checks
              and what the report does not show.
            </p>
          )}

          <div className="rp-levels" role="radiogroup" aria-label="Reading level">
            <div className="mode-tabs">
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
            {result && jobId && (
              <>
                <a className="rp-dl" href={api.reportExportUrl(jobId, level)} download>
                  Download Markdown
                </a>
                <ReproduceButton
                  spec={{ kind: "report", job_id: jobId, mode: "sample" }}
                  runtime="The combined script runs DLA, the lens and both sweeps; about 30–60 s for --mode sample."
                  compact
                />
              </>
            )}
            {result && (
              <button className="ghost" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
                {showSettings ? "Hide settings" : "Change settings"}
              </button>
            )}
          </div>
          {result && (
            <p className="rp-read">
              Curious-level readability: {result.lint.curious_readability.flesch_reading_ease} Flesch ·{" "}
              {result.lint.curious_readability.mean_sentence_words} words per sentence on average.
            </p>
          )}

          {settingsOpen && (
            <div className="rp-settings">
              <div className="rp-field">
                <label htmlFor="report-task">Task</label>
                <select id="report-task" value={taskId} onChange={(e) => setTaskId(e.target.value)} disabled={running}>
                  {tasks.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="rp-field">
                <label htmlFor="report-study">Saved study (optional)</label>
                <select id="report-study" value={studyJobId} onChange={(e) => setStudyJobId(e.target.value)} disabled={running}>
                  <option value="">None - single prompt only</option>
                  {studies.map((s) => (
                    <option key={s.id} value={s.id}>
                      study {s.id} ({String((s.config as Record<string, unknown>).n_prompts ?? "?")} prompts)
                    </option>
                  ))}
                </select>
              </div>
              <div className="rp-actions">
                <button className="primary" onClick={generate} disabled={!ready || !taskId || running}>
                  Generate
                </button>
                <button onClick={cancel} disabled={!running}>
                  Cancel
                </button>
              </div>
              <p className="rp-hint">
                A full report takes about 20 seconds on CPU (two patching sweeps, a logit lens, direct logit
                attribution and one prediction pass). No language model writes this text; it is template output
                from measured values.
              </p>
            </div>
          )}

          {running && job && (
            <div className="sw-progress" role="status">
              <span>
                {Math.round((job.progress ?? 0) * 100)}% · {job.progress_message || "running"}
              </span>
              <div>
                <i style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} />
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

          {!result && !running && (
            <div className="state-banner info" role="status" style={{ marginTop: "1rem" }}>
              No report yet. Choose a task and press Generate. Optionally attach a completed study for the
              multi-prompt "How sure can we be?" section.
            </div>
          )}

          {result && (
            <>
              <InShort sections={result.sections} level={level} />
              {result.sections.map((s, i) => (
                <SectionCard key={s.id} section={s} level={level} index={i} firstFigure={firstFigure[i]} />
              ))}
              <GlossaryList level={level} />
            </>
          )}
        </article>
      </div>
    </main>
  );
}
