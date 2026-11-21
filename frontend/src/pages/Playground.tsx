import { useEffect, useRef, useState } from "react";
import {
  api, type ContinueResult, type ExperimentRecord, type Health, type Prediction,
  type QuestionResult, type SavedExperiment, type Tokenization,
} from "../api/client";
import TopPredictions from "../components/TopPredictions";
import ContinuationPanel from "./playground/ContinuationPanel";
import ExperimentForm from "./playground/ExperimentForm";
import Hero from "./playground/Hero";
import HistoryPanel from "./playground/HistoryPanel";
import QuestionPanel from "./playground/QuestionPanel";
import ResultsPanel from "./playground/ResultsPanel";
import { triedKey, type Tried } from "./playground/insights";
import { PRESET } from "./playground/presets";
import { readParams } from "./playground/url";
import { useIoiPreset } from "./playground/useIoiPreset";
import { interventionsOnly } from "./playground/history";
import type { RunState } from "./playground/types";

type Tab = "patch" | "predict" | "ask" | "history";
const TABS: { id: Tab; label: string }[] = [
  { id: "patch", label: "Patch" },
  { id: "predict", label: "Predict & continue" },
  { id: "ask", label: "Ask a question" },
  { id: "history", label: "History" },
];

export default function Playground({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const params = useRef(readParams()).current;
  const [tab, setTab] = useState<Tab>("patch");
  const [clean, setClean] = useState(PRESET.clean);
  const [corrupted, setCorrupted] = useState(PRESET.corrupted);
  const [hook, setHook] = useState(params.hook ?? "blocks.5.hook_out");
  const [position, setPosition] = useState(params.position ?? PRESET.s2_position);
  const [tokenization, setTokenization] = useState<Tokenization | null>(null);
  const [baseline, setBaseline] = useState<Prediction | null>(null);
  const [candA, setCandA] = useState(PRESET.candidate_a);
  const [candB, setCandB] = useState(PRESET.candidate_b);
  const [comparison, setComparison] = useState<{ D: number; prob_a: number; prob_b: number } | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [contResult, setContResult] = useState<ContinueResult | null>(null);
  const [contBusy, setContBusy] = useState(false);
  const [contMode, setContMode] = useState<"greedy" | "sample">("greedy");
  const [contTemp, setContTemp] = useState(0.9);
  const [contTopK, setContTopK] = useState(20);
  const [contSeed, setContSeed] = useState(1234);
  const [history, setHistory] = useState<SavedExperiment[]>([]);
  const [qContext, setQContext] = useState("The father asked his son for water because the father was sick.");
  const [qQuestion, setQQuestion] = useState("Who was sick?");
  const [qCandidates, setQCandidates] = useState(" father, son");
  const [qResult, setQResult] = useState<QuestionResult | null>(null);
  const [qBusy, setQBusy] = useState(false);
  const [qError, setQError] = useState<string | null>(null);
  const [delConfirm, setDelConfirm] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<{ id: string; delta: number | null; recovery: number | null }[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [record, setRecord] = useState<ExperimentRecord | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [tried, setTried] = useState<Tried>({});
  const pollRef = useRef<number | null>(null);

  /** remember a finished run's measured recovery so the block strip can show it */
  const noteTried = (r: ExperimentRecord) => {
    const res = r.results;
    if (!res) return;
    const pos = res.patch.positions[0];
    if (pos === undefined) return;
    setTried((t) => ({ ...t, [triedKey(res.patch.hook, pos)]: res.metrics.Recovery ?? null }));
  };

  // live tokenization + baseline for the clean prompt
  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(() => {
      api
        .predict(clean, 6, candA, candB)
        .then((r) => {
          setTokenization(r.tokenization);
          setBaseline(r.prediction);
          setComparison(r.comparison ? { D: r.comparison.D, prob_a: r.comparison.prob_a, prob_b: r.comparison.prob_b } : null);
          setComparisonError(r.comparison_error ?? null);
        })
        .catch(() => {});
    }, 350);
    return () => clearTimeout(t);
  }, [clean, candA, candB, ready]);

  // poll experiment status while active
  useEffect(() => {
    if (!jobId || runState !== "running") return;
    pollRef.current = window.setInterval(() => {
      api
        .experiment(jobId)
        .then((r) => {
          setRecord(r);
          if (r.status === "complete") {
            setRunState("complete");
            noteTried(r);
            api.listExperiments().then((lr) => setHistory(interventionsOnly(lr.saved).slice(0, 10))).catch(() => {});
          }
          if (r.status === "failed" || r.status === "cancelled") setRunState(r.status);
        })
        .catch((e) => {
          setError(String(e.message ?? e));
          setRunState("failed");
        });
    }, 400);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId, runState]);

  useIoiPreset(params, {
    setClean,
    setCorrupted,
    setCandA,
    setCandB,
    setPosition,
  });

  const runExperiment = () => {
    setError(null);
    setRunState("running");
    setRecord(null);
    api
      .startExperiment({
        clean_text: clean,
        corrupted_text: corrupted,
        hook,
        positions: [position],
        heads: null,
        candidate_a: candA,
        candidate_b: candB,
      })
      .then((r) => setJobId(r.job_id))
      .catch((e) => {
        setError(String(e.message ?? e));
        setRunState("failed");
      });
  };

  const refreshHistory = () => {
    api
      .listExperiments()
      .then((r) => setHistory(interventionsOnly(r.saved).slice(0, 10)))
      .catch(() => {});
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refreshHistory, [ready]);

  const loadSaved = (id: string) => {
    api
      .getSavedExperiment(id)
      .then((r) => {
        if (r.results) {
          const c = r.results.config;
          // restore the stored setup too, so the page shows exactly what was run
          setClean(c.clean_text);
          setCorrupted(c.corrupted_text);
          setCandA(c.candidate_a);
          setCandB(c.candidate_b);
          setHook(c.hook);
          if (c.positions[0] !== undefined) setPosition(c.positions[0]);
          const rec = { ...r, status: "complete" as const, error: r.error ?? null };
          setRecord(rec);
          noteTried(rec);
          setJobId(id);
          setRunState("complete");
          setTab("patch");
        }
      })
      .catch((e) => setError(String(e.message ?? e)));
  };
  const runQuestion = () => {
    setQBusy(true);
    setQError(null);
    api
      .question({
        context: qContext,
        question: qQuestion,
        candidates: qCandidates.split(",").map((c) => (c.startsWith(" ") ? c : " " + c.trim())).filter((c) => c.length > 1),
      })
      .then(setQResult)
      .catch((e) => setQError(String(e.message ?? e)))
      .finally(() => setQBusy(false));
  };
  const toggleCompare = (id: string) => {
    setCompareIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : ids.length >= 2 ? [ids[1], id] : [...ids, id]));
  };
  const runCompare = () => {
    Promise.all(compareIds.map((id) => api.getSavedExperiment(id)))
      .then((rs) =>
        setCompareData(
          rs.map((r) => ({
            id: r.id,
            delta: r.results?.metrics?.Delta ?? null,
            recovery: r.results?.metrics?.Recovery ?? null,
          })),
        ),
      )
      .catch((e) => setError(String(e.message ?? e)));
  };
  const deleteSaved = (id: string) => {
    api
      .deleteExperiment(id)
      .then(() => {
        setDelConfirm(null);
        refreshHistory();
        if (jobId === id) {
          setRecord(null);
          setRunState("idle");
        }
      })
      .catch((e) => setError(String(e.message ?? e)));
  };

  const runContinuation = (nTokens: number) => {
    setContBusy(true);
    const base = contResult?.text_final ?? clean;
    api
      .continue(base, nTokens, {
        mode: contMode,
        temperature: contTemp,
        top_k: contTopK,
        seed: contSeed,
      })
      .then((r) => {
        if (contResult && base !== clean) {
          // step mode: append the new tokens to the displayed run
          setContResult({
            ...r,
            text_start: contResult.text_start,
            steps: [...contResult.steps, ...r.steps.map((s) => ({ ...s, step: contResult.steps.length + s.step }))],
          });
        } else {
          setContResult(r);
        }
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setContBusy(false));
  };

  const result = runState === "complete" ? (record?.results ?? null) : null;

  return (
    <main className="page pg-page">
      <header className="pg-head">
        <h1>Intervention Playground</h1>
        <p>
          Swap one piece of the model's memory and watch the answer change. This runs three <em>real</em> model
          passes: the clean prompt, a corrupted variant, and a patched run that copies one internal activation from
          the clean run into the corrupted one. Every number is measured from those passes.
        </p>
      </header>

      {!ready && (
        <div className="state-banner info" style={{ marginBottom: "1rem" }}>
          <span className="spinner" aria-hidden /> Waiting for the model. Experiments need the loaded weights.
        </div>
      )}

      <div className="pg-tabs" role="tablist" aria-label="Playground tools">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? "on" : ""}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "patch" && (
        <div className="pg-bench">
          <ExperimentForm
            ready={ready}
            clean={clean}
            setClean={setClean}
            corrupted={corrupted}
            setCorrupted={setCorrupted}
            candA={candA}
            setCandA={setCandA}
            candB={candB}
            setCandB={setCandB}
            hook={hook}
            setHook={setHook}
            position={position}
            setPosition={setPosition}
            tokenization={tokenization}
            comparison={comparison}
            comparisonError={comparisonError}
            runState={runState}
            jobId={jobId}
            runExperiment={runExperiment}
            setRunState={setRunState}
            setRecord={setRecord}
            setJobId={setJobId}
          />
          <div className="pg-main">
            <Hero
              prompt={clean}
              tokens={tokenization?.tokens ?? []}
              position={position}
              setPosition={setPosition}
              hook={hook}
              setHook={setHook}
              tried={tried}
              runState={runState}
              result={result}
            />
            {runState === "running" && (
              <div className="pg-card">
                <div className="state-banner info">
                  <span className="spinner" aria-hidden /> Executing the real model three times (clean capture,
                  corrupted, patched). This takes a few seconds on CPU.
                </div>
              </div>
            )}
            {runState === "failed" && (
              <div className="pg-card">
                <div className="state-banner error" role="alert">
                  <strong>Experiment failed:</strong> {error ?? record?.error}. No results are shown because none
                  were produced. Check that the two prompts tokenize to the same length and the position is valid.
                </div>
              </div>
            )}
            <ResultsPanel
              runState={runState}
              record={record}
              setRunState={setRunState}
              setRecord={setRecord}
              setJobId={setJobId}
            />
            {runState === "cancelled" && (
              <div className="pg-card">
                <div className="state-banner warn">Experiment cancelled before completion. Nothing was saved.</div>
              </div>
            )}
            <div className="pg-card">
              <h2>Baseline (live, clean prompt)</h2>
              <p className="pg-note">An independent live prediction of the clean prompt, for reference.</p>
              <TopPredictions candidates={baseline?.top_predictions ?? null} metric="prob" maxRows={6} />
            </div>
          </div>
        </div>
      )}

      {tab === "predict" && (
        <ContinuationPanel
          ready={ready}
          contMode={contMode}
          setContMode={setContMode}
          contTemp={contTemp}
          setContTemp={setContTemp}
          contTopK={contTopK}
          setContTopK={setContTopK}
          contSeed={contSeed}
          setContSeed={setContSeed}
          contResult={contResult}
          setContResult={setContResult}
          contBusy={contBusy}
          runContinuation={runContinuation}
        />
      )}

      {tab === "ask" && (
        <QuestionPanel
          ready={ready}
          qContext={qContext}
          setQContext={setQContext}
          qQuestion={qQuestion}
          setQQuestion={setQQuestion}
          qCandidates={qCandidates}
          setQCandidates={setQCandidates}
          qResult={qResult}
          setQResult={setQResult}
          qBusy={qBusy}
          qError={qError}
          runQuestion={runQuestion}
        />
      )}

      {tab === "history" && (
        <HistoryPanel
          history={history}
          compareIds={compareIds}
          toggleCompare={toggleCompare}
          delConfirm={delConfirm}
          setDelConfirm={setDelConfirm}
          loadSaved={loadSaved}
          deleteSaved={deleteSaved}
          compareData={compareData}
          runCompare={runCompare}
        />
      )}
    </main>
  );
}
