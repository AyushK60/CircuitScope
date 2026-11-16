import { useEffect, useRef, useState } from "react";
import {
  api, type ContinueResult, type ExperimentRecord, type Health, type Prediction,
  type QuestionResult, type SavedExperiment, type Tokenization,
} from "../api/client";
import TopPredictions from "../components/TopPredictions";
import ContinuationPanel from "./playground/ContinuationPanel";
import ExperimentForm from "./playground/ExperimentForm";
import GuidedWizard from "./playground/GuidedWizard";
import HistoryPanel from "./playground/HistoryPanel";
import QuestionPanel from "./playground/QuestionPanel";
import ResultsPanel from "./playground/ResultsPanel";
import { PRESET } from "./playground/presets";
import { readParams } from "./playground/url";
import { useIoiPreset } from "./playground/useIoiPreset";
import type { RunState } from "./playground/types";

export default function Playground({ health }: { health: Health | null }) {
  const ready = health?.model_loaded === true;
  const params = useRef(readParams()).current;
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
  const [wizardStep, setWizardStep] = useState(0);
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
  const pollRef = useRef<number | null>(null);

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
    if (!jobId || (runState !== "running")) return;
    pollRef.current = window.setInterval(() => {
      api
        .experiment(jobId)
        .then((r) => {
          setRecord(r);
          if (r.status === "complete") {
            setRunState("complete");
            api.listExperiments().then((lr) => setHistory(lr.saved.filter((x) => x.status === "complete").slice(0, 10))).catch(() => {});
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
        heads: hook.endsWith("hook_z") ? null : null,
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
      .then((r) => setHistory(r.saved.filter((x) => x.status === "complete").slice(0, 10)))
      .catch(() => {});
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refreshHistory, [ready]);

  const loadSaved = (id: string) => {
    api
      .getSavedExperiment(id)
      .then((r) => {
        if (r.results) {
          setRecord({ ...r, status: "complete" as const, error: r.error ?? null });
          setJobId(id);
          setRunState("complete");
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

  return (
    <main className="page">
      <h1>Intervention Playground</h1>
      <p style={{ color: "var(--text-1)", maxWidth: "75ch" }}>
        This workbench runs three <em>real</em> model passes: the clean prompt, a corrupted variant, and a
        patched run that copies one internal activation from the clean run into the corrupted run. Every
        number below is measured from those passes.
      </p>

      {!ready && (
        <div className="state-banner info" style={{ marginBottom: "1rem" }}>
          <span className="spinner" aria-hidden /> Waiting for the model. Experiments need the loaded weights.
        </div>
      )}

      {ready && (
        <GuidedWizard
          comparison={comparison}
          comparisonError={comparisonError}
          wizardStep={wizardStep}
          setWizardStep={setWizardStep}
          runState={runState}
          runExperiment={runExperiment}
          clean={clean}
          corrupted={corrupted}
          candA={candA}
          candB={candB}
          setClean={setClean}
          setCorrupted={setCorrupted}
          setCandA={setCandA}
          setCandB={setCandB}
          setRunState={setRunState}
          setRecord={setRecord}
          hook={hook}
        />
      )}
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
      {runState === "running" && (
        <div className="panel">
          <div className="state-banner info">
            <span className="spinner" aria-hidden /> Executing the real model three times (clean capture,
            corrupted, patched). This takes a few seconds on CPU.
          </div>
        </div>
      )}
      {runState === "failed" && (
        <div className="panel">
          <div className="state-banner error" role="alert">
            <strong>Experiment failed:</strong> {error ?? record?.error} — no results are shown because none
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
        <div className="panel">
          <div className="state-banner warn">Experiment cancelled before completion. Nothing was saved.</div>
        </div>
      )}
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
      <div className="panel">
        <h2>Baseline (live, clean prompt)</h2>
        <p className="inspector-meta">Independent live prediction of the clean prompt for reference.</p>
        <TopPredictions candidates={baseline?.top_predictions ?? null} metric="prob" maxRows={6} />
      </div>
    </main>
  );
}