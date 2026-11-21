import { api, type ExperimentRecord } from "../../api/client";
import TokenRow from "../../components/TokenRow";
import TopPredictions from "../../components/TopPredictions";
import ReproduceButton from "../../components/ReproduceButton";
import MetricCard from "./MetricCard";
import type { RunState } from "./types";

/** "2 · Measured effect" and "3 · Compare all three runs" panels, rendered
 *  only when a completed experiment with results is present. */
export default function ResultsPanel({
  runState,
  record,
  setRunState,
  setRecord,
  setJobId,
}: {
  runState: RunState;
  record: ExperimentRecord | null;
  setRunState: (s: RunState) => void;
  setRecord: (r: ExperimentRecord | null) => void;
  setJobId: (id: string | null) => void;
}) {
  const res = record?.results ?? null;
  const m = res?.metrics ?? null;
  if (!(runState === "complete" && res && m)) return null;
  return (
    <>
      <div className="pg-card">
        <h2>2 · Measured effect</h2>
        <div className="metric-cards">
          <MetricCard
            name="D · logit(A) − logit(B)"
            value={m.D_patched?.toFixed(3) ?? "-"}
            sub={`clean ${m.D_clean?.toFixed(3)} · corrupted ${m.D_corrupted?.toFixed(3)}`}
          />
          <MetricCard
            name="Δ effect of patch"
            value={m.Delta?.toFixed(3) ?? "-"}
            sub="D_patched − D_corrupted"
          />
          <MetricCard
            name="Recovery"
            value={m.Recovery != null ? m.Recovery.toFixed(3) : "n/a"}
            sub={m.Recovery_note ?? "(D_p − D_corr)/(D_cl − D_corr), unclamped"}
          />
        </div>
        <p className="inspector-meta">
          {m.Recovery != null && Math.abs(m.Recovery) < 0.05
            ? "The intervention had a negligible measured effect on this example - reported as measured."
            : m.Recovery != null && m.Recovery > 0.5
              ? "The patched activation carried most of the clean-vs-corrupted information for this metric."
              : "Sign convention: positive D means candidate A scores above candidate B at the final position."}
        </p>
      </div>

      <div className="pg-card">
        <h2>3 · Compare all three runs</h2>
        <div className="grid-3">
          <div className="run-column clean">
            <h4>Clean run · {res.config.clean_text.length > 30 ? "…" : ""}{res.config.clean_text.slice(-28)}</h4>
            <TopPredictions candidates={res.runs.clean} metric="prob" maxRows={5} highlight={[res.config.candidate_a, res.config.candidate_b]} />
          </div>
          <div className="run-column corrupted">
            <h4>Corrupted run · …{res.config.corrupted_text.slice(-28)}</h4>
            <TopPredictions candidates={res.runs.corrupted} metric="prob" maxRows={5} highlight={[res.config.candidate_a, res.config.candidate_b]} />
          </div>
          <div className="run-column patched">
            <h4>Patched run · {res.patch.hook} @ pos {res.patch.positions.join(",")}</h4>
            <TopPredictions candidates={res.runs.patched} metric="prob" maxRows={5} highlight={[res.config.candidate_a, res.config.candidate_b]} />
          </div>
        </div>
        <div className="grid-2" style={{ marginTop: "1rem", alignItems: "start" }}>
          <div>
            <h3>Corrupted tokens (patched position highlighted)</h3>
            <TokenRow
              tok={{ tokens: res.tokens_corrupted, token_ids: [], n_tokens: res.tokens_corrupted.length }}
              highlightPositions={res.patch.positions}
            />
          </div>
          <div>
            <h3>Experiment record</h3>
            <table className="kv-table">
              <tbody>
                <tr><td>Experiment ID</td><td>{record?.id}</td></tr>
                <tr><td>Model</td><td>{res.provenance.model_id} · {res.provenance.device} · torch {res.provenance.torch}</td></tr>
                <tr><td>Instrumentation</td><td>{res.provenance.library}</td></tr>
                <tr><td>Patch source</td><td>{res.patch.source}</td></tr>
                <tr><td>Elapsed</td><td>{res.provenance.elapsed_seconds.toFixed(2)} s (3 runs)</td></tr>
                <tr><td>Candidates</td><td>A {JSON.stringify(res.config.candidate_a)} · B {JSON.stringify(res.config.candidate_b)}</td></tr>
              </tbody>
            </table>
            <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.8rem" }}>
              <a href={api.exportUrl(record!.id, "json")} target="_blank" rel="noreferrer">
                <button>Export JSON</button>
              </a>
              <a href={api.exportUrl(record!.id, "csv")} target="_blank" rel="noreferrer">
                <button>Export CSV</button>
              </a>
              <button
                onClick={() => {
                  setRunState("idle");
                  setRecord(null);
                  setJobId(null);
                }}
              >
                Restore baseline
              </button>
            </div>
            <ReproduceButton
              spec={{ kind: "intervention", job_id: record!.id, mode: "sample" }}
              runtime="One clean, one corrupted and one patched raw forward pass; a few seconds."
              compact
            />
          </div>
        </div>
        <p className="inspector-meta" style={{ marginTop: "0.8rem" }}>
          <strong>What this establishes (and what it does not):</strong> replacing this activation at
          these positions changed the model's measured preference under this specific intervention.
          It is evidence about this example and hook - not a proof that one component "understands"
          anything, and not a statistical claim over other inputs.
        </p>
      </div>
    </>
  );
}
