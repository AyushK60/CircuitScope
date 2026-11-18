import { useState } from "react";
import { api, type ExplainResponse } from "../../api/client";
import { ClaimText } from "./Claims";
import type { ReadingLevel } from "../../lib/glossary";

/**
 * "In plain words" box (Stage 4 Task 6): a short plain-words rendering of a
 * completed sweep or study job, fetched from POST /api/explain (the server
 * reads the stored results; the client never supplies data). Appended to the
 * Sweeps and Study pages without changing their existing content or tests.
 */
export default function PlainWords({ jobId, level = "curious" }: { jobId: string; level?: ReadingLevel }) {
  const [claims, setClaims] = useState<ExplainResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setBusy(true);
    setError(null);
    api
      .explain(jobId, level)
      .then(setClaims)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  return (
    <div className="plain-words" style={{ marginTop: "0.8rem" }}>
      {!claims && (
        <button onClick={load} disabled={busy}>
          {busy ? "Writing…" : "In plain words"}
        </button>
      )}
      {error && (
        <div className="state-banner error" role="alert" style={{ marginTop: "0.4rem" }}>
          {error}
        </div>
      )}
      {claims && (
        <div className="plain-words-body">
          <h3>In plain words</h3>
          {claims.claims.map((c) => (
            <div key={c.id} className="claim">
              <span className={`claim-type claim-type-${c.claim_type}`}>{c.claim_type}</span>{" "}
              <ClaimText text={c.text} level={level} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
