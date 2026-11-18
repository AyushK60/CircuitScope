import { resolveEvidence } from "../../lib/evidence";
import { parseTermMarkers, type ReadingLevel } from "../../lib/glossary";
import Term from "../../components/Term";

export interface ReportClaim {
  id: string;
  level: ReadingLevel;
  text: string;
  claim_type: "measured" | "interpretation" | "caveat";
  evidence: string[];
}

export interface ReportSection {
  id: string;
  title: string;
  question: string;
  claims: ReportClaim[];
  data: Record<string, unknown>;
  evidence_tier: string;
  analogy: { id: string; title: string; text: string; breaks_when: string } | null;
  trust: {
    measured: string;
    against: { label: string; value: unknown; note?: string }[];
    checks: { id: string; label: string; value: unknown; tolerance: unknown; ok: boolean; comparison: string }[];
    warnings: string[];
    not_shown: string[];
  };
}

/** Render claim text with {{term:id|display}} markers as interactive Terms. */
export function ClaimText({ text, level }: { text: string; level: ReadingLevel }) {
  const parts = parseTermMarkers(text);
  return (
    <>
      {parts.map((p, i) =>
        p.kind === "text" ? (
          <span key={i}>{p.value}</span>
        ) : (
          <Term key={i} id={p.termId!} label={p.display} level={level} />
        ),
      )}
    </>
  );
}

/** Per-claim "where does this come from?" disclosure (Task 6). */
export function EvidenceDisclosure({ claim, data }: { claim: ReportClaim; data: Record<string, unknown> }) {
  const items = claim.evidence.map((p) => resolveEvidence(data, p));
  return (
    <details className="evidence-disclosure">
      <summary className="inspector-meta">Where does this come from?</summary>
      <ul className="evidence-list">
        {items.map((it) => (
          <li key={it.path}>
            <code className="mono">{it.path}</code>
            {it.kind === "data" && it.value !== undefined && (
              <>
                {" = "}
                <span className="mono">{it.value}</span>
              </>
            )}
            {it.kind === "note" && <span className="inspector-meta"> (fixed note)</span>}
            {it.kind === "data" && it.value === undefined && (
              <span className="inspector-meta"> (not present in stored data)</span>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

export function ClaimsList({
  claims,
  data,
  level,
}: {
  claims: ReportClaim[];
  data: Record<string, unknown>;
  level: ReadingLevel;
}) {
  const visible = claims.filter((c) => c.level === level);
  if (visible.length === 0) return null;
  return (
    <div className="claims-list">
      {visible.map((c) => (
        <div key={`${c.id}:${c.level}`} className="claim">
          <span className={`claim-type claim-type-${c.claim_type}`}>{c.claim_type}</span>{" "}
          <ClaimText text={c.text} level={level} />
          <EvidenceDisclosure claim={c} data={data} />
        </div>
      ))}
    </div>
  );
}
