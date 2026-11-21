import type { ReadingLevel } from "../../lib/glossary";
import { ClaimText, EvidenceDisclosure, type ReportSection } from "./Claims";

/** Claims as paragraphs: a small type label, the text, and the evidence in the margin beside it. */
export default function ArticleClaims({ section, level }: { section: ReportSection; level: ReadingLevel }) {
  const visible = section.claims.filter((c) => c.level === level);
  return (
    <>
      {visible.map((c) => (
        <div className="rp-claim" key={`${c.id}:${c.level}`}>
          <p className="claim">
            <span className={`claim-type claim-type-${c.claim_type}`}>{c.claim_type}</span>{" "}
            <ClaimText text={c.text} level={level} />
          </p>
          <aside className="rp-margin">
            <EvidenceDisclosure claim={c} data={section.data} />
          </aside>
        </div>
      ))}
    </>
  );
}
