import type { Tokenization } from "../api/client";

/** Exact tokenizer output. Optionally highlights subject positions (amber)
 *  and the source of a patched activation (teal). */
export default function TokenRow({
  tok,
  highlightPositions = [],
  subjectPositions = [],
}: {
  tok: Tokenization | null;
  highlightPositions?: number[];
  subjectPositions?: number[];
}) {
  if (!tok) return <div className="skeleton" style={{ height: "1.6rem" }} />;
  return (
    <div className="token-row" role="list" aria-label="Tokenized input">
      {tok.tokens.map((t, i) => (
        <span
          key={i}
          role="listitem"
          className={`token${highlightPositions.includes(i) ? " hl" : ""}${subjectPositions.includes(i) ? " subject" : ""}`}
          title={`position ${i}${tok.token_ids?.[i] !== undefined ? ` · id ${tok.token_ids[i]}` : ""}`}
        >
          {t}
          <span className="idx">{i}</span>
        </span>
      ))}
    </div>
  );
}
