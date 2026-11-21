import type { ReactNode } from "react";
import type { ReadingLevel } from "../../lib/glossary";
import { ClaimText, type ReportSection } from "./Claims";
import { MiniBlocks, MiniProb, MiniPush, type PushRow } from "./MiniCharts";

/* "In short": four of the report's own claims (picked by id, at the reading level in use), each with a
 * small graph drawn from the same stored data. Nothing is written here; the claims are the server's. */

const section = (sections: ReportSection[], id: string) => sections.find((s) => s.id === id);
const claim = (s: ReportSection | undefined, id: string | null, level: ReadingLevel) =>
  s?.claims.find((c) => c.level === level && (id === null || c.id === id));

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export default function InShort({ sections, level }: { sections: ReportSection[]; level: ReadingLevel }) {
  const items: { key: string; text: string; graph: ReactNode; tag?: string }[] = [];

  // 1. the answer
  const answer = section(sections, "answer");
  const answerClaim = claim(answer, "answer.top_token", level);
  if (answer && answerClaim) {
    const p = (answer.data.prediction ?? {}) as Record<string, unknown>;
    const a = num(p.prob_a);
    const b = num(p.prob_b);
    items.push({
      key: "answer",
      text: answerClaim.text,
      graph:
        a !== null && b !== null ? (
          <MiniProb
            rows={[
              { label: String(p.candidate_a ?? "A").trim(), value: a },
              { label: String(p.candidate_b ?? "B").trim(), value: b },
            ]}
          />
        ) : null,
    });
  }

  // 2. the word that matters late on
  const words = section(sections, "which_words");
  const wordsClaim = claim(words, "words.late", level);
  if (words && wordsClaim) {
    const sweep = (words.data.sweep ?? {}) as { recovery?: number[][] };
    const late = (words.data.late ?? {}) as { layer?: number };
    const perBlock = (sweep.recovery ?? []).map((row) => Math.max(...row.filter((v) => Number.isFinite(v)), 0));
    items.push({
      key: "words",
      text: wordsClaim.text,
      graph: perBlock.length ? <MiniBlocks values={perBlock} highlight={late.layer} /> : null,
    });
  }

  // 3. the strongest direct push
  const parts = section(sections, "which_parts");
  const partsClaim = claim(parts, "parts.dla_top.1", level);
  if (parts && partsClaim) {
    const dla = (parts.data.dla ?? {}) as { top?: { label: string; value: number }[]; most_negative?: { label: string; value: number } };
    const rows: PushRow[] = [...(dla.top ?? []).slice(0, 4), ...(dla.most_negative ? [dla.most_negative] : [])].map((r) => ({
      label: `${r.label}`,
      value: r.value,
    }));
    items.push({ key: "parts", text: partsClaim.text, graph: rows.length ? <MiniPush rows={rows} /> : null });
  }

  // 4. how sure
  const sure = section(sections, "how_sure");
  const sureClaim = claim(sure, null, level);
  if (sureClaim) {
    items.push({ key: "sure", text: sureClaim.text, graph: <span className="rp-chip">1 sentence</span> });
  }

  if (items.length === 0) return null;
  return (
    <section className="rp-short" aria-labelledby="rp-short-title">
      <h2 id="rp-short-title">In short</h2>
      <ol>
        {items.map((it, i) => (
          <li key={it.key}>
            <i>{String(i + 1).padStart(2, "0")}</i>
            <span>
              <ClaimText text={it.text} level={level} />
            </span>
            <span className="rp-spark">{it.graph}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
