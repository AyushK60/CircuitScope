import type { AgreementResult, DlaResult, LensResult, SweepResult } from "../../api/client";

/* Plain-language callouts computed from measured results. Each one states a fact that can be read
 * straight off the numbers on screen; nothing here is a claim about what the model "is doing". */

export interface Callout {
  lead: string;
  text: string;
}

const finite = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);
const quote = (s: string) => `“${s.trim() || "␣"}”`;

export function patchingCallouts(r: SweepResult): Callout[] {
  const cells: { ri: number; ci: number; v: number }[] = [];
  r.recovery.forEach((row, ri) =>
    row.forEach((v, ci) => {
      if (finite(v)) cells.push({ ri, ci, v });
    }),
  );
  if (!cells.length) return [];
  const best = cells.reduce((a, b) => (b.v > a.v ? b : a));
  if (best.v < 0.1) {
    return [{ lead: "Little effect:", text: `no single site restores more than ${best.v.toFixed(2)} of the clean answer.` }];
  }
  const half = best.v * 0.5;

  if (r.mode === "layer_head") {
    return [...cells]
      .sort((a, b) => b.v - a.v)
      .slice(0, 3)
      .map((c, i) => ({
        lead: i === 0 ? "Strongest:" : i === 1 ? "Next:" : "Then:",
        text: `${r.row_labels[c.ri]}, ${r.col_labels[c.ci]} (${c.v.toFixed(2)})`,
      }));
  }

  const rows = r.recovery.map((row, ri) => ({ ri, v: row[best.ci] })).filter((x) => finite(x.v) && x.v >= half);
  const lo = Math.min(...rows.map((x) => x.ri));
  const hi = Math.max(...rows.map((x) => x.ri));
  const out: Callout[] = [
    {
      lead: "Strongest:",
      text: `${quote(r.col_labels[best.ci])}, ${r.row_labels[lo]}${hi > lo ? ` to ${r.row_labels[hi]}` : ""} (peak ${best.v.toFixed(2)})`,
    },
  ];
  const lastCol = r.col_labels.length - 1;
  if (best.ci !== lastCol) {
    const first = r.recovery.findIndex((row) => finite(row[lastCol]) && row[lastCol] >= half);
    if (first >= 0) out.push({ lead: "Hand-over:", text: `${quote(r.col_labels[lastCol])} carries it from ${r.row_labels[first]}` });
  }
  return out;
}

export function dlaCallouts(d: DlaResult): Callout[] {
  const parts = d.components.filter((c) => finite(c.value));
  if (!parts.length) return [];
  const top = parts.reduce((a, b) => (b.value > a.value ? b : a));
  const low = parts.reduce((a, b) => (b.value < a.value ? b : a));
  const out: Callout[] = [];
  if (top.value > 0) out.push({ lead: "Top pusher:", text: `${top.name} (${top.value > 0 ? "+" : ""}${top.value.toFixed(2)})` });
  if (low.value < 0) out.push({ lead: "Top blocker:", text: `${low.name} (${low.value.toFixed(2)})` });
  out.push({
    lead: d.reconciliation.ok ? "Adds up:" : "Mismatch:",
    text: `parts sum to ${d.reconciliation.sum_of_terms.toFixed(3)}, model says ${d.reconciliation.model_D.toFixed(3)}`,
  });
  return out;
}

export function lensCallouts(l: LensResult): Callout[] {
  const tops = l.stages.map((s) => s.top[0]?.token);
  const last = tops.length - 1;
  if (last < 0 || tops[last] === undefined) return [];
  let from = last;
  while (from > 0 && tops[from - 1] === tops[last]) from--;
  const where = l.stages[from].stage;
  const word = quote(tops[last]!);
  if (from === 0) return [{ lead: "Settled early:", text: `${word} is the top guess at every stage` }];
  if (from === last) return [{ lead: "Late:", text: `${word} only becomes the top guess at the very last stage` }];
  return [{ lead: "Settles:", text: `${word} is the top guess from ${where} onward` }];
}

export function agreementCallouts(r: AgreementResult): Callout[] {
  const out: Callout[] = [];
  const a = r.stats.atp.pearson_heads;
  const d = r.stats.dla.pearson_heads;
  if (finite(a)) out.push({ lead: "Cheap approximation:", text: `attribution patching tracks exact patching at ${a.toFixed(3)}` });
  if (finite(d)) out.push({ lead: "Direct pushes only:", text: `DLA agrees at ${d.toFixed(3)}` });
  return out;
}
