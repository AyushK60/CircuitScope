import type { ReportResult } from "../../api/client";
import { sectionOf } from "./scenes";

/** One short headline and one hint per scene, computed from the report's own
 *  numbers. Longer wording lives in the full report. */

export interface SceneCopy {
  headline: string;
  hint: string;
}

const num = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
const quote = (t: string) => `“${t.trim()}”`;

export function agreementWord(pearson: number | null | undefined): string {
  if (pearson == null) return "in a way that was not measured";
  if (pearson >= 0.9) return "closely";
  if (pearson >= 0.5) return "roughly";
  return "only weakly";
}

/** Words whose best restoration over all blocks is above one half (BOS excluded). */
export function hotWords(sweep: { recovery?: number[][]; tokens?: string[] } | undefined): { index: number; token: string; best: number }[] {
  if (!sweep?.recovery?.length || !sweep.tokens) return [];
  const nCols = Math.max(...sweep.recovery.map((r) => r.length));
  const out: { index: number; token: string; best: number }[] = [];
  for (let c = 0; c < nCols; c++) {
    const best = Math.max(...sweep.recovery.map((row) => (Number.isFinite(row[c]) ? row[c] : -Infinity)));
    const token = sweep.tokens[c] ?? "";
    if (c === 0 && token.startsWith("<|")) continue;
    if (best > 0.5) out.push({ index: c, token, best });
  }
  return out;
}

export function joinNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function sceneCopy(id: string, report: ReportResult): SceneCopy {
  const missing: SceneCopy = { headline: "This report has no data for this scene.", hint: "Open the full report for what it does contain." };
  const data = (sectionId: string) => (sectionOf(report, sectionId)?.data ?? null) as Record<string, unknown> | null;

  if (id === "answer") {
    const pred = (data("answer")?.prediction ?? null) as Record<string, unknown> | null;
    if (!pred || typeof pred.top_token !== "string") return missing;
    return { headline: `GPT-2 picks ${quote(pred.top_token)}.`, hint: "Each dot is a 1% chance. The dots that are not coloured belong to other words." };
  }
  if (id === "where") {
    const jump = ((data("where_formed")?.lens ?? {}) as { max_jump?: { stage?: string; D_before?: number; D_after?: number } }).max_jump;
    if (!jump?.stage) return missing;
    const a = num(jump.D_before), b = num(jump.D_after);
    return {
      headline: `The answer takes shape at ${jump.stage}.`,
      hint: a != null && b != null ? `The gap between the two candidates jumps from ${a.toFixed(1)} to ${b.toFixed(1)} there.` : "That is where the gap between the two candidates grows most.",
    };
  }
  if (id === "words") {
    const sweep = (data("which_words")?.sweep ?? undefined) as { recovery?: number[][]; tokens?: string[] } | undefined;
    if (!sweep?.recovery?.length) return missing;
    const hot = hotWords(sweep);
    return {
      headline: hot.length ? `The answer leans on ${joinNames(hot.map((w) => quote(w.token)))}.` : "No single word restores most of the answer.",
      hint: "Brighter word: putting it back restores more of the answer.",
    };
  }
  if (id === "parts") {
    const rec = (data("which_parts")?.top_recovery ?? []) as { label: string; recovery: number }[];
    if (rec.length === 0) return missing;
    const a = rec[0];
    return {
      headline: rec.length > 1 ? `Heads ${a.label} and ${rec[1].label} matter most.` : `Head ${a.label} matters most.`,
      hint: `Patching head ${a.label} alone restores ${Math.round(a.recovery * 100)}% of the gap.`,
    };
  }
  if (id === "methods") {
    const stats = (data("methods_agree")?.stats ?? {}) as Record<string, { pearson_heads?: number | null }>;
    if (!stats.atp && !stats.dla) return missing;
    return {
      headline: `The fast method agrees ${agreementWord(stats.atp?.pearson_heads)} with the exact test.`,
      hint: `Direct attribution agrees ${agreementWord(stats.dla?.pearson_heads)}.`,
    };
  }
  if (id === "check") {
    return { headline: "You can check every number.", hint: "Run the script to recompute them with independent code." };
  }
  return missing;
}
