import type { KnownHeadsReference, Rq1Block, Rq2 } from "../../api/client";
import type { Callout } from "../sweeps/insights";
import { headGroups, topHeads } from "./heads";

/* Plain-language callouts computed from a stored study's numbers. Each states a fact that can be read
 * straight off the figure beside it; none of them interprets what a head "does". */

export const GROUP_LABEL: Record<string, string> = {
  name_movers: "name mover",
  negative_name_movers: "negative name mover",
  s_inhibition: "S-inhibition",
  induction: "induction",
  duplicate_token: "duplicate token",
  previous_token: "previous token",
  backup_name_movers: "backup name mover",
};

const f2 = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "n/a");

export function agreementCallouts(rq1: Rq1Block): Callout[] {
  const out: Callout[] = [];
  const atp = rq1.methods.atp.heads.signed.mean;
  const dla = rq1.methods.dla.heads.signed.mean;
  out.push({
    lead: "Attribution patching:",
    text: `${f2(atp)} on heads${rq1.beats_control.atp.heads ? ", above the control" : ", not above the control"}`,
  });
  out.push({
    lead: "Direct logit attribution:",
    text: `${f2(dla)} on heads${rq1.beats_control.dla.heads ? ", above the control" : ", not above the control"}`,
  });
  return out;
}

export function topHeadCallouts(vector: number[] | null, known: KnownHeadsReference | undefined): Callout[] {
  const top = topHeads(vector, 10);
  if (top.length === 0) return [];
  const groups = headGroups(known);
  const tagOf = (h: string) => groups[h]?.map((g) => GROUP_LABEL[g] ?? g).join("/");
  const lead = top[0];
  const out: Callout[] = [
    { lead: "Leader:", text: `head ${lead.head}${tagOf(lead.head) ? ` (${tagOf(lead.head)})` : ""}, ${lead.value.toFixed(3)}` },
  ];
  const counts: Record<string, number> = {};
  for (const t of top) for (const g of groups[t.head] ?? []) counts[g] = (counts[g] ?? 0) + 1;
  const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  if (best && best[1] >= 2) {
    out.push({ lead: `${GROUP_LABEL[best[0]] ?? best[0]} heads:`, text: `${best[1]} of the top 10` });
  }
  return out;
}

export function stabilityCallouts(rq2: Rq2, key: "k5" | "k10"): Callout[] {
  const names = rq2.condition_names;
  const m = rq2.topk[key].matrix;
  const k = key === "k5" ? 5 : 10;
  const label = (n: string) => n.replace(/_/g, " ");
  let most: { a: string; b: string; v: number } | null = null;
  let least: { a: string; b: string; v: number } | null = null;
  names.forEach((a, i) =>
    names.forEach((b, j) => {
      const v = m[i]?.[j];
      if (i >= j || v === null || v === undefined) return;
      if (!most || v > most.v) most = { a, b, v };
      if (!least || v < least.v) least = { a, b, v };
    }),
  );
  const out: Callout[] = [];
  const hi = most as { a: string; b: string; v: number } | null;
  const lo = least as { a: string; b: string; v: number } | null;
  if (hi) out.push({ lead: "Most alike:", text: `${label(hi.a)} and ${label(hi.b)} share ${hi.v} of ${k}` });
  if (lo && hi && lo.v < hi.v) out.push({ lead: "Least alike:", text: `${label(lo.a)} and ${label(lo.b)} share ${lo.v} of ${k}` });
  return out;
}
