import type { KnownHeadsReference } from "../../api/client";

// Helpers shared by the Study page and its tests (kept out of the component
// file so react-refresh sees only component exports there).

export function headGroups(known: KnownHeadsReference | undefined): Record<string, string[]> {
  const map: Record<string, string[]> = {};
  for (const g of known?.groups ?? []) {
    for (const h of g.heads ?? []) {
      (map[h] ??= []).push(g.group);
    }
  }
  return map;
}

export function topHeads(vector: number[] | null, k = 10): { head: string; value: number }[] {
  if (!vector) return [];
  return vector
    .map((v, i) => ({ head: `${Math.floor(i / 12)}.${i % 12}`, value: v }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k);
}
