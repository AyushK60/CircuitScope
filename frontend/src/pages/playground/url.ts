/** Reads #/playground?hook=...&position=... so the Explorer can carry a
 *  selected component into the laboratory. */
export function readParams(): { hook?: string; position?: number } {
  const q = window.location.hash.split("?")[1];
  if (!q) return {};
  const p = new URLSearchParams(q);
  const position = p.get("position");
  return {
    hook: p.get("hook") ?? undefined,
    position: position != null && !Number.isNaN(Number(position)) ? Number(position) : undefined,
  };
}
