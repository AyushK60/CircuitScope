/** Plain-language readings of an experiment, computed from the measured numbers
 *  (never hand-written claims about a result). */

/** "blocks.5.hook_out" → "block 5 (residual stream)" */
export function describeHook(hook: string): string {
  const m = hook.match(/blocks\.(\d+)\./);
  const n = m ? m[1] : "?";
  if (hook.includes(".mlp.")) return `block ${n} (MLP output)`;
  if (hook.endsWith("attn.hook_z")) return `block ${n} (attention z)`;
  if (hook.includes(".attn.")) return `block ${n} (attention output)`;
  return `block ${n} (residual stream)`;
}

/** The block number when the hook is a residual-stream site, else null. */
export function residualBlock(hook: string): number | null {
  const m = hook.match(/^blocks\.(\d+)\.hook_out$/);
  return m ? Number(m[1]) : null;
}

export function recoveryPercent(r: number | null | undefined): string {
  return r == null ? "n/a" : `${(r * 100).toFixed(1)}%`;
}

/** How to read a recovery value, in words. */
export function recoveryVerb(r: number | null | undefined): string {
  if (r == null) return "has no defined recovery here, because the clean and corrupted runs scored the same";
  if (Math.abs(r) < 0.05) return "had almost no measured effect";
  if (r < 0) return "pushed the answer further from the clean one";
  if (r > 0.8) return "brings back most of the answer";
  if (r > 0.4) return "brings back part of the answer";
  return "brings back a little of the answer";
}

export function headline(hook: string, token: string, r: number | null | undefined): string {
  const word = token.trim() || token;
  return `Patching ${describeHook(hook)} at “${word}” ${recoveryVerb(r)}.`;
}

/** Measured recoveries from this session, keyed `${hook}@${position}`. */
export type Tried = Record<string, number | null>;
export const triedKey = (hook: string, position: number) => `${hook}@${position}`;
