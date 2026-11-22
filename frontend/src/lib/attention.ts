/** Mean over query rows of the largest attention weight: 1 means every token puts all its
 *  attention on one other token, 1/n means it spreads evenly. Descriptive only. */
export function pointedness(pattern: number[][]): number {
  if (pattern.length === 0) return 0;
  return pattern.reduce((s, row) => s + Math.max(...row), 0) / pattern.length;
}
