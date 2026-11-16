export type HoverFn = (h: { x: number; y: number; title: string; lines: string[] } | null) => void;

export function hoverFromEvent(e: React.MouseEvent, title: string, lines: string[]): { x: number; y: number; title: string; lines: string[] } {
  const rect = (e.currentTarget as Element).closest(".canvas-shell")?.getBoundingClientRect();
  return { x: (e.clientX - (rect?.left ?? 0)) + 14, y: (e.clientY - (rect?.top ?? 0)) + 14, title, lines };
}
