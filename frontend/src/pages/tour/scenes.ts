import type { ReportResult, ReportSection } from "../../api/client";

/** Pure tour logic, kept separate so Vitest can drive it with fake timers. */

export function sectionOf(report: ReportResult, id: string): ReportSection | undefined {
  return report.sections.find((s) => s.id === id);
}

export const AUTO_ADVANCE_MS = 14000;

export interface TourScene {
  id: string;
  title: string;
  sectionId: string;
  visual: "answer" | "lens" | "words" | "parts" | "methods" | "check";
}

export const TOUR_SCENES: TourScene[] = [
  { id: "answer", title: "The answer", sectionId: "answer", visual: "answer" },
  { id: "where", title: "Where it forms", sectionId: "where_formed", visual: "lens" },
  { id: "words", title: "Which words matter", sectionId: "which_words", visual: "words" },
  { id: "parts", title: "Which parts matter", sectionId: "which_parts", visual: "parts" },
  { id: "methods", title: "Do the methods agree?", sectionId: "methods_agree", visual: "methods" },
  { id: "check", title: "How to check it", sectionId: "limits", visual: "check" },
];

export function nextScene(index: number, total = TOUR_SCENES.length): number {
  return Math.min(index + 1, total - 1);
}

export function prevScene(index: number): number {
  return Math.max(index - 1, 0);
}

export function isLastScene(index: number, total = TOUR_SCENES.length): boolean {
  return index >= total - 1;
}

/** Auto-play is off when the user asks for reduced motion, or pauses. */
export function shouldAutoAdvance(reducedMotion: boolean, paused: boolean): boolean {
  return !reducedMotion && !paused;
}

/** Provenance commits may carry a "+dirty" suffix; compare the commits only. */
export function stripDirty(commit: string | null | undefined): string {
  if (!commit) return "";
  return commit.replace(/\+dirty$/i, "").trim();
}

export function commitMatches(
  savedCommit: string | null | undefined,
  currentCommit: string | null | undefined,
): boolean {
  const a = stripDirty(savedCommit);
  const b = stripDirty(currentCommit);
  return a.length > 0 && a === b;
}

export interface SavedReportCandidate {
  id: string;
  task_id: string | null;
  generated_at: string | null;
  git_commit: string | null;
}

/**
 * Pick the newest saved report for the task whose provenance commit matches
 * the current commit (ignoring "+dirty"). Returns null when none matches, so
 * the tour can honestly generate a fresh report instead.
 */
export function pickSavedReport(
  candidates: SavedReportCandidate[],
  currentCommit: string | null,
  taskId: string,
): SavedReportCandidate | null {
  const matching = candidates
    .filter((c) => (c.task_id == null || c.task_id === taskId) && commitMatches(c.git_commit, currentCommit))
    .sort((a, b) => String(b.generated_at ?? "").localeCompare(String(a.generated_at ?? "")));
  return matching[0] ?? null;
}

export function progressPercent(index: number, total = TOUR_SCENES.length): number {
  return total <= 0 ? 0 : Math.round(((index + 1) / total) * 100);
}
