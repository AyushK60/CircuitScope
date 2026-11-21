// Reading-level persistence. localStorage may throw in some
// environments (privacy mode, disabled storage); the level then simply does
// not persist and defaults to curious.
import type { ReadingLevel } from "./glossary";

export const LEVEL_STORAGE_KEY = "cs.report.level";

export function loadLevel(): ReadingLevel {
  try {
    const v = window.localStorage.getItem(LEVEL_STORAGE_KEY);
    if (v === "curious" || v === "student" || v === "expert") return v;
  } catch {
    /* unavailable -> default */
  }
  return "curious";
}

export function saveLevel(level: ReadingLevel): void {
  try {
    window.localStorage.setItem(LEVEL_STORAGE_KEY, level);
  } catch {
    /* ignore: level persistence is best-effort */
  }
}
