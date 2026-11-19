import { describe, it, expect, afterEach, vi, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";
import {
  AUTO_ADVANCE_MS,
  TOUR_SCENES,
  commitMatches,
  isLastScene,
  nextScene,
  pickSavedReport,
  prevScene,
  progressPercent,
  shouldAutoAdvance,
  stripDirty,
} from "./scenes";
import Tour from "../Tour";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      tasks: vi.fn(),
      verifyStatus: vi.fn(),
      listExperiments: vi.fn(),
      getSavedExperiment: vi.fn(),
      startReport: vi.fn(),
      reportStatus: vi.fn(),
    },
  };
});

import { api } from "../../api/client";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("tour scene logic pure functions", () => {
  it("has exactly six scenes in the order the spec gives", () => {
    expect(TOUR_SCENES.map((s) => s.id)).toEqual([
      "answer",
      "where",
      "words",
      "parts",
      "methods",
      "check",
    ]);
  });

  it("auto-advances about every 14 seconds (about 90 s for six scenes)", () => {
    expect(AUTO_ADVANCE_MS).toBe(14000);
    expect(AUTO_ADVANCE_MS * 6).toBeGreaterThanOrEqual(80_000);
    expect(AUTO_ADVANCE_MS * 6).toBeLessThanOrEqual(100_000);
  });

  it("next and previous clamp at the ends", () => {
    expect(nextScene(0)).toBe(1);
    expect(nextScene(5)).toBe(5);
    expect(prevScene(0)).toBe(0);
    expect(prevScene(3)).toBe(2);
    expect(isLastScene(5)).toBe(true);
    expect(isLastScene(4)).toBe(false);
  });

  it("never auto-advances with reduced motion or when paused", () => {
    expect(shouldAutoAdvance(true, false)).toBe(false);
    expect(shouldAutoAdvance(false, true)).toBe(false);
    expect(shouldAutoAdvance(false, false)).toBe(true);
  });

  it("progress percent counts the current scene", () => {
    expect(progressPercent(0)).toBe(17);
    expect(progressPercent(5)).toBe(100);
    expect(progressPercent(0, 0)).toBe(0);
  });

  it("ignores +dirty when comparing commits and never matches empty commits", () => {
    expect(stripDirty("abc123+dirty")).toBe("abc123");
    expect(commitMatches("abc123+dirty", "abc123")).toBe(true);
    expect(commitMatches("abc123", "def456")).toBe(false);
    expect(commitMatches("", "")).toBe(false);
    expect(commitMatches(null, "abc")).toBe(false);
  });

  it("picks the newest saved report for the task at the current commit only", () => {
    const picked = pickSavedReport(
      [
        { id: "old", task_id: "ioi", generated_at: "2026-11-01", git_commit: "c1" },
        { id: "new", task_id: "ioi", generated_at: "2026-11-10", git_commit: "c1+dirty" },
        { id: "other-commit", task_id: "ioi", generated_at: "2026-11-18", git_commit: "c2" },
        { id: "other-task", task_id: "x", generated_at: "2026-11-18", git_commit: "c1" },
      ],
      "c1",
      "ioi",
    );
    expect(picked?.id).toBe("new");
    expect(pickSavedReport([], "c1", "ioi")).toBeNull();
  });
});

describe("Tour auto-advance with fake timers", () => {
  it("advances one scene per interval using the same timing rule as the page", () => {
    vi.useFakeTimers();
    let scene = 0;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const schedule = () => {
      if (isLastScene(scene) || !shouldAutoAdvance(false, false)) return;
      timers.push(setTimeout(() => {
        scene = nextScene(scene);
        schedule();
      }, AUTO_ADVANCE_MS));
    };
    schedule();
    vi.advanceTimersByTime(AUTO_ADVANCE_MS - 1);
    expect(scene).toBe(0);
    vi.advanceTimersByTime(1);
    expect(scene).toBe(1);
    vi.advanceTimersByTime(AUTO_ADVANCE_MS * 10);
    expect(scene).toBe(5);
  });
});

function savedReport(commit: string) {
  return {
    task_id: "ioi",
    generated_at: "2026-11-19T09:00:00Z",
    provenance: { git_commit: commit },
    sections: [],
  };
}

function mockBackendForSavedReport(commit: string) {
  vi.mocked(api.tasks).mockResolvedValue({ tasks: [{ id: "ioi" }] } as never);
  vi.mocked(api.verifyStatus).mockResolvedValue({
    checks: [],
    current_git_commit: commit,
  } as never);
  vi.mocked(api.listExperiments).mockResolvedValue({
    saved: [{ id: "rep1", config: { job_type: "report" } }],
  } as never);
  vi.mocked(api.getSavedExperiment).mockResolvedValue({
    results: savedReport(commit),
  } as never);
}

function setReducedMotion(matches: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })) as unknown as typeof window.matchMedia;
}

describe("Tour component", () => {
  beforeEach(() => {
    vi.mocked(api.tasks).mockReset();
    vi.mocked(api.verifyStatus).mockReset();
    vi.mocked(api.listExperiments).mockReset();
    vi.mocked(api.getSavedExperiment).mockReset();
  });

  it("uses a saved report whose commit matches and walks through six scenes with Next", async () => {
    setReducedMotion(true);
    mockBackendForSavedReport("abc123");
    render(<Tour health={{ model_loaded: true } as never} />);

    expect(await screen.findByRole("heading", { name: TOUR_SCENES[0].title })).toBeTruthy();
    for (let i = 1; i < TOUR_SCENES.length; i++) {
      fireEvent.click(screen.getByRole("button", { name: /Next/ }));
      expect(await screen.findByRole("heading", { name: TOUR_SCENES[i].title })).toBeTruthy();
    }
    expect(screen.getByRole("button", { name: "Finish" })).toBeTruthy();
  });

  it("does not auto-advance when reduced motion is requested", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    setReducedMotion(true);
    mockBackendForSavedReport("abc123");
    render(<Tour health={{ model_loaded: true } as never} />);
    await screen.findByRole("heading", { name: TOUR_SCENES[0].title });
    act(() => {
      vi.advanceTimersByTime(AUTO_ADVANCE_MS * 2);
    });
    expect(screen.getByRole("heading", { name: TOUR_SCENES[0].title })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /auto-play/ })).toBeNull();
  });

  it("auto-advances without reduced motion and pauses on request", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    setReducedMotion(false);
    mockBackendForSavedReport("abc123");
    render(<Tour health={{ model_loaded: true } as never} />);
    await screen.findByRole("heading", { name: TOUR_SCENES[0].title });
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      vi.advanceTimersByTime(AUTO_ADVANCE_MS + 100);
    });
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: TOUR_SCENES[1].title })).toBeTruthy(),
    );
    fireEvent.click(screen.getByRole("button", { name: "Pause auto-play" }));
    act(() => {
      vi.advanceTimersByTime(AUTO_ADVANCE_MS * 3);
    });
    expect(screen.getByRole("heading", { name: TOUR_SCENES[1].title })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Resume auto-play" })).toBeTruthy();
  });

  it("shows the honest error state instead of placeholder numbers when the backend fails", async () => {
    setReducedMotion(true);
    vi.mocked(api.tasks).mockRejectedValue(new Error("backend unreachable"));
    vi.mocked(api.verifyStatus).mockResolvedValue({ checks: [], current_git_commit: null } as never);
    render(<Tour health={null} />);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("backend unreachable");
    expect(screen.queryByRole("heading", { name: TOUR_SCENES[0].title })).toBeNull();
  });
});
