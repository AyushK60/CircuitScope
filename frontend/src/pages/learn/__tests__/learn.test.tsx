import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import Learn from "../../Learn";
import { CHAPTERS, TOPICS } from "../topics";

afterEach(cleanup);

describe("Learn", () => {
  it("covers every topic once, grouped into four chapters, each with a live link", () => {
    expect(CHAPTERS).toHaveLength(4);
    expect(new Set(TOPICS.map((t) => t.id)).size).toBe(TOPICS.length);
    for (const t of TOPICS) {
      expect(t.chapter).toBeGreaterThanOrEqual(0);
      expect(t.chapter).toBeLessThan(CHAPTERS.length);
      expect(t.live.href.startsWith("#/")).toBe(true);
    }
  });

  it("shows the topics at the chosen depth", () => {
    render(<Learn />);
    expect(screen.getByRole("heading", { name: "Attention head" })).toBeTruthy();
    // default depth is "more context"
    expect(screen.getByText(/computes 'how much should I look at each earlier token/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: /Technical/ }));
    expect(screen.getByText(/hook_pattern stores A per head/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: /Simple/ }));
    expect(screen.getByText("Looking at something isn't the same as using it.")).toBeTruthy();
  });

  it("offers a see-it-live link for each topic and keeps the plain-network contrast", () => {
    render(<Learn />);
    expect(screen.getAllByText(/See it live:/)).toHaveLength(TOPICS.length);
    expect(screen.getByText(/Not GPT-2\./)).toBeTruthy();
    expect(screen.getByText("All terms (glossary)")).toBeTruthy();
  });
});
