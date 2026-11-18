import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import Term from "../Term";
import { GLOSSARY } from "../../lib/glossary";

afterEach(cleanup);

describe("Term (glossary tooltip)", () => {
  it("renders a keyboard-focusable button with the display label", () => {
    render(<Term id="score_gap" />);
    const btn = screen.getByRole("button", { name: "score gap" });
    expect(btn).toBeTruthy();
    expect(btn.getAttribute("aria-expanded")).toBe("false");
  });

  it("opens on click and links the popover via aria-describedby", () => {
    render(
      <p>
        The <Term id="score_gap" /> grows.
      </p>,
    );
    const btn = screen.getByRole("button", { name: "score gap" });
    expect(screen.queryByRole("note")).toBeNull();
    fireEvent.click(btn);
    const pop = screen.getByRole("note");
    expect(pop).toBeTruthy();
    expect(pop.textContent).toContain(GLOSSARY.score_gap.curious);
    expect(btn.getAttribute("aria-describedby")).toBe(pop.id);
    expect(btn.getAttribute("aria-expanded")).toBe("true");
  });

  it("closes on Escape while focus stays usable", () => {
    render(<Term id="recovery" />);
    const btn = screen.getByRole("button", { name: "recovery" });
    fireEvent.click(btn);
    expect(screen.getByRole("note")).toBeTruthy();
    fireEvent.keyDown(btn, { key: "Escape" });
    expect(screen.queryByRole("note")).toBeNull();
    expect(btn.getAttribute("aria-expanded")).toBe("false");
  });

  it("toggles closed on a second click", () => {
    render(<Term id="patching" />);
    const btn = screen.getByRole("button", { name: "patching" });
    fireEvent.click(btn);
    expect(screen.getByRole("note")).toBeTruthy();
    fireEvent.click(btn);
    expect(screen.queryByRole("note")).toBeNull();
  });

  it("shows the requested level's definition", () => {
    render(<Term id="token" level="expert" />);
    fireEvent.click(screen.getByRole("button", { name: "token" }));
    expect(screen.getByRole("note").textContent).toContain("byte-level BPE");
  });

  it("falls back to plain text for an unknown id (defect caught by tests)", () => {
    render(<Term id="definitely_unknown" label="thing" />);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByText("thing")).toBeTruthy();
  });
});
