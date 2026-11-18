import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import LensSparkline from "../LensSparkline";
import { EvidenceDisclosure, type ReportClaim } from "../Claims";
import { loadLevel, saveLevel, LEVEL_STORAGE_KEY } from "../../../lib/level";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

const STAGES = [
  { stage: "embeddings", D: -1.2 },
  { stage: "block 0", D: -0.9 },
  { stage: "block 5", D: 0.8 },
  { stage: "block 7", D: 2.3 },
  { stage: "block 11", D: 2.51 },
];

describe("LensSparkline", () => {
  it("renders an svg trace of the score gap across stages", () => {
    render(<LensSparkline stages={STAGES} />);
    const img = screen.getByRole("img", { name: /Score gap D across 5 lens stages/ });
    expect(img).toBeTruthy();
    expect(img.textContent).toContain("-1.20");
    expect(img.textContent).toContain("2.51");
  });

  it("says so when there is not enough data (no invented points)", () => {
    render(<LensSparkline stages={[{ stage: "embeddings", D: null }]} />);
    expect(screen.getByRole("status").textContent).toMatch(/Not enough lens stages/);
  });
});

const CLAIM: ReportClaim = {
  id: "answer.gap",
  level: "curious",
  text: 'The score gap is 2.51. A positive gap means the model prefers " Mary".',
  claim_type: "measured",
  evidence: ["prediction.D", "prediction.candidate_a", "fixed:answer.note"],
};

describe("EvidenceDisclosure (where does this come from?)", () => {
  it("resolves cited numbers from the section data", () => {
    render(
      <EvidenceDisclosure
        claim={CLAIM}
        data={{ prediction: { D: 2.5128, candidate_a: " Mary" } }}
      />,
    );
    fireEvent.click(screen.getByText("Where does this come from?"));
    const list = screen.getByRole("list");
    expect(list.textContent).toContain("prediction.D");
    expect(list.textContent).toContain("2.5128");
    expect(list.textContent).toContain(" Mary");
    // fixed notes are labelled, never resolved as data
    expect(list.textContent).toContain("fixed:answer.note");
    expect(list.textContent).toContain("(fixed note)");
  });

  it("marks paths that are absent from the stored data", () => {
    render(<EvidenceDisclosure claim={CLAIM} data={{}} />);
    fireEvent.click(screen.getByText("Where does this come from?"));
    expect(screen.getByRole("list").textContent).toContain("not present in stored data");
  });
});

describe("reading level persistence", () => {
  it("saves and restores the level", () => {
    saveLevel("student");
    expect(loadLevel()).toBe("student");
    saveLevel("curious");
    expect(loadLevel()).toBe("curious");
  });

  it("defaults to curious for a corrupt stored value", () => {
    window.localStorage.setItem(LEVEL_STORAGE_KEY, "poetic");
    expect(loadLevel()).toBe("curious");
  });

  it("does not crash when localStorage throws (privacy mode)", () => {
    const throwing = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    } as unknown as Storage;
    vi.stubGlobal("localStorage", throwing);
    expect(loadLevel()).toBe("curious");
    expect(() => saveLevel("expert")).not.toThrow();
  });
});
