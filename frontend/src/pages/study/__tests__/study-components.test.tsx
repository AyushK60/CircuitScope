import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Rq1Table from "../Rq1Table";
import OverlapMatrix from "../OverlapMatrix";
import TopHeads from "../TopHeads";
import { headGroups, topHeads } from "../heads";
import { isInterventionRecord } from "../../../pages/playground/history";
import type { Rq1Block, Rq2, KnownHeadsReference } from "../../../api/client";
import type { SavedExperiment } from "../../../api/client";

afterEach(cleanup);

const rq1Block: Rq1Block = {
  n_prompts: 40,
  prompt_ids: [],
  methods: {
    atp: {
      heads: {
        signed: { mean: 0.62, ci_low: 0.51, ci_high: 0.7, n: 40, n_invalid: 0 },
        abs: { mean: 0.64, ci_low: 0.55, ci_high: 0.72, n: 40, n_invalid: 0 },
        n_components: 144,
        low_resolution: false,
      },
      mlps: {
        signed: { mean: 0.2, ci_low: 0.1, ci_high: 0.3, n: 40, n_invalid: 0 },
        abs: { mean: 0.22, ci_low: 0.12, ci_high: 0.31, n: 40, n_invalid: 0 },
        n_components: 12,
        low_resolution: true,
      },
      all: {
        signed: { mean: 0.58, ci_low: 0.49, ci_high: 0.66, n: 40, n_invalid: 0 },
        abs: { mean: 0.6, ci_low: 0.52, ci_high: 0.68, n: 40, n_invalid: 0 },
        n_components: 156,
        low_resolution: false,
      },
    },
    dla: {
      heads: {
        signed: { mean: 0.05, ci_low: -0.05, ci_high: 0.15, n: 40, n_invalid: 0 },
        abs: { mean: 0.06, ci_low: -0.04, ci_high: 0.16, n: 40, n_invalid: 0 },
        n_components: 144,
        low_resolution: false,
      },
      mlps: {
        signed: { mean: 0.1, ci_low: 0.0, ci_high: 0.2, n: 40, n_invalid: 0 },
        abs: { mean: 0.11, ci_low: 0.01, ci_high: 0.21, n: 40, n_invalid: 0 },
        n_components: 12,
        low_resolution: true,
      },
      all: {
        signed: { mean: 0.06, ci_low: -0.04, ci_high: 0.16, n: 40, n_invalid: 0 },
        abs: { mean: 0.07, ci_low: -0.03, ci_high: 0.17, n: 40, n_invalid: 0 },
        n_components: 156,
        low_resolution: false,
      },
    },
  },
  null: {
    heads: { p2_5: -0.17, p97_5: 0.18, note: "" },
    mlps: { p2_5: -0.5, p97_5: 0.5, note: "" },
    all: { p2_5: -0.08, p97_5: 0.08, note: "" },
  },
  beats_control: {
    atp: { heads: true, mlps: false, all: true },
    dla: { heads: false, mlps: false, all: false },
  },
  beats_control_rule: "CI lower bound (signed) > null 97.5th percentile",
};

describe("Rq1Table", () => {
  it("renders one row per method x group with CI, null percentile and beats flag", () => {
    render(<Rq1Table block={rq1Block} />);
    const table = screen.getByRole("table");
    expect(table).toBeTruthy();
    // AtP heads row: mean 0.620, CI and beats yes
    expect(screen.getByText("0.620")).toBeTruthy();
    expect(screen.getByText("[0.510, 0.700]")).toBeTruthy();
    expect(screen.getByText(/Null 97.5%/i)).toBeTruthy();
    // 2 methods x 3 groups = 6 "yes/no" cells
    const yesCells = screen.getAllByText("yes");
    expect(yesCells.length).toBe(2);
    const noCells = screen.getAllByText("no");
    expect(noCells.length).toBe(4);
    // MLP group is flagged low resolution
    expect(screen.getAllByText(/low resolution/i).length).toBeGreaterThanOrEqual(2);
  });
});

const condNames = ["swap_D", "third_D"];
const rq2: Rq2 = {
  condition_names: condNames,
  n_per_condition: { swap_D: 4, third_D: 4 },
  spearman: { matrix: [[1, 0.5], [0.5, 1]], ci_low: [[1, 0.1], [0.1, 1]], ci_high: [[1, 0.9], [0.9, 1]], note: "" },
  topk: {
    k5: { matrix: [[5, 3], [3, 5]], ci_low: [[5, 1], [1, 5]], ci_high: [[5, 5], [5, 5]] },
    k10: { matrix: [[10, 7], [7, 10]], ci_low: [[10, 3], [3, 10]], ci_high: [[10, 10], [10, 10]] },
    expected_random: { k5: 25 / 144, k10: 100 / 144 },
  },
  prompt_averaged_vectors: {
    // head 9.9 = component index 9*12+9 = 117 is the top head
    swap_D: Array.from({ length: 144 }, (_, i) => (i === 117 ? 2.0 : i === 99 ? 1.5 : 0.001 * i)),
    third_D: Array.from({ length: 144 }, (_, i) => (i === 99 ? 1.4 : 0.001 * i)),
  },
};

describe("OverlapMatrix", () => {
  it("renders the matrix with the random expectation in the caption", () => {
    render(
      <OverlapMatrix data={rq2.topk.k5} k={5} expectedRandom={25 / 144} conditionNames={condNames} />,
    );
    expect(screen.getByText(/Random expectation k·k\/144 = 0.17 for k=5/)).toBeTruthy();
    expect(screen.getAllByText("swap_D").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("third_D").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("3").length).toBe(2); // off-diagonal overlap (symmetric)
  });
});

const known: KnownHeadsReference = {
  verified_against_paper: false,
  note: "transcribed from memory",
  groups: [{ group: "name_movers", heads: ["9.9"] }],
};

describe("TopHeads", () => {
  it("ranks heads by signed score and badges known groups", () => {
    render(<TopHeads rq2={rq2} known={known} />);
    expect(screen.getAllByText("swap_D").length).toBeGreaterThan(0);
    // head 9.9 is component index 9*12+9 = 117 -> top chip with value 1.500
    // (chip text is split across nodes; match via the known-group tooltip)
    const chip = screen.getByTitle("name_movers");
    expect(chip.textContent).toContain("9.9 = 2.000");
    expect(screen.getByText("name_movers")).toBeTruthy();
  });

  it("handles a condition with no usable prompts", () => {
    const empty: Rq2 = { ...rq2, prompt_averaged_vectors: { swap_D: null, third_D: null } };
    render(<TopHeads rq2={empty} known={known} />);
    expect(screen.getAllByText(/no usable prompts/i).length).toBe(2);
  });
});

describe("helpers", () => {
  it("headGroups maps heads to their group names", () => {
    expect(headGroups(known)).toEqual({ "9.9": ["name_movers"] });
    expect(headGroups(undefined)).toEqual({});
  });
  it("topHeads sorts descending and labels L.H", () => {
    const top = topHeads([0.1, 0.5, 0.3, -1, 2], 3);
    expect(top.map((t) => t.head)).toEqual(["0.4", "0.1", "0.2"]);
    expect(top[0].value).toBe(2);
  });
  it("topHeads handles null vectors", () => {
    expect(topHeads(null, 5)).toEqual([]);
  });
});

describe("history compatibility with study records", () => {
  it("a study record is NOT an intervention (filtered from the Playground panel)", () => {
    const studyRec = { status: "complete", config: { job_type: "study" } } as unknown as SavedExperiment;
    const sweepRec = { status: "complete", config: { job_type: "sweep" } } as unknown as SavedExperiment;
    const legacy = { status: "complete", config: {} } as unknown as SavedExperiment;
    expect(isInterventionRecord(studyRec)).toBe(false);
    expect(isInterventionRecord(sweepRec)).toBe(false);
    expect(isInterventionRecord(legacy)).toBe(true);
  });
});
