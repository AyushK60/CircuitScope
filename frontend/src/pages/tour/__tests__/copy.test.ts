import { describe, expect, it } from "vitest";
import type { ReportResult } from "../../../api/client";
import { agreementWord, hotWords, joinNames, sceneCopy } from "../copy";

const report = {
  sections: [
    { id: "answer", data: { prediction: { top_token: " Mary", prob_a: 0.66, prob_b: 0.05 } } },
    { id: "where_formed", data: { lens: { max_jump: { stage: "block 9", D_before: 3.49, D_after: 10.38 } } } },
    {
      id: "which_words",
      data: {
        sweep: {
          tokens: ["<|endoftext|>", "John", " and", " to"],
          recovery: [[0, 1.0, 0.1, 0.0], [0, 0.2, 0.0, 0.97]],
        },
      },
    },
    { id: "which_parts", data: { top_recovery: [{ label: "8.10", recovery: 0.297 }, { label: "9.9", recovery: 0.24 }] } },
    { id: "methods_agree", data: { stats: { atp: { pearson_heads: 0.98 }, dla: { pearson_heads: 0.63 } } } },
  ],
} as unknown as ReportResult;

describe("tour scene copy", () => {
  it("states the answer from the measured top token", () => {
    expect(sceneCopy("answer", report).headline).toBe("GPT-2 picks “Mary”.");
  });

  it("names the block where the gap grows most, with the real numbers", () => {
    const c = sceneCopy("where", report);
    expect(c.headline).toBe("The answer takes shape at block 9.");
    expect(c.hint).toContain("3.5 to 10.4");
  });

  it("names the words whose best restoration is above one half and skips the start token", () => {
    expect(hotWords((report.sections[2].data as { sweep: never }).sweep).map((w) => w.token)).toEqual(["John", " to"]);
    expect(sceneCopy("words", report).headline).toBe("The answer leans on “John” and “to”.");
  });

  it("names the top heads and their recovery", () => {
    const c = sceneCopy("parts", report);
    expect(c.headline).toBe("Heads 8.10 and 9.9 matter most.");
    expect(c.hint).toContain("30%");
  });

  it("describes agreement from the Pearson number", () => {
    expect(agreementWord(0.98)).toBe("closely");
    expect(agreementWord(0.63)).toBe("roughly");
    expect(agreementWord(0.2)).toBe("only weakly");
    expect(sceneCopy("methods", report).headline).toContain("closely");
    expect(sceneCopy("methods", report).hint).toContain("roughly");
  });

  it("says so when the report has no data for a scene", () => {
    expect(sceneCopy("where", { sections: [] } as unknown as ReportResult).headline).toContain("no data");
  });

  it("joins names in plain English", () => {
    expect(joinNames(["a"])).toBe("a");
    expect(joinNames(["a", "b", "c"])).toBe("a, b and c");
  });
});
