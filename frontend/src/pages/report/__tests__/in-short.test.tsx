import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import InShort from "../InShort";
import type { ReportSection } from "../Claims";

afterEach(cleanup);

const trust = { measured: "", against: [], checks: [], warnings: [], not_shown: [] };
const claim = (id: string, level: "curious" | "student", text: string, type: "measured" | "caveat" = "measured") => ({
  id,
  level,
  text,
  claim_type: type,
  evidence: [],
});
const section = (id: string, claims: ReturnType<typeof claim>[], data: Record<string, unknown> = {}): ReportSection => ({
  id,
  title: id,
  question: "?",
  claims,
  data,
  evidence_tier: "single prompt, live",
  analogy: null,
  trust,
});

const sections: ReportSection[] = [
  section(
    "answer",
    [claim("answer.top_token", "curious", "Top word is Mary."), claim("answer.top_token", "student", "Student: Mary.")],
    { prediction: { candidate_a: " Mary", candidate_b: " John", prob_a: 0.662, prob_b: 0.054 } },
  ),
  section("which_words", [claim("words.late", "curious", "Late on, the word to matters.")], {
    sweep: { recovery: [[0.1, 0.2], [0.9, 0.3], [0.2, 0.97]] },
    late: { layer: 2 },
  }),
  section("which_parts", [claim("parts.dla_top.1", "curious", "Head 9.6 pushes hardest.")], {
    dla: { top: [{ label: "9.6", value: 1.65 }, { label: "9.9", value: 1.29 }], most_negative: { label: "10.7", value: -1.03 } },
  }),
  section("how_sure", [claim("not_run.study", "curious", "Not run yet: no study.", "caveat")]),
];

describe("InShort", () => {
  it("shows the report's own claims with a small graph for each", () => {
    render(<InShort sections={sections} level="curious" />);
    expect(screen.getByText("Top word is Mary.")).toBeInTheDocument();
    expect(screen.getByText("Head 9.6 pushes hardest.")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Mary 66\.2%, John 5\.4%/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Best single-site recovery in each of 3 blocks, highest 0\.97/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /9\.6 1\.65, 9\.9 1\.29, 10\.7 -1\.03/ })).toBeInTheDocument();
    expect(screen.getByText("1 sentence")).toBeInTheDocument();
  });

  it("follows the reading level and skips items it has no claim for", () => {
    render(<InShort sections={sections} level="student" />);
    expect(screen.getByText("Student: Mary.")).toBeInTheDocument();
    expect(screen.queryByText("Head 9.6 pushes hardest.")).toBeNull();
  });
});
