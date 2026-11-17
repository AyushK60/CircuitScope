import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import SweepHeatmap from "../SweepHeatmap";
import { isInterventionRecord, interventionsOnly } from "../../pages/playground/history";
import type { SavedExperiment } from "../../api/client";

afterEach(cleanup);

const recovery = [
  [0.1, -0.2, 1.0],
  [0.0, 0.9148, -0.05],
];
const dPatched = [
  [0.7, -1.3, 6.69],
  [0.0, 5.85, -0.33],
];

describe("SweepHeatmap", () => {
  it("renders the diverging legend and axis labels", () => {
    render(
      <SweepHeatmap
        rowLabels={["block 0", "block 1"]}
        colLabels={["p0", "p1", "p2"]}
        recovery={recovery}
        dPatched={dPatched}
        direction="clean -> corrupted (denoising)"
      />,
    );
    const legend = screen.getByRole("img", { name: /Diverging Recovery scale centred at 0/ });
    expect(legend).toBeTruthy();
    expect(screen.getByText(/Recovery \(0 = no effect, 1 = fully restores clean\)/)).toBeTruthy();
    expect(screen.getByText("direction: clean -> corrupted (denoising)")).toBeTruthy();
    // svg row labels are present
    expect(screen.getByRole("img", { name: /2 rows \(blocks\) by 3 columns/ })).toBeTruthy();
  });

  it("toggle renders the accessible numbers table with the same values", () => {
    render(
      <SweepHeatmap
        rowLabels={["block 0", "block 1"]}
        colLabels={["p0", "p1", "p2"]}
        recovery={recovery}
        dPatched={dPatched}
      />,
    );
    expect(screen.queryByRole("table")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show numbers" }));
    const table = screen.getByRole("table");
    expect(table).toBeTruthy();
    // the 1.0 cell reads 1.000 (e2e relies on this)
    expect(screen.getByText("1.000")).toBeTruthy();
    expect(screen.getByText("0.915")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show heatmap" }));
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("hover tooltip shows exact Recovery and D_patched from props", () => {
    render(
      <SweepHeatmap
        rowLabels={["block 0", "block 1"]}
        colLabels={["p0", "p1", "p2"]}
        recovery={recovery}
        dPatched={dPatched}
      />,
    );
    const cell = screen.getByRole("gridcell", { name: /block 1 \/ p1/ });
    fireEvent.mouseEnter(cell);
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("block 1 / p1");
    expect(status.textContent).toContain("0.9148"); // exact Recovery, not rounded to 3dp
    expect(status.textContent).toContain("5.8500"); // exact D_patched
    fireEvent.mouseLeave(cell);
    expect(screen.queryByRole("status")).toBeNull();
  });
});

describe("history compatibility (sweep records must not break the Playground)", () => {
  const base = {
    id: "abc123",
    created_at: 0,
    status: "complete" as const,
    error: null,
  };
  const interventionRecord: SavedExperiment = {
    ...base,
    config: { clean_text: "a", hook: "blocks.5.hook_out", positions: [9] },
  };
  const sweepRecord: SavedExperiment = {
    ...base,
    config: { task_id: "ioi_john_mary", mode: "layer_position", job_type: "sweep" },
  };

  it("a record is an intervention iff config.job_type is absent or 'intervention'", () => {
    expect(isInterventionRecord(interventionRecord)).toBe(true);
    expect(isInterventionRecord({ ...interventionRecord, config: { job_type: "intervention" } })).toBe(true);
    expect(isInterventionRecord(sweepRecord)).toBe(false);
    expect(isInterventionRecord({ ...sweepRecord, config: { job_type: "something_new" } })).toBe(false);
  });

  it("interventionsOnly filters sweeps out and keeps legacy records", () => {
    const kept = interventionsOnly([interventionRecord, sweepRecord, { ...interventionRecord, status: "failed" as const }]);
    expect(kept.map((r) => r.id)).toEqual(["abc123"]);
  });
});
