import { describe, it, expect, afterEach, vi, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import ReproduceButton from "../ReproduceButton";
import AgreementScatter, { type ScatterPoint } from "../AgreementScatter";
import VerificationCard from "../../pages/verify/VerificationCard";
import type { VerifyCheckEntry } from "../../api/client";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      reproduce: vi.fn(),
    },
  };
});

import { api } from "../../api/client";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ReproduceButton", () => {
  beforeEach(() => {
    vi.mocked(api.reproduce).mockReset();
  });

  it("states what the script checks and does not check", () => {
    render(<ReproduceButton spec={{ kind: "dla", mode: "sample" }} runtime="about 30 s" />);
    expect(screen.getByRole("button", { name: "Reproduce in Python" })).toBeTruthy();
    expect(screen.getByText("about 30 s")).toBeTruthy();
    expect(screen.getByText(/does not check the scientific interpretation/)).toBeTruthy();
  });

  it("downloads the script through the API and announces the filename", async () => {
    const blob = new Blob(["print('x')"], { type: "text/x-python" });
    vi.mocked(api.reproduce).mockResolvedValue({ blob, filename: "circuitscope_repro_dla_1.py" });
    const createObjectURL = vi.fn(() => "blob:fake");
    const revokeObjectURL = vi.fn();
    Object.assign(URL, { createObjectURL, revokeObjectURL });

    render(<ReproduceButton spec={{ kind: "dla", mode: "sample" }} />);
    fireEvent.click(screen.getByRole("button", { name: "Reproduce in Python" }));

    await waitFor(() =>
      expect(screen.getByRole("status").textContent).toContain("circuitscope_repro_dla_1.py"),
    );
    expect(api.reproduce).toHaveBeenCalledWith({ kind: "dla", mode: "sample" });
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:fake");
  });

  it("shows a visible error when the server refuses the request", async () => {
    vi.mocked(api.reproduce).mockRejectedValue(new Error("422: unknown kind"));
    render(<ReproduceButton spec={{ kind: "intervention", mode: "sample" }} />);
    fireEvent.click(screen.getByRole("button", { name: "Reproduce in Python" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("422: unknown kind");
  });

  it("copies the script text to the clipboard", async () => {
    const blob = new Blob(["import json\n"], { type: "text/x-python" });
    vi.mocked(api.reproduce).mockResolvedValue({ blob, filename: "f.py" });
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    render(<ReproduceButton spec={{ kind: "sweep", mode: "full" }} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy script" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("import json\n"));
  });
});

const points: ScatterPoint[] = [
  { label: "L0 H1", kind: "head", layer: 0, head: 1, x: 0.1, y: 0.12 },
  { label: "L5 H3", kind: "head", layer: 5, head: 3, x: 0.8, y: 0.7 },
  { label: "L9 MLP", kind: "mlp", layer: 9, head: null, x: -0.2, y: -0.1 },
];

describe("AgreementScatter", () => {
  it("draws one marker per point with shape-based encoding (circles heads, squares MLPs)", () => {
    const { container } = render(
      <AgreementScatter points={points} title="AtP vs exact" xLabel="exact" yLabel="AtP" />,
    );
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("aria-label")).toContain("2 attention heads (circles) and 1 MLPs (squares)");
    expect(container.querySelectorAll("circle").length).toBe(2);
    expect(container.querySelectorAll("rect").length).toBeGreaterThanOrEqual(1);
  });

  it("toggles an accessible numbers table", () => {
    render(<AgreementScatter points={points} title="AtP vs exact" xLabel="exact" yLabel="AtP" />);
    const toggle = screen.getByRole("button", { name: "Show numbers table" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(toggle);
    const hide = screen.getByRole("button", { name: "Hide numbers table" });
    expect(hide.getAttribute("aria-expanded")).toBe("true");
  });

  it("shows an empty state with no points", () => {
    render(<AgreementScatter points={[]} title="t" xLabel="x" yLabel="y" />);
    expect(screen.getByText("No scatter data yet.")).toBeTruthy();
  });
});

function entry(overrides: Partial<VerifyCheckEntry>): VerifyCheckEntry {
  return {
    check_id: "dla_lens",
    title: "DLA and logit lens",
    what: "Recomputes the DLA terms.",
    tolerance: 2e-3,
    runtime: "about 20 s",
    manual: false,
    script: "scripts/reference_dla_lens.py",
    current_script_sha256: "abc",
    status: "never run",
    ok: null,
    stale: false,
    result: null,
    ...overrides,
  };
}

const passResult = {
  check_id: "dla_lens",
  ok: true,
  exit_code: 0,
  max_diff: 1.5e-6,
  tolerance: 2e-3,
  elapsed_seconds: 12.3,
  script_sha256: "abc",
  git_commit: "deadbee",
  model_revision: "rev",
  stdout_tail: "max |api - reference| = 1.5e-06",
  checked_at: "2026-11-19T10:00:00Z",
  error: null,
} as unknown as NonNullable<VerifyCheckEntry["result"]>;

function renderCard(e: VerifyCheckEntry, running = false) {
  return render(
    <VerificationCard
      entry={e}
      running={running}
      progress={0.5}
      progressMessage="running the reference script"
      onRun={() => {}}
      onCancel={() => {}}
    />,
  );
}

describe("VerificationCard statuses", () => {
  it("says 'Never run' in words", () => {
    renderCard(entry({}));
    expect(screen.getByRole("status", { name: "Status: Never run" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run this check" })).toBeTruthy();
  });

  it("says 'Pass' on the stamp, with the max difference and how far inside the limit it is", () => {
    renderCard(entry({ status: "pass", ok: true, result: passResult }));
    const status = screen.getByRole("status", { name: "Status: Pass" });
    expect(status.textContent).toBe("Pass");
    expect(screen.getByText("1.500e-6")).toBeTruthy();
    expect(screen.getByText("deadbee")).toBeTruthy();
    expect(screen.getByText("1.5e-6", { exact: false })).toBeTruthy();
    expect(screen.getByText(/inside the limit/)).toBeTruthy();
  });

  it("says 'Fail' and flags a gap over the limit", () => {
    renderCard(entry({ status: "fail", ok: false, result: { ...passResult, ok: false, max_diff: 0.5 } as never }));
    expect(screen.getByRole("status", { name: "Status: Fail" }).textContent).toBe("Fail");
    expect(screen.getByText(/over the limit/)).toBeTruthy();
  });

  it("marks a result stale when the script changed, even if it passed before", () => {
    renderCard(entry({ status: "pass", ok: true, stale: true, result: passResult }));
    expect(screen.getByRole("status", { name: "Status: Stale result" }).textContent).toBe("Stale");
    expect(screen.getByText(/script changed after this result was recorded/)).toBeTruthy();
  });

  it("explains that the manual feedforward check needs a browser", () => {
    renderCard(entry({ check_id: "feedforward", manual: true }));
    expect(screen.getByRole("status", { name: "Status: Manual check" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Run this check" })).toBeNull();
  });

  it("shows progress and a cancel control while running", () => {
    renderCard(entry({}), true);
    expect(screen.getByText(/50% - running the reference script/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
  });
});
