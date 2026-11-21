import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import SiteNav from "../SiteNav";
import { NAV_GROUPS, ROUTE_NAMES, TAB_ROUTES } from "../../nav";

afterEach(cleanup);

function setup(route: Parameters<typeof SiteNav>[0]["route"] = "playground") {
  const navigate = vi.fn();
  render(<SiteNav route={route} navigate={navigate} status={{ cls: "ok", label: "GPT-2 ready · cpu" }} onRefresh={() => {}} />);
  return navigate;
}

describe("site navigation", () => {
  it("lists all nine pages once, in three groups", () => {
    expect(ROUTE_NAMES).toHaveLength(9);
    expect(new Set(ROUTE_NAMES).size).toBe(9);
    expect(NAV_GROUPS.map((g) => g.name)).toEqual(["Understand", "Experiment", "Evidence"]);
    setup();
    const main = screen.getByRole("navigation", { name: "Main" });
    expect(within(main).getAllByRole("link")).toHaveLength(9);
    expect(within(main).getAllByRole("group")).toHaveLength(3);
  });

  it("marks the current page and navigates on click", () => {
    const navigate = setup("sweeps");
    const main = screen.getByRole("navigation", { name: "Main" });
    expect(within(main).getByRole("link", { name: "Sweeps" }).getAttribute("aria-current")).toBe("page");
    expect(within(main).getByRole("link", { name: "Report" }).getAttribute("aria-current")).toBeNull();
    fireEvent.click(within(main).getByRole("link", { name: "Report" }));
    expect(navigate).toHaveBeenCalledWith("report");
  });

  it("keeps the live model status for assistive tech", () => {
    setup();
    expect(screen.getByRole("status").textContent).toContain("GPT-2 ready");
  });

  it("phone tab bar has four tabs plus a More sheet with every page, closed by Escape", () => {
    setup("tour");
    const tabs = screen.getByRole("navigation", { name: "Quick links" });
    expect(within(tabs).getAllByRole("link")).toHaveLength(TAB_ROUTES.length);
    const more = within(tabs).getByRole("button", { name: "More" });
    expect(more.className).toContain("active"); // Tour lives under More
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.click(more);
    const sheet = screen.getByRole("dialog", { name: "All pages" });
    expect(within(sheet).getAllByRole("link")).toHaveLength(9);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
