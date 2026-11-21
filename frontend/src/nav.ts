export type Route =
  | "landing"
  | "explorer"
  | "playground"
  | "learn"
  | "lab"
  | "sweeps"
  | "study"
  | "report"
  | "verify"
  | "tour";

export const NAV_LABELS: Record<Exclude<Route, "landing">, string> = {
  explorer: "Model Explorer",
  lab: "NN Lab",
  learn: "Learn",
  tour: "Tour",
  playground: "Playground",
  sweeps: "Sweeps",
  study: "Study",
  report: "Report",
  verify: "Verify",
};

/** Shorter names for the narrow phone tab bar. */
export const TAB_LABELS: Partial<Record<Exclude<Route, "landing">, string>> = { explorer: "Explorer" };

/** The nine pages, grouped by what a visitor is doing. */
export const NAV_GROUPS: { name: string; routes: Exclude<Route, "landing">[] }[] = [
  { name: "Understand", routes: ["explorer", "lab", "learn", "tour"] },
  { name: "Experiment", routes: ["playground", "sweeps", "study"] },
  { name: "Evidence", routes: ["report", "verify"] },
];

/** Pages that get a tab on phones; the rest live under "More". */
export const TAB_ROUTES: Exclude<Route, "landing">[] = ["explorer", "playground", "sweeps", "report"];

export const ROUTE_NAMES: Route[] = NAV_GROUPS.flatMap((g) => g.routes);
