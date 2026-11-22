import { describe, expect, it } from "vitest";
import { pointedness } from "./attention";

describe("pointedness", () => {
  it("is 1 when every row puts all its weight on one token", () => {
    expect(pointedness([[1, 0], [0, 1]])).toBe(1);
  });
  it("is the mean of each row's largest weight", () => {
    expect(pointedness([[0.5, 0.5], [0.8, 0.2]])).toBeCloseTo(0.65);
  });
  it("is 0 for an empty pattern", () => {
    expect(pointedness([])).toBe(0);
  });
});
