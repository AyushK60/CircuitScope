import { test, expect } from "@playwright/test";

// The complete guided journey against the real backend: landing prediction →
// explorer inspection → intervention experiment → measured comparison.

test.describe("CircuitScope core journey", () => {
  let consoleErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    consoleErrors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(String(err)));
  });

  test("landing shows live model status and real prediction", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("status")).toContainText("GPT-2 ready", { timeout: 30_000 });
    // auto demo runs when model is ready
    await expect(page.getByText("Tokens the model actually receives")).toBeVisible();
    await expect(page.getByText("<|endoftext|>").first()).toBeVisible();
    // measured probability appears (Mary is top for the preset example)
    await expect(page.getByText("66.2").first()).toBeVisible({ timeout: 30_000 });
  });

  test("explorer: model → block → head heatmap with real shapes", async ({ page }) => {
    await page.goto("/#/explorer");
    await expect(page.getByText("gpt2 · 12 blocks · 12 heads")).toBeVisible({ timeout: 30_000 });
    // expand block 5
    await page.getByRole("button", { name: "Transformer Block 5" }).click();
    await expect(page.getByText("Computation order (verified pre-LN)")).toBeVisible();
    // open head 4
    await page.getByRole("button", { name: "Head 4", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Attention head 4 of block 5" })).toBeVisible();
    // heatmap provenance from the real hook
    await expect(page.getByText("blocks.5.attn.hook_pattern").first()).toBeVisible();
    await expect(page.getByText(/12 × 14 × 14/)).toBeVisible();
    // inspector shows token positions and scale honestly
    await expect(page.getByText("softmax attention weights")).toBeVisible();
  });

  test("playground: full intervention experiment produces measured comparison", async ({ page }) => {
    await page.goto("/#/playground");
    await page.getByRole("button", { name: /Run experiment/ }).click();
    // three real model runs on CPU take a few seconds
    await expect(page.getByRole("heading", { name: "2 · Measured effect" })).toBeVisible({
      timeout: 60_000,
    });
    // metrics consistent with verified reference: D_patched ≈ 1.943, Recovery ≈ 0.915
    await expect(page.getByText("1.94", { exact: false }).first()).toBeVisible();
    await expect(page.getByText("0.91", { exact: false }).first()).toBeVisible();
    // all three runs show the real candidates
    await expect(page.getByRole("heading", { name: /Clean run/ })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Corrupted run/ })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Patched run/ })).toBeVisible();
    await expect(page.getByText("blocks.5.hook_out").first()).toBeVisible();
    // exports exist
    await expect(page.getByRole("link", { name: "Export JSON" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Export CSV" })).toBeVisible();
  });

  test("learn page renders all topics at three depths", async ({ page }) => {
    await page.goto("/#/learn");
    await expect(page.getByRole("heading", { name: "Attention head", exact: false })).toBeVisible();
    await page.getByRole("tab", { name: "Technical" }).click();
    await expect(page.getByText("hook_pattern stores A per head")).toBeVisible();
  });

  test("mobile viewport has no horizontal overflow", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 690 });
    await page.goto("/");
    await expect(page.getByRole("status")).toContainText("GPT-2 ready", { timeout: 30_000 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(2);
    // explorer (fixed-width SVG diagrams) must also fit
    await page.goto("/#/explorer");
    await expect(page.getByText("Measured residual stream (data view)")).toBeVisible({
      timeout: 30_000,
    });
    const overflowExplorer = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflowExplorer).toBeLessThanOrEqual(2);
  });

  test("flow playback: guided replay with recorded stages and honest timing", async ({ page }) => {
    await page.goto("/#/explorer");
    await page.getByRole("tab", { name: "Computation Flow" }).click();
    await page.getByRole("button", { name: "▶ Play" }).click();
    // playback advances through real block stages showing measured heatmaps
    await expect(page.getByRole("heading", { name: /Transformer block/ })).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText(/blocks\.0\.attn\.hook_pattern/).first()).toBeVisible();
    // honest timing label is always present
    await expect(page.getByText(/not measured GPU time/)).toBeVisible();
    // step controls work
    await page.getByRole("button", { name: "Step forward" }).click();
    // jumping to the final stage shows the completed-run state
    await page.getByRole("tab", { name: "logits" }).click();
    await expect(page.getByText(/Completed run/)).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("heading", { name: "Stage · Output scores" })).toBeVisible();
  });

  test("anatomy data view: measured residual norms across all blocks", async ({ page }) => {
    await page.goto("/#/explorer");
    await expect(page.getByText("Measured residual stream (data view)")).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.getByText("MEASURED", { exact: true })).toBeVisible();
    for (let l = 0; l < 12; l++) {
      await expect(page.getByRole("button", { name: new RegExp(`Block ${l}: mean residual norm`) })).toBeVisible();
    }
  });

  test("no major console errors across the journey", async ({ page }) => {
    for (const route of ["/", "/#/explorer", "/#/playground", "/#/learn"]) {
      await page.goto(route);
      await page.waitForTimeout(1500);
    }
    const significant = consoleErrors.filter(
      (e) => !e.includes("favicon") && !e.includes("Download the React DevTools"),
    );
    expect(significant).toEqual([]);
  });
});
