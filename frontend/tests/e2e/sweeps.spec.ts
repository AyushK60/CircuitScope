import { test, expect } from "@playwright/test";

// e2e: the Sweeps page runs the real layer-position patching sweep on
// the IOI task, shows the measured heatmap, and the DLA reconciliation line
// reports ok. All numbers come from the real backend.

test.describe("Sweeps page", () => {
  let consoleErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    consoleErrors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(String(err)));
  });

  test("runs the layer_position sweep, shows the heatmap, and DLA reconciles", async ({ page }) => {
    await page.goto("/#/sweeps");
    await expect(page.getByRole("status").first()).toContainText("GPT-2 ready", { timeout: 30_000 });

    // defaults: task select = ioi_john_mary (first task), mode = layer_position
    await expect(page.getByLabel("Task")).toHaveValue("ioi_john_mary");
    const modes = page.getByRole("group", { name: "Sweep mode" });
    await expect(modes.getByRole("button", { name: "layer × position" })).toHaveClass(/active/);

    // The sweep starts by itself: 168 real patched runs take several seconds on CPU and the
    // heatmap appears on completion.
    const heatmap = page.getByRole("img", { name: /Patching sweep heatmap, 12 rows \(blocks\) by 14 columns/ });
    await expect(heatmap).toBeVisible({ timeout: 120_000 });

    // "Run again" starts a second one; Cancel is available while it runs
    await page.locator("section#sw-patch").getByRole("button", { name: "Run again" }).click();
    await expect(page.getByRole("button", { name: "Cancel" }).first()).toBeEnabled({ timeout: 10_000 });
    await expect(heatmap).toBeVisible({ timeout: 120_000 });

    // baselines and the fixed patching direction are shown
    await expect(page.getByText("D_clean")).toBeVisible();
    await expect(page.getByText(/direction: clean -> corrupted \(denoising\)/)).toBeVisible();

    // "Show numbers" reveals the accessible table; the block 11 / final token
    // cell reads exactly 1.000 (Recovery invariant for the last block at the
    // final position).
    await page.getByRole("button", { name: "Show numbers" }).click();
    await expect(page.getByRole("table")).toBeVisible();
    const row11 = page.locator("table.sweep-numbers tbody tr").filter({ hasText: "block 11" });
    await expect(row11.locator("td").last()).toHaveText("1.000");

    // DLA step: runs by itself when scrolled into view; pick both prompts, each shows the reconciliation line ok
    await page.locator("section#sw-push").scrollIntoViewIfNeeded();
    await expect(page.getByText(/Components sum to/).first()).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText(/clean prompt · D =/)).toBeVisible();
    await page.getByRole("button", { name: "corrupted prompt" }).click();
    await expect(page.getByText(/corrupted prompt · D =/)).toBeVisible({ timeout: 60_000 });
    await expect(page.locator(".state-banner.ok").filter({ hasText: "Components sum to" })).toHaveCount(1, { timeout: 10_000 });
    // the direct-effect-only caveat is displayed
    await expect(page.getByText(/Direct effect only/).first()).toBeVisible();

    const significant = consoleErrors.filter(
      (e) => !e.includes("favicon") && !e.includes("Download the React DevTools"),
    );
    expect(significant).toEqual([]);
  });
});
