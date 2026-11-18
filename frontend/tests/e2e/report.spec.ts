import { test, expect } from "@playwright/test";

// Stage 4 e2e: the Report page generates a real Circuit Report (no study,
// ~20-30 s on CPU: two patching sweeps + lens + DLA + prediction), renders
// every section, opens a glossary term by keyboard, and offers the Markdown
// download. All numbers come from the real backend.

test.describe("Report page", () => {
  test("generates the default report, shows all sections, glossary keyboard popover, download link", async ({
    page,
  }) => {
    test.setTimeout(240_000);
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(String(err)));

    await page.goto("/#/report");
    await expect(page.getByRole("status").first()).toContainText("GPT-2 ready", { timeout: 30_000 });

    // default task select = the first (only) task; no study attached
    await expect(page.getByLabel("Task")).toHaveValue("ioi_john_mary");
    await page.getByRole("button", { name: "Generate" }).click();

    // progress appears while the job runs; Cancel is enabled
    await expect(page.getByRole("button", { name: "Cancel" })).toBeEnabled({ timeout: 10_000 });

    // all seven section headings exist once the report completes
    for (const heading of [
      "What did the model answer?",
      "Where does the answer form?",
      "Which words matter?",
      "Which parts of the model matter?",
      "How does this compare with the published circuit?",
      "How sure can we be?",
      "What this does not show",
    ]) {
      await expect(page.getByRole("heading", { name: heading })).toBeVisible({ timeout: 200_000 });
    }

    // no study attached -> the how_sure section says so instead of inventing numbers
    await expect(page.getByText(/Not run yet: the multi-prompt evidence/)).toBeVisible();

    // a glossary term opens by keyboard and closes with Escape
    const term = page.getByRole("button", { name: "score gap" }).first();
    await term.focus();
    await page.keyboard.press("Enter");
    const pop = page.getByRole("note");
    await expect(pop).toBeVisible();
    await expect(term).toHaveAttribute("aria-describedby", /.+/);
    await page.keyboard.press("Escape");
    await expect(pop).toHaveCount(0);

    // "where does this come from?" disclosure shows evidence paths (the first
    // claim cites prediction.top_token / prediction.top_prob)
    await page.getByText("Where does this come from?").first().click();
    await expect(page.getByText("prediction.top_token").first()).toBeVisible();

    // Markdown download link exists
    await expect(page.getByText("Download Markdown")).toBeVisible();

    const significant = consoleErrors.filter(
      (e) => !e.includes("favicon") && !e.includes("Download the React DevTools"),
    );
    expect(significant).toEqual([]);
  });

  test("390 px viewport has no horizontal overflow", async ({ page }) => {
    test.setTimeout(240_000);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/#/report");
    await expect(page.getByRole("status").first()).toContainText("GPT-2 ready", { timeout: 30_000 });
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.getByRole("heading", { name: "What did the model answer?" })).toBeVisible({
      timeout: 200_000,
    });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(2);
  });
});
