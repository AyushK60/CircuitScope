import { test, expect, type Page } from "@playwright/test";

// Stage 5 e2e: the verification panel runs the fast dla_lens check to a pass,
// the agreement section measures and draws real scatter plots, the guided tour
// walks six scenes, a Reproduce download arrives as a .py file, and the new
// pages do not overflow horizontally at 390 px. All numbers come from the
// real backend (GPT-2 on CPU); nothing here asserts a hardcoded result value.

function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  return errors;
}

function significant(errors: string[]): string[] {
  return errors.filter((e) => !e.includes("favicon") && !e.includes("Download the React DevTools"));
}

test.describe("Verify page", () => {
  test("runs the fast dla_lens check to a pass and shows the tolerance and commit", async ({ page }) => {
    test.setTimeout(240_000);
    const errors = trackErrors(page);
    await page.goto("/#/verify");
    await expect(page.getByRole("heading", { name: "Verify", exact: true })).toBeVisible();

    const card = page.locator("section.verify-card").filter({
      has: page.getByRole("heading", { name: "Direct logit attribution and the logit lens" }),
    });
    await expect(card).toBeVisible();
    await card.getByRole("button", { name: "Run this check" }).click();

    // progress and cancel appear while the reference script runs
    await expect(card.getByRole("button", { name: "Cancel" })).toBeVisible({ timeout: 10_000 });

    // a pass: colour-independent word and icon, max difference, commit shown
    await expect(card.getByRole("status", { name: "Status: Pass" })).toBeVisible({ timeout: 200_000 });
    await expect(card.getByText("Max |app − reference|")).toBeVisible();
    await expect(card.getByText("Verified at commit")).toBeVisible();
    await expect(card.locator("p.inspector-meta").filter({ hasText: "tolerance" })).toBeVisible();

    // the caveat that a pass does not validate the science is on the page
    await expect(page.getByText(/does not validate the scientific interpretation/)).toBeVisible();

    expect(significant(errors)).toEqual([]);
  });

  test("has no horizontal page overflow at 390 px", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/#/verify");
    await expect(page.getByRole("heading", { name: "Verify", exact: true })).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });
});

test.describe("Sweeps page: agreement section", () => {
  test("runs the agreement measurement and draws both scatter plots with the statistics", async ({ page }) => {
    test.setTimeout(240_000);
    const errors = trackErrors(page);
    await page.goto("/#/sweeps");
    await expect(page.getByRole("status").first()).toContainText("GPT-2 ready", { timeout: 30_000 });

    const section = page.locator("section[aria-labelledby='agreement-heading']");
    await expect(section.getByRole("heading", { name: "Do the three methods agree?" })).toBeVisible();
    // The previous check's heavy-job slot can still be releasing for a moment,
    // so a click may get the honest "already running" 422. Retry the click until accepted.
    const scatters = section.getByRole("img", { name: /attention heads \(circles\) and .* MLPs \(squares\)/ });
    await expect(async () => {
      await section.getByRole("button", { name: "Run agreement measurement" }).click();
      await expect(scatters).toHaveCount(2, { timeout: 15_000 });
    }).toPass({ timeout: 200_000 });
    await expect(section.getByRole("button", { name: "Run agreement measurement" })).toBeEnabled();

    // numbers table toggles and is accessible
    await section.getByRole("button", { name: "Show numbers table" }).first().click();
    await expect(section.getByRole("button", { name: "Hide numbers table" }).first()).toBeVisible();

    // statistics caption is present and the fixed explanatory caveat shows
    await expect(section.getByText(/Agreement statistics over the 144 heads/)).toBeVisible();

    // The browser logs each rejected (busy-slot, HTTP 422) click as a resource error. Those
    // are the intended retry path above, so only that status is excused here.
    expect(
      significant(errors).filter((e) => !e.includes("status of 422")),
    ).toEqual([]);
  });
});

test.describe("Guided tour", () => {
  test("walks through six scenes with Next and finishes", async ({ page }) => {
    test.setTimeout(300_000);
    const errors = trackErrors(page);
    // prefers-reduced-motion keeps the walk manual, so the test is not timing-dependent
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/#/tour");

    const titles = [
      "The answer",
      "Where it forms",
      "Which words matter",
      "Which parts matter",
      "Do the methods agree?",
      "How to check it",
    ];
    await expect(page.getByRole("heading", { name: titles[0] })).toBeVisible({ timeout: 200_000 });
    for (let i = 1; i < titles.length; i++) {
      await page.getByRole("button", { name: "Next →" }).click();
      await expect(page.getByRole("heading", { name: titles[i] })).toBeVisible();
      await expect(page.getByText(`Scene ${i + 1} of 6`)).toBeVisible();
    }
    await expect(page.getByRole("button", { name: "Finish" })).toBeVisible();

    // the last scene shows live verification status or "never run", never invented numbers
    const liveStatus = page.getByRole("heading", { name: "Live verification status" }).locator("xpath=following-sibling::*[1]");
    await expect(liveStatus.getByText(/(pass|fail|never run|stale|manual)/i).first()).toBeVisible();
    expect(significant(errors)).toEqual([]);
  });

  test("has no horizontal page overflow at 390 px", async ({ page }) => {
    test.setTimeout(300_000);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/#/tour");
    await expect(page.getByRole("heading", { name: "The answer" })).toBeVisible({ timeout: 200_000 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });
});

test.describe("Reproduce in Python", () => {
  test("a sweep result offers a download of a standalone .py script", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/#/sweeps");
    await expect(page.getByRole("status").first()).toContainText("GPT-2 ready", { timeout: 30_000 });
    await page.getByRole("button", { name: "Run sweep" }).click();
    await expect(page.getByRole("img", { name: /Patching sweep heatmap/ })).toBeVisible({ timeout: 120_000 });

    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Reproduce in Python" }).first().click();
    const file = await download;
    expect(file.suggestedFilename()).toMatch(/^circuitscope_repro_sweep_.*\.py$/);
  });
});
