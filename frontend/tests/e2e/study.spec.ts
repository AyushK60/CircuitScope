import { test, expect } from "@playwright/test";

// Stage 3 e2e: the Study page runs a small REAL study (n=4, a couple of
// minutes on CPU) and renders the pre-registered tables. Explicit timeout:
// an n=4 study runs ~790 patched runs per prompt (~35 s/prompt).

test.describe("Study page", () => {
  test("runs an n=4 study and renders the RQ1/top-heads/overlap tables", async ({ page }) => {
    test.setTimeout(900_000); // 15 min: generous for CI CPUs
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(String(err)));

    await page.goto("/#/study");
    await expect(page.getByRole("status").first()).toContainText("GPT-2 ready", { timeout: 30_000 });

    // set the small study size and run
    await page.getByLabel("n_prompts (4–100)").fill("4");
    await page.getByLabel("seed").fill("21");
    await page.getByRole("button", { name: "Run study" }).click();

    // progress appears while running (cancel enabled)
    await expect(page.getByRole("button", { name: "Cancel" })).toBeEnabled({ timeout: 10_000 });

    // on completion the headline numbers appear
    await expect(page.getByText("Model accuracy")).toBeVisible({ timeout: 840_000 });
    await expect(page.getByText("Model accuracy")).toBeVisible();

    // headline: 4 prompts, exclusions reported
    await expect(page.getByText(/^Prompts$/)).toBeVisible();
    await expect(page.getByText(/excluded per condition: swap/)).toBeVisible();

    // RQ1 table renders with methods and CI header (a second RQ1 table exists
    // inside the closed <details> subset block, hence .first())
    await expect(page.getByText("RQ1 — do the cheap methods agree with exact patching?")).toBeVisible();
    await expect(page.getByText("Mean rho (signed)").first()).toBeVisible();
    await expect(page.getByText(/AtP \(attribution patching\)/).first()).toBeVisible();

    // top-10 heads per condition with badges
    await expect(page.getByRole("heading", { name: "Top-10 heads per condition" })).toBeVisible();

    // overlap matrices with the random expectation
    await expect(page.getByText(/Top-5 head overlap/)).toBeVisible();
    await expect(page.getByText(/Random expectation k·k\/144 = 0.17 for k=5/)).toBeVisible();
    await expect(page.getByText(/Random expectation k·k\/144 = 0.69 for k=10/)).toBeVisible();

    // caveats panel with the fixed texts
    await expect(page.getByText(/Attribution patching \(AtP\) is a first-order approximation/)).toBeVisible();
    await expect(page.getByText(/DLA measures direct effects only/)).toBeVisible();
    await expect(page.getByText(/Denoising and ablation answer different questions/)).toBeVisible();

    const significant = consoleErrors.filter(
      (e) => !e.includes("favicon") && !e.includes("Download the React DevTools"),
    );
    expect(significant).toEqual([]);
  });
});
