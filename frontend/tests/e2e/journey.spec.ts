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
    // the hero runs the example sentence once the model is ready
    await expect(page.getByText("MEASURED · LIVE")).toBeVisible({ timeout: 30_000 });
    // measured probability appears (Mary is top for the preset example)
    await expect(page.getByText("66.2").first()).toBeVisible({ timeout: 30_000 });
    // removing a word re-runs the model
    await page.getByRole("button", { name: /Hide the word John/ }).last().click();
    await expect(page.getByText(/Without “John” the model's top word is/)).toBeVisible({ timeout: 30_000 });
  });

  test("explorer: model → block → attention anatomy → head heatmap", async ({ page }) => {
    await page.goto("/#/explorer?mode=map");
    await expect(page.getByText("gpt2 · 12 blocks · 12 heads")).toBeVisible({ timeout: 30_000 });
    // pick block 5 on the ribbon, then look inside it
    await page.getByRole("button", { name: "Open transformer block 5" }).click();
    await page.getByRole("button", { name: "Look inside block 5" }).click();
    await expect(page.getByText("Block 5 · verified pre-LN order")).toBeVisible({ timeout: 15_000 });
    // open attention anatomy
    await page.getByRole("button", { name: "Open attention anatomy" }).click();
    await expect(page.getByText("Attention anatomy · block 5")).toBeVisible();
    // pick head 4 → measured heatmap with real provenance
    await page.getByRole("button", { name: "Inspect head 4" }).click();
    await expect(page.getByRole("heading", { name: "Head 4 of block 5" })).toBeVisible();
    await expect(page.getByText("blocks.5.attn.hook_pattern").first()).toBeVisible();
    await expect(page.getByText(/12 × 14 × 14/)).toBeVisible();
    await expect(page.getByText("softmax weights")).toBeVisible();
  });

  test("explorer→playground handoff carries the selected hook", async ({ page }) => {
    await page.goto("/#/playground?hook=blocks.7.attn.hook_out&position=6");
    await expect(page.getByRole("heading", { name: "Intervention Playground" })).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.getByLabel("Activation to replace (clean → corrupted)")).toHaveValue(
      "blocks.7.attn.hook_out",
    );
    await expect(page.getByText(/attention · patch here/)).toBeVisible();
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
    await expect(page.getByText("Stage 1 of 16", { exact: true })).toBeVisible({ timeout: 30_000 });
    const overflowExplorer = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflowExplorer).toBeLessThanOrEqual(2);
    await page.getByRole("tab", { name: "Full map" }).click();
    await expect(page.getByText("Measured residual stream")).toBeVisible({ timeout: 30_000 });
    const overflowMap = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflowMap).toBeLessThanOrEqual(2);
  });

  test("guided tour: stages replay recorded data with honest pacing", async ({ page }) => {
    await page.goto("/#/explorer");
    await page.getByRole("button", { name: "▶ Play the walk" }).click();
    // playback advances through real block stages showing measured heatmaps
    await expect(page.getByRole("heading", { name: /Block 0: read, think, write back/ })).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText(/blocks\.0\.attn\.hook_pattern/).first()).toBeVisible();
    // honest pacing label is always present
    await expect(page.getByText(/not GPU timing/)).toBeVisible();
    // step controls work
    await page.getByRole("button", { name: "Next stage" }).click();
    // jumping to the final stage shows the answer
    await page.getByRole("tab", { name: "Answer" }).click();
    await expect(page.getByRole("heading", { name: "The answer: a score for every word" })).toBeVisible();
  });

  test("full map data view: measured residual norms across all blocks", async ({ page }) => {
    await page.goto("/#/explorer?mode=map");
    await expect(page.getByText("Measured residual stream")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("STRUCTURE + MEASURED")).toBeVisible();
    for (let l = 0; l < 12; l++) {
      await expect(page.getByRole("button", { name: new RegExp(`Block ${l}: mean residual norm`) })).toBeVisible();
    }
  });

  test("explorer→playground handoff shows intervention overlay", async ({ page }) => {
    await page.goto("/#/explorer?hook=blocks.7.attn.hook_out&position=6");
    await expect(page.getByText("Intervention overlay:").first()).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/INTERVENTION TARGET/).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Open transformer block 7" })).toBeVisible();
  });

  test("playground: custom candidates update the live comparison", async ({ page }) => {
    await page.goto("/#/playground");
    await expect(page.getByText("Guided experiment")).toBeVisible({ timeout: 30_000 });
    const candA = page.getByLabel("Candidate A");
    await candA.fill(" them");
    await expect(page.getByText("D · logit(A) − logit(B)").first()).toBeVisible();
    await expect(page.getByText(/P\(A\)/).first()).toContainText("them");
  });

  test("playground: greedy continuation generates real text", async ({ page }) => {
    await page.goto("/#/playground");
    await page.getByRole("tab", { name: "Predict & continue" }).click();
    await expect(page.getByText(/GPT-2 Small predicts text continuations/)).toBeVisible({ timeout: 30_000 });
    await page.getByRole("button", { name: "Generate 5 tokens" }).click();
    await expect(page.getByText(/greedy \(argmax\)/)).toBeVisible({ timeout: 30_000 });
    // the continuation extends the prompt (measured tokens rendered)
    await expect(page.getByText(/step 1:/)).toBeVisible();
    // sampling mode exposes its parameters honestly
    await page.getByLabel("mode").selectOption("sample");
    await expect(page.getByText(/temp 0\.90/)).toBeVisible();
  });

  test("neural network lab: 4-6-6-5-3 network computes live", async ({ page }) => {
    await page.goto("/#/lab");
    await expect(page.getByRole("heading", { name: "Neural Network Lab" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/Separate from GPT-2\./)).toBeVisible();
    // exact connection count stated in the accessible description
    await expect(page.getByRole("img", { name: /105 connections/ })).toBeVisible();
    // moving an input changes the computed output probabilities
    const slider = page.getByLabel(/in·1/);
    await slider.fill("-2");
    await expect(page.getByText(/class A/).first()).toBeVisible();
    // neuron inspection shows the real pre-activation
    await page.getByRole("button", { name: /Neuron layer 3 index 1/ }).click();
    await expect(page.getByText(/pre-activation z/)).toBeVisible();
    await expect(page.getByText(/ReLU: max\(0, z\)/)).toBeVisible();
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
