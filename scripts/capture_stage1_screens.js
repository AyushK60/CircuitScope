/* Stage 1 "before/after" screenshot capture. NOT part of the test suite.
 *
 * Usage:
 *   node scripts/capture_stage1_screens.js <output-dir>
 *
 * Requires the CircuitScope stack to be running: backend on 127.0.0.1:8000
 * and the Vite dev server on localhost:5199 (same as frontend/playwright.config.ts).
 * Captures 1280x800 PNGs of: Landing, Explorer (model/block/attention),
 * Playground (idle + after one real experiment).
 */
const path = require("path");
const fs = require("fs");
// resolve from the frontend install; this script lives outside any package
const { chromium } = require(path.join(__dirname, "..", "frontend", "node_modules", "playwright"));

const BASE = "http://localhost:5199";
const OUT = process.argv[2] || path.join(__dirname, "..", "outputs", "stage1_before");

async function waitForReady(page) {
  await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded" });
  await page.getByRole("status").filter({ hasText: "GPT-2 ready" }).waitFor({ timeout: 120000 });
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.setDefaultTimeout(120000);

  // Landing (model ready + auto demo measured probabilities visible)
  await waitForReady(page);
  await page.getByText("66.2").first().waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "landing.png"), fullPage: false });
  console.log("saved landing.png");

  // Explorer — model level
  await page.goto(`${BASE}/#/explorer`, { waitUntil: "domcontentloaded" });
  await page.getByText("gpt2 · 12 blocks · 12 heads").waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "explorer_model.png"), fullPage: false });
  console.log("saved explorer_model.png");

  // Explorer — block level
  await page.getByRole("button", { name: "Open transformer block 5" }).click();
  await page.getByText("Block 5 · verified pre-LN order").waitFor({ timeout: 30000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, "explorer_block.png"), fullPage: false });
  console.log("saved explorer_block.png");

  // Explorer — attention level
  await page.getByRole("button", { name: "Open attention anatomy" }).click();
  await page.getByText("Attention anatomy · block 5").waitFor({ timeout: 30000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, "explorer_attention.png"), fullPage: false });
  console.log("saved explorer_attention.png");

  // Playground — idle
  await page.goto(`${BASE}/#/playground`, { waitUntil: "domcontentloaded" });
  await page.getByText("Guided experiment").waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "playground_idle.png"), fullPage: false });
  console.log("saved playground_idle.png");

  // Playground — after one real experiment (3 real CPU model passes)
  await page.getByRole("button", { name: /Run experiment/ }).click();
  await page.getByRole("heading", { name: "2 · Measured effect" }).waitFor({ timeout: 120000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "playground_result.png"), fullPage: false });
  console.log("saved playground_result.png");

  await browser.close();
  console.log("done:", OUT);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
