/* README screenshot capture. NOT part of the test suite.
 *
 * Usage:
 *   start the stack first:  start.bat
 *   then:                   node scripts\capture_readme_screens.js
 *
 * Reads the frontend URL from outputs\run\pids.json (written by start.ps1) and
 * drives the REAL UI with Playwright: every screenshot shows live pages, and
 * the Sweeps, Report and Verify shots each trigger their real measurement
 * through the page's own button, so every pixel comes from the real model.
 * Writes docs\screenshots\readme_*.png (1280x800).
 */
const path = require("path");
const fs = require("fs");
// resolve Playwright from the frontend install; this script lives outside any package
const { chromium } = require(path.join(__dirname, "..", "frontend", "node_modules", "playwright"));

const REPO = path.join(__dirname, "..");
const OUT = path.join(REPO, "docs", "screenshots");

function frontendUrl() {
  if (process.env.CAPTURE_URL) return process.env.CAPTURE_URL.replace(/\/$/, "");
  const pidsPath = path.join(REPO, "outputs", "run", "pids.json");
  if (!fs.existsSync(pidsPath)) {
    console.error("outputs\\run\\pids.json not found. Start the stack first:  start.bat");
    process.exit(1);
  }
  const record = JSON.parse(fs.readFileSync(pidsPath, "utf8").replace(/^\uFEFF/, ""));
  return record.frontend.url;
}

(async () => {
  const BASE = frontendUrl();
  console.log("capturing from", BASE);
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.setDefaultTimeout(240000);

  const save = async (name) => {
    await page.waitForTimeout(800); // let transitions settle
    const file = path.join(OUT, name);
    await page.screenshot({ path: file, fullPage: false });
    console.log("saved", name);
  };

  // 1. Landing - model ready, live prediction demo measured.
  // Smaller viewport than the other shots: keeps the PNG under ~300 KB.
  await page.setViewportSize({ width: 1024, height: 640 });
  await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded" });
  await page.getByRole("status").filter({ hasText: "GPT-2 ready" }).waitFor({ timeout: 120000 });
  await save("readme_1_landing.png");
  await page.setViewportSize({ width: 1280, height: 800 });

  // 2. Explorer - the full map: the ribbon with the measured block sizes
  await page.goto(`${BASE}/#/explorer?mode=map`, { waitUntil: "domcontentloaded" });
  await page.getByText("gpt2 · 12 blocks · 12 heads").waitFor({ timeout: 60000 });
  await page.getByRole("button", { name: "Block 4: mean residual norm" }).waitFor({ timeout: 120000 });
  await page.getByRole("button", { name: "Open transformer block 3" }).scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: "Open transformer block 3" }).click();
  await save("readme_2_explorer.png");

  // 3. Sweeps - the page runs the real layer x position patching sweep on load; wait for the heatmap
  await page.goto(`${BASE}/#/sweeps`, { waitUntil: "domcontentloaded" });
  await page
    .getByRole("img", { name: /Patching sweep heatmap/ })
    .waitFor({ timeout: 240000 });
  await save("readme_3_sweeps.png");

  // 4. Report - generate the real Circuit Report. Wait for the LAST section
  // heading: sections render only once the report job is complete, so a
  // mid-run wait here would leave the job engine busy for later steps.
  await page.goto(`${BASE}/#/report`, { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Generate" }).click();
  await page
    .getByRole("heading", { name: "What this does not show" })
    .waitFor({ timeout: 240000 });
  await save("readme_4_report.png");

  // 5. Verify - run the first reference check (DLA + logit lens vs raw HF).
  // Wait for the Cancel button first: the card may show a stored "Pass" from
  // an earlier run, and only a visible Cancel proves THIS check is running.
  await page.goto(`${BASE}/#/verify`, { waitUntil: "domcontentloaded" });
  const card = page.locator("section", { hasText: "Direct logit attribution and the logit lens" }).first();
  await card.getByRole("button", { name: "Run this check" }).click();
  await card.getByRole("button", { name: "Cancel" }).waitFor({ timeout: 15000 });
  await card.getByRole("status", { name: "Status: Pass" }).waitFor({ timeout: 240000 });
  await save("readme_5_verify.png");

  // 6. Tour - first scene (real prediction), the tour generates its own report.
  // reducedMotion keeps the walk manual (same trick as the e2e spec), so the
  // scene stays on screen until a human clicks Next.
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${BASE}/#/tour`, { waitUntil: "domcontentloaded" });
  await page.getByRole("heading", { name: "The answer" }).waitFor({ timeout: 240000 });
  await save("readme_6_tour.png");

  await browser.close();
  console.log("done:", OUT);
})().catch((e) => {
  console.error("capture failed:", e);
  process.exit(1);
});
