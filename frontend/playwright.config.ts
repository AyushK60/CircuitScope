import { defineConfig } from "@playwright/test";

// E2E against the real local stack: backend on 8000, Vite dev server on 5173.
// Start both before running: see docs/EXECUTION_STATE.md launch procedure.
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 120_000,
  use: {
    baseURL: "http://localhost:5173",
    viewport: { width: 1280, height: 800 },
  },
  expect: { timeout: 30_000 },
  reporter: [["list"]],
});
