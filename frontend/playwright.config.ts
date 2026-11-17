import { defineConfig } from "@playwright/test";

// E2E against the real local stack: backend on 8000, Vite dev server on a dedicated port (5199) so it never collides with other local projects on 5173.
// Servers are started automatically, or reused if already running.
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 120_000,
  use: {
    baseURL: "http://localhost:5199",
    viewport: { width: 1280, height: 800 },
  },
  expect: { timeout: 30_000 },
  reporter: [["list"]],
  // All e2e tests share one backend whose job engine runs ONE heavy job at a
  // time; parallel test files would 422 each other's submissions.
  workers: 1,
  webServer: [
    {
      command: ".venv\\Scripts\\python -m backend.app.main",
      cwd: "..",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: true,
      timeout: 180_000,
    },
    {
      command: "npm run dev -- --port 5199 --strictPort",
      url: "http://localhost:5199",
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
