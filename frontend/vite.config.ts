/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // The shared glossary (examples/glossary.json) lives outside frontend/; allow
  // the dev server to serve it (the production build bundles it regardless).
  server: { fs: { allow: [".."] } },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    globals: true,
  },
});
