import { defineConfig } from "vitest/config";

// The first-party JS files are splash.js and note-guide.js, so coverage is scoped to them.
// Vendored files are excluded by the include list, not an omission list, so
// a re-vendor cannot silently enter the coverage denominator.
export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["tests/js/**/*.test.js"],
    coverage: {
      provider: "v8",
      include: ["pikaraoke/static/js/splash.js", "pikaraoke/static/js/note-guide.js"],
      reporter: ["text", "html"],
      reportsDirectory: "coverage/js",
    },
  },
});
