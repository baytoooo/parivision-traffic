// @ts-check
import { defineConfig } from "astro/config";

// Static output for Vercel (project root = site/). The only dynamic part is the
// demo page, which talks to the API in src/config.ts from the browser.
export default defineConfig({
  output: "static",
  trailingSlash: "ignore",
  build: { inlineStylesheets: "auto" },
  devToolbar: { enabled: false },
});
