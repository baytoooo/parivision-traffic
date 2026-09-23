// @ts-check
import { defineConfig } from "astro/config";

// Static output for Vercel (project root = site/). The only dynamic part is the
// demo page, which runs the pipeline in the visitor's browser (src/pipeline/).
export default defineConfig({
  output: "static",
  trailingSlash: "ignore",
  build: { inlineStylesheets: "auto" },
  devToolbar: { enabled: false },
  vite: {
    // the pipeline's worker imports onnxruntime-web on demand, which needs a module worker
    worker: { format: "es" },
    // onnxruntime-web finds its .wasm next to its own module; pre-bundling would move the module
    optimizeDeps: { exclude: ["onnxruntime-web"] },
  },
});
