// @ts-check
import { defineConfig } from "astro/config";

// Static output for Vercel (project root = site/). The only dynamic part is the
// demo page, which runs the pipeline in the visitor's browser (src/pipeline/).
export default defineConfig({
  output: "static",
  trailingSlash: "ignore",
  build: { inlineStylesheets: "auto" },
  devToolbar: { enabled: false },
  // the headers public/vercel.json sets in production: a cross-origin isolated page gets
  // SharedArrayBuffer, which threaded onnxruntime-web and ffmpeg.wasm's core need
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  vite: {
    // the pipeline's worker imports onnxruntime-web on demand, which needs a module worker
    worker: { format: "es" },
    // onnxruntime-web finds its .wasm next to its own module, and @ffmpeg/ffmpeg its worker;
    // pre-bundling would move the module
    optimizeDeps: { exclude: ["onnxruntime-web", "@ffmpeg/ffmpeg"] },
  },
});
