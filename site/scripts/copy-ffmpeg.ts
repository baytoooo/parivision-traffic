// Copies ffmpeg.wasm's multi-threaded core from node_modules to public/ffmpeg/, where the demo
// loads it when it has to convert a clip (src/scripts/transcode.ts). The page is cross-origin
// isolated (COEP require-corp), so the core cannot come from a CDN. Runs before `dev` and `build`.
//
//   node scripts/copy-ffmpeg.ts

import { copyFile, mkdir } from "node:fs/promises";

const FILES = ["ffmpeg-core.js", "ffmpeg-core.wasm", "ffmpeg-core.worker.js"];

// the ES module build (dist/esm/): @ffmpeg/ffmpeg's worker is a module worker and import()s the core
const from = new URL(".", import.meta.resolve("@ffmpeg/core-mt"));
const to = new URL("../public/ffmpeg/", import.meta.url);
await mkdir(to, { recursive: true });
await Promise.all(FILES.map((name) => copyFile(new URL(name, from), new URL(name, to))));
