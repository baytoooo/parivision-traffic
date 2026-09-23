// The browser pipeline on a video file, from the command line (Node + ffmpeg + onnxruntime-node).
//
//   node scripts/run-pipeline.ts <clip.mp4> <out.json> [--seconds 120] [--video /media/demo/clip.mp4]
//
// Does what worker.ts and local_api.ts do in the page, with ffmpeg decoding instead of a <video>:
// frames at 5 fps scaled to 1920 px wide, the first one in grey at 480 px for the alignment, the
// detector on a 960 px copy, the lamp patches read at full size. Writes the ClipResult the page
// builds (with the overlay), so the output can be the replay for ?mock=1, or be compared with
// the Python pipeline on the same clip. --video sets the result's video URL.

import { spawn } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { basename } from "node:path";
import * as ort from "onnxruntime-node";
import { alignToReference } from "../src/pipeline/align.ts";
import { Analyser } from "../src/pipeline/analyse.ts";
import { OnnxDetector } from "../src/pipeline/detector.ts";
import { matInv } from "../src/pipeline/geometry.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import { lampPatches, lampScores } from "../src/pipeline/signal.ts";

const FPS = 5;
const W = 1920;

const args = process.argv.slice(2);
const opt = (name: string, fallback: string) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const [clip, out] = args;
if (!clip || !out) {
  console.error("usage: node scripts/run-pipeline.ts <clip.mp4> <out.json> [--seconds 120] [--video URL]");
  process.exit(2);
}
const maxSeconds = Number(opt("--seconds", "120"));
const PUB = new URL("../public/pipeline/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;

/** Frames of the clip at FPS, RGB, W px wide, as ffmpeg decodes them. */
async function* frames(path: string, height: number): AsyncGenerator<Uint8Array> {
  const size = W * height * 3;
  const ff = spawn("ffmpeg", ["-v", "error", "-i", path, "-t", String(maxSeconds), "-vf", `fps=${FPS},scale=${W}:${height}:flags=area`,
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], { stdio: ["ignore", "pipe", "inherit"] });
  let buf = Buffer.alloc(0);
  for await (const chunk of ff.stdout) {
    buf = Buffer.concat([buf, chunk as Buffer]);
    while (buf.length >= size) {
      yield new Uint8Array(buf.subarray(0, size));
      buf = buf.subarray(size);
    }
  }
}

/** Area-average downscale of an RGB image by an integer factor, to RGB or grey. */
function shrink(rgb: Uint8Array, w: number, h: number, k: number, grey: boolean): Uint8Array {
  const ow = Math.floor(w / k), oh = Math.floor(h / k), ch = grey ? 1 : 3;
  const outBuf = new Uint8Array(ow * oh * ch);
  for (let y = 0; y < oh; y++)
    for (let x = 0; x < ow; x++) {
      let r = 0, g = 0, b = 0;
      for (let dy = 0; dy < k; dy++)
        for (let dx = 0; dx < k; dx++) {
          const i = ((y * k + dy) * w + x * k + dx) * 3;
          r += rgb[i]; g += rgb[i + 1]; b += rgb[i + 2];
        }
      const n = k * k, o = (y * ow + x) * ch;
      if (grey) outBuf[o] = Math.round((0.299 * r + 0.587 * g + 0.114 * b) / n);
      else [outBuf[o], outBuf[o + 1], outBuf[o + 2]] = [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
    }
  return outBuf;
}

async function probe(path: string): Promise<{ width: number; height: number; duration: number }> {
  const ff = spawn("ffprobe", ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "json", path]);
  let text = "";
  for await (const chunk of ff.stdout) text += chunk;
  const j = JSON.parse(text);
  return { width: j.streams[0].width, height: j.streams[0].height, duration: Number(j.format.duration) };
}

const scene = await loadScene(read);
const detector = await OnnxDetector.create(ort, new Uint8Array(await read("model.onnx")), scene, ["cpu"]);
const info = await probe(clip);
const height = Math.round((W * info.height) / info.width / 2) * 2;
const refs = {
  reference_day: new Uint8Array(await readFile(new URL("../tests/fixtures/refs/reference_day.gray", import.meta.url))),
  reference_dusk: new Uint8Array(await readFile(new URL("../tests/fixtures/refs/reference_dusk.gray", import.meta.url))),
};

let analyser: Analyser | null = null;
let boxes: [number, number, number, number][] = [];
let k = 0;
for await (const rgb of frames(clip, height)) {
  if (!analyser) {
    const a = alignToReference(shrink(rgb, W, height, 4, true), refs, [W, height]);
    analyser = new Analyser(scene, FPS, [W, height]);
    analyser.setAlignment(a);
    boxes = lampPatches(matInv(a.H), scene.c.signal_lamps);
    console.error(`aligned to ${a.reference}, score ${a.score.toFixed(2)}, ok ${a.ok}`);
  }
  const small = shrink(rgb, W, height, 2, false);
  const dets = await detector.detect(small, W / 2, height / 2, 3, W);
  const scores = lampScores(rgb, W, height, 3, boxes) as [number, number, number];
  analyser.push(k / FPS, dets, scores);
  k++;
}
if (!analyser) throw new Error("no frames decoded");
const duration = Math.min(info.duration, maxSeconds);
const result = analyser.finish(duration, basename(clip));
// the page's ClipResult (local_api.ts toClipResult): the overlay also names the tracker groups
const video = opt("--video", "");
const { fps: _fps, overlay, ...rest } = result;
const groups = Object.keys(scene.c.tracking.GROUPS);
await writeFile(out, JSON.stringify({ ...rest, video, overlay: { ...overlay, groups } }));
console.error(`${k} frames, ${result.events.length} events: ${JSON.stringify(result.events)}`);
