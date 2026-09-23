// The ONNX detector (onnxruntime-node) on det_frame.rgb against what Ultralytics gets from the same model.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { after, test } from "node:test";
import * as ort from "onnxruntime-node";
import { letterbox, OnnxDetector, parseOutput } from "../src/pipeline/detector.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import type { Detection } from "../src/pipeline/types.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;
const scene = await loadScene(read);
const model = await readFile(new URL("model.onnx", PUB));
const det = await OnnxDetector.create(ort, model, scene, ["cpu"]);
// without it onnxruntime-node can abort at process exit
after(() => det.release());

type Box = [number, number, number, number, number, number];

/** Pairs every expected box with a distinct detection of the same class, corners within tol px, score within 0.01. */
function unmatched(got: Detection[], want: Box[], tol = 1): Box[] {
  const used = new Set<number>();
  return want.filter(([x1, y1, x2, y2, conf, cls]) => {
    const i = got.findIndex(
      (d, j) =>
        !used.has(j) &&
        d.cls === cls &&
        Math.abs(d.conf - conf) <= 0.01 &&
        Math.max(Math.abs(d.x1 - x1), Math.abs(d.y1 - y1), Math.abs(d.x2 - x2), Math.abs(d.y2 - y2)) <= tol,
    );
    if (i >= 0) used.add(i);
    return i < 0;
  });
}

test("letterbox pads a 960x540 frame with 2 rows of 114 at the top and bottom", () => {
  const px = new Uint8Array(960 * 540 * 4).map((_, i) => (i % 4 === 3 ? 255 : i % 251));
  const lb = letterbox(px, 960, 540, 4, [544, 960]);
  assert.deepEqual([lb.padTop, lb.padLeft, lb.gain], [2, 0, 1]);
  assert.equal(lb.data.length, 3 * 544 * 960);
  const plane = 544 * 960;
  for (const y of [0, 1, 542, 543]) for (let c = 0; c < 3; c++) assert.equal(lb.data[c * plane + y * 960 + 5], Math.fround(114 / 255));
  // pixel (x=7, y=3) of the frame is at row 5 of the input, channel-first, RGBA alpha dropped
  const s = (3 * 960 + 7) * 4;
  for (let c = 0; c < 3; c++) assert.equal(lb.data[c * plane + 5 * 960 + 7], Math.fround(px[s + c] / 255));
});

test("letterbox of a 4:3 frame resizes and centres it like LetterBox", () => {
  const px = new Uint8Array(720 * 540 * 3).fill(200);
  const lb = letterbox(px, 720, 540, 3, [544, 960]);
  // r = 544/540, new_unpad = (725, 544), dw = 235 -> left round(117.4), right round(117.6)
  assert.deepEqual([lb.padTop, lb.padLeft, lb.gain], [0, 117, 544 / 540]);
  const v = Math.fround(200 / 255), pad = Math.fround(114 / 255);
  assert.equal(lb.data[300 * 960 + 116], pad);
  assert.equal(lb.data[300 * 960 + 117], v);
  assert.equal(lb.data[300 * 960 + 117 + 724], v);
  assert.equal(lb.data[300 * 960 + 117 + 725], pad);
});

test("parseOutput: threshold, class filter, unpad, clip, work scale", () => {
  const out = Float32Array.from([
    [-3, 1, 100, 546, 0.9, 2], // clipped to the frame
    [10, 12, 20, 22, 0.1, 0], // score not > conf
    [10, 12, 20, 22, 0.5, 9], // traffic light: not kept
    [10, 12, 20, 22, 0.5, 0],
  ].flat());
  const lb = { padTop: 2, padLeft: 0, gain: 1 };
  assert.deepEqual(parseOutput(out, lb, 960, 540, 0.1, [0, 2], 2), [
    { x1: 0, y1: 0, x2: 200, y2: 1080, conf: Math.fround(0.9), cls: 2 },
    { x1: 20, y1: 20, x2: 40, y2: 40, conf: 0.5, cls: 0 },
  ]);
  assert.equal(parseOutput(out, lb, 960, 540, 0.1, null, 1).length, 3);
});

for (const clip of ["C3905", "C3902"]) {
  test(`${clip}: boxes match Ultralytics on the same ONNX model`, async (t) => {
    const FIX = new URL(`./fixtures/${clip}/`, import.meta.url);
    const rgb = new Uint8Array(await readFile(new URL("det_frame.rgb", FIX)));
    const expected = JSON.parse(await readFile(new URL("det_expected.json", FIX), "utf8"));
    const [w, h]: [number, number] = expected.size;
    const want: Box[] = expected.boxes;
    assert.equal(rgb.length, w * h * 3);

    // like for like: Ultralytics' boxes are in 960x540 pixels, all classes, conf 0.1
    const lb = letterbox(rgb, w, h, 3, scene.c.detector.input);
    const t0 = performance.now();
    const out = await det.infer(lb);
    const ms = performance.now() - t0;
    const got = parseOutput(out, lb, w, h, 0.1, null, 1);
    assert.equal(got.length, want.length, "number of boxes");
    assert.deepEqual(unmatched(got, want), []);

    // detect(): the scene's classes and conf, in 1920-wide work pixels
    const t1 = performance.now();
    const work = await det.detect(rgb, w, h, 3, 1920);
    const ms2 = performance.now() - t1;
    const keep = new Set(scene.c.detector.keep_classes);
    const wantWork = want
      .filter(([, , , , conf, cls]) => keep.has(cls) && conf > scene.c.detector.conf)
      .map(([x1, y1, x2, y2, conf, cls]): Box => [x1 * 2, y1 * 2, x2 * 2, y2 * 2, conf, cls]);
    assert.equal(work.length, wantWork.length);
    assert.deepEqual(unmatched(work, wantWork, 2), [], "1 px at 960 is 2 work px");
    t.diagnostic(`${det.provider}: ${got.length} boxes; first inference ${ms.toFixed(0)} ms, detect() ${ms2.toFixed(0)} ms`);
  });
}
