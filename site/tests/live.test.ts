// The live page's pure parts (src/scripts/liveloop.ts) and the Analyser accessors the worker's
// `live` reply reads. The accessors must give, frame by frame, what finish() later lists.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { Analyser } from "../src/pipeline/analyse.ts";
import { roundHalfEven } from "../src/pipeline/geometry.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import type { Detection, OverlayFrame } from "../src/pipeline/types.ts";
import { CLASS_NAMES, countInView, COUNTED, detectorWidth, frameShape, Pacer, RateMeter, Rolling, sameShape } from "../src/scripts/liveloop.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;
const scene = loadScene(read);

test("Pacer sends one frame at a time, at most one per gap", () => {
  const p = new Pacer(200);
  assert.equal(p.wait(1000), 0, "the first frame leaves at once");
  p.sent(1000);
  assert.equal(p.wait(1050), Infinity, "nothing leaves while a frame is out");
  assert.throws(() => p.sent(1060), /no reply/);
  // a fast worker: the reply is back after 50 ms, the next frame waits for the rest of the gap
  p.replied();
  assert.equal(p.wait(1050), 150);
  assert.equal(p.wait(1250), 0);
  // a slow worker: the reply takes 700 ms, the next frame leaves at once
  p.sent(1250);
  p.replied();
  assert.equal(p.wait(1950), 0);
});

test("RateMeter gives the rate of the last frames and forgets old ones", () => {
  const m = new RateMeter(3000);
  assert.equal(m.rate(0), null);
  m.tick(0);
  assert.equal(m.rate(0), null, "one frame is no rate");
  for (let k = 1; k <= 20; k++) m.tick(k * 200);
  assert.ok(Math.abs(m.rate(4000)! - 5) < 1e-9, `${m.rate(4000)}`);
  // slower: one frame every 500 ms for 4 s, which pushes the fast ones out of the window
  for (let k = 1; k <= 8; k++) m.tick(4000 + k * 500);
  assert.ok(Math.abs(m.rate(8000)! - 2) < 1e-9, `${m.rate(8000)}`);
  assert.equal(m.rate(20000), null, "a stalled worker has no rate");
});

test("Rolling keeps the last span seconds", () => {
  const r = new Rolling(60);
  assert.equal(r.start, 0);
  for (let k = 0; k <= 50; k++) r.push(k * 0.2, k);
  assert.equal(r.points.length, 51);
  assert.equal(r.start, 0, "the window does not start before 0");
  for (let k = 51; k <= 400; k++) r.push(k * 0.2, k);
  assert.ok(Math.abs(r.start - 20) < 1e-9, `${r.start}`);
  assert.ok(r.points[0][0] >= r.start - 1e-9 && r.points.at(-1)![0] === 80);
  assert.equal(r.points.length, 301);
});

test("countInView counts the six classes and nothing else", async () => {
  const boxes = [2, 2, 2, 5, 7, 3, 0, 0, 1, 16, 2].map((cls) => ({ cls }));
  assert.deepEqual(countInView(boxes), { car: 4, bus: 1, truck: 1, motorcycle: 1, person: 2, bicycle: 1 });
  assert.deepEqual(countInView([]), Object.fromEntries(COUNTED.map((n) => [n, 0])));
  // the names cover every class the detector keeps, with the ids scene.json gives them
  const c = (await scene).c;
  assert.deepEqual(Object.keys(CLASS_NAMES).map(Number), c.detector.keep_classes);
  const ids = (c as typeof c & { trajectories: Record<string, number> }).trajectories;
  for (const [key, id] of Object.entries(ids)) assert.equal(CLASS_NAMES[id], key.toLowerCase(), key);
  assert.equal(CLASS_NAMES[c.tracking.BICYCLE as number], "bicycle");
});

test("detectorWidth fits any frame into the detector input without a resize", async () => {
  const input = (await scene).c.detector.input;
  assert.deepEqual(input, [544, 960]);
  // 16:9 as the demo draws clips; 4:3 and upright phones narrower
  assert.equal(detectorWidth(1920, 1080, input), 960);
  assert.equal(detectorWidth(1280, 720, input), 960);
  assert.equal(detectorWidth(640, 480, input), 725);
  assert.equal(detectorWidth(720, 1280, input), 306);
  for (const [w, h] of [[1920, 1080], [2560, 1080], [1440, 900], [1280, 1024], [640, 480], [1080, 1920], [3840, 2160], [1366, 768]]) {
    const dw = detectorWidth(w, h, input);
    const dh = roundHalfEven((dw * h) / w); // FrameReader's height
    assert.ok(dw <= input[1] && dh <= input[0], `${w}x${h}: ${dw}x${dh}`);
    assert.ok(dw === input[1] || dh >= input[0] - 1, `${w}x${h}: ${dw}x${dh} leaves the input unfilled`);
  }
});

test("frameShape follows the picture's shape, not its size", async () => {
  const input = (await scene).c.detector.input;
  const shape = (w: number, h: number) => frameShape(w, h, input, 1920);
  assert.deepEqual(shape(1920, 1080), { work: [1920, 1080], det: [960, 540] });
  assert.deepEqual(shape(640, 480), { work: [1920, 1440], det: [725, 544] });
  // a camera that sends small pictures first, then the size asked for: the run goes on
  assert.ok(sameShape(shape(640, 360), shape(1920, 1080)));
  assert.ok(sameShape(shape(1280, 720), shape(3840, 2160)));
  // a new shape: the work frame or the detector's frame changes
  assert.ok(!sameShape(shape(640, 480), shape(1920, 1080)), "4:3 to 16:9");
  assert.ok(!sameShape(shape(1080, 1920), shape(1920, 1080)), "a phone turned");
  assert.ok(!sameShape(shape(1280, 1000), shape(1280, 1001)), "a window one pixel taller");
});

test("Analyser.lastFrame and lastRisk follow each push and match finish()", async () => {
  const fixture = async (name: string) => JSON.parse(await readFile(new URL(`./fixtures/C3905/${name}`, import.meta.url), "utf8"));
  const det = await fixture("detections.json");
  const { H, work_size } = await fixture("alignment.json");
  const a = new Analyser(await scene, det.fps, work_size);
  assert.deepEqual([a.lastFrame, a.lastRisk], [null, null]);
  a.setAlignment({ H: (H as number[][]).flat(), score: 1, ok: true, reference: "fixture" });
  const frames: (OverlayFrame | null)[] = [];
  const risks: (number | null)[] = [];
  for (const fr of (det.frames as { t: number; boxes: number[][] }[]).slice(0, 200)) {
    const dets: Detection[] = fr.boxes.map(([x1, y1, x2, y2, conf, cls]) => ({ x1, y1, x2, y2, conf, cls }));
    a.push(fr.t, dets, null);
    frames.push(a.lastFrame);
    risks.push(a.lastRisk);
    assert.equal(a.lastFrame?.t, fr.t);
  }
  const r = a.finish(det.frames[199].t + 0.2, "C3905");
  assert.deepEqual(frames, r.overlay.frames);
  assert.deepEqual(risks, r.risk.map(([, v]) => v));
  assert.ok(frames.some((f) => f!.boxes.length > 5), "the fixture has traffic");
  assert.ok(risks.some((v) => v! > 0), "and some risk");
});
