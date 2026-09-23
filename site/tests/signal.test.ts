// signal.py on real pixels and on the clip's recorded lamp scores, against the port.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { matInv, type Mat3 } from "../src/pipeline/geometry.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import { fillPhases, lampPatches, lampScores, phaseFromScores, type Box } from "../src/pipeline/signal.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const scene = loadScene(async (name) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer);
const fixture = async (clip: string, name: string) => readFile(new URL(`./fixtures/${clip}/${name}`, import.meta.url));
const json = async (clip: string, name: string) => JSON.parse((await fixture(clip, name)).toString("utf8"));

for (const clip of ["C3905", "C3902"]) {
  test(`${clip}: lamp boxes from the clip's homography`, async () => {
    const H: Mat3 = (await json(clip, "alignment.json")).H.flat();
    const lamps = await json(clip, "lamps.json");
    assert.deepEqual(lampPatches(matInv(H), (await scene).c.signal_lamps), lamps.boxes);
  });

  test(`${clip}: lamp scores on full-size crops`, async () => {
    const lamps = await json(clip, "lamps.json");
    const boxes: Box[] = lamps.boxes;
    for (const f of lamps.frames) {
      const rgb = new Uint8Array(await fixture(clip, f.file.replace(".png", ".rgb")));
      const [w, h] = f.size;
      assert.equal(rgb.length, w * h * 3);
      const got = lampScores(rgb, w, h, 3, boxes, f.origin);
      got.forEach((s, k) => assert.ok(Math.abs(s - f.scores[k]) <= 0.5, `${f.file} lamp ${k}: ${s} vs ${f.scores[k]}`));
      // the same crop as RGBA gives the same scores
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) rgba.set(rgb.subarray(i * 3, i * 3 + 3), i * 4);
      assert.deepEqual(lampScores(rgba, w, h, 4, boxes, f.origin), got);
    }
  });

  test(`${clip}: phase per frame and fill_phases`, async () => {
    const minContrast = (await scene).c.signal_min_contrast;
    const sig = await json(clip, "signal.json");
    const raw = sig.scores.map((s: number[]) => phaseFromScores(s, minContrast));
    const bad = raw.filter((p: string, i: number) => p !== sig.raw[i]).length;
    assert.equal(bad, 0, `${bad} of ${raw.length} raw phases differ`);
    assert.deepEqual(fillPhases(sig.raw, sig.times), sig.phases);
  });
}

test("fill_phases: blips, flashing green, long dark spells", () => {
  const t = Array.from({ length: 12 }, (_, i) => i);
  // a one-sample flip is removed; a dark spell of 2 s keeps the phase before it
  assert.deepEqual(
    fillPhases(["red", "green", "red", "red", "off", "off", "off", "green", "off", "green", "green", "green"], t),
    ["red", "red", "red", "red", "red", "red", "red", "green", "green", "green", "green", "green"],
  );
  // a leading dark spell takes the phase after it; one longer than maxOff is unknown
  assert.deepEqual(fillPhases(["off", "off", "red"], [0, 1, 2]), ["red", "red", "red"]);
  assert.deepEqual(fillPhases(["red", "off", "off", "off"], [0, 1, 3, 6]), ["red", "unknown", "unknown", "unknown"]);
  assert.deepEqual(fillPhases(["off", "off"], [0, 1]), ["unknown", "unknown"]);
});

test("phase_from_scores: lamp furthest above its threshold", () => {
  const mc = [20, 20, 6];
  assert.equal(phaseFromScores([30, 0, 0], mc), "red");
  assert.equal(phaseFromScores([30, 0, 12], mc), "green"); // 12 / 6 = 2 beats 30 / 20 = 1.5
  assert.equal(phaseFromScores([19.75, 19.75, 5.75], mc), "off");
  assert.equal(phaseFromScores([20, 0, 0], mc), "red");
});
