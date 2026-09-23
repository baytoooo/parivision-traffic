// alignToReference against the homography the Python pipeline holds for each clip
// (alignment.json: SIFT + RANSAC on the clip's median background), and against frames that do
// not show the junction.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { alignToReference, MIN_SCORE, rgbaToGray, type AlignResult } from "../src/pipeline/align.ts";
import { matInv, warpPoints, type Mat3 } from "../src/pipeline/geometry.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import { lampPatches, lampScores, phaseFromScores } from "../src/pipeline/signal.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const FIX = new URL("./fixtures/", import.meta.url);
const scene = loadScene(async (name) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer);
const bytes = async (path: string) => new Uint8Array(await readFile(new URL(path, FIX)));
const json = async (path: string) => JSON.parse(await readFile(new URL(path, FIX), "utf8"));
const refs = async () => ({ day: await bytes("refs/reference_day.gray"), dusk: await bytes("refs/reference_dusk.gray") });

// scene.py JUNCTION_BOX's four corners (the polygon is not in scene.json)
const JUNCTION_CORNERS: [number, number][] = [[400, 655], [1160, 575], [1160, 1080], [600, 1080]];

// What Python's analyse() gets for the same first frame: registration.align_best(frame, references())
// on the 1920-wide frame. alignment.json holds the clip's cached homography instead (SIFT on a median
// background of a proxy video), which sits 1.5 to 2.5 px off this one, hence the looser tolerances there.
const FIRST_FRAME_H: Record<string, Mat3> = {
  C3905: [0.987598915, -0.0176471149, 16.0260451, 0.0215934217, 0.986497635, -24.3867724, 1.04965158e-6, -2.09160452e-6, 1],
  C3902: [0.980415972, -0.00977170912, 61.8692299, 0.0147576755, 0.98660399, -38.4052908, -4.98505671e-6, 1.74748122e-6, 1],
};

/** The fixture clip's first frame aligned once, shared by the tests below; with its run time. */
const aligned = new Map<string, Promise<[AlignResult, number]>>();
function alignClip(clip: string): Promise<[AlignResult, number]> {
  if (!aligned.has(clip))
    aligned.set(clip, (async () => {
      const a = await json(`${clip}/alignment.json`);
      const gray = await bytes(`${clip}/frame0.gray`);
      assert.equal(gray.length, a.frame_size[0] * a.frame_size[1]);
      const r = await refs();
      const t0 = performance.now();
      return [alignToReference(gray, r, a.work_size), performance.now() - t0];
    })());
  return aligned.get(clip)!;
}

/** Largest distance, in work pixels, between where two homographies put reference points. */
function apart(pts: [number, number][], Ha: Mat3, Hb: Mat3): number {
  const a = warpPoints(pts, matInv(Ha)), b = warpPoints(pts, matInv(Hb));
  return Math.max(...a.map((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1])));
}

for (const clip of ["C3905", "C3902"]) {
  test(`${clip}: reference points land where the Python homography puts them`, async (t) => {
    const s = (await scene).c;
    const a = await json(`${clip}/alignment.json`);
    const [r, ms] = await alignClip(clip);
    t.diagnostic(`${r.reference} score ${r.score.toFixed(3)} in ${ms.toFixed(0)} ms`);
    assert.ok(r.ok, `score ${r.score}`);
    const Hpy: Mat3 = a.H.flat();
    const lamps = Object.values(s.signal_lamps);
    const others = [...s.stop_line_sb, s.median_nose, ...JUNCTION_CORNERS];
    for (const p of others) assert.ok(apart([p], r.H, Hpy) <= 4, `${p}: ${apart([p], r.H, Hpy).toFixed(2)} px apart`);
    for (const p of lamps) assert.ok(apart([p], r.H, Hpy) <= 3, `lamp ${p}: ${apart([p], r.H, Hpy).toFixed(2)} px apart`);
    const d = apart([...others, ...lamps], r.H, FIRST_FRAME_H[clip]);
    t.diagnostic(`vs alignment.json ${apart([...others, ...lamps], r.H, Hpy).toFixed(2)} px, vs align_best on this frame ${d.toFixed(2)} px`);
    assert.ok(d <= 1, `${d.toFixed(2)} px from Python's align_best on the same frame`);
  });

  test(`${clip}: the lamp boxes from the found homography read the same phases`, async () => {
    const s = (await scene).c;
    const [r] = await alignClip(clip);
    const boxes = lampPatches(matInv(r.H), s.signal_lamps);
    const lamps = await json(`${clip}/lamps.json`);
    for (const f of lamps.frames) {
      const rgb = await bytes(`${clip}/${f.file.replace(".png", ".rgb")}`);
      const got = phaseFromScores(lampScores(rgb, f.size[0], f.size[1], 3, boxes, f.origin), s.signal_min_contrast);
      assert.equal(got, phaseFromScores(f.scores, s.signal_min_contrast), f.file);
    }
  });
}

/** Deterministic pseudo-random numbers in [0, 1). */
function lcg(seed: number): () => number {
  return () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 2 ** 32;
}

test("frames that are not this junction are not ok", async () => {
  const [W, H] = [480, 270];
  const g = await bytes("C3905/frame0.gray");
  const rnd = lcg(7);
  const mirrored = new Uint8Array(W * H), upsideDown = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      mirrored[y * W + x] = g[y * W + W - 1 - x];
      upsideDown[y * W + x] = g[(H - 1 - y) * W + x];
    }
  // the frame cut into 8x5 tiles of 60x54 px, shuffled
  const order = Array.from({ length: 40 }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const shuffled = new Uint8Array(W * H);
  order.forEach((src, dst) => {
    const [sx, sy, dx, dy] = [(src % 8) * 60, Math.floor(src / 8) * 54, (dst % 8) * 60, Math.floor(dst / 8) * 54];
    for (let y = 0; y < 54; y++) shuffled.set(g.subarray((sy + y) * W + sx, (sy + y) * W + sx + 60), (dy + y) * W + dx);
  });
  const noise = new Uint8Array(W * H).map(() => Math.floor(rnd() * 256));
  const flat = new Uint8Array(W * H).fill(90);
  const r = await refs();
  for (const [name, img] of Object.entries({ mirrored, upsideDown, shuffled, noise, flat })) {
    const res = alignToReference(img, r, [1920, 1080]);
    assert.equal(res.ok, false, `${name}: score ${res.score}`);
    assert.deepEqual(res.H, [1, 0, 0, 0, 1, 0, 0, 0, 1], name);
    assert.ok(res.score < MIN_SCORE - 0.05, `${name}: score ${res.score} too close to ${MIN_SCORE}`);
  }
});

test("rgbaToGray: BT.601 weights, grey stays grey", () => {
  const rgba = new Uint8Array([10, 10, 10, 255, 200, 200, 200, 0, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
  assert.deepEqual([...rgbaToGray(rgba)], [10, 200, 76, 150, 29]);
});
