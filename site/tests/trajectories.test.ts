// trajectories.build on the Python tracker output (tracks.json) gives trajectories.json.
//
// The fixtures are rounded (tracks.json rows to 3 decimals, alignment.json H to 8, trajectories.json
// boxes to 2 and the rest to 3), so the Python code itself, fed these files, lands up to 0.0051 px
// from trajectories.json on boxes and 0.0036 on foot points (H's 1e-6 perspective terms keep only
// 2-3 digits). The tolerances below are those bounds. Fed the same numbers, the port and the Python
// code agree to 4e-12 px (boxes and conf bit for bit).
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { collect } from "../src/pipeline/tracker.ts";
import type { Track } from "../src/pipeline/tracker.ts";
import type { SceneConstants } from "../src/pipeline/scene.ts";
import type { TrackRow } from "../src/pipeline/types.ts";
import { at, build, COCO, isPerson, isVehicle, speed } from "../src/pipeline/trajectories.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const fixture = async (clip: string, name: string) =>
  JSON.parse(await readFile(new URL(`./fixtures/${clip}/${name}`, import.meta.url), "utf8"));

const TOL: Record<string, number> = { t: 1e-9, box: 6e-3, foot: 5e-3, vel: 5e-3, height: 5e-3, conf: 1e-3 };

for (const clip of ["C3905", "C3902"]) {
  test(`build matches trajectories.json (${clip})`, async () => {
    const scene: SceneConstants = JSON.parse(await readFile(new URL("scene.json", PUB), "utf8"));
    const { frames } = await fixture(clip, "tracks.json");
    const { H } = await fixture(clip, "alignment.json");
    const expected = await fixture(clip, "trajectories.json");
    // tracking.collect over the frames in order: one Track per id, in order of first appearance
    const tracks = new Map<number, Track>();
    for (const { t, rows } of frames as { t: number; rows: TrackRow[] }[]) collect(tracks, t, rows, scene.tracking);
    const got = build(tracks.values(), (H as number[][]).flat());

    assert.deepEqual(got.map((tr) => tr.tid), expected.map((tr: { tid: number }) => tr.tid));
    const worst: Record<string, number> = {};
    const close: Record<string, number> = {}, count: Record<string, number> = {};
    got.forEach((tr, i) => {
      const e = expected[i];
      assert.equal(tr.group, e.group, `group of ${tr.tid}`);
      assert.equal(tr.cls, e.cls, `cls of ${tr.tid}`);
      for (const key of Object.keys(TOL)) {
        const a = (tr[key as keyof typeof tr] as (number | number[])[]).flat();
        const b = (e[key] as (number | number[])[]).flat();
        assert.equal(a.length, b.length, `${key} length of ${tr.tid}`);
        a.forEach((v, j) => {
          const d = Math.abs(v - b[j]);
          worst[key] = Math.max(worst[key] ?? 0, d);
          count[key] = (count[key] ?? 0) + 1;
          if (d <= 1e-3) close[key] = (close[key] ?? 0) + 1;
        });
      }
    });
    for (const [key, tol] of Object.entries(TOL)) assert.ok(worst[key] <= tol, `${key}: worst ${worst[key]} > ${tol}`);
    // the bulk sits within the files' own rounding (boxes are stored with 2 decimals, so they are left out)
    for (const key of ["foot", "vel", "height", "conf"]) assert.ok(close[key] / count[key] > 0.8, `${key}: ${close[key]} of ${count[key]} within 1e-3`);
  });
}

// Hand-made tracks for the paths the two clips never take: uneven sampling (np.gradient's
// non-uniform branch), an even smoothing window, a single-sample track (zero velocity), a track
// under min_len, a class tie (np.unique + argmax: the smaller id wins) and a perspective H.
// Expected: trajectories.build on the same rows (tracking.collect, then build), to 6 decimals.
test("build on hand-made tracks matches trajectories.py (uneven dt, even window, 1-sample track, class tie)", () => {
  const H = [1.02, 0.013, -4.5, -0.008, 0.97, 12.25, 2.1e-5, -3.4e-5, 1.0];
  const T = [0, 0.1, 0.3, 0.4, 0.7, 0.8];
  const tracks = () => {
    const out = new Map<number, Track>();
    T.forEach((t, i) => {
      const rows: TrackRow[] = [[100.3 + 7.7 * i, 200.15 + 3.1 * i * i, 180.9 + 7.9 * i, 260.45 + 3.3 * i * i, 1000001, 0.81 + 0.01 * i, i % 2 ? 2 : 7]];
      if (i >= 3) rows.push([640.25 - 11.3 * i, 300.5, 660.75 - 11.1 * i, 371.2 + 0.37 * i, 2000005, 0.55, 0]);
      if (i === 5) rows.push([1500.1, 800.2, 1530.3, 880.4, 2000009, 0.4, 0]);
      collect(out, t, rows);
    });
    return out;
  };
  const car = { foot: [[147.825626, 259.578473], [152.720182, 265.311561], [159.297337, 275.538057], [167.584565, 291.556729], [174.264691, 306.952751], [179.313666, 320.438393]],
    vel: [[48.945566, 57.330877], [43.592303, 55.264745], [66.210112, 123.835308], [67.720982, 132.97006], [43.434084, 113.972333], [50.48975, 134.856419]],
    height: [50.36662, 50.536417, 51.062594, 51.94637, 53.189844, 54.796029] };
  const expected: Record<string, { tid: number; cls: number; foot: number[][]; vel: number[][]; height: number[] }[]> = {
    // build(tracks, H): only the car has 5+ samples
    default: [{ tid: 1000001, cls: 2, ...car }],
    // build(tracks, H, min_len=1, smooth=4)
    ml1_sm4: [
      { tid: 1000001, cls: 2, foot: [[144.989148, 257.201106], [149.036964, 260.367394], [155.15516, 267.533753], [163.376604, 280.316873], [171.726761, 299.561901], [178.03798, 316.418953]],
        vel: [[40.478163, 31.662878], [37.182435, 33.052518], [65.006619, 97.164734], [68.619293, 111.910924], [54.292605, 142.465416], [63.112188, 168.570524]], height: car.height },
      { tid: 2000005, cls: 0, foot: [[625.637273, 368.526247], [618.126942, 368.886162], [610.616611, 369.246076]],
        vel: [[-25.034435, 1.199715], [-62.586088, 2.999288], [-75.103306, 3.599145]], height: [70.405496, 70.785219, 71.165135] },
      { tid: 2000009, cls: 0, foot: [[1549.527365, 852.508935]], vel: [[0, 0]], height: [79.922992] },
    ],
  };
  const runs = { default: build(tracks().values(), H), ml1_sm4: build(tracks().values(), H, 1, 4) };
  for (const [name, got] of Object.entries(runs)) {
    const exp = expected[name];
    assert.deepEqual(got.map((tr) => [tr.tid, tr.cls]), exp.map((e) => [e.tid, e.cls]), name);
    got.forEach((tr, i) => {
      for (const key of ["foot", "vel", "height"] as const) {
        const a = (tr[key] as (number | number[])[]).flat(), b = (exp[i][key] as (number | number[])[]).flat();
        assert.equal(a.length, b.length, `${name} ${tr.tid} ${key}`);
        a.forEach((v, j) => assert.ok(Math.abs(v - b[j]) <= 1e-6, `${name} ${tr.tid} ${key}[${j}]: ${v} vs ${b[j]}`));
      }
    });
  }
});

test("at() breaks ties like np.argmin (the earlier sample) and allows 1e-6 s at the ends", () => {
  const tr = { t: [0, 0.5, 1] };
  // Trajectory.at on the same times: 0 1 0 None None
  assert.equal(at(tr, 0.25), 0);
  assert.equal(at(tr, 0.75), 1);
  assert.equal(at(tr, -1e-7), 0);
  assert.equal(at(tr, 1 + 2e-6), null);
  assert.equal(at(tr, -2e-6), null);
});

test("class ids and helpers follow trajectories.py", async () => {
  const scene = JSON.parse(await readFile(new URL("scene.json", PUB), "utf8"));
  assert.deepEqual({ ...COCO }, Object.fromEntries(Object.keys(COCO).map((k) => [k, scene.trajectories[k]])));
  const tr = { cls: 2, t: [1, 1.2, 1.4], vel: [[3, 4], [0, 0], [-6, 8]] };
  assert.deepEqual(speed(tr), [5, 0, 10]);
  assert.equal(isVehicle(tr), true);
  assert.equal(isPerson(tr), false);
  assert.equal(at(tr, 1.25), 1);
  assert.equal(at(tr, 0.5), null);
  assert.equal(at(tr, 1.4 + 1e-7), 2);
});
