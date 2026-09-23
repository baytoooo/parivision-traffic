// The ByteTrack port against the Python tracker: fixtures/<clip>/detections.json in, tracks.json out,
// frame by frame. The exporter also replays the risk model (risk.py Anticipator), whose own
// MultiTracker (1.5 s buffer) is updated after the main one on every frame and draws ids from the
// same Ultralytics counter, so the ids in tracks.json only line up with that second tracker running.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import type { TestContext } from "node:test";
import { loadScene } from "../src/pipeline/scene.ts";
import { collect, groupOf, linearAssignment, MultiTracker, trackLabel } from "../src/pipeline/tracker.ts";
import type { Track } from "../src/pipeline/tracker.ts";
import type { Detection, TrackRow } from "../src/pipeline/types.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;
const fixture = async (clip: string, name: string) =>
  JSON.parse(await readFile(new URL(`./fixtures/${clip}/${name}`, import.meta.url), "utf8"));

/** risk.py: Anticipator builds MultiTracker(fps=self.fps / self.stride, buffer_sec=1.5); stride is 1 at 5 fps. */
const RISK_BUFFER_SEC = 1.5;

interface DetFile {
  fps: number;
  width: number;
  height: number;
  frames: { t: number; boxes: number[][] }[];
}

const sameRow = (g: number[], e: number[]) =>
  [0, 1, 2, 3].every((c) => Math.abs(g[c] - e[c]) <= 0.01) && Math.abs(g[5] - e[5]) <= 6e-4 && g[6] === e[6];

/**
 * Rows equal to Python's: `exact` in the same order with the same ids; `relabelled` allowing a
 * one-to-one renaming of ids (a track born in one run and not in the other shifts every later id).
 * The renaming pairs ids by how many rows they share, most first.
 */
function compare(got: TrackRow[][], expected: number[][][]) {
  let total = 0, exact = 0, firstDiff = -1;
  const shared = new Map<string, number>();
  expected.forEach((exp, k) => {
    const rows = got[k];
    total += Math.max(exp.length, rows.length);
    if (rows.length !== exp.length && firstDiff < 0) firstDiff = k;
    const free = rows.slice();
    exp.forEach((e, i) => {
      if (rows[i] && rows[i][4] === e[4] && sameRow(rows[i], e)) exact++;
      else if (firstDiff < 0) firstDiff = k;
      const j = free.findIndex((g) => g && sameRow(g, e));
      if (j < 0) return;
      const key = `${e[4]} ${free[j][4]}`;
      shared.set(key, (shared.get(key) ?? 0) + 1);
      free[j] = undefined as unknown as TrackRow;
    });
  });
  let relabelled = 0;
  const usedE = new Set<string>();
  const usedG = new Set<string>();
  for (const [key, n] of [...shared].sort((a, b) => b[1] - a[1])) {
    const [eid, gid] = key.split(" ");
    if (usedE.has(eid) || usedG.has(gid)) continue;
    usedE.add(eid);
    usedG.add(gid);
    relabelled += n;
  }
  return { total, exact, relabelled, firstDiff };
}

/** True when detections.json holds the float32 values themselves rather than 3-decimal roundings. */
const fullPrecision = (d: DetFile) => d.frames.some((f) => f.boxes.some((b) => Number(b[4].toFixed(3)) !== b[4]));

for (const clip of ["C3905", "C3902"]) {
  test(`${clip}: MultiTracker reproduces tracks.json`, async (t: TestContext) => {
    const scene = await loadScene(read);
    const cfg = scene.c.tracking;
    const d: DetFile = await fixture(clip, "detections.json");
    const ref = await fixture(clip, "tracks.json");
    assert.equal(ref.frames.length, d.frames.length);

    const main = new MultiTracker(d.fps, cfg.buffer_sec, cfg);
    const risk = new MultiTracker(d.fps, RISK_BUFFER_SEC, cfg, main.ids);
    const got = d.frames.map((f) => {
      const dets: Detection[] = f.boxes.map(([x1, y1, x2, y2, conf, cls]) => ({ x1, y1, x2, y2, conf, cls }));
      const rows = main.update(dets, [d.height, d.width]);
      risk.update(dets, [d.height, d.width]);
      return rows;
    });
    const r = compare(got, ref.frames.map((f: { rows: number[][] }) => f.rows));
    const pct = (n: number) => ((100 * n) / r.total).toFixed(2);
    t.diagnostic(
      `${clip}: ${r.total} rows; identical with ids ${r.exact} (${pct(r.exact)}%), first difference at frame ${r.firstDiff}; ` +
        `identical up to id renaming ${r.relabelled} (${pct(r.relabelled)}%)`,
    );
    // detections.json holds the detector's float32 values exactly, so every row must match, ids included
    assert.ok(fullPrecision(d), "detections.json must be written at full precision (tools/export_parity_fixtures.py)");
    assert.equal(r.exact, r.total, `rows identical with ids: ${pct(r.exact)}%`);
  });
}

test("ByteTrack edge cases match Ultralytics: thresholds, degenerate boxes, lost buffer, a removed track's last frame", async () => {
  // 5 fps, buffer 2 s: track_buffer 10 frames. Expected rows from Python's MultiTracker on the same
  // detections (float32 as Detections holds them); every box is static, so the Kalman boxes are exact.
  const cfg = (await loadScene(read)).c.tracking;
  const S9 = Math.fround(0.9), S8 = Math.fround(0.8);
  const AT40 = Math.fround(0.4), BELOW40 = 0.3999999761581421, BELOW30 = 0.29999998211860657;
  const AT10 = Math.fround(0.1), ABOVE10 = 0.10000000894069672;
  const A = [100, 100, 200, 180], B = [400, 100, 500, 180], CD = [700, 300, 760, 420];
  const E = [100, 500, 180, 560], F = [300, 500, 380, 560], G = [500, 500, 580, 560];
  const frame = (k: number): number[][] => {
    const b: number[][] = [];
    if ([1, 2, 3, 14, 15].includes(k)) b.push([...A, 0.9, 2]); // back 11 frames after its last one: matched before the buffer check, same id
    if ([1, 2, 3, 15, 17, 18].includes(k)) b.push([...B, 0.9, 2]); // removed at 14 but still in the lost pool at 15: same id; lost at 16 -> gone, new id
    if (k <= 3) b.push([...CD, 0.8, 0], [...CD, 0.8, 2]); // a person and a car on one box: two groups, two tracks
    if (k === 1)
      b.push([...E, AT40, 7], [...F, BELOW40, 7], [...G, AT10, 7], [900, 500, 900, 560, 0.9, 2], [950, 500, 990, 500, 0.9, 2]);
    if (k === 2) b.push([...E, BELOW30, 7]); // a low detection keeps E (second association, plain IoU)
    if (k === 3) b.push([...E, ABOVE10, 7]);
    if (k === 4) b.push([...E, AT10, 7]); // 0.1 exactly is not above the low threshold (float32 compare): E is lost
    return b;
  };
  const first = (sE: number) => [
    [...A, 1000001, S9, 2], [...B, 1000002, S9, 2], [...CD, 1000003, S8, 2], [...E, 1000004, sE, 7], [...CD, 2000005, S8, 0],
  ];
  const expected: Record<number, number[][]> = {
    1: first(AT40), 2: first(BELOW30), 3: first(ABOVE10),
    14: [[...A, 1000001, S9, 2]],
    15: [[...A, 1000001, S9, 2], [...B, 1000002, S9, 2]],
    18: [[...B, 1000006, S9, 2]],
  };
  const trk = new MultiTracker(5, 2.0, cfg);
  for (let k = 1; k <= 18; k++) {
    const dets: Detection[] = frame(k).map(([x1, y1, x2, y2, conf, cls]) => ({ x1, y1, x2, y2, conf, cls }));
    assert.deepEqual(trk.update(dets, [1080, 1920]), expected[k] ?? [], `frame ${k}`);
  }
});

test("collect, groupOf and trackLabel rebuild the tracks trajectories.json was made from", async () => {
  const scene = await loadScene(read);
  const cfg = scene.c.tracking;
  assert.deepEqual(Object.keys(cfg.GROUPS), ["vehicle", "person", "bicycle", "animal"]);
  for (const clip of ["C3905", "C3902"]) {
    const ref = await fixture(clip, "tracks.json");
    const trajs: { tid: number; group: string; cls: number; t: number[] }[] = await fixture(clip, "trajectories.json");
    const tracks = new Map<number, Track>();
    for (const f of ref.frames) collect(tracks, f.t, f.rows);
    for (const tr of tracks.values()) assert.equal(tr.group, groupOf(tr.tid, cfg));
    for (const tj of trajs) {
      const tr = tracks.get(tj.tid);
      assert.ok(tr, `${clip}: track ${tj.tid} missing`);
      assert.equal(tr.group, tj.group);
      assert.equal(trackLabel(tr), tj.cls, `${clip}: label of ${tj.tid}`);
      assert.equal(tr.t.length, tj.t.length);
    }
  }
  const tr: Track = { tid: 1000001, group: "vehicle", t: [], box: [], conf: [], cls: [7, 2, 7, 2, 5] };
  assert.equal(trackLabel(tr), 2); // tie between 2 and 7: np.unique + argmax takes the smaller class
  assert.equal(groupOf(2000017, cfg), "person");
  assert.equal(groupOf(4000001, cfg), "animal");
  assert.equal(groupOf(17, cfg), "animal"); // below the first offset: Python's list(GROUPS)[-1]
});

test("linearAssignment matches lap.lapjv(extend_cost=True, cost_limit)", () => {
  // x (row -> column or -1) as lap.lapjv returns it for these float32 matrices
  const cases: { n: number; m: number; thr: number; cost: number[]; x: number[] }[] = [
    { n: 2, m: 3, thr: 0.8, cost: [0.1, 0.5, 0.9, 0.2, 0.15, 1], x: [0, 1] },
    { n: 3, m: 2, thr: 0.5, cost: [0.6, 1, 0.3, 0.45, 1, 1], x: [-1, 0, -1] },
    { n: 2, m: 2, thr: 0.8, cost: [0.5, 0.5, 0.5, 0.5], x: [0, 1] },
    { n: 3, m: 3, thr: 0.7, cost: [1, 1, 1, 1, 0.9, 1, 1, 1, 1], x: [-1, -1, -1] },
  ];
  for (const c of cases) {
    const r = linearAssignment(c.cost, c.n, c.m, c.thr);
    const x = new Array(c.n).fill(-1);
    for (const [i, j] of r.matches) x[i] = j;
    assert.deepEqual(x, c.x);
    assert.deepEqual(r.unmatchedA, x.flatMap((j, i) => (j < 0 ? [i] : [])));
  }
  assert.deepEqual(linearAssignment([], 2, 0, 0.8), { matches: [], unmatchedA: [0, 1], unmatchedB: [] });
});
