// What the page can still find on a view that is not our junction (the first frame, and the keyframes'
// median at the end, did not register). The zebras, the stop line, the lamps and the carriageways are
// drawn for our junction only, so those rules stay off. Two need no map of the place:
//   - wrong-way driving, against the direction the traffic takes in each part of the frame, learned
//     from the clip itself (browser only; checked with the same settings in Python on our four clips
//     at the demo's 5 frames per second, as if the junction were unknown: 948 of 1072 real vehicle
//     tracks run backwards found, and no alarm in the 18 minutes of normal traffic);
//   - collisions (rules.ts collisions), in metres from the size of the vehicles in the clip, as
//     tools/crash_check.py checks the accident rule on other cameras.
// The risk model runs the same way as crash_check.replay: the whole frame counts as road, there is no
// median, and metres come from the vehicle sizes.

import { runs } from "./segments.ts";
import { COCO, isVehicle, type Classes } from "./trajectories.ts";
import type { Detection, Evidence, Trajectory } from "./types.ts";

/** Median sqrt(box area) of a car in metres on our own clips (tools/crash_check.py vehicle_metres()). */
export const VEHICLE_METRES = 2.3737;

export const FLOW = {
  cell: 40,      // work px per side of a cell of the direction map
  rel: 0.3,      // moving: faster than this many of its own box sizes (sqrt of the area) per second
  minR: 0.9,     // a cell has one direction when the mean of its unit headings is at least this long
  minN: 30,      // samples of moving vehicles in the cell, this track left out
  minTracks: 4,  // distinct other tracks in the cell
  angle: 135,    // degrees away from the cell's direction
  minDur: 2.0,   // s against the flow
  gap: 0.5,      // s of break a run bridges
};

type Cell = { sx: number; sy: number; n: number; ids: Set<number> };

function boxScale(tr: Trajectory, i: number): number {
  const [x1, y1, x2, y2] = tr.box[i];
  return Math.sqrt(Math.max((x2 - x1) * (y2 - y1), 1));
}

function movingSamples(tr: Trajectory, rel: number): boolean[] {
  return tr.vel.map(([vx, vy], i) => Math.sqrt(vx * vx + vy * vy) > rel * boxScale(tr, i));
}

const cellKey = (p: readonly number[], cell: number) => `${Math.floor(p[0] / cell)},${Math.floor(p[1] / cell)}`;

/** Motor vehicles only: cyclists ride along the zebras with the pedestrians. */
export function learnedWrongWay(trajectories: Trajectory[], k: Classes = COCO, f = FLOW) {
  const movers = trajectories.filter((tr) => isVehicle(tr, k));
  const field = new Map<string, Cell>();
  const moving = new Map<number, boolean[]>();
  for (const tr of movers) {
    const mv = movingSamples(tr, f.rel);
    moving.set(tr.tid, mv);
    tr.foot.forEach((p, i) => {
      if (!mv[i]) return;
      const [vx, vy] = tr.vel[i];
      const s = Math.sqrt(vx * vx + vy * vy);
      const key = cellKey(p, f.cell);
      let c = field.get(key);
      if (!c) field.set(key, (c = { sx: 0, sy: 0, n: 0, ids: new Set() }));
      c.sx += vx / s;
      c.sy += vy / s;
      c.n += 1;
      c.ids.add(tr.tid);
    });
  }
  const cosMax = Math.cos((f.angle * Math.PI) / 180);
  const out: Evidence[] = [];
  for (const tr of movers) {
    const mv = moving.get(tr.tid)!;
    if (mv.filter(Boolean).length < 5) continue;
    // this track's own share of each cell, left out when it is judged
    const own = new Map<string, [number, number, number]>();
    tr.foot.forEach((p, i) => {
      if (!mv[i]) return;
      const [vx, vy] = tr.vel[i];
      const s = Math.sqrt(vx * vx + vy * vy);
      const o = own.get(cellKey(p, f.cell)) ?? [0, 0, 0];
      own.set(cellKey(p, f.cell), [o[0] + vx / s, o[1] + vy / s, o[2] + 1]);
    });
    const bad = tr.foot.map((p, i) => {
      if (!mv[i]) return false;
      const key = cellKey(p, f.cell);
      const c = field.get(key);
      if (!c) return false;
      const o = own.get(key) ?? [0, 0, 0];
      const sx = c.sx - o[0], sy = c.sy - o[1], n = c.n - o[2];
      const others = c.ids.size - (c.ids.has(tr.tid) ? 1 : 0);
      if (n < f.minN || others < f.minTracks) return false;
      const len = Math.sqrt(sx * sx + sy * sy);
      if (len / n < f.minR) return false;
      const [vx, vy] = tr.vel[i];
      const s = Math.sqrt(vx * vx + vy * vy);
      return (vx * sx + vy * sy) / (s * len) < cosMax;
    });
    for (const [s, e] of runs(tr.t, bad, f.gap)) {
      if (e - s >= f.minDur) out.push({ label: "wrong_way", start: s, end: e, actors: [tr.tid], note: "against the traffic here" });
    }
  }
  return out;
}

/** np.percentile, linear interpolation. */
function percentile(v: number[], q: number): number {
  const s = [...v].sort((a, b) => a - b);
  const pos = ((s.length - 1) * q) / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/**
 * tools/crash_check.py vehicle_scale: metres per work pixel at (x, y), from a plane fitted to the size
 * of the confident vehicle boxes of the clip (sqrt of the area against the bottom centre).
 */
export function vehicleScale(frames: Detection[][], k: Classes = COCO): (x: number, y: number) => number {
  const vehicles = new Set([k.CAR, k.MOTORCYCLE, k.BUS, k.TRUCK]);
  const pts: [number, number, number][] = [];
  for (const dets of frames)
    for (const d of dets)
      if (d.conf > 0.4 && vehicles.has(d.cls)) pts.push([(d.x1 + d.x2) / 2, d.y2, Math.sqrt((d.x2 - d.x1) * (d.y2 - d.y1))]);
  if (pts.length < 20) return () => VEHICLE_METRES / 60.0;
  // least squares for size = c0 + cx * x + cy * y, through the 3 x 3 normal equations
  const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (const [x, y, s] of pts) {
    const a = [1, x, y];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) M[r][c] += a[r] * a[c];
      M[r][3] += a[r] * s;
    }
  }
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < 3; r++) {
      if (r === c || M[c][c] === 0) continue;
      const f = M[r][c] / M[c][c];
      for (let j = c; j < 4; j++) M[r][j] -= f * M[c][j];
    }
  }
  const coef = M.map((row, i) => (row[i] ? row[3] / row[i] : 0));
  const lo = percentile(pts.map((p) => p[2]), 5);
  return (x, y) => VEHICLE_METRES / Math.max(lo, coef[0] + coef[1] * x + coef[2] * y);
}
