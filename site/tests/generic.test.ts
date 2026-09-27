// The rules of generic.ts, for a view that is not our junction: the traffic direction learned from
// the clip, and metres per pixel from the size of its vehicles.
import assert from "node:assert/strict";
import { test } from "node:test";
import { learnedWrongWay, VEHICLE_METRES, vehicleScale } from "../src/pipeline/generic.ts";
import type { Detection, Trajectory } from "../src/pipeline/types.ts";

/** A car driving along y at vx px/s from x0, sampled every 0.2 s for `dur` s. */
function car(tid: number, x0: number, y: number, vx: number, t0 = 0, dur = 6): Trajectory {
  const t: number[] = [], box: number[][] = [], foot: number[][] = [], vel: number[][] = [];
  for (let s = 0; s <= dur + 1e-9; s += 0.2) {
    const x = x0 + vx * s;
    t.push(t0 + s);
    box.push([x - 40, y - 50, x + 40, y + 10]);
    foot.push([x, y]);
    vel.push([vx, 0]);
  }
  return { tid, group: "vehicle", cls: 2, t, box, foot, vel, height: t.map(() => 60), conf: t.map(() => 0.9) } as unknown as Trajectory;
}

test("a car against the traffic of its lane is wrong-way; the lane's own traffic is not", () => {
  const lane = Array.from({ length: 40 }, (_, i) => car(1000 + i, 100, 500, 200, i * 1.5));
  const against = car(2000, 1300, 500, -200, 70);
  const ev = learnedWrongWay([...lane, against]);
  assert.deepEqual(ev.map((e) => e.actors), [[2000]]);
  assert.ok(ev[0].end - ev[0].start >= 2);
  assert.deepEqual(learnedWrongWay(lane), []);
});

test("two lanes of opposite traffic give no alarm", () => {
  const east = Array.from({ length: 40 }, (_, i) => car(1000 + i, 100, 500, 200, i * 1.5));
  const west = Array.from({ length: 40 }, (_, i) => car(3000 + i, 1300, 700, -200, i * 1.5));
  assert.deepEqual(learnedWrongWay([...east, ...west]), []);
});

test("vehicleScale fits the box size across the frame", () => {
  // boxes that grow with y: sqrt(area) = 20 + 0.1 * y
  const frames: Detection[][] = [];
  for (let y = 200; y <= 1000; y += 20) {
    const s = 20 + 0.1 * y;
    frames.push([{ x1: 500 - s / 2, y1: y - s, x2: 500 + s / 2, y2: y, conf: 0.9, cls: 2 } as Detection]);
  }
  const mpp = vehicleScale(frames);
  assert.ok(Math.abs(mpp(500, 600) - VEHICLE_METRES / 80) < 1e-9);
  assert.equal(vehicleScale(frames.slice(0, 5))(0, 0), VEHICLE_METRES / 60);
});
