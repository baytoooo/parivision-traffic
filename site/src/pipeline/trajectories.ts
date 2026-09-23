// Tracks as smoothed trajectories in reference coordinates (trajectories.py).
//
// Every box is reduced to its foot point (bottom centre; for vehicles a bit above the
// bottom edge) and mapped into the reference view with H. The foot track is smoothed
// with scipy's uniform_filter1d (mode "nearest") and differentiated like np.gradient.
// Boxes are float32 in Python (tracking.Track.box), so box arithmetic is rounded to
// float32 here too (Math.fround) and only the warped points are float64.

import { warpPoints } from "./geometry.ts";
import type { Mat3 } from "./geometry.ts";
import { trackLabel } from "./tracker.ts";
import type { Track } from "./tracker.ts";
import type { Trajectory } from "./types.ts";

export type { Track };

/** The COCO ids trajectories.py names (detector.py); scene.json `trajectories` holds the same table. */
export interface Classes {
  PERSON: number;
  CAR: number;
  MOTORCYCLE: number;
  BUS: number;
  TRUCK: number;
}

/** COCO ids of the detector. tests/trajectories.test.ts checks them against scene.json `trajectories`. */
export const COCO: Classes = { PERSON: 0, CAR: 2, MOTORCYCLE: 3, BUS: 5, TRUCK: 7 };

const f32 = Math.fround;

export function speed(tr: Pick<Trajectory, "vel">): number[] {
  return tr.vel.map(([vx, vy]) => Math.sqrt(vx * vx + vy * vy));
}

function isVehicleClass(cls: number, k: Classes): boolean {
  return cls === k.CAR || cls === k.BUS || cls === k.TRUCK || cls === k.MOTORCYCLE;
}

export function isVehicle(tr: Pick<Trajectory, "cls">, k: Classes = COCO): boolean {
  return isVehicleClass(tr.cls, k);
}

export function isPerson(tr: Pick<Trajectory, "cls">, k: Classes = COCO): boolean {
  return tr.cls === k.PERSON;
}

/** Index of the sample closest to t if the track exists then. */
export function at(tr: Pick<Trajectory, "t">, t: number): number | null {
  const ts = tr.t;
  if (t < ts[0] - 1e-6 || t > ts[ts.length - 1] + 1e-6) return null;
  let best = 0;
  for (let i = 1; i < ts.length; i++) if (Math.abs(ts[i] - t) < Math.abs(ts[best] - t)) best = i;
  return best;
}

/** Bottom centre of each box, raised by 15% of the box height for vehicles; float32 like NumPy. */
export function footPoints(box: number[][], isVehicle: boolean): number[][] {
  return box.map(([x1, y1, x2, y2]) => {
    const cx = f32(f32(x1 + x2) / 2);
    const fy = isVehicle ? f32(y2 - f32(f32(0.15) * f32(y2 - y1))) : y2;
    return [cx, fy];
  });
}

/** scipy.ndimage.uniform_filter1d(x, size, mode="nearest") on one column, with scipy's running sum. */
export function uniformFilter1d(x: number[], size: number): number[] {
  const n = x.length;
  const size1 = Math.floor(size / 2);
  // the line extended by size1 copies of the first value and size - size1 - 1 of the last
  const line = (i: number) => x[Math.min(n - 1, Math.max(0, i - size1))];
  const out = new Array<number>(n);
  let tmp = 0;
  for (let l = 0; l < size; l++) tmp += line(l);
  out[0] = tmp / size;
  for (let l = 1; l < n; l++) {
    tmp += line(l + size - 1) - line(l - 1);
    out[l] = tmp / size;
  }
  return out;
}

/** np.gradient(f, t) along the samples, edge_order 1 (uniform spacing takes NumPy's scalar branch). */
export function gradient(f: number[], t: number[]): number[] {
  const n = f.length;
  const dx = t.slice(1).map((v, i) => v - t[i]);
  const uniform = dx.every((d) => d === dx[0]);
  const out = new Array<number>(n);
  for (let i = 1; i < n - 1; i++) {
    if (uniform) out[i] = (f[i + 1] - f[i - 1]) / (2 * dx[0]);
    else {
      const dx1 = dx[i - 1], dx2 = dx[i];
      const a = -dx2 / (dx1 * (dx1 + dx2));
      const b = (dx2 - dx1) / (dx1 * dx2);
      const c = dx1 / (dx2 * (dx1 + dx2));
      out[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
    }
  }
  out[0] = (f[1] - f[0]) / dx[0];
  out[n - 1] = (f[n - 1] - f[n - 2]) / dx[n - 2];
  return out;
}

/** Turn raw tracker output into trajectories. H maps work pixels to reference pixels. */
export function build(tracks: Iterable<Track>, H: Mat3, minLen = 5, smooth = 5, k: Classes = COCO): Trajectory[] {
  const out: Trajectory[] = [];
  for (const tr of tracks) {
    if (tr.t.length < minLen) continue;
    const order = tr.t.map((_, i) => i).sort((a, b) => tr.t[a] - tr.t[b]);
    const t = order.map((i) => tr.t[i]);
    const box = order.map((i) => tr.box[i].map(f32));
    const cls = trackLabel(tr);
    const foot = warpPoints(footPoints(box, isVehicleClass(cls, k)), H);
    const top = warpPoints(box.map(([x1, y1, x2]) => [f32(f32(x1 + x2) / 2), y1]), H);
    const height = foot.map(([x, y], i) => {
      const dx = x - top[i][0], dy = y - top[i][1];
      return Math.sqrt(dx * dx + dy * dy);
    });
    const size = Math.min(smooth, t.length);
    const fx = uniformFilter1d(foot.map((p) => p[0]), size);
    const fy = uniformFilter1d(foot.map((p) => p[1]), size);
    const vx = t.length > 1 ? gradient(fx, t) : fx.map(() => 0);
    const vy = t.length > 1 ? gradient(fy, t) : fy.map(() => 0);
    out.push({
      tid: tr.tid,
      group: tr.group,
      cls,
      t,
      box,
      foot: fx.map((x, i) => [x, fy[i]]),
      vel: vx.map((v, i) => [v, vy[i]]),
      height,
      conf: order.map((i) => f32(tr.conf[i])),
    });
  }
  return out;
}
