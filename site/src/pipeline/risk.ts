// Part B's accident anticipation, the observe() path of risk.py's Anticipator: one frame's
// detections -> ByteTrack (1.5 s buffer) -> per-user history of foot points in the reference
// view -> constant-velocity hazard of every closing pair (pair_hazard), boosted by hard braking
// and gated by how long the pair has looked dangerous -> the worst pair, smoothed with an EMA.
//
// The browser feeds frames at its own pace, so step()'s frame decimation and time budget are
// not ported. Speeds use scene.metres_per_px at each user's latest foot point; a pair's
// positions share one scale taken midway between them.

import { roundHalfEven, warpPoints } from "./geometry.ts";
import type { Mat3 } from "./geometry.ts";
import type { Scene } from "./scene.ts";
import { MultiTracker } from "./tracker.ts";
import type { Detection, TrackRow } from "./types.ts";

const f32 = Math.fround;

/** risk.py's module constants, from scene.json `risk`. */
export interface RiskConstants {
  PERSON: number;
  TARGET_HZ: number;
  HISTORY: number;
  HORIZON: number;
  STEP: number;
  MIN_SPEED: number;
  CLOSING_MIN: number;
  CLOSING_FULL: number;
  TTC_HALF: number;
  PERSIST_SEC: number;
  RADIUS_M: Record<string, number>;
  EMA: number;
}

type Vec = [number, number];

const norm = (v: Vec) => Math.sqrt(v[0] * v[0] + v[1] * v[1]); // np.linalg.norm
const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1];
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** scene.metres_per_px: metres per reference pixel at (x, y), from the fitted person height. */
export function metresPerPx(scene: Scene, x: number, y: number): number {
  const [c0, cx, cy] = scene.c.person_height_px;
  return 1.7 / Math.max(15.0, c0 + cx * x + cy * y);
}

/**
 * pair_hazard: hazard in [0, 1] for two road users (positions in m, velocities in m/s, radii in m).
 * Constant-velocity closest approach within HORIZON s; the hazard rises with how soon the
 * footprints touch and how deep they overlap, scaled by the closing speed.
 */
export function pairHazard(p1: Vec, v1: Vec, r1: number, p2: Vec, v2: Vec, r2: number, k: RiskConstants): number {
  const relP: Vec = [p2[0] - p1[0], p2[1] - p1[1]];
  const relV: Vec = [v2[0] - v1[0], v2[1] - v1[1]];
  const dist = norm(relP) + 1e-6;
  const closing = -dot(relP, relV) / dist;
  if (closing < k.CLOSING_MIN) return 0.0;
  if (Math.min(norm(v1), norm(v2)) < k.MIN_SPEED) {
    // one of them stands still: only a dead-centre course counts (cars pass parked cars all day)
    r1 = 0.4 * r1;
    r2 = 0.4 * r2;
  }
  const vv = dot(relV, relV) + 1e-6;
  const tStar = clip(-dot(relP, relV) / vv, 0.0, k.HORIZON);
  const at = (s: number) => norm([relP[0] + s * relV[0], relP[1] + s * relV[1]]);
  const gap = at(tStar) - (r1 + r2);
  if (gap >= 0.0) return 0.0;
  // time at which the footprints first touch: np.arange(0, t_star + 1e-9, STEP)
  const n = Math.ceil((tStar + 1e-9) / k.STEP);
  let ttc = tStar;
  for (let i = 0; i < n; i++) {
    if (at(i * k.STEP) < r1 + r2) {
      ttc = i * k.STEP;
      break;
    }
  }
  const depth = Math.min(1.0, -gap / (r1 + r2));
  const soon = 1.0 / (1.0 + Math.exp(2.5 * (ttc - k.TTC_HALF))); // 0.5 at TTC_HALF seconds
  return clip(soon * (0.4 + 0.6 * depth) * Math.min(1.0, closing / k.CLOSING_FULL), 0.0, 1.0);
}

/** Anticipator._braking: 0..1, how hard the user decelerated over the history window (ps in m). */
export function braking(ts: number[], ps: Vec[]): number {
  if (ts.length < 6) return 0.0;
  const n = ts.length;
  const mid = n >> 1;
  const vOld = norm([ps[mid][0] - ps[0][0], ps[mid][1] - ps[0][1]]) / Math.max(1e-3, ts[mid] - ts[0]);
  const vNew = norm([ps[n - 1][0] - ps[mid][0], ps[n - 1][1] - ps[mid][1]]) / Math.max(1e-3, ts[n - 1] - ts[mid]);
  const decel = (vOld - vNew) / Math.max(1e-3, (ts[n - 1] - ts[0]) / 2);
  return vOld > 3.0 ? clip((decel - 4.0) / 6.0, 0.0, 1.0) : 0.0; // m/s^2: >4 is hard braking
}

/** Slope of the least-squares line through (x, y): np.polyfit(x, y, 1)[0]. */
function slope(x: number[], y: number[]): number {
  const n = x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
  }
  return sxy / sxx;
}

/** One history sample: time, foot point in reference px, footprint radius in m, COCO class. */
type Sample = [number, Vec, number, number];

interface User {
  /** Velocity in reference px/s. */
  v: Vec;
  r: number;
  cls: number;
  speed: number;
  braking: number;
  /** Latest foot point in reference px. */
  q: Vec;
  tid: number;
}

export class Anticipator {
  readonly fps: number;
  readonly stride: number;
  readonly width: number;
  readonly height: number;
  score = 0.0;
  private k: RiskConstants;
  private scene: Scene;
  private tracker: MultiTracker;
  private H: Mat3 | null = null;
  private history = new Map<number, Sample[]>();
  private streaks = new Map<string, number>();

  constructor(meta: { fps: number; width: number; height: number }, scene: Scene) {
    this.scene = scene;
    this.k = scene.c.risk as unknown as RiskConstants;
    this.fps = meta.fps || 25.0;
    this.stride = Math.max(1, roundHalfEven(this.fps / this.k.TARGET_HZ));
    this.width = meta.width || 3840;
    this.height = meta.height || 2160;
    this.tracker = new MultiTracker(this.fps / this.stride, Number(scene.c.risk.TRACK_BUFFER_SEC ?? 1.5), scene.c.tracking);
  }

  /** The homography from work pixels to the reference view (Python: self.alignment.H). */
  setAlignment(H: Mat3): void {
    this.H = H;
  }

  /** Tracker + hazard update from one frame's detections (work pixels); returns the smoothed score. */
  observe(dets: Detection[], shape: [number, number], t: number): number {
    const rows = this.tracker.update(dets, shape);
    this.remember(rows, t);
    const raw = this.hazard(t);
    this.score = clip((1 - this.k.EMA) * this.score + this.k.EMA * raw, 0.0, 1.0);
    return this.score;
  }

  private remember(rows: TrackRow[], t: number): void {
    const live = new Set<number>();
    if (rows.length) {
      if (!this.H) throw new Error("Anticipator: setAlignment() before observe()");
      // bottom centre of the float32 boxes, as NumPy computes it on the tracker's float32 rows
      const foot = warpPoints(rows.map(([x1, , x2, y2]) => [f32(x1 + x2) / 2, y2]), this.H);
      rows.forEach((row, i) => {
        const tid = Math.trunc(row[4]);
        const cls = Math.trunc(row[6]);
        const h = this.history.get(tid) ?? [];
        h.push([t, foot[i] as Vec, this.k.RADIUS_M[String(cls)] ?? 1.0, cls]);
        this.history.set(tid, h.slice(-this.k.HISTORY));
        live.add(tid);
      });
    }
    for (const [tid, h] of this.history) {
      if (!live.has(tid) && t - h[h.length - 1][0] > 1.0) this.history.delete(tid);
    }
  }

  private hazard(t: number): number {
    const k = this.k;
    const users: User[] = [];
    for (const [tid, h] of this.history) {
      const last = h[h.length - 1];
      if (h.length < 4 || t - last[0] > 0.25) continue;
      const ts = h.map((s) => s[0]);
      const ps = h.map((s): Vec => [s[1][0], s[1][1]]); // reference pixels
      const dt = ts.map((x) => x - ts[ts.length - 1]);
      // px/s, least squares: robust to box jitter
      const v: Vec = [slope(dt, ps.map((p) => p[0])), slope(dt, ps.map((p) => p[1]))];
      const m = metresPerPx(this.scene, last[1][0], last[1][1]); // for this user's own speed and braking only
      users.push({ v, r: last[2], cls: last[3], speed: norm(v) * m, braking: braking(ts, ps.map((q): Vec => [q[0] * m, q[1] * m])), q: last[1], tid });
    }
    let best = 0.0;
    const streak = new Map<string, number>();
    for (let i = 0; i < users.length; i++) {
      const a = users[i];
      for (let j = i + 1; j < users.length; j++) {
        const b = users[j];
        if (a.cls === k.PERSON && b.cls === k.PERSON) continue;
        // one scale for both users, taken between them: positions scaled by each user's own scale
        // would not share a frame, and users far apart would look close
        const m = metresPerPx(this.scene, (a.q[0] + b.q[0]) / 2, (a.q[1] + b.q[1]) / 2);
        const pa: Vec = [a.q[0] * m, a.q[1] * m], pb: Vec = [b.q[0] * m, b.q[1] * m];
        if (Math.max(a.speed, b.speed) < k.MIN_SPEED || norm([pb[0] - pa[0], pb[1] - pa[1]]) > 30.0) continue;
        // moving car vs parked or queued car: never on its own
        if (Math.min(a.speed, b.speed) < k.MIN_SPEED && a.cls !== k.PERSON && b.cls !== k.PERSON) continue;
        if ((a.cls === k.PERSON && !this.onRoad(a.q)) || (b.cls === k.PERSON && !this.onRoad(b.q))) continue;
        const ca = this.carriageway(a.q), cb = this.carriageway(b.q);
        if (ca !== cb && ca !== 0 && cb !== 0) continue; // opposite sides of the median
        const hz = pairHazard(pa, [a.v[0] * m, a.v[1] * m], a.r, pb, [b.v[0] * m, b.v[1] * m], b.r, k);
        if (hz <= 0.0) continue;
        const key = `${Math.min(a.tid, b.tid)} ${Math.max(a.tid, b.tid)}`;
        const since = this.streaks.get(key) ?? t; // when the pair started to look dangerous
        streak.set(key, since);
        if (t - since >= k.PERSIST_SEC - 1e-6) best = Math.max(best, Math.min(1.0, hz * (1.0 + 0.5 * Math.max(a.braking, b.braking))));
      }
    }
    this.streaks = streak;
    return best;
  }

  /** _road_mask: the carriageway including the crossings. */
  private onRoad(p: Vec): boolean {
    return this.scene.mask("road", p[0], p[1]) === 1;
  }

  /** _carriageways: 0 elsewhere, 1 southbound, 2 northbound (nb is filled last, so it wins an overlap). */
  private carriageway(p: Vec): number {
    if (this.scene.zone("nb", p[0], p[1])) return 2;
    return this.scene.zone("sb", p[0], p[1]) ? 1 : 0;
  }
}
