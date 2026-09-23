// Multi-object tracking (tracking.py: MultiTracker, Track, collect, group_of) on the ByteTrack
// of Ultralytics 8.4.159 it wraps: trackers/byte_tracker.py, basetrack.py, utils/kalman_filter.py
// (KalmanFilterXYAH), utils/matching.py, utils/stracks.py, and lap.lapjv for the assignment.
//
// NumPy runs part of this in float32: detection boxes and scores, the IoU and fused cost
// matrices, and the thresholds they are compared with (NEP 50 casts a Python float to the
// array's float32). Those steps go through Math.fround so every threshold falls the same
// way; the Kalman state is float64 as in Python.

import { roundHalfEven } from "./geometry.ts";
import type { SceneConstants } from "./scene.ts";
import type { Detection, Group, TrackRow } from "./types.ts";

const f32 = Math.fround;

type TrackingConstants = SceneConstants["tracking"];

// basetrack.TrackState
const NEW = 0;
const TRACKED = 1;
const LOST = 2;
const REMOVED = 3;

/**
 * BaseTrack._count. In Python it is a class attribute: every tracker of the process draws ids
 * from it, and constructing any BYTETracker resets it to 0. Pass one counter to several
 * MultiTrackers to reproduce a process that runs them side by side.
 */
export interface IdCounter {
  count: number;
}

// ---------------------------------------------------------------- Kalman filter (XYAH)

const STD_POS = 1.0 / 20;
const STD_VEL = 1.0 / 160;

/** KalmanFilterXYAH.initiate: measurement (x, y, a, h) -> mean (8), covariance (8x8 row-major). */
function kfInitiate(m: number[]): [number[], number[]] {
  const mean = [m[0], m[1], m[2], m[3], 0, 0, 0, 0];
  const h = m[3];
  const std = [2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h, 10 * STD_VEL * h, 10 * STD_VEL * h, 1e-5, 10 * STD_VEL * h];
  const cov = new Array(64).fill(0);
  for (let i = 0; i < 8; i++) cov[i * 9] = std[i] * std[i];
  return [mean, cov];
}

/** KalmanFilterXYAH.multi_predict for one state (constant velocity, dt = 1). */
function kfPredict(mean: number[], cov: number[]): [number[], number[]] {
  const h = mean[3];
  const std = [STD_POS * h, STD_POS * h, 1e-2, STD_POS * h, STD_VEL * h, STD_VEL * h, 1e-5, STD_VEL * h];
  const m = mean.slice();
  for (let i = 0; i < 4; i++) m[i] = mean[i] + mean[i + 4];
  // F P F^T: F adds row/column i+4 to row/column i for i < 4
  const fp = cov.slice();
  for (let i = 0; i < 4; i++) for (let k = 0; k < 8; k++) fp[i * 8 + k] = cov[i * 8 + k] + cov[(i + 4) * 8 + k];
  const p = fp.slice();
  for (let i = 0; i < 8; i++) for (let l = 0; l < 4; l++) p[i * 8 + l] = fp[i * 8 + l] + fp[i * 8 + l + 4];
  for (let i = 0; i < 8; i++) p[i * 9] += std[i] * std[i];
  return [m, p];
}

/**
 * Solve A X = B in place for a 4x4 A and a 4 x 8 B (both row-major), by LU with partial pivoting
 * like LAPACK gesv. X is left in B.
 */
function solve4x8(a: Float64Array, b: Float64Array): void {
  for (let c = 0; c < 4; c++) {
    let piv = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(a[r * 4 + c]) > Math.abs(a[piv * 4 + c])) piv = r;
    if (piv !== c) {
      for (let k = 0; k < 4; k++) {
        const t = a[c * 4 + k];
        a[c * 4 + k] = a[piv * 4 + k];
        a[piv * 4 + k] = t;
      }
      for (let k = 0; k < 8; k++) {
        const t = b[c * 8 + k];
        b[c * 8 + k] = b[piv * 8 + k];
        b[piv * 8 + k] = t;
      }
    }
    for (let r = c + 1; r < 4; r++) {
      const f = a[r * 4 + c] / a[c * 4 + c];
      for (let k = c; k < 4; k++) a[r * 4 + k] -= f * a[c * 4 + k];
      for (let k = 0; k < 8; k++) b[r * 8 + k] -= f * b[c * 8 + k];
    }
  }
  for (let c = 3; c >= 0; c--) {
    for (let k = 0; k < 8; k++) {
      let s = b[c * 8 + k];
      for (let j = c + 1; j < 4; j++) s -= a[c * 4 + j] * b[j * 8 + k];
      b[c * 8 + k] = s / a[c * 4 + c];
    }
  }
}

/** KalmanFilterXYAH.update (project + correct), without the NSA confidence scaling ByteTrack does not use. */
function kfUpdate(mean: number[], cov: number[], meas: number[]): [number[], number[]] {
  const h = mean[3];
  const std = [STD_POS * h, STD_POS * h, 1e-1, STD_POS * h];
  const S = new Float64Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) S[i * 4 + j] = cov[i * 8 + j] + (i === j ? std[i] * std[i] : 0);
  // kalman_gain = solve(S, P[:, :4].T).T: KT (4x8) holds K transposed, K[i][j] = KT[j * 8 + i]
  const KT = new Float64Array(32);
  for (let j = 0; j < 4; j++) for (let i = 0; i < 8; i++) KT[j * 8 + i] = cov[i * 8 + j];
  solve4x8(S.slice(), KT);
  const innov = [meas[0] - mean[0], meas[1] - mean[1], meas[2] - mean[2], meas[3] - mean[3]];
  const m = mean.slice();
  for (let i = 0; i < 8; i++) {
    let s = 0;
    for (let j = 0; j < 4; j++) s += innov[j] * KT[j * 8 + i];
    m[i] = mean[i] + s;
  }
  // covariance - K S K^T, evaluated as K (S K^T) like np.linalg.multi_dot
  const SKT = new Float64Array(32);
  for (let a = 0; a < 4; a++)
    for (let c = 0; c < 8; c++) {
      let s = 0;
      for (let b = 0; b < 4; b++) s += S[a * 4 + b] * KT[b * 8 + c];
      SKT[a * 8 + c] = s;
    }
  const p = cov.slice();
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      let s = 0;
      for (let a = 0; a < 4; a++) s += KT[a * 8 + r] * SKT[a * 8 + c];
      p[r * 8 + c] = cov[r * 8 + c] - s;
    }
  return [m, p];
}

// ---------------------------------------------------------------- STrack

/** STrack: one detection, and once activated a Kalman track. */
class STrack {
  trackId = 0;
  isActivated = false;
  state = NEW;
  score: number;
  cls: number;
  startFrame = 0;
  frameId = 0;
  trackletLen = 0;
  /** _tlwh: the detection box, float32. */
  private tlwh0: number[];
  mean: number[] | null = null;
  cov: number[] | null = null;

  /** From one float32 detection row, via Boxes.xywh -> parse_bboxes -> xywh2ltwh. */
  constructor(cx: number, cy: number, w: number, h: number, score: number, cls: number) {
    this.tlwh0 = [f32(cx - w / 2), f32(cy - h / 2), w, h];
    this.score = score;
    this.cls = cls;
  }

  /** tlwh: the Kalman state when there is one (float64), else the detection (float32). */
  tlwh(): number[] {
    if (!this.mean) return this.tlwh0.slice();
    const [x, y, a, h] = this.mean;
    const w = a * h;
    return [x - w / 2, y - h / 2, w, h];
  }

  xyxy(): number[] {
    const [x, y, w, h] = this.tlwh();
    // a detection's box is a float32 array, a track's float64
    return this.mean ? [x, y, w + x, h + y] : [x, y, f32(w + x), f32(h + y)];
  }

  /** tlwh_to_xyah of a detection box (float32 arithmetic). */
  xyah(): number[] {
    const [x, y, w, h] = this.tlwh0;
    return [f32(x + w / 2), f32(y + h / 2), f32(w / h), h];
  }

  activate(frameId: number, ids: IdCounter): void {
    this.trackId = ++ids.count;
    [this.mean, this.cov] = kfInitiate(this.xyah());
    this.trackletLen = 0;
    this.state = TRACKED;
    if (frameId === 1) this.isActivated = true;
    this.frameId = frameId;
    this.startFrame = frameId;
  }

  reActivate(det: STrack, frameId: number): void {
    [this.mean, this.cov] = kfUpdate(this.mean!, this.cov!, det.xyah());
    this.trackletLen = 0;
    this.state = TRACKED;
    this.isActivated = true;
    this.frameId = frameId;
    this.score = det.score;
    this.cls = det.cls;
  }

  update(det: STrack, frameId: number): void {
    this.frameId = frameId;
    this.trackletLen += 1;
    [this.mean, this.cov] = kfUpdate(this.mean!, this.cov!, det.xyah());
    this.state = TRACKED;
    this.isActivated = true;
    this.score = det.score;
    this.cls = det.cls;
  }
}

/** STrack.multi_predict: a track that is not Tracked has its height velocity zeroed first. */
function multiPredict(tracks: STrack[]): void {
  for (const t of tracks) {
    const mean = t.mean!.slice();
    if (t.state !== TRACKED) mean[7] = 0;
    [t.mean, t.cov] = kfPredict(mean, t.cov!);
  }
}

// ---------------------------------------------------------------- matching

const EPS32 = f32(1e-7);

/** matching.iou_distance: 1 - IoU (bbox_ioa with iou=True), in float32. Row-major n x m. */
function iouDistance(a: STrack[], b: STrack[]): number[] {
  const A = a.map((t) => t.xyxy().map(f32));
  const B = b.map((t) => t.xyxy().map(f32));
  const out = new Array(A.length * B.length);
  for (let i = 0; i < A.length; i++) {
    const [ax1, ay1, ax2, ay2] = A[i];
    const areaA = f32(f32(ax2 - ax1) * f32(ay2 - ay1));
    for (let j = 0; j < B.length; j++) {
      const [bx1, by1, bx2, by2] = B[j];
      const iw = Math.max(f32(Math.min(ax2, bx2) - Math.max(ax1, bx1)), 0);
      const ih = Math.max(f32(Math.min(ay2, by2) - Math.max(ay1, by1)), 0);
      const inter = f32(iw * ih);
      const areaB = f32(f32(bx2 - bx1) * f32(by2 - by1));
      const union = f32(f32(areaB + areaA) - inter);
      out[i * B.length + j] = f32(1 - f32(inter / f32(union + EPS32)));
    }
  }
  return out;
}

/** matching.fuse_score: 1 - (1 - cost) * score, in float32. */
function fuseScore(cost: number[], dets: STrack[]): number[] {
  const m = dets.length;
  return cost.map((c, k) => f32(1 - f32(f32(1 - c) * dets[k % m].score)));
}

// lap.lapjv (Jonker-Volgenant, dense), step for step as lap's lapjv.cpp so that ties resolve
// the same way. cost is a row-major n x n matrix.
const LARGE = 1000000;

function ccrrt(n: number, cost: Float64Array, freeRows: Int32Array, x: Int32Array, y: Int32Array, v: Float64Array): number {
  for (let i = 0; i < n; i++) {
    x[i] = -1;
    v[i] = LARGE;
    y[i] = 0;
  }
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const c = cost[i * n + j];
      if (c < v[j]) {
        v[j] = c;
        y[j] = i;
      }
    }
  const unique = new Uint8Array(n).fill(1);
  let j = n;
  do {
    j--;
    const i = y[j];
    if (x[i] < 0) x[i] = j;
    else {
      unique[i] = 0;
      y[j] = -1;
    }
  } while (j > 0);
  let nFree = 0;
  for (let i = 0; i < n; i++) {
    if (x[i] < 0) freeRows[nFree++] = i;
    else if (unique[i]) {
      const jj = x[i];
      let min = LARGE;
      for (let j2 = 0; j2 < n; j2++) {
        if (j2 === jj) continue;
        const c = cost[i * n + j2] - v[j2];
        if (c < min) min = c;
      }
      v[jj] -= min;
    }
  }
  return nFree;
}

function carr(n: number, cost: Float64Array, nFreeRows: number, freeRows: Int32Array, x: Int32Array, y: Int32Array, v: Float64Array): number {
  let current = 0;
  let newFree = 0;
  let rrCnt = 0;
  while (current < nFreeRows) {
    rrCnt++;
    const freeI = freeRows[current++];
    let j1 = 0;
    let v1 = cost[freeI * n] - v[0];
    let j2 = -1;
    let v2 = LARGE;
    for (let j = 1; j < n; j++) {
      const c = cost[freeI * n + j] - v[j];
      if (c < v2) {
        if (c >= v1) {
          v2 = c;
          j2 = j;
        } else {
          v2 = v1;
          v1 = c;
          j2 = j1;
          j1 = j;
        }
      }
    }
    let i0 = y[j1];
    const v1New = v[j1] - (v2 - v1);
    const v1Lowers = v1New < v[j1];
    if (rrCnt < current * n) {
      if (v1Lowers) v[j1] = v1New;
      else if (i0 >= 0 && j2 >= 0) {
        j1 = j2;
        i0 = y[j2];
      }
      if (i0 >= 0) {
        if (v1Lowers) freeRows[--current] = i0;
        else freeRows[newFree++] = i0;
      }
    } else if (i0 >= 0) freeRows[newFree++] = i0;
    x[freeI] = j1;
    y[j1] = freeI;
  }
  return newFree;
}

function findDense(n: number, lo: number, d: Float64Array, cols: Int32Array): number {
  let hi = lo + 1;
  let mind = d[cols[lo]];
  for (let k = hi; k < n; k++) {
    const j = cols[k];
    if (d[j] <= mind) {
      if (d[j] < mind) {
        hi = lo;
        mind = d[j];
      }
      cols[k] = cols[hi];
      cols[hi++] = j;
    }
  }
  return hi;
}

/** _scan_dense; returns a free column, or -1 after storing the new lo/hi in span. */
function scanDense(n: number, cost: Float64Array, span: [number, number], d: Float64Array, cols: Int32Array, pred: Int32Array, y: Int32Array, v: Float64Array): number {
  let [lo, hi] = span;
  while (lo !== hi) {
    let j = cols[lo++];
    const i = y[j];
    const mind = d[j];
    const h = cost[i * n + j] - v[j] - mind;
    for (let k = hi; k < n; k++) {
      j = cols[k];
      const cred = cost[i * n + j] - v[j] - h;
      if (cred < d[j]) {
        d[j] = cred;
        pred[j] = i;
        if (cred === mind) {
          if (y[j] < 0) return j;
          cols[k] = cols[hi];
          cols[hi++] = j;
        }
      }
    }
  }
  span[0] = lo;
  span[1] = hi;
  return -1;
}

function findPathDense(n: number, cost: Float64Array, startI: number, y: Int32Array, v: Float64Array, pred: Int32Array): number {
  const span: [number, number] = [0, 0];
  let finalJ = -1;
  let nReady = 0;
  const cols = new Int32Array(n);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    cols[i] = i;
    pred[i] = startI;
    d[i] = cost[startI * n + i] - v[i];
  }
  while (finalJ === -1) {
    if (span[0] === span[1]) {
      nReady = span[0];
      span[1] = findDense(n, span[0], d, cols);
      for (let k = span[0]; k < span[1]; k++) {
        const j = cols[k];
        if (y[j] < 0) finalJ = j;
      }
    }
    if (finalJ === -1) finalJ = scanDense(n, cost, span, d, cols, pred, y, v);
  }
  const mind = d[cols[span[0]]];
  for (let k = 0; k < nReady; k++) {
    const j = cols[k];
    v[j] += d[j] - mind;
  }
  return finalJ;
}

function caDense(n: number, cost: Float64Array, nFree: number, freeRows: Int32Array, x: Int32Array, y: Int32Array, v: Float64Array): void {
  const pred = new Int32Array(n);
  for (let f = 0; f < nFree; f++) {
    const freeI = freeRows[f];
    let i = -1;
    let k = 0;
    let j = findPathDense(n, cost, freeI, y, v, pred);
    while (i !== freeI) {
      i = pred[j];
      y[j] = i;
      const tmp = x[i];
      x[i] = j;
      j = tmp;
      if (++k >= n) throw new Error("lapjv: augmentation does not finish");
    }
  }
}

/**
 * matching.linear_assignment(cost, thresh): lap.lapjv(cost, extend_cost=True, cost_limit=thresh).
 * The n x m matrix is extended to (n + m) square with cost_limit / 2 for leaving a row or a column
 * unmatched, so a pair is matched only if that beats leaving both alone.
 */
export function linearAssignment(
  cost: ArrayLike<number>,
  nRows: number,
  nCols: number,
  thresh: number,
): { matches: [number, number][]; unmatchedA: number[]; unmatchedB: number[] } {
  if (nRows === 0 || nCols === 0)
    return { matches: [], unmatchedA: [...Array(nRows).keys()], unmatchedB: [...Array(nCols).keys()] };
  const n = nRows + nCols;
  const c = new Float64Array(n * n).fill(thresh / 2);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i < nRows && j < nCols) c[i * n + j] = cost[i * nCols + j];
      else if (i >= nRows && j >= nCols) c[i * n + j] = 0;
    }
  const x = new Int32Array(n);
  const y = new Int32Array(n);
  const freeRows = new Int32Array(n);
  const v = new Float64Array(n);
  let nFree = ccrrt(n, c, freeRows, x, y, v);
  for (let pass = 0; nFree > 0 && pass < 2; pass++) nFree = carr(n, c, nFree, freeRows, x, y, v);
  if (nFree > 0) caDense(n, c, nFree, freeRows, x, y, v);
  const matches: [number, number][] = [];
  const unmatchedA: number[] = [];
  const unmatchedB: number[] = [];
  for (let i = 0; i < nRows; i++) {
    if (x[i] < nCols) matches.push([i, x[i]]);
    else unmatchedA.push(i);
  }
  for (let j = 0; j < nCols; j++) if (y[j] >= nRows) unmatchedB.push(j);
  return { matches, unmatchedA, unmatchedB };
}

// ---------------------------------------------------------------- track pools (utils/stracks.py)

function jointStracks(a: STrack[], b: STrack[]): STrack[] {
  const ids = new Set(a.map((t) => t.trackId));
  return a.concat(b.filter((t) => !ids.has(t.trackId)));
}

function subStracks(a: STrack[], b: STrack[]): STrack[] {
  const ids = new Set(b.map((t) => t.trackId));
  return a.filter((t) => !ids.has(t.trackId));
}

const DUP_THRESH = f32(0.15);

/** remove_duplicate_stracks: of two tracks with IoU > 0.85 the younger one goes (ties: the one in a). */
function removeDuplicateStracks(a: STrack[], b: STrack[]): [STrack[], STrack[]] {
  const pdist = iouDistance(a, b);
  const dupa = new Set<number>();
  const dupb = new Set<number>();
  for (let p = 0; p < a.length; p++)
    for (let q = 0; q < b.length; q++) {
      if (!(pdist[p * b.length + q] < DUP_THRESH)) continue;
      const timep = a[p].frameId - a[p].startFrame;
      const timeq = b[q].frameId - b[q].startFrame;
      if (timep > timeq) dupb.add(q);
      else dupa.add(p);
    }
  return [a.filter((_, i) => !dupa.has(i)), b.filter((_, i) => !dupb.has(i))];
}

// ---------------------------------------------------------------- BYTETracker

interface ByteArgs {
  trackHigh: number;
  trackLow: number;
  newTrack: number;
  match: number;
  fuse: boolean;
  trackBuffer: number;
}

/** One detection row as the float32 array tracking.py hands to Boxes. */
interface DetRow {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
}

class ByteTracker {
  tracked: STrack[] = [];
  lost: STrack[] = [];
  removed: STrack[] = [];
  frameId = 0;
  private args: ByteArgs;
  private maxFramesLost: number;
  private ids: IdCounter;

  constructor(args: ByteArgs, ids: IdCounter) {
    this.args = args;
    // this Ultralytics version takes no frame_rate: max_frames_lost is track_buffer as is
    this.maxFramesLost = args.trackBuffer;
    this.ids = ids;
    ids.count = 0; // BYTETracker.__init__ -> reset_id()
  }

  private dists(tracks: STrack[], dets: STrack[]): number[] {
    const d = iouDistance(tracks, dets);
    return this.args.fuse ? fuseScore(d, dets) : d;
  }

  private applyMatches(matches: [number, number][], pool: STrack[], dets: STrack[], activated: STrack[], refind: STrack[]): void {
    for (const [it, id] of matches) {
      const track = pool[it];
      if (track.state === TRACKED) {
        track.update(dets[id], this.frameId);
        activated.push(track);
      } else {
        track.reActivate(dets[id], this.frameId);
        refind.push(track);
      }
    }
  }

  /** BYTETracker.update: returns rows [x1, y1, x2, y2, track_id, score, cls] (float32). */
  update(rows: DetRow[]): number[][] {
    this.frameId += 1;
    const activated: STrack[] = [];
    const refind: STrack[] = [];
    const lostNow: STrack[] = [];
    const removedNow: STrack[] = [];
    const a = this.args;

    // _split_detections + init_track (Boxes.xywh in float32)
    const high: STrack[] = [];
    const low: STrack[] = [];
    for (const r of rows) {
      const w = f32(r.x2 - r.x1);
      const h = f32(r.y2 - r.y1);
      if (!(w > 0 && h > 0)) continue;
      const s = r.conf;
      const make = () => new STrack(f32(r.x1 + r.x2) / 2, f32(r.y1 + r.y2) / 2, w, h, s, r.cls);
      if (s >= a.trackHigh) high.push(make());
      else if (s > a.trackLow && s < a.trackHigh) low.push(make());
    }

    const unconfirmed = this.tracked.filter((t) => !t.isActivated);
    const confirmed = this.tracked.filter((t) => t.isActivated);
    const pool = jointStracks(confirmed, this.lost);
    multiPredict(pool);

    // first association: confirmed + lost tracks with high detections, fused IoU
    const first = linearAssignment(this.dists(pool, high), pool.length, high.length, a.match);
    this.applyMatches(first.matches, pool, high, activated, refind);

    // second association: still-tracked leftovers with low detections, plain IoU
    const rTracked = first.unmatchedA.map((i) => pool[i]).filter((t) => t.state === TRACKED);
    let uTrack: number[];
    if (rTracked.length && low.length) {
      const second = linearAssignment(iouDistance(rTracked, low), rTracked.length, low.length, 0.5);
      this.applyMatches(second.matches, rTracked, low, activated, refind);
      uTrack = second.unmatchedA;
    } else uTrack = [...rTracked.keys()];
    for (const it of uTrack) {
      const track = rTracked[it];
      if (track.state !== LOST) {
        track.state = LOST;
        lostNow.push(track);
      }
    }

    // unconfirmed tracks (one frame old) with the high detections left over
    const dets = first.unmatchedB.map((i) => high[i]);
    let uDet: number[];
    if (!unconfirmed.length) uDet = [...dets.keys()];
    else {
      const third = linearAssignment(this.dists(unconfirmed, dets), unconfirmed.length, dets.length, 0.7);
      for (const [it, id] of third.matches) {
        unconfirmed[it].update(dets[id], this.frameId);
        activated.push(unconfirmed[it]);
      }
      for (const it of third.unmatchedA) {
        unconfirmed[it].state = REMOVED;
        removedNow.push(unconfirmed[it]);
      }
      uDet = third.unmatchedB;
    }

    for (const inew of uDet) {
      const track = dets[inew];
      if (track.score < a.newTrack) continue;
      track.activate(this.frameId, this.ids);
      activated.push(track);
    }

    for (const track of this.lost) {
      if (this.frameId - track.frameId > this.maxFramesLost) {
        track.state = REMOVED;
        removedNow.push(track);
      }
    }

    // merge_track_pools. The lost pool is cleaned with the removed history *before* this
    // frame's removals are added, so a stale track lingers one more frame, as in Python.
    this.tracked = this.tracked.filter((t) => t.state === TRACKED);
    this.tracked = jointStracks(this.tracked, activated);
    this.tracked = jointStracks(this.tracked, refind);
    this.lost = subStracks(this.lost, this.tracked);
    this.lost.push(...lostNow);
    this.lost = subStracks(this.lost, this.removed);
    [this.tracked, this.lost] = removeDuplicateStracks(this.tracked, this.lost);
    this.removed.push(...removedNow);
    if (this.removed.length > 1000) this.removed = this.removed.slice(-1000);

    return this.tracked
      .filter((t) => t.isActivated)
      .map((t) => [...t.xyxy().map(f32), t.trackId, t.score, t.cls]);
  }
}

// ---------------------------------------------------------------- tracking.py

/**
 * One ByteTrack per object group (tracking.GROUPS, in scene.json order); ids are offset by
 * (group index + 1) * 1_000_000 so they never collide. The group trackers share one id counter.
 */
export class MultiTracker {
  private groups: [number[], ByteTracker][];
  readonly ids: IdCounter;

  /** `ids`: pass another tracker's counter to share it (Python's BaseTrack._count is process-wide). */
  constructor(fps: number, bufferSec: number, cfg: TrackingConstants, ids: IdCounter = { count: 0 }) {
    const b = cfg.bytetrack;
    const args: ByteArgs = {
      trackHigh: f32(b.track_high_thresh),
      trackLow: f32(b.track_low_thresh),
      newTrack: f32(b.new_track_thresh),
      match: b.match_thresh,
      fuse: b.fuse_score,
      trackBuffer: Math.max(1, roundHalfEven(bufferSec * fps)),
    };
    this.ids = ids;
    this.groups = Object.values(cfg.GROUPS).map((classes) => [classes, new ByteTracker(args, ids)]);
  }

  /** Rows [x1, y1, x2, y2, track_id, score, cls] for the active tracks, float32 like the Python array. */
  update(dets: Detection[], shape: [number, number]): TrackRow[] {
    void shape; // Boxes' orig_shape; ByteTrack does not read it
    const out: TrackRow[] = [];
    this.groups.forEach(([classes, trk], gi) => {
      const mine = dets.filter((d) => classes.includes(d.cls));
      if (!mine.length && !trk.tracked.length && !trk.lost.length) {
        trk.frameId += 1; // nothing to associate; just keep the frame clock in step
        return;
      }
      const rows = mine.map((d) => ({ x1: f32(d.x1), y1: f32(d.y1), x2: f32(d.x2), y2: f32(d.y2), conf: f32(d.conf), cls: f32(d.cls) }));
      const offset = (gi + 1) * 1_000_000;
      for (const r of trk.update(rows)) out.push([r[0], r[1], r[2], r[3], f32(r[4] + offset), r[5], r[6]]);
    });
    return out;
  }
}

/** A tracked object's samples, as tracking.collect gathers them (tracking.py Track). */
export interface Track {
  tid: number;
  group: Group;
  t: number[];
  /** [x1, y1, x2, y2] in work pixels (float32 values). */
  box: number[][];
  conf: number[];
  cls: number[];
}

type GroupTable = Pick<TrackingConstants, "GROUPS">;

/** tracking.group_of: the group a track id belongs to, from its million offset. */
export function groupOf(tid: number, cfg: GroupTable): Group {
  return Object.keys(cfg.GROUPS).at(Math.floor(Math.trunc(tid) / 1_000_000) - 1) as Group;
}

// The group order of scene.json tracking.GROUPS (and of the Group type), for collect without cfg.
const GROUP_ORDER: GroupTable = { GROUPS: { vehicle: [], person: [], bicycle: [], animal: [] } };

/** tracking.collect: append one frame of tracker rows to the tracks, keyed by id. */
export function collect(tracks: Map<number, Track>, t: number, rows: TrackRow[], cfg: GroupTable = GROUP_ORDER): void {
  for (const [x1, y1, x2, y2, id, score, cls] of rows) {
    const tid = Math.trunc(id);
    let tr = tracks.get(tid);
    if (!tr) {
      tr = { tid, group: groupOf(tid, cfg), t: [], box: [], conf: [], cls: [] };
      tracks.set(tid, tr);
    }
    tr.t.push(t);
    tr.box.push([f32(x1), f32(y1), f32(x2), f32(y2)]);
    tr.conf.push(score);
    tr.cls.push(Math.trunc(cls));
  }
}

/** Track.label: the majority class over the track; ties go to the smallest class (np.unique + argmax). */
export function trackLabel(tr: Track): number {
  const counts = new Map<number, number>();
  for (const c of tr.cls) counts.set(c, (counts.get(c) ?? 0) + 1);
  let best = NaN;
  let most = -1;
  for (const [c, n] of [...counts].sort((p, q) => p[0] - q[0])) {
    if (n > most) {
      best = c;
      most = n;
    }
  }
  return best;
}
