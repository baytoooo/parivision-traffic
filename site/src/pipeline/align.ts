// Registration for the in-browser demo; no Python equivalent (registration.py matches SIFT
// features, which the browser cannot run). The upload's first frame, grey and a quarter of the
// work size, is matched against each grey quarter-size reference on edge maps:
//   1. exhaustive search over scale and shift at 1/8 of that size, maximising the normalised
//      cross-correlation (NCC) of the edge maps;
//   2. Gauss-Newton refinement, coarse to fine, of the warp together with a gain and offset for
//      the contrast, with robust (Tukey) weights so that traffic, which differs from frame to
//      frame, drops out. The warp is affine at the two coarse levels and a full homography at
//      the two fine ones: besides shift and zoom, clips differ by about a degree of roll and a
//      little perspective, worth 10 and 4 px at the frame edges.
// The reference with the higher NCC wins; `ok` says whether that NCC is high enough for the
// frame to show this junction at all. On frames of the four sample clips the result is within
// 0.6 px (work pixels) of the homography Python's analyse() gets from SIFT on the same frame.

import { matInv, matMul, type Mat3 } from "./geometry.ts";

export interface AlignResult {
  /** Work pixels -> reference pixels, row-major; a plain rescale when not ok (as registration.py). */
  H: Mat3;
  /** NCC of the edge maps after alignment, -1..1. */
  score: number;
  ok: boolean;
  /** Key of the reference that matched best ("" when there was none). */
  reference: string;
}

/** The grey frame and the references are this many times smaller than work / reference pixels. */
export const ALIGN_DOWNSCALE = 4;
/** Lowest score of a frame of this junction. Frames of the four sample clips score 0.58 to 0.79
 * against their better reference; mirrored, upside-down or tile-shuffled frames and noise at
 * most 0.21 (tests/align.test.ts). */
export const MIN_SCORE = 0.35;

const LEVELS = 4; // 480x270 .. 60x33
const SCALES = { min: 0.94, max: 1.06, step: 0.01 };
const MAX_SHIFT = 48; // grey-frame pixels
const CANDIDATES = 3; // coarse maxima refined at levels 3 and 2; the best goes on to 1 and 0
const ITERATIONS = [5, 6, 6, 8]; // Gauss-Newton steps at level 0 (grey size) .. 3
const HOMOGRAPHY_BELOW = 2; // levels under this one refine all 8 homography entries, others 6
const TUKEY_C = 4.685;
const MEDIAN_SAMPLES = 8192; // residuals the robust scale is taken from

interface Img {
  w: number;
  h: number;
  d: Float32Array;
}

/** Copy with a one-pixel replicated border, so the 3x3 filters below need no bounds checks. */
function pad(src: Uint8Array | Float32Array, w: number, h: number): Float32Array {
  const W = w + 2;
  const out = new Float32Array(W * (h + 2));
  for (let y = -1; y <= h; y++) {
    const sy = Math.min(h - 1, Math.max(0, y)), row = (y + 1) * W;
    out.set(src.subarray(sy * w, sy * w + w), row + 1);
    out[row] = src[sy * w];
    out[row + W - 1] = src[sy * w + w - 1];
  }
  return out;
}

/** [1 2 1] smoothing, Sobel, then the square root of the gradient magnitude, so that a few very
 * strong edges (headlights, white cars) do not outweigh the kerbs and markings. */
function edgeMap(src: Uint8Array, w: number, h: number): Img {
  const W = w + 2;
  let p = pad(src, w, h);
  const s = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = (y + 1) * W + x + 1;
      s[y * w + x] =
        (p[i - W - 1] + 2 * p[i - W] + p[i - W + 1] + 2 * p[i - 1] + 4 * p[i] + 2 * p[i + 1] + p[i + W - 1] + 2 * p[i + W] + p[i + W + 1]) / 16;
    }
  p = pad(s, w, h);
  const m = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = (y + 1) * W + x + 1;
      const gx = p[i - W + 1] + 2 * p[i + 1] + p[i + W + 1] - p[i - W - 1] - 2 * p[i - 1] - p[i + W - 1];
      const gy = p[i + W - 1] + 2 * p[i + W] + p[i + W + 1] - p[i - W - 1] - 2 * p[i - W] - p[i - W + 1];
      m[y * w + x] = Math.sqrt(Math.sqrt(gx * gx + gy * gy));
    }
  return { w, h, d: m };
}

/** 2x2 box average; an odd last row or column is dropped. Pixel centres: x_fine = 2 x + 0.5. */
function halve(a: Img): Img {
  const w = a.w >> 1, h = a.h >> 1;
  const d = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = 2 * y * a.w + 2 * x;
      d[y * w + x] = (a.d[i] + a.d[i + 1] + a.d[i + a.w] + a.d[i + a.w + 1]) / 4;
    }
  return { w, h, d };
}

/**
 * 1 where a reference shows the scene, 0 in the black corners left by warping it into the
 * reference view (dark pixels connected to the border), grown by a few pixels so that the
 * edge along that border is not matched either.
 */
function validMask(src: Uint8Array, w: number, h: number, grow = 4): Img {
  const bad = new Uint8Array(w * h);
  const stack: number[] = [];
  const push = (i: number) => {
    if (!bad[i] && src[i] <= 2) {
      bad[i] = 1;
      stack.push(i);
    }
  };
  for (let x = 0; x < w; x++) push(x), push((h - 1) * w + x);
  for (let y = 0; y < h; y++) push(y * w), push(y * w + w - 1);
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (i >= w) push(i - w);
    if (i < (h - 1) * w) push(i + w);
  }
  const d = new Float32Array(w * h).fill(1);
  for (let i = 0; i < w * h; i++) {
    if (!bad[i]) continue;
    const x = i % w, y = (i / w) | 0;
    for (let yy = Math.max(0, y - grow); yy <= Math.min(h - 1, y + grow); yy++)
      d.fill(0, yy * w + Math.max(0, x - grow), yy * w + Math.min(w - 1, x + grow) + 1);
  }
  return { w, h, d };
}

/** Central-difference gradients (one-sided at the border). */
function gradients(a: Img): [Float32Array, Float32Array] {
  const { w, h, d } = a;
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 1; x < w - 1; x++) gx[r + x] = (d[r + x + 1] - d[r + x - 1]) / 2;
    gx[r] = d[r + 1] - d[r];
    gx[r + w - 1] = d[r + w - 1] - d[r + w - 2];
    const up = y > 0 ? r - w : r, down = y < h - 1 ? r + w : r, k = y > 0 && y < h - 1 ? 0.5 : 1;
    for (let x = 0; x < w; x++) gy[r + x] = (d[down + x] - d[up + x]) * k;
  }
  return [gx, gy];
}

/**
 * Warps are 3x3 homographies (row-major, [8] = 1) between normalised coordinates, the same at
 * every level: a reference point is (x0 - rc) / S and a frame point (x0 - fc) / S, x0 being
 * pixel coordinates at full grey size (level 0) and S half the reference's grey width.
 */
interface Norm {
  S: number;
  rcx: number;
  rcy: number;
  fcx: number;
  fcy: number;
}

/** One pyramid level of one reference, with buffers for the warped frame samples. */
interface Level {
  /** 2^level: level pixel x is level-0 pixel f x + (f - 1) / 2. */
  f: number;
  ref: Img;
  mask: Uint8Array;
  frame: Img;
  fgx: Float32Array;
  fgy: Float32Array;
  // the reference pixels whose warp lands inside the frame, and what the frame has there
  idx: Int32Array;
  xn: Float32Array;
  yn: Float32Array;
  Xn: Float32Array;
  Yn: Float32Array;
  iD: Float32Array;
  v: Float32Array;
  gx: Float32Array;
  gy: Float32Array;
  res: Float32Array;
  tmp: Float32Array;
}

interface FrameLevel {
  frame: Img;
  fgx: Float32Array;
  fgy: Float32Array;
}

function framePyramid(frame: Img): FrameLevel[] {
  const out: FrameLevel[] = [];
  for (let k = 0, f = frame; k < LEVELS; k++, f = halve(f)) {
    const [fgx, fgy] = gradients(f);
    out.push({ frame: f, fgx, fgy });
  }
  return out;
}

function makeLevels(frames: FrameLevel[], ref: Img, valid: Img): Level[] {
  const out: Level[] = [];
  let r = ref, m = valid;
  for (let k = 0; k < LEVELS; k++) {
    if (k > 0) {
      r = halve(r);
      m = halve(m);
    }
    const n = r.w * r.h;
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = m.d[i] > 0.999 ? 1 : 0; // all children valid
    const buf = () => new Float32Array(n);
    out.push({
      f: 2 ** k, ref: r, mask, ...frames[k], idx: new Int32Array(n),
      xn: buf(), yn: buf(), Xn: buf(), Yn: buf(), iD: buf(), v: buf(), gx: buf(), gy: buf(), res: buf(), tmp: buf(),
    });
  }
  return out;
}

/**
 * Samples the frame (bilinear) at the warp of every valid reference pixel of the level that
 * lands inside the frame with a one-pixel margin; returns how many did.
 */
function sample(L: Level, H: Mat3, N: Norm, withGradient: boolean): number {
  const { f, ref, mask, frame, fgx, fgy, idx, xn, yn, Xn, Yn, iD, v, gx, gy } = L;
  const fw = frame.w, fh = frame.h, s = frame.d;
  const o = (f - 1) / 2, k = N.S / f;
  let n = 0;
  for (let y = 0; y < ref.h; y++) {
    const yr = (f * y + o - N.rcy) / N.S;
    for (let x = 0; x < ref.w; x++) {
      const i = y * ref.w + x;
      if (!mask[i]) continue;
      const xr = (f * x + o - N.rcx) / N.S;
      const id = 1 / (H[6] * xr + H[7] * yr + 1);
      const Xw = (H[0] * xr + H[1] * yr + H[2]) * id, Yw = (H[3] * xr + H[4] * yr + H[5]) * id;
      const X = k * Xw + (N.fcx - o) / f, Y = k * Yw + (N.fcy - o) / f;
      if (!(X >= 1 && Y >= 1 && X <= fw - 2 && Y <= fh - 2)) continue;
      const x0 = X | 0, y0 = Y | 0;
      const ax = X - x0, ay = Y - y0;
      const j = y0 * fw + x0;
      const w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;
      idx[n] = i;
      v[n] = w00 * s[j] + w01 * s[j + 1] + w10 * s[j + fw] + w11 * s[j + fw + 1];
      if (withGradient) {
        xn[n] = xr;
        yn[n] = yr;
        Xn[n] = Xw;
        Yn[n] = Yw;
        iD[n] = id;
        gx[n] = w00 * fgx[j] + w01 * fgx[j + 1] + w10 * fgx[j + fw] + w11 * fgx[j + fw + 1];
        gy[n] = w00 * fgy[j] + w01 * fgy[j + 1] + w10 * fgy[j + fw] + w11 * fgy[j + fw + 1];
      }
      n++;
    }
  }
  return n;
}

/** NCC between the reference and the warped frame over the overlap; -1 if under 30% of the view overlaps. */
function ncc(L: Level, H: Mat3, N: Norm): number {
  const n = sample(L, H, N, false);
  if (n < 0.3 * L.ref.w * L.ref.h) return -1;
  const t = L.ref.d, { idx, v } = L;
  let st = 0, si = 0, stt = 0, sii = 0, sti = 0;
  for (let k = 0; k < n; k++) {
    const tv = t[idx[k]], iv = v[k];
    st += tv;
    si += iv;
    stt += tv * tv;
    sii += iv * iv;
    sti += tv * iv;
  }
  const den = Math.sqrt((stt - (st * st) / n) * (sii - (si * si) / n));
  return den > 0 ? (sti - (st * si) / n) / den : -1;
}

/** Solves the symmetric positive definite system A x = b (row-major n x n) by Cholesky; null if singular. */
function solve(A: Float64Array, b: Float64Array, n: number): Float64Array | null {
  const Lm = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= Lm[i * n + k] * Lm[j * n + k];
      if (i === j) {
        if (!(s > 0)) return null;
        Lm[i * n + i] = Math.sqrt(s);
      } else Lm[i * n + j] = s / Lm[j * n + j];
    }
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= Lm[i * n + k] * x[k];
    x[i] = s / Lm[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= Lm[k * n + i] * x[k];
    x[i] = s / Lm[i * n + i];
  }
  return x;
}

/** Median of |r| over at most MEDIAN_SAMPLES evenly spaced entries of r[0..n), by quickselect. */
function medianAbs(r: Float32Array, n: number, tmp: Float32Array): number {
  const step = Math.max(1, Math.floor(n / MEDIAN_SAMPLES));
  let m = 0;
  for (let i = 0; i < n; i += step) tmp[m++] = Math.abs(r[i]);
  const k = m >> 1;
  let lo = 0, hi = m - 1;
  while (lo < hi) {
    const pivot = tmp[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (tmp[i] < pivot) i++;
      while (tmp[j] > pivot) j--;
      if (i <= j) {
        const t = tmp[i];
        tmp[i++] = tmp[j];
        tmp[j--] = t;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return tmp[k];
}

/**
 * Gauss-Newton on sum_x w(x) (g I(W(x; H)) + o - T(x))^2 over the warp H (its 6 affine entries,
 * or all 8), the gain g and the offset o, where T is the reference's edge map and I the frame's;
 * w are Tukey weights whose scale comes from the median residual of each step.
 */
function refine(L: Level, H0: Mat3, N: Norm, iterations: number, full: boolean): Mat3 {
  const H = [...H0];
  const t = L.ref.d, { idx, xn, yn, Xn, Yn, iD, v, gx, gy, res } = L;
  const np = full ? 10 : 8;
  const A = new Float64Array(np * np), b = new Float64Array(np), J = new Float64Array(np);
  const k = N.S / L.f; // level pixels per normalised unit
  let gain = NaN, offset = 0;
  for (let it = 0; it < iterations; it++) {
    const n = sample(L, H, N, true);
    if (n < 100) break;
    if (Number.isNaN(gain)) {
      // photometric start: least-squares fit of the frame's edges to the reference's
      let st = 0, si = 0, sii = 0, sti = 0;
      for (let q = 0; q < n; q++) {
        const tv = t[idx[q]];
        st += tv;
        si += v[q];
        sii += v[q] * v[q];
        sti += tv * v[q];
      }
      const vi = sii - (si * si) / n;
      gain = vi > 0 ? (sti - (st * si) / n) / vi : 1;
      if (!(gain > 0.05)) gain = 1;
      offset = (st - gain * si) / n;
    }
    for (let q = 0; q < n; q++) res[q] = gain * v[q] + offset - t[idx[q]];
    const c = TUKEY_C * Math.max(1e-6, 1.4826 * medianAbs(res, n, L.tmp));
    A.fill(0);
    b.fill(0);
    for (let q = 0; q < n; q++) {
      const u = res[q] / c;
      if (u <= -1 || u >= 1) continue;
      const w = (1 - u * u) * (1 - u * u);
      const x = xn[q], y = yn[q];
      const Gx = gain * k * gx[q] * iD[q], Gy = gain * k * gy[q] * iD[q];
      J[0] = Gx * x;
      J[1] = Gx * y;
      J[2] = Gx;
      J[3] = Gy * x;
      J[4] = Gy * y;
      J[5] = Gy;
      if (full) {
        const G = -(Gx * Xn[q] + Gy * Yn[q]);
        J[6] = G * x;
        J[7] = G * y;
      }
      J[np - 2] = v[q];
      J[np - 1] = 1;
      const wr = w * res[q];
      for (let i = 0; i < np; i++) {
        const wi = w * J[i];
        b[i] -= J[i] * wr;
        for (let j = 0; j <= i; j++) A[i * np + j] += wi * J[j];
      }
    }
    for (let i = 0; i < np; i++) for (let j = 0; j < i; j++) A[j * np + i] = A[i * np + j];
    const dp = solve(A, b, np);
    if (!dp) break;
    for (let i = 0; i < np - 2; i++) H[i] += dp[i];
    gain += dp[np - 2];
    offset += dp[np - 1];
    // converged: under 1/100 of a level pixel of shift and 1e-5 of scale
    if (Math.abs(dp[2]) * k < 0.01 && Math.abs(dp[5]) * k < 0.01 && Math.abs(dp[0]) + Math.abs(dp[4]) < 1e-5) break;
  }
  return H;
}

/**
 * Scale-and-shift search at one (coarse) level: for each scale the frame is resampled once onto
 * the reference grid widened by the largest shift, and every whole-pixel shift of that grid is
 * scored by NCC. Returns the best `keep` candidates that are not neighbours of each other.
 */
function coarseSearch(L: Level, N: Norm, maxShift: number, keep: number): Mat3[] {
  const { f, ref, mask, frame } = L;
  const r = Math.ceil(maxShift / f);
  const W = ref.w + 2 * r, Hh = ref.h + 2 * r;
  const g = new Float32Array(W * Hh), ok = new Uint8Array(W * Hh);
  const found: { H: Mat3; score: number; dx: number; dy: number }[] = [];
  const t = ref.d, n0 = ref.w * ref.h, o = (f - 1) / 2;
  for (let s = SCALES.min; s <= SCALES.max + 1e-9; s += SCALES.step) {
    ok.fill(0);
    for (let y = 0; y < Hh; y++)
      for (let x = 0; x < W; x++) {
        // reference level pixel (x - r, y - r) -> frame level pixel, scaled about the centres
        const X = (s * (f * (x - r) + o - N.rcx) + N.fcx - o) / f;
        const Y = (s * (f * (y - r) + o - N.rcy) + N.fcy - o) / f;
        if (!(X >= 0 && Y >= 0 && X <= frame.w - 1.001 && Y <= frame.h - 1.001)) continue;
        const x0 = X | 0, y0 = Y | 0, ax = X - x0, ay = Y - y0, j = y0 * frame.w + x0, d = frame.d;
        g[y * W + x] = (1 - ay) * ((1 - ax) * d[j] + ax * d[j + 1]) + ay * ((1 - ax) * d[j + frame.w] + ax * d[j + frame.w + 1]);
        ok[y * W + x] = 1;
      }
    for (let dy = -r; dy <= r; dy++)
      for (let dx = -r; dx <= r; dx++) {
        let n = 0, st = 0, si = 0, stt = 0, sii = 0, sti = 0;
        for (let y = 0; y < ref.h; y++) {
          const row = (y + r + dy) * W + r + dx;
          for (let x = 0; x < ref.w; x++) {
            const i = y * ref.w + x;
            if (!mask[i] || !ok[row + x]) continue;
            const tv = t[i], iv = g[row + x];
            n++;
            st += tv;
            si += iv;
            stt += tv * tv;
            sii += iv * iv;
            sti += tv * iv;
          }
        }
        if (n < 0.3 * n0) continue;
        const den = Math.sqrt((stt - (st * st) / n) * (sii - (si * si) / n));
        if (!(den > 0)) continue;
        // reference pixel x takes the frame where the scaled warp sends x + (dx, dy)
        const u = (s * f * dx) / N.S, v = (s * f * dy) / N.S;
        found.push({ H: [s, 0, u, 0, s, v, 0, 0, 1], score: (sti - (st * si) / n) / den, dx, dy });
      }
  }
  found.sort((a, b) => b.score - a.score);
  const out: typeof found = [];
  for (const c of found) {
    if (out.every((q) => Math.abs(q.dx - c.dx) > 1 || Math.abs(q.dy - c.dy) > 1)) out.push(c);
    if (out.length === keep) break;
  }
  return out.map((c) => c.H);
}

/**
 * Homography from the upload's work pixels to reference pixels, found on edge maps. `gray` is
 * the first frame at workSize / ALIGN_DOWNSCALE (480x270 for a 16:9 clip), `refs` the
 * references at refSize / ALIGN_DOWNSCALE (public/pipeline/reference_*.png as grey bytes).
 */
export function alignToReference(
  gray: Uint8Array,
  refs: Record<string, Uint8Array>,
  workSize: [number, number],
  refSize: [number, number] = [1920, 1080],
): AlignResult {
  const gw = Math.round(workSize[0] / ALIGN_DOWNSCALE);
  const gh = Math.round(gray.length / gw);
  const rw = Math.round(refSize[0] / ALIGN_DOWNSCALE), rh = Math.round(refSize[1] / ALIGN_DOWNSCALE);
  if (gw * gh !== gray.length) throw new Error(`grey frame of ${gray.length} px is not ${gw} px wide`);
  const N: Norm = { S: rw / 2, rcx: (rw - 1) / 2, rcy: (rh - 1) / 2, fcx: (gw - 1) / 2, fcy: (gh - 1) / 2 };
  const frames = framePyramid(edgeMap(gray, gw, gh));
  const top = LEVELS - 1;
  // per reference: every coarse candidate through levels 3 and 2, the best on through level 1
  const tried: { lv: Level[]; H: Mat3; score: number; name: string }[] = [];
  for (const [name, img] of Object.entries(refs)) {
    if (img.length !== rw * rh) throw new Error(`reference ${name} is not ${rw}x${rh}`);
    const lv = makeLevels(frames, edgeMap(img, rw, rh), validMask(img, rw, rh));
    let H: Mat3 | null = null, score = -Infinity;
    for (let c of coarseSearch(lv[top], N, MAX_SHIFT, CANDIDATES)) {
      for (let k = top; k >= 2; k--) c = refine(lv[k], c, N, ITERATIONS[k], k < HOMOGRAPHY_BELOW);
      const s = ncc(lv[1], c, N);
      if (s > score) [H, score] = [c, s];
    }
    if (!H) continue;
    H = refine(lv[1], H, N, ITERATIONS[1], 1 < HOMOGRAPHY_BELOW);
    tried.push({ lv, H, score: ncc(lv[1], H, N), name });
  }
  // as registration._fallback: a frame that matches no reference gets a plain rescale
  const rescale: Mat3 = [refSize[0] / workSize[0], 0, 0, 0, refSize[1] / workSize[1], 0, 0, 0, 1];
  if (!tried.length) return { H: rescale, score: -1, ok: false, reference: "" };
  // only the winner is refined at full grey size
  const best = tried.reduce((a, b) => (b.score > a.score ? b : a));
  const Hn = refine(best.lv[0], best.H, N, ITERATIONS[0], 0 < HOMOGRAPHY_BELOW);
  const score = ncc(best.lv[0], Hn, N);
  // normalised reference -> normalised frame, turned into work pixels -> reference pixels;
  // grey pixel centres sit at full-size X = F x + (F - 1) / 2
  const fx = workSize[0] / gw, fy = workSize[1] / gh;
  const rx = refSize[0] / rw, ry = refSize[1] / rh;
  const workToNorm: Mat3 = [
    1 / (fx * N.S), 0, (-(fx - 1) / (2 * fx) - N.fcx) / N.S,
    0, 1 / (fy * N.S), (-(fy - 1) / (2 * fy) - N.fcy) / N.S,
    0, 0, 1,
  ];
  const normToRef: Mat3 = [rx * N.S, 0, rx * N.rcx + (rx - 1) / 2, 0, ry * N.S, ry * N.rcy + (ry - 1) / 2, 0, 0, 1];
  const H = matMul(normToRef, matMul(matInv(Hn), workToNorm)).map((v, _, m) => v / m[8]);
  const ok = score >= MIN_SCORE && H.every(Number.isFinite);
  return { H: ok ? H : rescale, score, ok, reference: best.name };
}

/** Grey bytes from RGBA (a canvas's getImageData): the BT.601 weights of OpenCV's BGR2GRAY, which
 * made frame0.png and the references (equal to it within one grey level). */
export function rgbaToGray(rgba: Uint8Array | Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(rgba.length >> 2);
  for (let i = 0; i < out.length; i++)
    out[i] = (rgba[4 * i] * 19595 + rgba[4 * i + 1] * 38470 + rgba[4 * i + 2] * 7471 + 32768) >>> 16;
  return out;
}
