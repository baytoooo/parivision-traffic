// Vehicle signal phase from the three-lamp head on the median nose (signal.py): a box per
// lamp, the colour contrast inside it, the phase per frame and the offline clean-up.
// The Python frame is BGR; here pixels are RGB or RGBA as a canvas or decoder gives them.
// CausalPhase is not ported: analyse() uses fill_phases, and nothing else reads the signal online.

import { percentile, roundHalfEven, warpPoints, type Mat3 } from "./geometry.ts";

export type Phase = "red" | "yellow" | "green" | "off" | "unknown";
export type Box = [number, number, number, number];

export const LAMPS = ["red", "yellow", "green"] as const;
const CODES: Record<Phase, number> = { red: 0, yellow: 1, green: 2, off: 3, unknown: 4 };
const NAMES: Phase[] = ["red", "yellow", "green", "off", "unknown"];

/**
 * Work-pixel boxes [x0, y0, x1, y1) around the red, yellow and green lamp centres. Pass
 * matInv(H) as it comes, like np.linalg.inv(alignment.H): the radius scales with the
 * determinant of its top-left 2x2, so renormalising it would change the boxes slightly.
 */
export function lampPatches(HrefToWork: Mat3, lamps: Record<"red" | "yellow" | "green", [number, number]>, radiusRef = 2.5): Box[] {
  const scale = Math.sqrt(Math.abs(HrefToWork[0] * HrefToWork[4] - HrefToWork[1] * HrefToWork[3]));
  const r = Math.max(1.0, radiusRef * scale);
  return LAMPS.map((name) => {
    const [[x, y]] = warpPoints([lamps[name]], HrefToWork);
    return [roundHalfEven(x - r), roundHalfEven(y - r), roundHalfEven(x + r) + 1, roundHalfEven(y + r) + 1];
  });
}

/**
 * Colour contrast of each lamp: [redness of top, warmth of middle, greenness of bottom].
 * `pixels` is a crop of the work frame whose top-left corner is at `origin` (work pixels);
 * boxes are clipped to the crop, so it must hold them whole to match Python on the full frame.
 */
export function lampScores(
  pixels: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  channels: 3 | 4,
  boxes: Box[],
  origin: [number, number] = [0, 0],
): number[] {
  const out = [0, 0, 0];
  boxes.forEach(([bx0, by0, bx1, by1], k) => {
    const x0 = Math.max(0, bx0 - origin[0]), x1 = Math.min(width, bx1 - origin[0]);
    const y0 = Math.max(0, by0 - origin[1]), y1 = Math.min(height, by1 - origin[1]);
    const v: number[] = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = (y * width + x) * channels;
        const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
        v.push(k === 0 ? r - Math.max(g, b) : k === 1 ? r - b : g - Math.max(r, b));
      }
    }
    if (v.length) out[k] = Math.fround(percentile(v, 75));
  });
  return out;
}

/** The lamp furthest above its own threshold, or "off" when none reaches it (float32 like NumPy). */
export function phaseFromScores(scores: ArrayLike<number>, minContrast: ArrayLike<number>): Phase {
  let k = 0;
  let best = -Infinity;
  for (let i = 0; i < 3; i++) {
    const ratio = Math.fround(Math.fround(scores[i]) / Math.fround(minContrast[i]));
    if (ratio > best) [best, k] = [ratio, i];
  }
  return best >= 1.0 ? LAMPS[k] : "off";
}

/**
 * Offline clean-up: 1-sample blips removed, short dark spells take the phase before them.
 * Flashing green shows up as green/off alternation, so it stays green; a dark spell longer
 * than maxOff seconds becomes unknown.
 */
export function fillPhases(raw: string[], times: ArrayLike<number>, maxOff = 4): Phase[] {
  const codes = raw.map((p) => CODES[p as Phase]);
  const n = codes.length;
  const OFF = CODES.off;
  for (let i = 1; i < n - 1; i++) {
    // in place, as in Python: a flip fixed at i is what i + 1 compares with
    if (codes[i - 1] === codes[i + 1] && codes[i + 1] !== codes[i] && codes[i - 1] !== OFF) codes[i] = codes[i - 1];
  }
  const out = [...codes];
  let i = 0;
  while (i < n) {
    if (codes[i] !== OFF) {
      i++;
      continue;
    }
    let j = i;
    while (j < n && codes[j] === OFF) j++;
    const prev = i > 0 ? codes[i - 1] : j < n ? codes[j] : CODES.unknown;
    const span = j > i ? times[j - 1] - times[i] : 0;
    const fill = span <= maxOff ? prev : CODES.unknown;
    for (let k = i; k < j; k++) out[k] = fill;
    i = j;
  }
  return out.map((c) => NAMES[c]);
}
