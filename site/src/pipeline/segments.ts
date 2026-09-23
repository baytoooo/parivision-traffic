// Interval helpers: per-actor intervals -> one non-overlapping segment list per class (segments.py).

import { roundHalfEven } from "./geometry.ts";
import type { Seg } from "./types.ts";

/** Maximal runs of true in a sampled boolean signal; gaps up to maxGap seconds are bridged. */
export function runs(times: ArrayLike<number>, flags: ArrayLike<boolean | number>, maxGap: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < times.length; i++) {
    if (!flags[i]) continue;
    const t = times[i];
    const last = out[out.length - 1];
    if (last && t - last[1] <= maxGap + 1e-9) last[1] = t;
    else out.push([t, t]);
  }
  return out;
}

/** Merge overlapping intervals, and ones separated by at most gap seconds. */
export function union(intervals: [number, number][], gap = 0): [number, number][] {
  const merged: [number, number][] = [];
  for (const [s, e] of [...intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1])) {
    const last = merged[merged.length - 1];
    if (last && s <= last[1] + gap) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }
  return merged;
}

/** Python's round(x, 2): the exact binary value rounded to 2 decimals, halves to even. toFixed rounds
 * the exact value too but sends exact halves up; those are only x.125, x.375, x.625 and x.875. */
export function round2(v: number): number {
  const tie = Number.isInteger(v * 8) && !Number.isInteger(v * 4);
  return tie ? roundHalfEven(v * 100) / 100 : Number(v.toFixed(2));
}

/** Union per class, drop short blips, clip to the video, emit [start, end, label] sorted like Python lists. */
export function finalize(
  perClass: Record<string, [number, number][]>,
  duration: number,
  gap: Record<string, number>,
  minLen: Record<string, number>,
): Seg[] {
  const events: Seg[] = [];
  for (const [label, ivs] of Object.entries(perClass)) {
    for (const [s0, e0] of union(ivs, gap[label] ?? 0)) {
      const s = Math.max(0, s0);
      const e = Math.min(duration, e0);
      if (e - s >= (minLen[label] ?? 0)) events.push([round2(s), round2(e), label]);
    }
  }
  return events.sort((a, b) => a[0] - b[0] || a[1] - b[1] || (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : 0));
}
