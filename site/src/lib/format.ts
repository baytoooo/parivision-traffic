// Formatting helpers shared by build-time pages and browser scripts.
import type { Num } from "./types";

export function isNum(v: Num | unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** 91.49 -> "1:31.5" */
export function fmtTime(t: number, digits = 1): string {
  if (!Number.isFinite(t)) return "-";
  const neg = t < 0;
  t = Math.abs(t);
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  const ss = s.toFixed(digits).padStart(digits ? digits + 3 : 2, "0");
  return `${neg ? "-" : ""}${m}:${ss}`;
}

/** A metric that may still be "TBD". */
export function fmtNum(v: Num, digits = 3): string {
  if (isNum(v)) return v.toFixed(digits);
  if (typeof v === "string" && v.trim()) return v;
  return "TBD";
}

export function fmtInt(v: Num): string {
  if (isNum(v)) return String(Math.round(v));
  if (typeof v === "string" && v.trim()) return v;
  return "TBD";
}

export function fmtBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

/** Temporal IoU of two segments. */
export function iou(a: [number, number], b: [number, number]): number {
  const inter = Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
  const union = Math.max(a[1], b[1]) - Math.min(a[0], b[0]);
  return union > 0 ? inter / union : 0;
}

/** "TODO" or empty means the value is not filled in yet. */
export function isTodo(v: string | undefined | null): boolean {
  return !v || /^\s*(TODO|TBD)\b/i.test(v);
}

/** Merged alarm runs [start, end], as alarm_starts() in evaluate.py builds them before keeping only the starts. */
export function alarmRuns(curve: [number, number][], theta: number, mergeGap: number): [number, number][] {
  const runs: [number, number][] = [];
  let start: number | null = null;
  let end = 0;
  for (const [t, s] of curve) {
    if (s >= theta) {
      if (start === null) start = t;
      end = t;
    } else if (start !== null) {
      runs.push([start, end]);
      start = null;
    }
  }
  if (start !== null) runs.push([start, end]);
  const merged: [number, number][] = [];
  for (const [s, e] of runs) {
    const last = merged[merged.length - 1];
    if (last && s - last[1] < mergeGap) last[1] = e;
    else merged.push([s, e]);
  }
  return merged;
}
