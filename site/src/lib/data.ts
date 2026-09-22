// Build-time access to public/data. Pages read these files when the site is
// built, so replacing a file and rebuilding is enough; no code changes.
import fs from "node:fs";
import path from "node:path";
import type { Ablation, Clip, ClipResult, Eda, Example, Failure, Metrics, Team } from "./types";

const DATA_DIR = path.resolve(process.cwd(), "public", "data");

function read<T>(rel: string, fallback: T): T {
  const file = path.join(DATA_DIR, rel);
  if (!fs.existsSync(file)) return fallback;
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

export function readText(rel: string): string {
  const file = path.join(DATA_DIR, rel);
  return fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
}

export const loadClips = () => read<Clip[]>("clips.json", []);
export const loadMetrics = () => read<Metrics | null>("metrics.json", null);
export const loadAblations = () => read<Ablation[]>("ablations.json", []);
export const loadEda = () => read<Eda | null>("eda.json", null);
export const loadExamples = () => read<Example[]>("examples.json", []);
export const loadFailures = () => read<Failure[]>("failures.json", []);
export const loadTeam = () => read<Team>("team.json", { team: "PariVision", university: "", members: [] });

export function loadResult(clipId: string): ClipResult | null {
  return read<ClipResult | null>(`results/${clipId}.json`, null);
}

export function loadAllResults(): Record<string, ClipResult> {
  const out: Record<string, ClipResult> = {};
  for (const c of loadClips()) {
    const r = loadResult(c.id);
    if (r) out[c.id] = r;
  }
  return out;
}

/** Signal phase stats for one clip: complete green (incl. yellow) and red phases only. */
export function signalStats(signal: ClipResult["signal"]) {
  const phases: [number, number, "go" | "red"][] = [];
  for (const [s, e, l] of signal ?? []) {
    const k = l === "red" ? "red" : l === "green" || l === "yellow" ? "go" : null;
    if (!k) continue;
    const last = phases[phases.length - 1];
    if (last && last[2] === k) last[1] = e;
    else phases.push([s, e, k]);
  }
  const inner = phases.slice(1, -1);
  const greens = inner.filter((p) => p[2] === "go").map((p) => p[1] - p[0]);
  const reds = inner.filter((p) => p[2] === "red").map((p) => p[1] - p[0]);
  const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const unknown = (signal ?? []).filter((s) => s[2] === "unknown" || s[2] === "off").reduce((acc, s) => acc + (s[1] - s[0]), 0);
  return {
    cycles: Math.min(greens.length, reds.length),
    green: mean(greens),
    red: mean(reds),
    redShare: phases.length ? phases.filter((p) => p[2] === "red").reduce((a, p) => a + p[1] - p[0], 0) : NaN,
    unknown,
  };
}
