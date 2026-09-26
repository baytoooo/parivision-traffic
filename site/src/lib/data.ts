// Build-time access to public/data. Pages read these files when the site is
// built, so replacing a file and rebuilding is enough; no code changes.
import fs from "node:fs";
import path from "node:path";
import type { Ablation, Clip, ClipResult, Eda, ErrorAnalysis, Example, Failure, Metrics, Runtime, Team } from "./types";

const PUBLIC_DIR = path.resolve(process.cwd(), "public");
const DATA_DIR = path.join(PUBLIC_DIR, "data");

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
export const loadRuntime = () => read<Runtime | null>("runtime.json", null);
export const loadErrors = () => read<ErrorAnalysis | null>("errors.json", null);

/**
 * srcset for an image in public/ with smaller copies next to it (tools/make_site_data.py writes
 * name_480.jpg and name_960.jpg): "/media/a_480.jpg 480w, /media/a.jpg 960w". Copies that do not
 * exist are left out, so the page still works before the script has run.
 */
export function srcset(src: string, width: number, smaller: number[]): string | undefined {
  const parts = smaller
    .map((w) => ({ w, url: src.replace(/(\.[a-z]+)$/i, `_${w}$1`) }))
    .filter((p) => fs.existsSync(path.join(PUBLIC_DIR, p.url)))
    .map((p) => `${p.url} ${p.w}w`);
  return parts.length ? [...parts, `${src} ${width}w`].join(", ") : undefined;
}

/** Part A + Part B wall time over clip length, pooled over the clips in runtime.json. */
export function runtimeFactor(rt: Runtime | null): number | null {
  if (!rt || !rt.clips.length) return null;
  const total = rt.clips.reduce((a, c) => a + c.total_sec, 0);
  const dur = rt.clips.reduce((a, c) => a + c.duration, 0);
  return dur > 0 ? total / dur : null;
}

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
