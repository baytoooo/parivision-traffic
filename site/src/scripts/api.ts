// Demo API client. HttpApi talks to the real server at API_BASE_URL;
// MockApi replays a recorded job from /data/demo so the page works without it.
import type { ClipResult, Job, Sample } from "../lib/types";

export class ApiError extends Error {
  kind: "network" | "too_big" | "http" | "not_found" | "aborted" | "bad_response";
  status?: number;
  constructor(kind: ApiError["kind"], message: string, status?: number) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

export type Health = "ok" | "sleeping" | "unreachable";

export interface Api {
  readonly mock: boolean;
  health(): Promise<Health>;
  samples(): Promise<Sample[]>;
  submitFile(file: File, onProgress: (f: number) => void, signal: AbortSignal): Promise<string>;
  submitSample(name: string): Promise<string>;
  job(id: string): Promise<Job>;
  videoUrl(path: string): string;
}

async function fetchTimeout(url: string, ms: number, init: RequestInit = {}): Promise<Response> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: ctl.signal });
  } finally {
    clearTimeout(timer);
  }
}

export class HttpApi implements Api {
  readonly mock = false;
  constructor(private base: string) {}

  private url(path: string) {
    return `${this.base.replace(/\/+$/, "")}${path}`;
  }

  async health(): Promise<Health> {
    try {
      const r = await fetchTimeout(this.url("/api/health"), 8000, { cache: "no-store" });
      if (!r.ok) return "sleeping";
      const j = await r.json().catch(() => null);
      return j && j.ok ? "ok" : "sleeping";
    } catch (e) {
      return (e as Error).name === "AbortError" ? "sleeping" : "unreachable";
    }
  }

  async samples(): Promise<Sample[]> {
    const r = await fetchTimeout(this.url("/api/samples"), 10000);
    if (!r.ok) throw new ApiError("http", `The server answered ${r.status} for the sample list.`, r.status);
    const j = await r.json();
    if (!Array.isArray(j)) throw new ApiError("bad_response", "The sample list came back in an unexpected shape.");
    return j;
  }

  submitFile(file: File, onProgress: (f: number) => void, signal: AbortSignal): Promise<string> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", this.url("/api/jobs"));
      xhr.responseType = "text";
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status === 413) return reject(new ApiError("too_big", "The server refused the file as too large. The limit is 500 MB.", 413));
        if (xhr.status < 200 || xhr.status >= 300) {
          let detail = "";
          try {
            const j = JSON.parse(xhr.responseText);
            detail = j.detail || j.error || "";
          } catch {
            /* not JSON */
          }
          return reject(new ApiError("http", `The server answered ${xhr.status}${detail ? `: ${detail}` : "."}`, xhr.status));
        }
        try {
          const j = JSON.parse(xhr.responseText);
          if (!j.job_id) throw new Error();
          resolve(String(j.job_id));
        } catch {
          reject(new ApiError("bad_response", "The server accepted the file but did not return a job id."));
        }
      };
      xhr.onerror = () => reject(new ApiError("network", "The upload failed. Check your connection and try again."));
      xhr.onabort = () => reject(new ApiError("aborted", "Upload cancelled."));
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
      const fd = new FormData();
      fd.append("file", file, file.name);
      xhr.send(fd);
    });
  }

  async submitSample(name: string): Promise<string> {
    let r: Response;
    try {
      r = await fetchTimeout(this.url(`/api/jobs/sample/${encodeURIComponent(name)}`), 20000, { method: "POST" });
    } catch {
      throw new ApiError("network", "Could not reach the server to start the sample job.");
    }
    if (!r.ok) throw new ApiError("http", `The server answered ${r.status} when starting the sample.`, r.status);
    const j = await r.json().catch(() => null);
    if (!j?.job_id) throw new ApiError("bad_response", "The server did not return a job id.");
    return String(j.job_id);
  }

  async job(id: string): Promise<Job> {
    let r: Response;
    try {
      r = await fetchTimeout(this.url(`/api/jobs/${encodeURIComponent(id)}`), 15000, { cache: "no-store" });
    } catch {
      throw new ApiError("network", "Lost contact with the server.");
    }
    if (r.status === 404) throw new ApiError("not_found", "The server does not know this job any more. It may have restarted.", 404);
    if (!r.ok) throw new ApiError("http", `The server answered ${r.status}.`, r.status);
    const j = await r.json().catch(() => null);
    if (!j || typeof j.status !== "string") throw new ApiError("bad_response", "The job status came back in an unexpected shape.");
    return j as Job;
  }

  videoUrl(path: string): string {
    try {
      return new URL(path, `${this.base.replace(/\/+$/, "")}/`).toString();
    } catch {
      return path;
    }
  }
}

/** The stages our server reports (demo/app.py, demo/worker.py, pipeline.analyse), in order; the mock replays them. */
export const STAGES = [
  "waiting in the queue",
  "aligning the view",
  "detecting and tracking",
  "applying event rules",
  "rendering the annotated video",
];

const MOCK_TIMES = [1.0, 1.2, 5.8, 1.0, 1.4]; // seconds per stage

interface MockJob {
  started: number;
  fail: boolean;
}

export class MockApi implements Api {
  readonly mock = true;
  private jobs = new Map<string, MockJob>();
  private n = 0;
  private result: Promise<ClipResult> | null = null;
  constructor(private opts: { fail?: boolean } = {}) {}

  async health(): Promise<Health> {
    return "ok";
  }

  async samples(): Promise<Sample[]> {
    const r = await fetch("/data/demo/mock_samples.json");
    if (!r.ok) throw new ApiError("http", "Could not load the sample list.");
    return r.json();
  }

  submitFile(file: File, onProgress: (f: number) => void, signal: AbortSignal): Promise<string> {
    // pretend to upload: about 1.5 s, a bit longer for bigger files
    const total = 900 + Math.min(1600, file.size / 200_000);
    const t0 = performance.now();
    return new Promise((resolve, reject) => {
      const tick = () => {
        if (signal.aborted) return reject(new ApiError("aborted", "Upload cancelled."));
        const f = Math.min(1, (performance.now() - t0) / total);
        onProgress(f);
        if (f >= 1) resolve(this.newJob());
        else setTimeout(tick, 80);
      };
      tick();
    });
  }

  async submitSample(): Promise<string> {
    await new Promise((r) => setTimeout(r, 250));
    return this.newJob();
  }

  private newJob(): string {
    const id = `mock-${Date.now().toString(36)}-${++this.n}`;
    this.jobs.set(id, { started: performance.now(), fail: !!this.opts.fail });
    return id;
  }

  private loadResult() {
    if (!this.result)
      this.result = fetch("/data/demo/mock_result.json").then((r) => {
        if (!r.ok) throw new ApiError("http", "Could not load the recorded result.");
        return r.json();
      });
    return this.result;
  }

  async job(id: string): Promise<Job> {
    const j = this.jobs.get(id);
    if (!j) throw new ApiError("not_found", "Unknown job.", 404);
    const el = (performance.now() - j.started) / 1000;
    const total = MOCK_TIMES.reduce((a, b) => a + b, 0);
    if (j.fail && el > 5.5)
      return { status: "error", progress: 0.4, stage: "detecting and tracking", eta_sec: null, error: "Test error from mock mode (?mock=error): the detector ran out of memory.", result: null };
    if (el >= total) {
      const result = await this.loadResult();
      return { status: "done", progress: 1, stage: "done", eta_sec: 0, error: null, result };
    }
    let acc = 0;
    let i = 0;
    while (i < MOCK_TIMES.length && acc + MOCK_TIMES[i] <= el) acc += MOCK_TIMES[i++];
    return {
      status: i === 0 ? "queued" : "running",
      progress: Math.min(0.99, el / total),
      stage: STAGES[i],
      eta_sec: Math.max(1, Math.round(total - el)),
      error: null,
      result: null,
    };
  }

  videoUrl(path: string): string {
    return path;
  }
}
