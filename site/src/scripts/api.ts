// What the demo page needs from whatever analyses a clip. LocalApi (local_api.ts) runs the
// pipeline in this browser; MockApi replays a recorded job from /data/demo, for ?mock=1 and for
// browsers that cannot run the pipeline.
import type { ClipResult, Job, Sample } from "../lib/types";

export class ApiError extends Error {
  kind: "network" | "http" | "not_found" | "aborted" | "bad_response";
  status?: number;
  constructor(kind: ApiError["kind"], message: string, status?: number) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

/** "unsupported": this browser lacks something the pipeline needs (Web Workers, WebAssembly, ...). */
export type Health = "ok" | "unsupported";

export interface Api {
  readonly mock: boolean;
  health(): Promise<Health>;
  samples(): Promise<Sample[]>;
  /** Starts a job and resolves with its id; aborting `signal` stops the job. */
  submitFile(file: File, signal: AbortSignal): Promise<string>;
  submitSample(name: string, signal: AbortSignal): Promise<string>;
  job(id: string): Promise<Job>;
  videoUrl(path: string): string;
}

/** The stages a job reports as Job.stage, in order (named like those of the Python demo server,
 * demo/worker.py and pipeline.analyse); the page lists them and the mock replays them. */
export const STAGES = ["reading the clip", "loading the model", "aligning the view", "detecting and tracking", "applying event rules"];

const MOCK_TIMES = [0.8, 1.2, 1.0, 5.8, 1.0]; // seconds per stage

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

  async submitFile(_file: File, signal: AbortSignal): Promise<string> {
    return this.submitSample("", signal);
  }

  async submitSample(_name: string, signal: AbortSignal): Promise<string> {
    await new Promise((r) => setTimeout(r, 250));
    if (signal.aborted) throw new ApiError("aborted", "Cancelled.");
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
    while (i < MOCK_TIMES.length - 1 && acc + MOCK_TIMES[i] <= el) acc += MOCK_TIMES[i++];
    return {
      status: "running",
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
