// The demo page's Api when the pipeline runs in this browser (src/pipeline/, see its README).
// A job reads frames from a hidden <video> by seeking to t = k / 5 s (5 frames per second, as
// pipeline.py's "cpu" profile), draws each one at the detector's width and the signal head at
// work scale, and posts them to the pipeline's worker, which owns the ONNX session and the
// Analyser. The page polls job() as it polled the Python demo server; nothing leaves the device.

import { UPLOAD_MAX_SECONDS } from "../config";
import type { ClipResult, Counts, Job, Sample } from "../lib/types";
import { ALIGN_DOWNSCALE, rgbaToGray } from "../pipeline/align";
import { roundHalfEven } from "../pipeline/geometry";
import type { Backend, FromWorker, Pixels, ToWorker } from "../pipeline/messages";
import type { PipelineResult } from "../pipeline/types";
import { ApiError, STAGES, type Api, type Health } from "./api";

const FPS = 5; // pipeline.py PROFILES["cpu"]: analysed frames per second
const WORK_WIDTH = 1920; // pipeline.py WORK_WIDTH: the work frame, whose pixels the pipeline's boxes are in
const IN_FLIGHT = 2; // frames posted to the worker and not yet done
const SEEK_TIMEOUT_MS = 20000;
const REFERENCES = ["day", "dusk"]; // public/pipeline/reference_<name>.png
const [READ, LOAD, ALIGN, DETECT, RULES] = STAGES;
// where each stage starts on the progress bar; detection is most of the work
const AT: Record<string, number> = { [READ]: 0, [LOAD]: 0.04, [ALIGN]: 0.12, [DETECT]: 0.14, [RULES]: 0.97 };

const CANNOT_DECODE = "This browser cannot decode the clip. H.264 .mp4 files play in every browser; 4K HEVC (H.265) only in some.";

type Handler = (m: FromWorker) => void;

interface LocalJob {
  id: string;
  signal: AbortSignal;
  status: Job["status"];
  stage: string;
  progress: number;
  stageStarted: number;
  framesDone: number;
  framesTotal: number;
  error: string | null;
  result: ClipResult | null;
  /** Object URL of the clip; the result's video. */
  url: string | null;
  /** Worker messages of this job, by type. */
  on: Partial<Record<FromWorker["type"], Handler>>;
  /** Rejects when the job is cancelled or fails; every wait of the job races it. */
  stopped: Promise<never>;
  stop: (e: Error) => void;
}

interface Engine {
  backend: Backend;
  input: [number, number];
}

/** Resolves with `p`, unless the job stops first. */
function wait<T>(job: LocalJob, p: Promise<T>): Promise<T> {
  return Promise.race([p, job.stopped]);
}

async function webGpuAdapter(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

/** The two reference views as grey bytes (they are grey PNGs at a quarter of 1920x1080). */
async function loadReferences(): Promise<Record<string, Uint8Array>> {
  const out: Record<string, Uint8Array> = {};
  await Promise.all(
    REFERENCES.map(async (name) => {
      const img = new Image();
      img.src = `/pipeline/reference_${name}.png`;
      await img.decode();
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      out[name] = rgbaToGray(ctx.getImageData(0, 0, c.width, c.height).data);
    }),
  );
  return out;
}

async function fetchClip(url: string, signal: AbortSignal, onProgress: (f: number) => void): Promise<Blob> {
  let r: Response;
  try {
    r = await fetch(url, { signal });
  } catch {
    if (signal.aborted) throw new ApiError("aborted", "Cancelled.");
    throw new ApiError("network", "Could not download the sample clip. Check your connection and try again.");
  }
  if (!r.ok || !r.body) throw new ApiError("http", `Could not download the sample clip (HTTP ${r.status}).`, r.status);
  const total = Number(r.headers.get("content-length")) || 0;
  const reader = r.body.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress(Math.min(1, got / total));
  }
  return new Blob(chunks, { type: "video/mp4" });
}

/** A muted <video> in the page (some browsers decode nothing for a detached one), ready to seek. */
function openVideo(url: string): Promise<HTMLVideoElement> {
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.setAttribute("aria-hidden", "true");
  v.style.cssText = "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;z-index:-1";
  document.body.append(v);
  return new Promise((resolve, reject) => {
    const done = (err?: string) => {
      clearTimeout(timer);
      v.onloadeddata = v.onerror = null;
      if (err) {
        v.remove();
        reject(new Error(err));
      } else resolve(v);
    };
    const timer = setTimeout(() => done("The browser took too long to open the clip."), 30000);
    v.onloadeddata = () =>
      done(v.videoWidth && Number.isFinite(v.duration) && v.duration > 0 ? undefined : "This browser opens the file but cannot read its frames or its length.");
    v.onerror = () => done(CANNOT_DECODE);
    v.src = url;
  });
}

/** Seeks and resolves once the frame at `t` can be drawn. A background tab may stall seeking; only a
 * visible page times out. */
function seek(v: HTMLVideoElement, t: number): Promise<void> {
  if (!v.seeking && v.readyState >= 2 && Math.abs(v.currentTime - t) < 1e-3) return Promise.resolve();
  return new Promise((resolve, reject) => {
    let timer = 0;
    const cleanup = () => {
      clearTimeout(timer);
      v.removeEventListener("seeked", ok);
      v.removeEventListener("error", bad);
    };
    const ok = () => {
      cleanup();
      resolve();
    };
    const bad = () => {
      cleanup();
      reject(new Error(CANNOT_DECODE));
    };
    const arm = () => {
      timer = window.setTimeout(() => {
        if (document.hidden) return arm();
        cleanup();
        reject(new Error(`The browser stopped decoding the clip at ${t.toFixed(1)} s.`));
      }, SEEK_TIMEOUT_MS);
    };
    v.addEventListener("seeked", ok);
    v.addEventListener("error", bad);
    arm();
    v.currentTime = t;
  });
}

/** Draws the current video frame into the canvases the pipeline reads. */
class FrameReader {
  private video: HTMLVideoElement;
  private work: [number, number];
  private det: CanvasRenderingContext2D;
  private gray: CanvasRenderingContext2D;
  private lamp: CanvasRenderingContext2D;

  constructor(video: HTMLVideoElement, work: [number, number], detWidth: number) {
    this.video = video;
    this.work = work;
    const ctx = (w: number, h: number, quality: ImageSmoothingQuality) => {
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const g = c.getContext("2d", { willReadFrequently: true, alpha: false });
      if (!g) throw new Error("This browser gave no 2D canvas to draw the frames on.");
      g.imageSmoothingQuality = quality;
      return g;
    };
    // the detector gets the frame at its input width (960 x 540 for 16:9), as the letterbox expects
    this.det = ctx(detWidth, roundHalfEven((detWidth * video.videoHeight) / video.videoWidth), "medium");
    // the alignment gets the first frame grey at a quarter of the work size (480 x 270)
    this.gray = ctx(Math.round(work[0] / ALIGN_DOWNSCALE), Math.round(work[1] / ALIGN_DOWNSCALE), "high");
    this.lamp = ctx(1, 1, "medium");
  }

  grayFrame(): Uint8Array<ArrayBuffer> {
    const { canvas } = this.gray;
    this.gray.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    return rgbaToGray(this.gray.getImageData(0, 0, canvas.width, canvas.height).data) as Uint8Array<ArrayBuffer>;
  }

  detFrame(): Pixels {
    const { canvas } = this.det;
    this.det.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    return { data: this.det.getImageData(0, 0, canvas.width, canvas.height).data.buffer, width: canvas.width, height: canvas.height };
  }

  /** The work-pixel rectangle [x0, y0, x1, y1) of the frame at work scale, with its origin. */
  lampCrop([x0, y0, x1, y1]: [number, number, number, number]): Pixels & { origin: [number, number] } {
    const w = x1 - x0, h = y1 - y0;
    const { canvas } = this.lamp;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; // resets the context's settings
      canvas.height = h;
      this.lamp.imageSmoothingQuality = "medium";
    }
    const s = this.video.videoWidth / this.work[0]; // video pixels per work pixel
    this.lamp.drawImage(this.video, x0 * s, y0 * s, w * s, h * s, 0, 0, w, h);
    return { data: this.lamp.getImageData(0, 0, w, h).data.buffer, width: w, height: h, origin: [x0, y0] };
  }
}

function toClipResult(r: PipelineResult, groups: string[], video: string): ClipResult {
  return {
    clip: r.clip,
    duration: r.duration,
    events: r.events,
    risk: r.risk,
    signal: r.signal,
    evidence: r.evidence,
    counts: r.counts as Counts,
    aligned: r.aligned,
    video,
    overlay: { work: r.overlay.work, groups, frames: r.overlay.frames },
  };
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export class LocalApi implements Api {
  readonly mock = false;
  /** Where the detector runs: a guess from health(), then what the worker reports once the model is loaded. */
  backend: Backend | null = null;
  private worker: Worker | null = null;
  private engine: Promise<Engine> | null = null;
  private engineProgress: ((f: number) => void) | null = null;
  private engineReady: ((e: Engine) => void) | null = null;
  private engineFailed: ((e: Error) => void) | null = null;
  private jobs = new Map<string, LocalJob>();
  private active: LocalJob | null = null;
  private sampleList: Sample[] = [];
  private n = 0;

  async health(): Promise<Health> {
    if (
      typeof Worker === "undefined" ||
      typeof WebAssembly !== "object" ||
      typeof DecompressionStream === "undefined" ||
      !document.createElement("canvas").getContext("2d")
    )
      return "unsupported";
    // ?backend=wasm forces the WebAssembly path, which is what browsers without WebGPU get
    const forced = new URLSearchParams(location.search).get("backend");
    this.backend = forced === "wasm" ? "wasm" : (await webGpuAdapter()) ? "webgpu" : "wasm";
    return "ok";
  }

  async samples(): Promise<Sample[]> {
    let r: Response;
    try {
      r = await fetch("/data/demo/samples.json");
    } catch {
      throw new ApiError("network", "Could not load the list of sample clips.");
    }
    if (!r.ok) throw new ApiError("http", "Could not load the list of sample clips.", r.status);
    this.sampleList = await r.json();
    return this.sampleList;
  }

  async submitFile(file: File, signal: AbortSignal): Promise<string> {
    return this.start(file.name, signal, async () => file);
  }

  async submitSample(name: string, signal: AbortSignal): Promise<string> {
    const s = this.sampleList.find((x) => x.name === name);
    if (!s?.url) throw new ApiError("not_found", "There is no sample clip of that name.", 404);
    const url = s.url;
    return this.start(url.split("/").pop() || name, signal, (job, onProgress) => fetchClip(url, job.signal, onProgress));
  }

  async job(id: string): Promise<Job> {
    const j = this.jobs.get(id);
    if (!j) throw new ApiError("not_found", "This job is gone. Start the clip again.", 404);
    let eta: number | null = null;
    if (j.status === "running" && j.stage === DETECT && j.framesDone >= 3) {
      const perFrame = (performance.now() - j.stageStarted) / 1000 / j.framesDone;
      eta = Math.round(perFrame * (j.framesTotal - j.framesDone)) + 1;
    } else if (j.status === "running" && j.stage === RULES) eta = 1;
    return { status: j.status, progress: Math.round(j.progress * 1000) / 1000, stage: j.stage, eta_sec: eta, error: j.error, result: j.result };
  }

  videoUrl(path: string): string {
    return path;
  }

  // ---------------------------------------------------------------- jobs

  private start(clip: string, signal: AbortSignal, getClip: (job: LocalJob, onProgress: (f: number) => void) => Promise<Blob>): string {
    if (signal.aborted) throw new ApiError("aborted", "Cancelled.");
    // The page shows one result at a time and a new job hides the last one: free the old clips.
    for (const old of this.jobs.values()) if (old.url) URL.revokeObjectURL(old.url);
    this.jobs.clear();
    let stop!: (e: Error) => void;
    const stopped = new Promise<never>((_, reject) => (stop = reject));
    stopped.catch(() => undefined);
    const job: LocalJob = {
      id: `local-${Date.now().toString(36)}-${++this.n}`,
      signal,
      status: "running",
      stage: READ,
      progress: 0,
      stageStarted: performance.now(),
      framesDone: 0,
      framesTotal: 0,
      error: null,
      result: null,
      url: null,
      on: {},
      stopped,
      stop,
    };
    this.jobs.set(job.id, job);
    signal.addEventListener("abort", () => job.stop(new ApiError("aborted", "Cancelled.")), { once: true });
    void this.run(job, clip, getClip);
    return job.id;
  }

  private setStage(job: LocalJob, stage: string): void {
    job.stage = stage;
    job.progress = AT[stage];
    job.stageStarted = performance.now();
  }

  private async run(job: LocalJob, clip: string, getClip: (job: LocalJob, onProgress: (f: number) => void) => Promise<Blob>): Promise<void> {
    let video: HTMLVideoElement | null = null;
    this.active = job;
    try {
      const blob = await wait(job, getClip(job, (f) => (job.progress = AT[READ] + (AT[LOAD] - AT[READ]) * f)));
      job.url = URL.createObjectURL(blob);
      video = await wait(job, openVideo(job.url));
      const limit = Math.min(video.duration, UPLOAD_MAX_SECONDS);
      const work: [number, number] = [WORK_WIDTH, roundHalfEven((video.videoHeight * WORK_WIDTH) / video.videoWidth)];
      const times: number[] = [];
      for (let k = 0; k / FPS < limit; k++) times.push(k / FPS);
      job.framesTotal = times.length;

      this.setStage(job, LOAD);
      // the model's bytes; the ONNX Runtime engine loads after them, in the rest of the stage
      const engine = await wait(job, this.load((f) => (job.progress = AT[LOAD] + 0.75 * (AT[ALIGN] - AT[LOAD]) * f)));

      this.setStage(job, ALIGN);
      const frames = new FrameReader(video, work, engine.input[1]);
      await wait(job, seek(video, 0));
      const gray = frames.grayFrame();
      const aligned = await wait(job, this.call(job, { type: "align", job: job.id, gray, workSize: work, fps: FPS }, [gray.buffer], "aligned"));

      this.setStage(job, DETECT);
      // Seek and draw frame k + 1 while the worker detects frame k; at most IN_FLIGHT posted.
      let inFlight = 0;
      let wake: (() => void) | null = null;
      job.on["frame-done"] = () => {
        inFlight--;
        job.framesDone++;
        job.progress = AT[DETECT] + ((AT[RULES] - AT[DETECT]) * job.framesDone) / job.framesTotal;
        wake?.();
      };
      const slot = () => wait(job, new Promise<void>((r) => (wake = r)));
      for (let k = 0; k < times.length; k++) {
        await wait(job, seek(video, times[k]));
        const det = frames.detFrame();
        const lamp = frames.lampCrop(aligned.lampCrop);
        while (inFlight >= IN_FLIGHT) await slot();
        this.post({ type: "frame", job: job.id, index: k, t: times[k], det, lamp }, [det.data, lamp.data]);
        inFlight++;
      }
      while (inFlight > 0) await slot();

      this.setStage(job, RULES);
      const done = await wait(job, this.call(job, { type: "finish", job: job.id, duration: limit, clip }, [], "result"));
      job.result = toClipResult(done.result, done.groups, job.url);
      job.status = "done";
      job.stage = "done";
      job.progress = 1;
    } catch (e) {
      const cancelled = e instanceof ApiError && e.kind === "aborted";
      job.status = "error";
      job.error = cancelled ? "Cancelled." : message(e);
      job.stage = "failed";
      this.worker?.postMessage({ type: "cancel", job: job.id } satisfies ToWorker);
      if (job.url) URL.revokeObjectURL(job.url);
      job.url = null;
      job.stop(e instanceof Error ? e : new Error(message(e)));
    } finally {
      job.on = {};
      if (video) {
        video.removeAttribute("src");
        video.load(); // lets the browser drop the decoder; the object URL stays for the result
        video.remove();
      }
      if (this.active === job) this.active = null;
    }
  }

  /** Posts `msg` and resolves with the job's next message of type `type`. */
  private call<T extends FromWorker["type"]>(job: LocalJob, msg: ToWorker, transfer: Transferable[], type: T): Promise<Extract<FromWorker, { type: T }>> {
    return new Promise((resolve) => {
      job.on[type] = (m) => {
        delete job.on[type];
        resolve(m as Extract<FromWorker, { type: T }>);
      };
      this.post(msg, transfer);
    });
  }

  private post(msg: ToWorker, transfer: Transferable[] = []): void {
    if (!this.worker) throw new Error("The pipeline worker is not running.");
    this.worker.postMessage(msg, transfer);
  }

  // ---------------------------------------------------------------- worker

  /** Starts the worker and loads the scene and the model once; later jobs reuse them. */
  private load(onProgress: (f: number) => void): Promise<Engine> {
    this.engineProgress = onProgress;
    if (!this.engine) {
      this.engine = (async () => {
        const refs = await loadReferences();
        const worker = new Worker(new URL("../pipeline/worker.ts", import.meta.url), { type: "module" });
        this.worker = worker;
        worker.onmessage = (ev: MessageEvent<FromWorker>) => this.route(ev.data);
        worker.onerror = (ev) => this.crash(ev.message || "The pipeline worker stopped.");
        worker.onmessageerror = () => this.crash("The pipeline worker sent a message the page could not read.");
        const ready = new Promise<Engine>((resolve, reject) => {
          this.engineReady = resolve;
          this.engineFailed = reject;
        });
        const init: ToWorker = { type: "init", base: new URL("/pipeline/", location.href).href, refs, allowWebGpu: this.backend !== "wasm" };
        worker.postMessage(init, Object.values(refs).map((r) => r.buffer));
        const engine = await ready;
        this.backend = engine.backend;
        return engine;
      })();
      this.engine.catch(() => {
        // the next job starts over with a fresh worker
        this.worker?.terminate();
        this.worker = null;
        this.engine = null;
      });
    }
    return this.engine;
  }

  private route(m: FromWorker): void {
    switch (m.type) {
      case "loading":
        this.engineProgress?.(m.fraction);
        return;
      case "ready":
        this.engineReady?.({ backend: m.backend, input: m.input });
        return;
      case "error":
        if (m.job === null) this.engineFailed?.(new Error(`Could not load the pipeline: ${m.message}`));
        else this.jobs.get(m.job)?.stop(new Error(m.message));
        return;
      default:
        this.jobs.get(m.job)?.on[m.type]?.(m);
    }
  }

  /** An uncaught error in the worker: it may be in any state, so fail everything and start over next time. */
  private crash(text: string): void {
    this.engineFailed?.(new Error(text));
    this.active?.stop(new Error(text));
    this.worker?.terminate();
    this.worker = null;
    this.engine = null;
  }
}
