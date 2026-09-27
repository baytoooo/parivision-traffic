// The demo page's Api when the pipeline runs in this browser (src/pipeline/, see its README).
// A job reads frames from a hidden <video> by seeking to t = k / 5 s (5 frames per second, as
// pipeline.py's "cpu" profile), draws each one at the detector's width and the signal head at
// work scale, and posts them to the pipeline's worker, which owns the ONNX session and the
// Analyser. A clip the <video> cannot decode (the camera's own 10-bit 4:2:2 files in most browsers
// on Windows and Linux), or decodes only slowly, is first converted in the page by ffmpeg.wasm
// (transcode.ts). The page polls job() as it polled the Python demo server; nothing leaves the
// device. The live page (live.ts) uses the same worker through live(), with frames it grabs from a
// camera or a shared screen itself.

import { UPLOAD_MAX_SECONDS } from "../config";
import { readableNote } from "../lib/classes";
import type { ClipResult, Counts, Job, Sample } from "../lib/types";
import { ALIGN_DOWNSCALE, rgbaToGray } from "../pipeline/align";
import { roundHalfEven } from "../pipeline/geometry";
import type { Backend, FromWorker, Pixels, ToWorker } from "../pipeline/messages";
import type { PipelineResult } from "../pipeline/types";
import { ApiError, CONVERT, STAGES, type Api, type Health } from "./api";

const FPS = 5; // pipeline.py PROFILES["cpu"]: analysed frames per second
export const WORK_WIDTH = 1920; // pipeline.py WORK_WIDTH: the work frame, whose pixels the pipeline's boxes are in
const IN_FLIGHT = 2; // frames posted to the worker and not yet done
const SEEK_TIMEOUT_MS = 20000;
const OPEN_TIMEOUT_MS = 30000;
const FIRST_FRAME_MS = 8000; // from the clip's metadata to its first frame; longer means no decoder
// A browser may decode a clip in software: Chrome on a Mac does so for the camera's 10-bit 4:2:2
// files and then takes 0.2 to 0.4 s per seek at 4K (seconds on a busy machine), where converting
// the clip and analysing the copy takes 0.2 s per frame in all. Clips it decodes in hardware, 4K
// or not, seek in 10 to 70 ms.
const PROBE = [1, 2, 3].map((k) => k / FPS); // seeks timed before choosing
const SLOW_SEEK_MS = 150; // most of them slower than this: convert the clip first
// WebKit (Safari) draws black at t = 0 when a clip's first frame comes later, as after an empty edit
// at the start of many ffmpeg cuts; Chrome and Firefox draw that first frame. A job aligns on, and
// starts detecting at, the first frame that is not black, stepping 1 / FPS at a time.
const BLACK_MAX = 16; // a grey frame whose brightest pixel is darker than this is black
const BLACK_STEPS = 5;
const REFERENCES = ["day", "dusk"]; // public/pipeline/reference_<name>.png
const [READ, LOAD, ALIGN, DETECT, RULES] = STAGES;

/** A job's stages in order, each with where it starts on the progress bar. */
type Plan = [stage: string, start: number][];
// Detection is most of the work, unless the clip is converted first: for 4K that takes about
// twice as long as everything after it. Both plans read the clip in the same span, as a job only
// switches to the second after reading.
const PLAN: Plan = [[READ, 0], [LOAD, 0.04], [ALIGN, 0.12], [DETECT, 0.14], [RULES, 0.97]];
const PLAN_CONVERTED: Plan = [[READ, 0], [CONVERT, 0.04], [LOAD, 0.7], [ALIGN, 0.73], [DETECT, 0.74], [RULES, 0.99]];

/** openVideo's error when the browser cannot show a frame of the clip; the job then converts it. */
class CannotDecode extends Error {}

type Handler = (m: FromWorker) => void;

interface LocalJob {
  id: string;
  signal: AbortSignal;
  status: Job["status"];
  plan: Plan;
  stage: string;
  progress: number;
  stageStarted: number;
  /** How much of the current stage is done, 0 to 1. */
  stageDone: number;
  framesDone: number;
  framesTotal: number;
  error: string | null;
  result: ClipResult | null;
  /** The job analyses the clip up to this time, at most. */
  maxSeconds: number;
  /** Object URL of the clip; the result's video. */
  url: string | null;
  /** Object URL of a still of the first frame analysed; the result's poster. */
  poster: string | null;
  /** Worker messages of this job, by type. */
  on: Partial<Record<FromWorker["type"], Handler>>;
  /** Rejects when the job is cancelled or fails; every wait of the job races it. */
  stopped: Promise<never>;
  stop: (e: Error) => void;
}

export interface Engine {
  backend: Backend;
  /** The detector's input [h, w]. */
  input: [number, number];
}

type Reply<T extends FromWorker["type"]> = Extract<FromWorker, { type: T }>;

/** A live run (LocalApi.live): the page sends the frames, one at a time, and gets each one's
 * tracked boxes back. Every call rejects once the run has stopped or failed. */
export interface LiveJob {
  readonly engine: Engine;
  /** Registers the first frame (grey, work size / 4) and starts the Analyser. */
  align(gray: Uint8Array<ArrayBuffer>, workSize: [number, number]): Promise<Reply<"aligned">>;
  /** One frame at `t` s from the start of the stream; resolves with its tracked boxes and risk. */
  frame(index: number, t: number, det: Pixels, lamp: (Pixels & { origin: [number, number] }) | null): Promise<Reply<"live">>;
  /** Builds trajectories and applies the rules to everything sent so far; ends the run. */
  finish(duration: number, clip: string): Promise<Reply<"result">>;
  /** Drops the run; the worker skips what is left of it. */
  cancel(): void;
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
  if (!r.ok || !r.body) throw new ApiError("http", `Could not download the sample clip (HTTP ${r.status}).`);
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

/** A muted <video> in the page (some browsers decode nothing for a detached one), ready to seek.
 * Rejects with CannotDecode when the browser shows no frame of the clip. */
function openVideo(url: string): Promise<HTMLVideoElement> {
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.setAttribute("aria-hidden", "true");
  v.style.cssText = "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;z-index:-1";
  document.body.append(v);
  return new Promise((resolve, reject) => {
    let timer = 0;
    const done = (err?: string) => {
      clearTimeout(timer);
      v.onloadedmetadata = v.onloadeddata = v.onerror = null;
      if (err) {
        // a detached video that plays the audio keeps loading the clip, whose URL the job revokes
        closeVideo(v);
        reject(new CannotDecode(err));
      } else resolve(v);
    };
    // a background tab may not load the video at all; only a visible page times out
    const arm = (ms: number, err: string) => {
      timer = window.setTimeout(() => (document.hidden ? arm(ms, err) : done(err)), ms);
    };
    arm(OPEN_TIMEOUT_MS, "The browser took too long to open the clip.");
    v.onloadedmetadata = () => {
      if (!v.videoWidth) return done("This browser opens the file but cannot decode its picture.");
      clearTimeout(timer);
      arm(FIRST_FRAME_MS, "This browser opens the file but shows no frame of it.");
    };
    v.onloadeddata = () => done(Number.isFinite(v.duration) && v.duration > 0 ? undefined : "This browser cannot read the length of the clip.");
    v.onerror = () => done("This browser cannot decode the clip.");
    v.src = url;
  });
}

/** For openVideo(...).catch: null when the browser cannot decode the clip. */
function orNull(e: unknown): null {
  if (e instanceof CannotDecode) return null;
  throw e;
}

/** openVideo for a job; a video that opens only after the job has stopped is closed again. */
async function openFor(job: LocalJob, url: string): Promise<HTMLVideoElement> {
  const opening = openVideo(url);
  try {
    return await wait(job, opening);
  } catch (e) {
    opening.then(closeVideo, () => undefined);
    throw e;
  }
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
      reject(new Error(`The browser could not decode the clip at ${t.toFixed(1)} s.`));
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

/** Whether most seeks to `times` take longer than `limit` ms; it stops seeking once that is
 * settled. False when the page goes to the background, where seeking is slow for any decoder. */
async function seeksSlowly(v: HTMLVideoElement, times: number[], limit: number): Promise<boolean> {
  const most = Math.floor(times.length / 2) + 1;
  let slow = 0;
  let fast = 0;
  for (const t of times) {
    const t0 = performance.now();
    await seek(v, t);
    if (document.hidden) return false;
    if (performance.now() - t0 > limit) slow++;
    else fast++;
    if (slow >= most || fast >= most) break;
  }
  return slow >= most;
}

/** Whether a grey frame is all black: what WebKit draws before a clip's first frame. */
function isBlack(gray: Uint8Array): boolean {
  for (let i = 0; i < gray.length; i++) if (gray[i] >= BLACK_MAX) return false;
  return true;
}

/** Frees the browser's decoder of a video from openVideo and takes it off the page. */
function closeVideo(v: HTMLVideoElement): void {
  v.removeAttribute("src");
  v.load();
  v.remove();
}

/** Draws the current video frame into the canvases the pipeline reads. */
export class FrameReader {
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

  /** The frame last drawn for the detector as a JPEG (null if the browser cannot encode it). */
  still(): Promise<Blob | null> {
    return new Promise((resolve) => this.det.canvas.toBlob(resolve, "image/jpeg", 0.85));
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

function toClipResult(r: PipelineResult, groups: string[], video: string, poster: string | null): ClipResult {
  return {
    clip: r.clip,
    duration: r.duration,
    events: r.events,
    risk: r.risk,
    signal: r.signal,
    // the notes as the Results page words them ("west crossing", not "west")
    evidence: r.evidence.map((e) => ({ ...e, note: readableNote(e.note) })),
    counts: r.counts as Counts,
    aligned: r.aligned,
    mode: r.mode,
    video,
    ...(poster ? { poster } : {}),
    overlay: { work: r.overlay.work, groups, frames: r.overlay.frames },
  };
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export class LocalApi implements Api {
  readonly mock = false;
  /** ?transcode=1 converts every clip, also one this browser decodes, to try the conversion anywhere. */
  readonly alwaysConvert = new URLSearchParams(location.search).get("transcode") === "1";
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
    if (!r.ok) throw new ApiError("http", "Could not load the list of sample clips.");
    this.sampleList = await r.json();
    return this.sampleList;
  }

  async submitFile(file: File, signal: AbortSignal, maxSeconds = UPLOAD_MAX_SECONDS): Promise<string> {
    return this.start(file.name, signal, async () => file, Math.min(maxSeconds, UPLOAD_MAX_SECONDS));
  }

  async submitSample(name: string, signal: AbortSignal): Promise<string> {
    const s = this.sampleList.find((x) => x.name === name);
    if (!s?.url) throw new ApiError("not_found", "There is no sample clip of that name.");
    const url = s.url;
    return this.start(url.split("/").pop() || name, signal, (job, onProgress) => fetchClip(url, job.signal, onProgress));
  }

  async job(id: string): Promise<Job> {
    const j = this.jobs.get(id);
    if (!j) throw new ApiError("not_found", "This job is gone. Start the clip again.");
    let eta: number | null = null;
    if (j.status === "running" && j.stage === DETECT && j.framesDone >= 3) {
      const perFrame = (performance.now() - j.stageStarted) / 1000 / j.framesDone;
      eta = Math.round(perFrame * (j.framesTotal - j.framesDone)) + 1;
    } else if (j.status === "running" && j.stage === CONVERT && j.stageDone >= 0.02) {
      // the conversion's own time left; the stages after it follow
      const elapsed = (performance.now() - j.stageStarted) / 1000;
      eta = Math.round((elapsed * (1 - j.stageDone)) / j.stageDone) + 1;
    } else if (j.status === "running" && j.stage === RULES) eta = 1;
    return { status: j.status, progress: Math.round(j.progress * 1000) / 1000, stage: j.stage, eta_sec: eta, error: j.error, result: j.result };
  }

  // ---------------------------------------------------------------- jobs

  private start(
    clip: string,
    signal: AbortSignal,
    getClip: (job: LocalJob, onProgress: (f: number) => void) => Promise<Blob>,
    maxSeconds = UPLOAD_MAX_SECONDS,
  ): string {
    if (signal.aborted) throw new ApiError("aborted", "Cancelled.");
    // The page shows one result at a time and a new job hides the last one: free the old clips.
    for (const old of this.jobs.values()) for (const url of [old.url, old.poster]) if (url) URL.revokeObjectURL(url);
    this.jobs.clear();
    const job = this.newJob(signal);
    job.maxSeconds = maxSeconds;
    void this.run(job, clip, getClip);
    return job.id;
  }

  /** A running job, registered for the worker's replies; aborting `signal` stops it. */
  private newJob(signal: AbortSignal): LocalJob {
    let stop!: (e: Error) => void;
    const stopped = new Promise<never>((_, reject) => (stop = reject));
    stopped.catch(() => undefined);
    const job: LocalJob = {
      id: `local-${Date.now().toString(36)}-${++this.n}`,
      signal,
      status: "running",
      plan: PLAN,
      stage: READ,
      progress: 0,
      stageStarted: performance.now(),
      stageDone: 0,
      framesDone: 0,
      framesTotal: 0,
      error: null,
      result: null,
      maxSeconds: UPLOAD_MAX_SECONDS,
      url: null,
      poster: null,
      on: {},
      stopped,
      stop,
    };
    this.jobs.set(job.id, job);
    signal.addEventListener("abort", () => job.stop(new ApiError("aborted", "Cancelled.")), { once: true });
    return job;
  }

  /**
   * Starts a live run for the live page: loads the pipeline as a job does (`onProgress` gets the
   * fraction of the model downloaded) and returns the run. Aborting `signal` stops it, as does a
   * worker error. The page grabs and sends the frames itself.
   */
  async live(signal: AbortSignal, onProgress: (f: number) => void): Promise<LiveJob> {
    if (signal.aborted) throw new ApiError("aborted", "Cancelled.");
    const job = this.newJob(signal);
    this.active = job;
    // However the run ends (cancel, `signal`, a worker error or finish), forget it; unless it
    // finished, the worker drops what is left of it.
    job.stopped.catch(() => {
      if (job.status === "running") this.worker?.postMessage({ type: "cancel", job: job.id } satisfies ToWorker);
      job.on = {};
      this.jobs.delete(job.id);
      if (this.active === job) this.active = null;
    });
    const engine = await wait(job, this.load(onProgress)).catch((e: unknown) => {
      job.stop(e instanceof Error ? e : new Error(message(e)));
      throw e;
    });
    const ask = <T extends FromWorker["type"]>(msg: ToWorker, transfer: Transferable[], type: T) => wait(job, this.call(job, msg, transfer, type));
    return {
      engine,
      align: (gray, workSize) => ask({ type: "align", job: job.id, gray, workSize, fps: FPS }, [gray.buffer], "aligned"),
      frame: (index, t, det, lamp) =>
        ask({ type: "frame", job: job.id, index, t, det, lamp, live: true }, lamp ? [det.data, lamp.data] : [det.data], "live"),
      finish: async (duration, clip) => {
        const done = await ask({ type: "finish", job: job.id, duration, clip }, [], "result");
        job.status = "done";
        job.stop(new ApiError("aborted", "Finished."));
        return done;
      },
      cancel: () => job.stop(new ApiError("aborted", "Stopped.")),
    };
  }

  private setStage(job: LocalJob, stage: string): void {
    job.stage = stage;
    job.stageStarted = performance.now();
    this.advance(job, 0);
  }

  /** Puts the job's progress bar at fraction `f` of its current stage. */
  private advance(job: LocalJob, f: number): void {
    const i = job.plan.findIndex(([stage]) => stage === job.stage);
    if (i < 0) return;
    const start = job.plan[i][1];
    const end = i + 1 < job.plan.length ? job.plan[i + 1][1] : 1;
    job.stageDone = f;
    job.progress = start + (end - start) * f;
  }

  private async run(job: LocalJob, clip: string, getClip: (job: LocalJob, onProgress: (f: number) => void) => Promise<Blob>): Promise<void> {
    let video: HTMLVideoElement | null = null;
    this.active = job;
    try {
      const blob = await wait(job, getClip(job, (f) => this.advance(job, f)));
      job.url = URL.createObjectURL(blob);
      // the clip is converted first when this browser cannot decode it, decodes it slowly or
      // fails a seek; slow is true when it decodes every frame, only slowly
      video = this.alwaysConvert ? null : await openFor(job, job.url).catch(orNull);
      const slow = video ? await wait(job, seeksSlowly(video, PROBE, SLOW_SEEK_MS).catch(() => null)) : null;
      if (video && slow !== false) {
        closeVideo(video);
        video = null;
      }
      if (!video) {
        URL.revokeObjectURL(job.url);
        job.url = null;
        job.plan = PLAN_CONVERTED;
        this.setStage(job, CONVERT);
        // ffmpeg stops with the job, cancelled or failed
        const halt = new AbortController();
        job.stopped.catch(() => halt.abort());
        let playable: Blob = blob;
        try {
          const { transcode } = await wait(job, import("./transcode"));
          playable = await wait(job, transcode(blob, job.maxSeconds, (f) => this.advance(job, f), halt.signal));
        } catch (e) {
          // a clip this browser decodes, if slowly, is analysed as it is when the conversion fails
          if (slow !== true || (e instanceof ApiError && e.kind === "aborted")) throw e;
          job.plan = PLAN;
        }
        job.url = URL.createObjectURL(playable);
        video = await openFor(job, job.url);
      }
      const limit = Math.min(video.duration, job.maxSeconds);
      const work: [number, number] = [WORK_WIDTH, roundHalfEven((video.videoHeight * WORK_WIDTH) / video.videoWidth)];

      this.setStage(job, LOAD);
      // the model's bytes; the ONNX Runtime engine loads after them, in the rest of the stage
      const engine = await wait(job, this.load((f) => this.advance(job, 0.75 * f)));

      this.setStage(job, ALIGN);
      const frames = new FrameReader(video, work, engine.input[1]);
      // the first frame that is not black; if the first BLACK_STEPS are, the clip starts black and
      // is read from 0 all the same
      let first = 0;
      await wait(job, seek(video, 0));
      let gray = frames.grayFrame();
      for (let k = 1; isBlack(gray) && k <= BLACK_STEPS && k / FPS < limit; k++) {
        await wait(job, seek(video, k / FPS));
        gray = frames.grayFrame();
        if (!isBlack(gray)) first = k;
      }
      const aligned = await wait(job, this.call(job, { type: "align", job: job.id, gray, workSize: work, fps: FPS }, [gray.buffer], "aligned"));
      const times: number[] = [];
      for (let k = first; k / FPS < limit; k++) times.push(k / FPS);
      job.framesTotal = times.length;

      this.setStage(job, DETECT);
      // Seek and draw frame k + 1 while the worker detects frame k; at most IN_FLIGHT posted.
      let inFlight = 0;
      let wake: (() => void) | null = null;
      job.on["frame-done"] = () => {
        inFlight--;
        job.framesDone++;
        this.advance(job, job.framesDone / job.framesTotal);
        wake?.();
      };
      const slot = () => wait(job, new Promise<void>((r) => (wake = r)));
      // the result's poster: the first frame analysed, which the player shows until it plays (WebKit
      // shows black for a clip that starts after 0)
      let still: Promise<Blob | null> = Promise.resolve(null);
      for (let k = 0; k < times.length; k++) {
        await wait(job, seek(video, times[k]));
        const det = frames.detFrame();
        if (k === 0) still = frames.still();
        const lamp = frames.lampCrop(aligned.lampCrop);
        while (inFlight >= IN_FLIGHT) await slot();
        this.post({ type: "frame", job: job.id, index: k, t: times[k], det, lamp }, [det.data, lamp.data]);
        inFlight++;
      }
      while (inFlight > 0) await slot();

      this.setStage(job, RULES);
      const done = await wait(job, this.call(job, { type: "finish", job: job.id, duration: limit, clip }, [], "result"));
      const poster = await wait(job, still);
      if (poster) job.poster = URL.createObjectURL(poster);
      job.result = toClipResult(done.result, done.groups, job.url, job.poster);
      job.status = "done";
      job.stage = "done";
      job.progress = 1;
    } catch (e) {
      const cancelled = e instanceof ApiError && e.kind === "aborted";
      job.status = "error";
      job.error = cancelled ? "Cancelled." : message(e);
      job.stage = "failed";
      this.worker?.postMessage({ type: "cancel", job: job.id } satisfies ToWorker);
      for (const url of [job.url, job.poster]) if (url) URL.revokeObjectURL(url);
      job.url = job.poster = null;
      job.stop(e instanceof Error ? e : new Error(message(e)));
    } finally {
      job.on = {};
      if (video) closeVideo(video); // the object URL stays for the result
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
