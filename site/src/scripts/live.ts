// Live page controller. The source is a camera (getUserMedia) or a shared tab or window
// (getDisplayMedia, for a live traffic stream playing in another tab). The pipeline's worker runs
// through LocalApi.live: the first frame is registered against our reference views, then frames go
// to the worker one at a time, at most 5 a second (liveloop.ts), timed from the start of the run.
// The page draws the tracked boxes over the video and shows the counts in view and the processing
// rate. When the first frame matches our junction, the lamps are read too and the page shows the
// risk score and its last minute; at Stop the rules run over the whole run and the page lists the
// events. On any other view it only detects and tracks.

import { LIVE_MAX_SECONDS, RISK_MERGE_GAP, RISK_THETA } from "../config";
import { className } from "../lib/classes";
import { fmtTime } from "../lib/format";
import type { FromWorker, LiveBox } from "../pipeline/messages";
import type { Seg } from "../pipeline/types";
import { ApiError } from "./api";
import { CLASS_NAMES, countInView, COUNTED, frameShape, Pacer, RateMeter, Rolling, sameShape } from "./liveloop";
import { FrameReader, LocalApi, WORK_WIDTH, type LiveJob } from "./local_api";
import { drawTag, GROUP_RGB, OTHER_RGB, tagFont, workToCanvas } from "./player";
import { RiskCurve } from "./riskcurve";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

type Source = "camera" | "screen";
/** Why a run ended: Stop, the stream ended, the time limit, the picture of our junction changed
 * shape, or the page was left. */
type End = "stop" | "ended" | "limit" | "resized" | "left";
type State = "checking" | "unsupported" | "ready" | "asking" | "loading" | "aligning" | "running" | "stopping";

const RISK_WINDOW = 60; // seconds of risk line
const FIRST_FRAME_MS = 10000; // from the stream to its first picture
const NEW_FRAME_MS = 250; // how long to wait for a picture the worker has not seen yet
// Some cameras send a few smaller pictures before the size they were asked for: the view is
// checked once the picture has kept its size this long, or after STEADY_MAX_MS whatever it does.
const STEADY_MS = 1000;
const STEADY_MAX_MS = 5000;

interface Run {
  source: Source;
  stream: MediaStream;
  /** Stops the run before its frames flow: while the stream opens, the model loads or the view is checked. */
  abort: AbortController;
  job: LiveJob | null;
  /** The first frame matched our junction: lamps, events and risk are on. */
  aligned: boolean;
  end: End | null;
  /** Frames analysed so far. */
  frames: number;
  /** Time of the last frame analysed, s from the start of the run. */
  lastT: number;
  /** performance.now() when the first frame was taken; 0 before. */
  started: number;
  /** performance.now() when the picture last changed size (the video's resize event). */
  resizedAt: number;
  /** The camera or the capture is off. */
  released: boolean;
}

const local = new LocalApi();
let ready = false;
/** A permission prompt or the share picker is open. */
let opening = false;
/** The browser offers the page a camera, and screen sharing. */
const offers = { camera: false, screen: false };
let run: Run | null = null;
let rate = new RateMeter();
let rolling = new Rolling(RISK_WINDOW);
let curve: RiskCurve | null = null;
/** What the overlay shows: the boxes of the last frame analysed, in work pixels. */
let boxes: LiveBox[] = [];
let work: [number, number] | null = null;
let clock = 0;

const video = () => $<HTMLVideoElement>("video");
const backendName = (b = local.backend) => (b === "webgpu" ? "WebGPU" : "WebAssembly on the CPU");

// ------------------------------------------------------------------ state and messages
function setState(state: State, text: string) {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
  const idle = ready && !opening && run === null;
  $<HTMLButtonElement>("btn-camera").disabled = !idle || !offers.camera;
  $<HTMLButtonElement>("btn-screen").disabled = !idle || !offers.screen;
  for (const stop of document.querySelectorAll<HTMLButtonElement>("[data-stop]")) {
    stop.hidden = run === null;
    stop.disabled = !!run?.end;
  }
  // while a run is on, the bar under the video, with its Stop, stays at the bottom of the screen
  $("view").toggleAttribute("data-running", run !== null);
}

function readyText(): string {
  return `Ready. The detector runs on this device, on ${backendName()}.`;
}

function say(text: string, kind: "err" | "warn" | "ok" = "err") {
  const el = $("live-msg");
  el.textContent = text;
  el.dataset.kind = kind;
  el.hidden = !text;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** A readable reason for a source that did not start (getUserMedia and getDisplayMedia errors). */
function sourceError(e: unknown, source: Source): string {
  const name = e instanceof Error || e instanceof DOMException ? e.name : "";
  if (source === "camera")
    switch (name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        return "The camera is not allowed for this page. Allow it in the browser's site settings (the icon next to the address), then press the button again.";
      case "NotFoundError":
      case "DevicesNotFoundError":
      case "OverconstrainedError":
        return "This device has no camera the browser can use.";
      case "NotReadableError":
      case "TrackStartError":
        return "The camera is busy or blocked by the system. Close other apps that use it and try again.";
      case "SecurityError":
      case "NotSupportedError":
        return "This browser gives this page no camera. Browsers only offer cameras to pages served over HTTPS.";
      default:
        return `The camera did not start: ${errorText(e)}`;
    }
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Nothing was shared. The picker was closed, or sharing is blocked for this page. Press the button again and pick a tab or window.";
    case "NotFoundError":
      return "The browser found nothing to share.";
    case "NotReadableError":
      return "The system did not let the browser capture that tab or window. On a Mac, allow screen recording for the browser in System Settings.";
    case "NotSupportedError":
      return "This browser cannot share a tab or window with a page.";
    default:
      return `Sharing did not start: ${errorText(e)}`;
  }
}

// ------------------------------------------------------------------ sources
async function openSource(source: Source): Promise<MediaStream> {
  const md = navigator.mediaDevices as MediaDevices | undefined;
  if (!md || typeof (source === "camera" ? md.getUserMedia : md.getDisplayMedia) !== "function")
    throw new DOMException("No media devices", "NotSupportedError");
  if (source === "camera")
    return md.getUserMedia({ audio: false, video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: { ideal: "environment" } } });
  // Chrome brings a shared tab to the front unless told not to; this page should stay in front
  const Controller = (window as Window & { CaptureController?: new () => { setFocusBehavior(b: string): void } }).CaptureController;
  const controller = Controller ? new Controller() : undefined;
  // this tab is left out of the picker: shared with itself, it would analyse its own boxes
  const options = { video: { frameRate: { ideal: 30 } }, audio: false, selfBrowserSurface: "exclude", controller };
  const stream = await md.getDisplayMedia(options as DisplayMediaStreamOptions);
  const surface = (stream.getVideoTracks()[0]?.getSettings() as MediaTrackSettings & { displaySurface?: string }).displaySurface;
  try {
    if (surface === "browser" || surface === "window") controller?.setFocusBehavior("no-focus-change");
  } catch {
    // too late, or not offered for this surface: the browser decides
  }
  return stream;
}

function stopTracks(stream: MediaStream) {
  for (const track of stream.getTracks()) track.stop();
}

/** Turns the camera or the capture off, and keeps the last picture as the video's poster so that
 * the last boxes stay over it. */
function release(r: Run) {
  if (r.released) return;
  r.released = true;
  const v = video();
  if (v.videoWidth) {
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")?.drawImage(v, 0, 0);
    try {
      v.poster = c.toDataURL("image/jpeg", 0.85);
    } catch {
      // no still: the video goes black
    }
  }
  stopTracks(r.stream);
}

/** Resolves once the video shows a picture; rejects after FIRST_FRAME_MS or when the run stops. */
function firstPicture(r: Run, v: HTMLVideoElement): Promise<void> {
  return new Promise((resolve, reject) => {
    const events = ["loadeddata", "resize", "playing", "timeupdate"];
    const done = (err?: Error) => {
      clearTimeout(timer);
      for (const ev of events) v.removeEventListener(ev, check);
      r.abort.signal.removeEventListener("abort", stopped);
      if (err) reject(err);
      else resolve();
    };
    const check = () => {
      if (v.readyState >= 2 && v.videoWidth > 0) done();
    };
    const stopped = () => done(new ApiError("aborted", "Stopped."));
    const timer = window.setTimeout(() => done(new Error("The stream shows no picture.")), FIRST_FRAME_MS);
    for (const ev of events) v.addEventListener(ev, check);
    r.abort.signal.addEventListener("abort", stopped);
    check();
  });
}

/** Counts the pictures a video shows, where the browser reports them (requestVideoFrameCallback),
 * so that the worker is not sent the same picture twice. */
class FrameCounter {
  count = 0;
  private video: HTMLVideoElement;
  private supported: boolean;
  private handle = 0;
  private wake: (() => void) | null = null;

  constructor(v: HTMLVideoElement) {
    this.video = v;
    this.supported = "requestVideoFrameCallback" in v;
    if (this.supported) this.handle = v.requestVideoFrameCallback(this.shown);
  }

  private shown = () => {
    this.count++;
    this.wake?.();
    this.handle = this.video.requestVideoFrameCallback(this.shown);
  };

  /** Resolves once a picture after the `seen`-th has been shown, or after `ms` ms. */
  next(seen: number, ms: number): Promise<void> {
    if (!this.supported || this.count > seen) return Promise.resolve();
    return new Promise((resolve) => {
      const timer = window.setTimeout(() => this.wake?.(), ms);
      this.wake = () => {
        this.wake = null;
        clearTimeout(timer);
        resolve();
      };
    });
  }

  close() {
    if (this.supported) this.video.cancelVideoFrameCallback(this.handle);
    this.wake?.();
  }
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

// ------------------------------------------------------------------ a run
async function start(source: Source) {
  if (run || opening || !ready) return;
  // the permission prompt or the picker must open straight from the click, before any other wait
  const asked = openSource(source);
  opening = true;
  say("");
  $("events").hidden = true;
  setState("asking", source === "camera" ? "Waiting for the camera" : "Waiting for you to pick a tab or window");
  let stream: MediaStream;
  try {
    stream = await asked;
  } catch (e) {
    opening = false;
    setState("ready", readyText());
    say(sourceError(e, source));
    return;
  }
  opening = false;
  const r: Run = {
    source,
    stream,
    abort: new AbortController(),
    job: null,
    aligned: false,
    end: null,
    frames: 0,
    lastT: 0,
    started: 0,
    resizedAt: performance.now(),
    released: false,
  };
  run = r;
  for (const track of stream.getVideoTracks()) track.addEventListener("ended", () => stop("ended"));
  resetView();
  const view = $("view");
  view.hidden = false;
  view.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  const v = video();
  v.srcObject = stream;
  v.play().catch(() => undefined);
  setState("asking", "Waiting for the first picture");

  let failed = false;
  let step: "picture" | "load" | "run" = "picture";
  try {
    await firstPicture(r, v);
    step = "load";
    // the model goes on loading after a Stop, for the next run; this run no longer shows it
    const progress = (f: number) => {
      if (run !== r || r.end) return;
      setState("loading", f < 1 ? `Loading the model, ${Math.round(f * 100)}%` : "Starting the detector");
    };
    progress(0);
    r.job = await local.live(r.abort.signal, progress);
    $("backend").textContent = `on ${backendName(r.job.engine.backend)}`;
    step = "run";
    await analyse(r, r.job, v);
  } catch (e) {
    if (!(e instanceof ApiError && e.kind === "aborted")) {
      failed = true;
      const text = errorText(e);
      say(
        step === "picture"
          ? `${text} Check that the camera or the shared tab shows something, then try again.`
          : step === "load"
            ? `The pipeline did not load: ${text.replace(/^Could not load the pipeline: /, "")} Check the connection and try again.`
            : `The analysis stopped: ${text}`,
      );
    }
  }
  if (!failed) await wrapUp(r);
  close(r);
}

/** Resolves once the picture has kept its size for STEADY_MS, or after STEADY_MAX_MS, or when the
 * run stops; rejects when the video has shown no picture for all of STEADY_MAX_MS. */
async function steadySize(r: Run, v: HTMLVideoElement): Promise<void> {
  const t0 = performance.now();
  let size = `${v.videoWidth}x${v.videoHeight}`;
  while (!r.end) {
    const now = performance.now();
    // the video's resize event sets resizedAt too; this catches a browser that does not send it
    const s = `${v.videoWidth}x${v.videoHeight}`;
    if (s !== size) {
      size = s;
      r.resizedAt = now;
    }
    const shows = v.videoWidth > 0 && v.videoHeight > 0;
    if (shows && (now - r.resizedAt >= STEADY_MS || now - t0 >= STEADY_MAX_MS)) return;
    if (!shows && now - t0 >= STEADY_MAX_MS) throw new Error("The stream shows no picture.");
    await sleep(100);
  }
}

async function analyse(r: Run, job: LiveJob, v: HTMLVideoElement) {
  if (r.end) return;
  setState("aligning", "Checking the view");
  await steadySize(r, v);
  if (r.end) return;
  const shapeOf = (w: number, h: number) => frameShape(w, h, job.engine.input, WORK_WIDTH);
  let size: [number, number] = [v.videoWidth, v.videoHeight];
  let shape = shapeOf(...size);
  work = shape.work;
  let frames = new FrameReader(v, shape.work, shape.det[0]);
  const shown = new FrameCounter(v);
  try {
    r.started = performance.now();
    const gray = frames.grayFrame();
    const aligned = await job.align(gray, shape.work);
    r.aligned = aligned.ok;
    showAlignment(aligned.ok);
    startClock(r);
    setState("running", r.aligned ? "Running: detection, tracking, events and risk" : "Running: detection and tracking");

    const pacer = new Pacer();
    let seen = shown.count;
    while (!r.end) {
      await sleep(pacer.wait(performance.now()));
      await shown.next(seen, NEW_FRAME_MS);
      if (r.end) break;
      if (v.videoWidth !== size[0] || v.videoHeight !== size[1]) {
        if (!v.videoWidth || !v.videoHeight) {
          seen = shown.count; // no picture for now: wait for the next one
          continue;
        }
        size = [v.videoWidth, v.videoHeight];
        const next = shapeOf(...size);
        // a new resolution of the same shape changes nothing for the pipeline
        if (!sameShape(next, shape)) {
          // the lamp patches and the rules are drawn for the view the first frame showed
          if (r.aligned) {
            r.end = "resized";
            break;
          }
          // elsewhere detection and tracking go on in the new shape; tracks may break at the change
          shape = next;
          work = next.work;
          frames = new FrameReader(v, next.work, next.det[0]);
          boxes = [];
          drawBoxes();
        }
      }
      const now = performance.now();
      const t = Math.round(now - r.started) / 1000;
      if (t >= LIVE_MAX_SECONDS) {
        r.end = "limit";
        break;
      }
      seen = shown.count;
      const det = frames.detFrame();
      const lamp = r.aligned ? frames.lampCrop(aligned.lampCrop) : null;
      pacer.sent(now);
      const reply = await job.frame(r.frames, t, det, lamp);
      pacer.replied();
      r.frames++;
      r.lastT = t;
      rate.tick(now); // when the frame was taken: the pacer keeps those at least 200 ms apart
      showFrame(r, reply);
    }
  } finally {
    shown.close();
  }
}

/** After the loop: the events of an aligned run, or a line on what the run did. */
async function wrapUp(r: Run) {
  stopClock();
  release(r);
  const why: Record<End, string> = {
    stop: "",
    ended: "The stream ended, so the run stopped.",
    limit: `A live run stops after ${LIVE_MAX_SECONDS / 60} minutes. Start it again to go on.`,
    resized: "The picture changed shape (a turned phone or a resized window). The event rules need the view the run started with, so the run stopped. Start it again to go on.",
    left: "The run stopped when you left the page.",
  };
  if (r.end && why[r.end]) say(why[r.end], "warn");
  if (!r.job || !r.frames) return;
  const summary = `${fmtTime(r.lastT, 0)} of stream, ${plural(r.frames, "frame")}`;
  if (!r.aligned) {
    r.job.cancel();
    $("final").textContent = `Stopped after ${summary}.`;
    return;
  }
  setState("stopping", "Applying the event rules to the whole run");
  try {
    const done = await r.job.finish(r.lastT + 0.2, r.source === "camera" ? "camera" : "shared screen");
    $("final").textContent = `Stopped after ${summary}.`;
    showEvents(done.result.events, summary);
  } catch (e) {
    say(`The event rules did not finish: ${errorText(e)}`);
  }
}

function close(r: Run) {
  release(r);
  video().srcObject = null; // shows the still
  r.abort.abort();
  stopClock();
  run = null;
  setState("ready", readyText());
}

function stop(end: End) {
  const r = run;
  if (!r || r.end) return;
  r.end = end;
  // the camera goes off at once; the frame in the worker, if any, finishes first
  release(r);
  if (!r.job) r.abort.abort();
  setState("stopping", "Stopping");
}

// ------------------------------------------------------------------ display
function resetView() {
  boxes = [];
  work = null;
  drawBoxes();
  video().removeAttribute("poster");
  rolling = new Rolling(RISK_WINDOW);
  rate = new RateMeter();
  $("rate").textContent = "-";
  $("backend").textContent = `on ${backendName()}`;
  $("clock").textContent = "0:00";
  $("frames").textContent = "0 frames";
  $("final").textContent = "";
  $("align-msg").hidden = true;
  $("risk-panel").hidden = true;
  const box = $("risk-box");
  box.dataset.on = "wait";
  delete box.dataset.level;
  $("risk").textContent = "-";
  $("risk-note").textContent = "Waiting for the view check";
  for (const name of COUNTED) $(`count-${name}`).textContent = "0";
}

function showAlignment(ok: boolean) {
  const el = $("align-msg");
  el.dataset.ok = String(ok);
  el.textContent = ok
    ? "This looks like our junction: events and risk are on."
    : "This is not our junction, or not our camera's view of it. The event rules and the risk model are drawn for that one view, so here the page only detects and tracks. The boxes and counts still work.";
  el.hidden = false;
  $("risk-box").dataset.on = String(ok);
  $("risk").textContent = ok ? "-" : "off";
  $("risk-note").textContent = ok ? `Alarm at ${RISK_THETA}` : "Only on our junction's view";
}

function showFrame(r: Run, m: Extract<FromWorker, { type: "live" }>) {
  boxes = m.boxes;
  drawBoxes();
  const counts = countInView(m.boxes);
  for (const name of COUNTED) $(`count-${name}`).textContent = String(counts[name]);
  $("frames").textContent = plural(r.frames, "frame");
  showRate();
  if (!r.aligned) return;
  $("risk").textContent = m.risk.toFixed(2);
  $("risk-box").dataset.level = m.risk >= RISK_THETA ? "alarm" : "calm";
  rolling.push(m.t, m.risk);
  $("risk-panel").hidden = false;
  const data = { duration: RISK_WINDOW, start: rolling.start, risk: rolling.points };
  if (!curve) curve = new RiskCurve($("risk-curve"), data, { theta: RISK_THETA, mergeGap: RISK_MERGE_GAP, onSeek: () => undefined });
  else curve.setData(data);
  curve.setTime(m.t);
}

function showRate() {
  const fps = rate.rate(performance.now());
  $("rate").textContent = fps === null ? "-" : `${fps.toFixed(1)} fps`;
}

function startClock(r: Run) {
  stopClock();
  const tick = () => {
    $("clock").textContent = fmtTime((performance.now() - r.started) / 1000, 0);
    showRate();
  };
  tick();
  clock = window.setInterval(tick, 500);
}

function stopClock() {
  clearInterval(clock);
}

/** The tracked boxes of the last frame analysed, over the video, each tagged with its track id and class. */
function drawBoxes() {
  const c = $<HTMLCanvasElement>("overlay");
  const dpr = window.devicePixelRatio || 1;
  const cr = c.getBoundingClientRect();
  const W = Math.round(cr.width * dpr), H = Math.round(cr.height * dpr);
  if (c.width !== W || c.height !== H) {
    c.width = W;
    c.height = H;
  }
  const g = c.getContext("2d");
  if (!g) return;
  g.clearRect(0, 0, W, H);
  if (!work || !boxes.length || !W || !H) return;
  const { ox, oy, sx, sy } = workToCanvas(video(), cr, work, dpr);
  g.lineJoin = "round";
  g.lineWidth = 1.5 * dpr;
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.box;
    g.strokeStyle = `rgba(${GROUP_RGB[b.group] ?? OTHER_RGB}, 0.9)`;
    g.strokeRect(ox + x1 * sx, oy + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
  }
  // tags last, so that no box crosses one
  g.font = tagFont(dpr);
  g.textBaseline = "bottom";
  for (const b of boxes) {
    const text = `${b.id % 1_000_000} ${CLASS_NAMES[b.cls] ?? b.group}`;
    drawTag(g, text, ox + b.box[0] * sx, oy + b.box[1] * sy, GROUP_RGB[b.group] ?? OTHER_RGB, dpr, W);
  }
}

function showEvents(events: Seg[], summary: string) {
  const rows = $("event-rows");
  rows.textContent = "";
  $("events-sum").textContent = events.length
    ? `${plural(events.length, "event")} in ${summary}. Times are from the start of the run.`
    : `No events in ${summary}. Most rules need a road user to do the same thing for a few seconds, and the signal rules need the lamps in view.`;
  $("events-table").hidden = !events.length;
  for (const [s, e, label] of [...events].sort((a, b) => a[0] - b[0])) {
    const tr = document.createElement("tr");
    const cell = (text: string, cls: string) => {
      const td = document.createElement("td");
      td.className = cls;
      td.textContent = text;
      tr.append(td);
    };
    cell(className(label), "cls-name");
    cell(fmtTime(s), "t");
    cell(fmtTime(e), "t");
    cell(`${(e - s).toFixed(1)} s`, "num");
    rows.append(tr);
  }
  const section = $("events");
  section.hidden = false;
  section.scrollIntoView({ block: "nearest" });
  section.focus({ preventScroll: true });
}

function legend() {
  const box = $("legend");
  box.textContent = "";
  for (const [group, rgb] of Object.entries(GROUP_RGB)) {
    const span = document.createElement("span");
    const sw = document.createElement("i");
    sw.style.borderColor = `rgb(${rgb})`;
    span.append(sw, group);
    box.append(span);
  }
}

// ------------------------------------------------------------------ wiring
export async function initLive() {
  legend();
  $("btn-camera").addEventListener("click", () => start("camera"));
  $("btn-screen").addEventListener("click", () => start("screen"));
  for (const b of document.querySelectorAll("[data-stop]")) b.addEventListener("click", () => stop("stop"));
  new ResizeObserver(() => drawBoxes()).observe($("overlay"));
  video().addEventListener("resize", () => {
    if (run) run.resizedAt = performance.now();
  });
  // Leaving the page ends the run, so that a page the browser keeps and later shows again (the
  // back-forward cache) does not go on posting the last picture of a dead stream.
  addEventListener("pagehide", () => stop("left"));
  // Browsers slow down pages that are not in front, and may stop drawing their videos.
  document.addEventListener("visibilitychange", () => {
    if (run && !run.end && document.hidden)
      say("This tab was in the background, where the browser slows the analysis down. Keep it in front for the full rate.", "warn");
  });

  const md = navigator.mediaDevices as MediaDevices | undefined;
  offers.camera = typeof md?.getUserMedia === "function";
  offers.screen = typeof md?.getDisplayMedia === "function";
  // phones and some browsers cannot share a screen
  $("btn-screen").hidden = !offers.screen;
  $("screen-hint").hidden = !offers.screen;
  setState("checking", "Checking what this browser can run");
  if ((await local.health()) !== "ok") {
    setState("unsupported", "This browser cannot run the pipeline");
    say("Live mode needs Web Workers, WebAssembly and built-in gzip decompression, and this browser lacks one of them. Try a recent Chrome, Edge, Firefox or Safari.");
    return;
  }
  if (!offers.camera && !offers.screen) {
    setState("unsupported", "This browser gives the page no camera");
    say("Live mode needs a camera or screen sharing, and this browser offers neither to this page. Browsers only offer them to pages served over HTTPS.");
    return;
  }
  ready = true;
  setState("ready", readyText());
  if (!offers.camera) say("This browser offers the page no camera. You can still share a tab or window.", "warn");
}
