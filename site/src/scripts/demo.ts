// Demo page controller: what this browser can run, file or sample, job progress, result.
import { RISK_MERGE_GAP, RISK_THETA, UPLOAD_MAX_SECONDS } from "../config";
import { fmtBytes, fmtTime } from "../lib/format";
import type { ClipResult, Job, Sample } from "../lib/types";
import { ApiError, CONVERT, MockApi, STAGES, type Api } from "./api";
import { LocalApi } from "./local_api";
import { Player } from "./player";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const params = new URLSearchParams(location.search);
const mockParam = params.get("mock");
const forceMock = mockParam !== null && mockParam !== "0";

/** online: the pipeline runs here; offline: it cannot, so the page replays a stored job. */
type EngineState = "checking" | "online" | "offline" | "mock";

const local = new LocalApi();
let api: Api = forceMock ? new MockApi({ fail: mockParam === "error" }) : local;
let engineState: EngineState = "checking";
let file: File | null = null;
let fileOk = false;
let running = false;
/** Whether the running job converts its clip first, which adds a stage to the list. */
let converting = false;
let abort: AbortController | null = null;
let player: Player | null = null;
let lastResult: { result: ClipResult; source: string; jobId: string } | null = null;

// ------------------------------------------------------------------ what runs here
const backendName = () => (local.backend === "webgpu" ? "WebGPU" : "WebAssembly on the CPU");

function setEngine(state: EngineState, detail = "") {
  engineState = state;
  const box = $("server");
  box.dataset.state = state;
  const text: Record<EngineState, string> = {
    checking: "Checking what this browser can run",
    online: `Runs in this browser, on ${backendName()}`,
    offline: "This browser cannot run the pipeline: replay mode",
    mock: "Replay mode",
  };
  $("server-text").textContent = text[state];
  $("server-detail").textContent = detail;
  $("replay-note").hidden = !(state === "offline" || state === "mock");
  document.querySelectorAll<HTMLButtonElement>("[data-needs-engine]").forEach((b) => (b.disabled = state === "checking" || running));
  updateAnalyse();
}

function onlineDetail() {
  return local.backend === "webgpu"
    ? "Nothing is uploaded. The detector runs on this device's graphics chip through WebGPU."
    : "Nothing is uploaded. This browser offers no WebGPU, so the detector runs on the CPU through WebAssembly. It works, only slower.";
}

async function checkEngine() {
  if (forceMock) {
    setEngine("mock", "Started with ?mock=1. Nothing is analysed; the page plays back a stored example job.");
    return loadSamples();
  }
  setEngine("checking");
  if ((await local.health()) !== "ok") {
    api = new MockApi();
    setEngine("offline", "It needs Web Workers, WebAssembly and built-in gzip decompression, and this browser lacks one of them. The page plays back a stored example job instead.");
    return loadSamples();
  }
  api = local;
  setEngine("online", onlineDetail());
  loadSamples();
}

async function loadSamples() {
  const box = $("samples");
  box.textContent = "";
  const note = $("samples-note");
  note.textContent = "Loading sample clips";
  let samples: Sample[] = [];
  try {
    samples = await api.samples();
  } catch (e) {
    note.textContent = e instanceof Error ? e.message : "Could not load the sample clips.";
    return;
  }
  note.textContent = !samples.length
    ? "No sample clips right now."
    : api.mock
      ? "In replay mode every clip gives the same stored result."
      : "Cuts from the organisers' footage, about 11 MB each. The clip downloads to this browser and is analysed here.";
  for (const s of samples) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sample";
    b.dataset.needsEngine = "";
    b.innerHTML = `<span class="s-label"></span><span class="s-meta mono"></span>`;
    b.querySelector(".s-label")!.textContent = s.label;
    b.querySelector(".s-meta")!.textContent = `${s.name}, ${s.seconds} s`;
    b.addEventListener("click", () => startSample(s));
    box.append(b);
  }
  setEngine(engineState, $("server-detail").textContent || "");
}

// ------------------------------------------------------------------ file choice
function updateAnalyse() {
  const btn = $<HTMLButtonElement>("btn-analyse");
  btn.disabled = !file || !fileOk || running || engineState === "checking";
}

function fileMsg(text: string, kind: "ok" | "warn" | "err") {
  const el = $("file-msg");
  el.textContent = text;
  el.dataset.kind = kind;
  el.hidden = !text;
}

async function readDuration(f: File): Promise<{ duration: number; width: number; height: number } | null> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    const url = URL.createObjectURL(f);
    let done = false;
    const finish = (r: { duration: number; width: number; height: number } | null) => {
      if (done) return;
      done = true;
      URL.revokeObjectURL(url);
      v.removeAttribute("src");
      v.load();
      resolve(r);
    };
    v.preload = "metadata";
    v.muted = true;
    v.onloadedmetadata = () => finish(Number.isFinite(v.duration) ? { duration: v.duration, width: v.videoWidth, height: v.videoHeight } : null);
    v.onerror = () => finish(null);
    setTimeout(() => finish(null), 6000);
    v.src = url;
  });
}

async function chooseFile(f: File | null) {
  file = f;
  fileOk = false;
  const info = $("file-info");
  if (!f) {
    info.hidden = true;
    fileMsg("", "ok");
    updateAnalyse();
    return;
  }
  info.hidden = false;
  $("file-name").textContent = f.name;
  $("file-size").textContent = fmtBytes(f.size);
  $("file-dur").textContent = "reading";
  updateAnalyse();
  const isMp4 = /\.mp4$/i.test(f.name) || f.type === "video/mp4";
  if (!isMp4) {
    $("file-dur").textContent = "-";
    return fileMsg("Pick an .mp4 file. Other containers are not supported.", "err");
  }
  const meta = await readDuration(f);
  if (file !== f) return;
  fileOk = true;
  if (!meta || !meta.width) {
    $("file-dur").textContent = "unknown";
    fileMsg(`This browser cannot play the file, so we first convert it here, in the browser. For ${UPLOAD_MAX_SECONDS / 60} minutes of 4K that takes a few minutes.`, "warn");
    return updateAnalyse();
  }
  $("file-dur").textContent = `${fmtTime(meta.duration)}, ${meta.width}x${meta.height}`;
  if (local.alwaysConvert && !api.mock) fileMsg("Started with ?transcode=1: we convert the clip in this browser first, although this browser can play it.", "warn");
  else if (meta.duration > UPLOAD_MAX_SECONDS + 0.5)
    fileMsg(`This clip is ${fmtTime(meta.duration, 0)} long. Only the first ${fmtTime(UPLOAD_MAX_SECONDS, 0)} is analysed.`, "warn");
  else if (meta.width >= 3000) fileMsg("4K works but is slower. If this browser decodes the clip slowly, as Chrome does with the camera's own 10-bit files, we convert it here first.", "warn");
  else fileMsg("Ready.", "ok");
  updateAnalyse();
}

// ------------------------------------------------------------------ job
function show(section: "idle" | "job" | "error" | "result") {
  $("job").hidden = section !== "job";
  $("job-error").hidden = section !== "error";
  $("result").hidden = section !== "result";
}

function setRunning(v: boolean) {
  running = v;
  document.querySelectorAll<HTMLButtonElement>("[data-needs-engine]").forEach((b) => (b.disabled = v || engineState === "checking"));
  $<HTMLInputElement>("file-input").disabled = v;
  $("drop").classList.toggle("disabled", v);
  updateAnalyse();
}

function renderStages(stage: string, status: Job["status"] | "starting") {
  const ol = $("stages");
  ol.textContent = "";
  const cur = stage.toLowerCase();
  if (cur === CONVERT) converting = true;
  const stages = converting ? [STAGES[0], CONVERT, ...STAGES.slice(1)] : STAGES;
  let idx = stages.findIndex((s) => s === cur);
  if (idx < 0) idx = stages.findIndex((s) => cur.includes(s) || s.includes(cur));
  const curItem = status === "starting" ? 0 : status === "done" ? stages.length : idx;
  stages.forEach((name, i) => {
    const li = document.createElement("li");
    li.textContent = name;
    li.className = curItem < 0 ? "pending" : i < curItem ? "done" : i === curItem ? "active" : "pending";
    if (i === curItem) li.setAttribute("aria-current", "step");
    ol.append(li);
  });
  if (curItem < 0 && stage) {
    const li = document.createElement("li");
    li.className = "active";
    li.setAttribute("aria-current", "step");
    li.textContent = stage;
    ol.append(li);
  }
}

function setProgress(f: number, stage: string, eta: number | null, status: Job["status"] | "starting") {
  const pct = Math.round(Math.max(0, Math.min(1, f)) * 100);
  const bar = $("bar");
  bar.style.setProperty("--p", `${pct}%`);
  bar.setAttribute("aria-valuenow", String(pct));
  bar.setAttribute("aria-valuetext", `${pct}%, ${stage}`);
  $("pct").textContent = `${pct}%`;
  $("stage").textContent = stage;
  $("eta").textContent =
    eta === null || eta === undefined ? "estimating time left" : eta <= 1 ? "almost done" : eta < 90 ? `about ${Math.round(eta)} s left` : `about ${Math.round(eta / 60)} min left`;
  renderStages(stage, status);
}

let elapsedTimer = 0;
function startElapsed() {
  const t0 = performance.now();
  clearInterval(elapsedTimer);
  const tick = () => ($("elapsed").textContent = `${Math.round((performance.now() - t0) / 1000)} s`);
  tick();
  elapsedTimer = window.setInterval(tick, 500);
}

function fail(msg: string) {
  clearInterval(elapsedTimer);
  setRunning(false);
  $("job-error-msg").textContent = msg;
  show("error");
  $("job-error").focus();
}

function idle() {
  clearInterval(elapsedTimer);
  setRunning(false);
  show("idle");
}

// Browsers slow down pages that are not in front, and seeking a hidden video can stall.
document.addEventListener("visibilitychange", () => {
  if (!running || api.mock || !document.hidden) return;
  const warn = $("job-warn");
  warn.textContent = "This tab was in the background, where the browser slows the analysis down. Keep it in front to finish sooner.";
  warn.hidden = false;
});

async function runJob(source: string, start: (signal: AbortSignal) => Promise<string>) {
  if (running) return;
  abort = new AbortController();
  const signal = abort.signal;
  converting = false;
  setRunning(true);
  show("job");
  $("job-title").textContent = source;
  $("job-mode").textContent = api.mock ? "replay" : "on this device";
  $("job-warn").hidden = true;
  startElapsed();
  setProgress(0, STAGES[0], null, "starting");
  $("job").scrollIntoView({ block: "nearest" });

  let jobId: string;
  try {
    jobId = await start(signal);
  } catch (e) {
    if (e instanceof ApiError && e.kind === "aborted") return idle();
    return fail(e instanceof Error ? e.message : "Could not start the job.");
  }

  for (;;) {
    if (signal.aborted) return idle();
    let job: Job;
    try {
      job = await api.job(jobId);
    } catch (e) {
      return fail(e instanceof Error ? e.message : "Lost track of the job.");
    }
    if (signal.aborted) return idle();
    if (job.status === "error") return fail(job.error || "The analysis stopped without saying why.");
    if (job.status === "done") {
      if (!job.result) return fail("The job finished but returned no result.");
      setProgress(1, "done", 0, "done");
      clearInterval(elapsedTimer);
      setRunning(false);
      // the model may have fallen back from WebGPU to WebAssembly while it loaded
      if (api === local) setEngine("online", onlineDetail());
      return showResult(job.result, source, jobId);
    }
    setProgress(job.progress ?? 0, job.stage || job.status, job.eta_sec, job.status);
    await new Promise((r) => setTimeout(r, api.mock ? 400 : 500));
  }
}

function startUpload() {
  if (!file || !fileOk) return;
  const f = file;
  runJob(f.name, (signal) => api.submitFile(f, signal));
}

function startSample(s: Sample) {
  runJob(`${s.label} (${s.name})`, (signal) => api.submitSample(s.name, signal));
}

// ------------------------------------------------------------------ result
function showResult(result: ClipResult, source: string, jobId: string) {
  lastResult = { result, source, jobId };
  show("result");
  const video = result.video ?? "";
  $("result-title").textContent = source;
  $("result-mode").textContent = api.mock
    ? "Stored example result from replay mode, not an analysis of your file."
    : [
        `Analysed on this device with ${backendName()}: ${result.risk.length} frames, 5 per second.`,
        result.aligned === false
          ? "The first frame did not match our view of the junction, so the rules looked in the wrong places and these events are not to be trusted."
          : "",
      ].join(" ");
  const root = document.querySelector<HTMLElement>('[data-player="demo"]')!;
  // the clip's own first frame instead of a stock poster, which the tracked boxes would not fit
  const poster = result.overlay ? undefined : "/media/demo_poster.jpg";
  const src = { result, video, poster, title: `${source}, ${result.duration.toFixed(1)} s` };
  if (!player) player = new Player(root, src, { theta: RISK_THETA, mergeGap: RISK_MERGE_GAP });
  else player.load(src);
  $("result").scrollIntoView({ block: "start" });
  $("result-heading").focus({ preventScroll: true });
}

function downloadJson() {
  if (!lastResult) return;
  const { result, source, jobId } = lastResult;
  // the result as the Python demo server returned it: no object URL, no per-frame boxes
  const data: Partial<ClipResult> = { ...result };
  delete data.video;
  delete data.overlay;
  const payload = { job_id: jobId, source, mode: api.mock ? "mock" : `browser, ${local.backend}`, ...data };
  const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const slug = source.replace(/\.[a-z0-9]+$/i, "").replace(/[^a-z0-9]+/gi, "_").replace(/^_|_$/g, "").slice(0, 40) || "clip";
  a.download = `parivision_${slug}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ------------------------------------------------------------------ wiring
export function initDemo() {
  const input = $<HTMLInputElement>("file-input");
  const drop = $("drop");
  input.addEventListener("change", () => chooseFile(input.files?.[0] ?? null));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      if (!running) drop.classList.add("over");
    }),
  );
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    if (running) return;
    const f = (e as DragEvent).dataTransfer?.files?.[0];
    if (f) chooseFile(f);
  });
  $("btn-clear").addEventListener("click", () => {
    input.value = "";
    chooseFile(null);
  });
  $("btn-analyse").addEventListener("click", startUpload);
  $("btn-cancel").addEventListener("click", () => abort?.abort());
  $("btn-again").addEventListener("click", () => {
    show("idle");
    document.getElementById("inputs")?.scrollIntoView({ block: "start" });
  });
  $("btn-err-back").addEventListener("click", () => show("idle"));
  $("btn-download").addEventListener("click", downloadJson);
  show("idle");
  checkEngine();
}
