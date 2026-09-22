// Demo page controller: server status, upload or sample, job progress, result.
import { API_BASE_URL, RISK_MERGE_GAP, RISK_THETA, UPLOAD_MAX_BYTES, UPLOAD_MAX_SECONDS } from "../config";
import { fmtBytes, fmtTime } from "../lib/format";
import type { ClipResult, Job, Sample } from "../lib/types";
import { ApiError, HttpApi, MockApi, STAGES, type Api } from "./api";
import { Player } from "./player";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const params = new URLSearchParams(location.search);
const mockParam = params.get("mock");
const forceMock = mockParam !== null && mockParam !== "0";
const DEV = import.meta.env.DEV;
const WAKE_LIMIT_S = 90;
const UNREACHABLE_LIMIT_S = DEV ? 0 : 30;

type ServerState = "checking" | "online" | "waking" | "offline" | "mock";

let api: Api = forceMock ? new MockApi({ fail: mockParam === "error" }) : new HttpApi(API_BASE_URL);
let serverState: ServerState = "checking";
let checkToken = 0;
let file: File | null = null;
let fileOk = false;
let running = false;
let abort: AbortController | null = null;
let player: Player | null = null;
let lastResult: { result: ClipResult; source: string; jobId: string } | null = null;

// ------------------------------------------------------------------ server status
function setServer(state: ServerState, detail = "") {
  serverState = state;
  const box = $("server");
  box.dataset.state = state;
  const text: Record<ServerState, string> = {
    checking: "Checking the analysis server",
    online: "Analysis server online",
    waking: "Waking up the server",
    offline: "Server not reachable: replay mode",
    mock: "Replay mode",
  };
  $("server-text").textContent = text[state];
  $("server-detail").textContent = detail;
  $("btn-use-mock").hidden = state !== "waking";
  $("btn-retry-server").hidden = state !== "offline";
  const replay = state === "offline" || state === "mock";
  $("replay-note").hidden = !replay;
  document.querySelectorAll<HTMLButtonElement>("[data-needs-server]").forEach((b) => (b.disabled = state === "checking" || state === "waking" || running));
  updateAnalyse();
}

async function checkServer() {
  const token = ++checkToken;
  if (forceMock) {
    api = new MockApi({ fail: mockParam === "error" });
    setServer("mock", "Started with ?mock=1. Nothing is uploaded; the page plays back a stored example job.");
    return loadSamples();
  }
  api = new HttpApi(API_BASE_URL);
  setServer("checking", API_BASE_URL.replace(/^https?:\/\//, ""));
  const t0 = performance.now();
  let first = true;
  for (;;) {
    const h = await (api as HttpApi).health();
    if (token !== checkToken) return;
    const el = (performance.now() - t0) / 1000;
    if (h === "ok") {
      setServer("online", API_BASE_URL.replace(/^https?:\/\//, ""));
      return loadSamples();
    }
    if ((h === "unreachable" && el >= UNREACHABLE_LIMIT_S) || el >= WAKE_LIMIT_S) {
      return fallbackToMock();
    }
    setServer(
      "waking",
      first
        ? "The server sleeps when nobody uses it. The first request wakes it; this can take up to a minute."
        : `Still waking, ${Math.round(el)} s so far. The server sleeps when nobody uses it.`,
    );
    first = false;
    await new Promise((r) => setTimeout(r, 4000));
    if (token !== checkToken) return;
  }
}

function fallbackToMock() {
  checkToken++;
  api = new MockApi();
  setServer("offline", "We could not reach the analysis server, so the page plays back a stored example job. Nothing you pick is uploaded.");
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
  note.textContent = samples.length ? "Short clips cut from the organisers' footage, hosted on our server." : "No sample clips on the server right now.";
  for (const s of samples) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sample";
    b.dataset.needsServer = "";
    b.innerHTML = `<span class="s-label"></span><span class="s-meta mono"></span>`;
    b.querySelector(".s-label")!.textContent = s.label;
    b.querySelector(".s-meta")!.textContent = `${s.name}, ${s.seconds} s`;
    b.addEventListener("click", () => startSample(s));
    box.append(b);
  }
  setServer(serverState, $("server-detail").textContent || "");
}

// ------------------------------------------------------------------ file choice
function updateAnalyse() {
  const btn = $<HTMLButtonElement>("btn-analyse");
  btn.disabled = !file || !fileOk || running || serverState === "checking" || serverState === "waking";
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
  if (f.size > UPLOAD_MAX_BYTES) {
    $("file-dur").textContent = "-";
    return fileMsg(`This file is ${fmtBytes(f.size)}. The limit is ${fmtBytes(UPLOAD_MAX_BYTES)}.`, "err");
  }
  const meta = await readDuration(f);
  if (file !== f) return;
  if (!meta) {
    $("file-dur").textContent = "unknown";
    fileOk = true;
    fileMsg("Your browser cannot read this file's length (4K 10-bit files often do this). The server will check it.", "warn");
    return updateAnalyse();
  }
  $("file-dur").textContent = `${fmtTime(meta.duration)}${meta.width ? `, ${meta.width}x${meta.height}` : ""}`;
  if (meta.duration > UPLOAD_MAX_SECONDS + 0.5) {
    return fileMsg(`This clip is ${fmtTime(meta.duration, 0)} long. The limit is ${fmtTime(UPLOAD_MAX_SECONDS, 0)}; cut it first.`, "err");
  }
  fileOk = true;
  if (meta.width >= 3000) fileMsg("4K works, but it takes several times longer than 1080p.", "warn");
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
  document.querySelectorAll<HTMLButtonElement>("[data-needs-server]").forEach((b) => (b.disabled = v || serverState === "checking" || serverState === "waking"));
  $<HTMLInputElement>("file-input").disabled = v;
  $("drop").classList.toggle("disabled", v);
  updateAnalyse();
}

function renderStages(stage: string, status: Job["status"] | "uploading") {
  const ol = $("stages");
  ol.textContent = "";
  const cur = stage.toLowerCase();
  let idx = STAGES.findIndex((s) => s === cur);
  if (idx < 0) idx = STAGES.findIndex((s) => cur.includes(s) || s.includes(cur));
  const items = ["uploading", ...STAGES];
  const curItem = status === "uploading" ? 0 : status === "done" ? items.length : idx >= 0 ? idx + 1 : -1;
  items.forEach((name, i) => {
    const li = document.createElement("li");
    li.textContent = name;
    if (curItem < 0) li.className = i === 0 ? "done" : "pending";
    else li.className = i < curItem ? "done" : i === curItem ? "active" : "pending";
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

function setProgress(f: number, stage: string, eta: number | null, status: Job["status"] | "uploading") {
  const pct = Math.round(Math.max(0, Math.min(1, f)) * 100);
  const bar = $("bar");
  bar.style.setProperty("--p", `${pct}%`);
  bar.setAttribute("aria-valuenow", String(pct));
  bar.setAttribute("aria-valuetext", `${pct}%, ${stage}`);
  $("pct").textContent = `${pct}%`;
  $("stage").textContent = stage;
  $("eta").textContent = eta === null || eta === undefined ? "estimating time left" : eta <= 1 ? "almost done" : `about ${Math.round(eta)} s left`;
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

async function runJob(source: string, start: (signal: AbortSignal) => Promise<string>) {
  if (running) return;
  abort = new AbortController();
  const signal = abort.signal;
  setRunning(true);
  show("job");
  $("job-title").textContent = source;
  $("job-mode").textContent = api.mock ? "replay" : "live";
  startElapsed();
  setProgress(0, "uploading", null, "uploading");
  $("job").scrollIntoView({ block: "nearest" });

  let jobId: string;
  try {
    jobId = await start(signal);
  } catch (e) {
    if (e instanceof ApiError && e.kind === "aborted") {
      clearInterval(elapsedTimer);
      setRunning(false);
      return show("idle");
    }
    return fail(e instanceof Error ? e.message : "Could not start the job.");
  }

  let misses = 0;
  for (;;) {
    if (signal.aborted) {
      clearInterval(elapsedTimer);
      setRunning(false);
      return show("idle");
    }
    let job: Job;
    try {
      job = await api.job(jobId);
      misses = 0;
      $("job-warn").hidden = true;
    } catch (e) {
      const err = e as ApiError;
      if (err.kind === "not_found") return fail(err.message);
      if (++misses > 8) return fail(`${err.message} We gave up after several tries.`);
      $("job-warn").hidden = false;
      $("job-warn").textContent = "Lost contact with the server, retrying.";
      await new Promise((r) => setTimeout(r, 1500 * misses));
      continue;
    }
    if (job.status === "error") return fail(job.error || "The server reported an error without a message.");
    if (job.status === "done") {
      if (!job.result) return fail("The job finished but returned no result.");
      setProgress(1, "done", 0, "done");
      clearInterval(elapsedTimer);
      setRunning(false);
      return showResult(job.result, source, jobId);
    }
    setProgress(job.progress ?? 0, job.stage || job.status, job.eta_sec, job.status);
    await new Promise((r) => setTimeout(r, api.mock ? 400 : 1200));
  }
}

function startUpload() {
  if (!file || !fileOk) return;
  const f = file;
  runJob(f.name, (signal) =>
    api.submitFile(
      f,
      (p) => setProgress(p * 0.1, p < 1 ? `uploading, ${Math.round(p * 100)}%` : "upload finished", null, "uploading"),
      signal,
    ),
  );
}

function startSample(s: Sample) {
  runJob(`${s.label} (${s.name})`, () => api.submitSample(s.name));
}

// ------------------------------------------------------------------ result
function showResult(result: ClipResult, source: string, jobId: string) {
  lastResult = { result, source, jobId };
  show("result");
  const video = result.video ? api.videoUrl(result.video) : "";
  $("result-title").textContent = source;
  $("result-mode").textContent = api.mock ? "Stored example result from replay mode, not an analysis of your file." : `Job ${jobId}`;
  const root = document.querySelector<HTMLElement>('[data-player="demo"]')!;
  const src = { result, video, poster: "/media/demo_poster.jpg", title: `${source}, ${result.duration.toFixed(1)} s` };
  if (!player) player = new Player(root, src, { theta: RISK_THETA, mergeGap: RISK_MERGE_GAP });
  else player.load(src);
  $("result").scrollIntoView({ block: "start" });
  $("result-heading").focus({ preventScroll: true });
}

function downloadJson() {
  if (!lastResult) return;
  const { result, source, jobId } = lastResult;
  const payload = { job_id: jobId, source, mode: api.mock ? "mock" : "live", ...result };
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
  $("btn-use-mock").addEventListener("click", fallbackToMock);
  $("btn-retry-server").addEventListener("click", () => checkServer());
  show("idle");
  checkServer();
}
