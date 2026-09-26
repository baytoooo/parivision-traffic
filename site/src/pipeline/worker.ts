// The demo's Web Worker (a module worker; the only file of this folder that needs a browser).
// It owns the ONNX session and the Analyser of the running job: src/scripts/local_api.ts draws
// frames from a <video> and posts them here, and for each one the worker runs the detector and
// the lamp reader and feeds the Analyser; at the end it posts the PipelineResult. A frame of the
// live page (src/scripts/live.ts) also gets its tracked boxes and risk back at once. onnxruntime-web
// runs on WebGPU when the browser offers an adapter and on WebAssembly (CPU) otherwise, or when
// WebGPU cannot create the session. Messages are typed in messages.ts.

import webgpuWasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import cpuWasm from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { alignToReference, halfGray, medianGray } from "./align.ts";
import { Analyser } from "./analyse.ts";
import { OnnxDetector, type OrtModule } from "./detector.ts";
import { matInv } from "./geometry.ts";
import type { Backend, FromWorker, LiveBox, ToWorker } from "./messages.ts";
import { Scene, type SceneConstants } from "./scene.ts";
import { lampPatches, lampScores, type Box } from "./signal.ts";
import { groupOf } from "./tracker.ts";

// The DOM lib types `self` as a Window; this is the part of DedicatedWorkerGlobalScope used here.
const scope = self as unknown as {
  postMessage(msg: FromWorker, options?: { transfer?: Transferable[] }): void;
  onmessage: ((ev: MessageEvent<ToWorker>) => void) | null;
  location: Location;
  crossOriginIsolated: boolean;
};
const post = (msg: FromWorker) => scope.postMessage(msg);

interface Engine {
  scene: Scene;
  detector: OnnxDetector;
  refs: Record<string, Uint8Array>;
}
interface RunningJob {
  id: string;
  analyser: Analyser;
  /** The three lamp patches in work pixels, from the latest registration. */
  boxes: Box[];
  workSize: [number, number];
  /** Video time of the next keyframe registration. */
  nextAlign: number;
  /** Grey keyframes (work size / 4), for the median registration at finish. */
  keyframes: Uint8Array[];
}

// pipeline.py: the camera can still be settling when a recording starts (C3896 moves 9 px over its
// first 40 s, enough to take the lamp patches off the lamps within seconds), so the view is
// registered again every 2 s for the first 30 s, then every 10 s; the lamps are read with the
// latest registration and the rules use one registration of the keyframes' median.
const REALIGN_EARLY: [number, number] = [2, 30];
const REALIGN_EVERY = 10;
// work pixels around the first frame's lamp patches that the page crops for every frame: room for
// the patches to follow the camera as it settles
const LAMP_MARGIN = 24;

let engine: Engine | null = null;
let job: RunningJob | null = null;

async function hasWebGpu(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

/** onnxruntime-web's WebGPU build (it also runs the WASM provider) or its smaller WASM-only build. */
async function runtime(webgpu: boolean): Promise<OrtModule> {
  const ort = webgpu ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm");
  // the .mjs glue is in the bundle; only the .wasm is fetched, from the file Vite emitted
  ort.env.wasm.wasmPaths = { wasm: new URL(webgpu ? webgpuWasm : cpuWasm, scope.location.href).href };
  // threads need SharedArrayBuffer, which a page only gets when it is cross-origin isolated
  ort.env.wasm.numThreads = scope.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.logLevel = "error";
  return ort;
}

/** A file's bytes; `onProgress` gets the fraction received while it downloads. */
async function download(url: string, onProgress?: (f: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  const r = await fetch(url);
  if (!r.ok || !r.body) throw new Error(`Could not load ${url.split("/").pop()} (HTTP ${r.status}).`);
  const total = Number(r.headers.get("content-length")) || 0;
  if (!onProgress || !total) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  let shown = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    const now = performance.now();
    if (now - shown > 100) {
      shown = now;
      onProgress(Math.min(1, got / total)); // a compressed transfer can report fewer bytes than arrive
    }
  }
  onProgress(1);
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/** scene.loadScene, except that scene.bin.gz may arrive inflated already: a server that sends it with
 * Content-Encoding: gzip (Vite's dev server does) has fetch undo the gzip before we see the bytes. */
async function sceneFiles(base: string): Promise<Scene> {
  const [json, gz] = await Promise.all([download(base + "scene.json"), download(base + "scene.bin.gz")]);
  const constants = JSON.parse(new TextDecoder().decode(json)) as SceneConstants;
  const raw =
    gz[0] === 0x1f && gz[1] === 0x8b
      ? new Uint8Array(await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer())
      : gz;
  return new Scene(constants, raw);
}

async function init(base: string, refs: Record<string, Uint8Array>, allowWebGpu: boolean): Promise<void> {
  const [scene, model, webgpu] = await Promise.all([
    sceneFiles(base),
    download(base + "model.onnx", (fraction) => post({ type: "loading", fraction })),
    allowWebGpu ? hasWebGpu() : Promise.resolve(false),
  ]);
  const ort = await runtime(webgpu);
  const detector = await OnnxDetector.create(ort, model, scene, webgpu ? ["webgpu", "wasm"] : ["wasm"]);
  engine = { scene, detector, refs };
  const backend: Backend = detector.provider === "webgpu" ? "webgpu" : "wasm";
  post({ type: "ready", backend, input: scene.c.detector.input });
}

function need(): Engine {
  if (!engine) throw new Error("The pipeline is not loaded.");
  return engine;
}

function align(m: Extract<ToWorker, { type: "align" }>): void {
  const e = need();
  const a = alignToReference(m.gray, e.refs, m.workSize, e.scene.c.ref_size);
  // pipeline.py: lamp_patches(np.linalg.inv(alignment.H)), from the first frame's alignment
  const boxes = lampPatches(matInv(a.H), e.scene.c.signal_lamps);
  const analyser = new Analyser(e.scene, m.fps, m.workSize);
  analyser.setAlignment(a);
  job = { id: m.job, analyser, boxes, workSize: m.workSize, nextAlign: REALIGN_EARLY[0], keyframes: [m.gray] };
  // the crop the page reads the lamps from: the patches and a small margin, inside the frame
  const [w, h] = m.workSize;
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
  const x0 = clamp(Math.min(...boxes.map((b) => b[0])) - LAMP_MARGIN, 0, w - 1);
  const y0 = clamp(Math.min(...boxes.map((b) => b[1])) - LAMP_MARGIN, 0, h - 1);
  const x1 = clamp(Math.max(...boxes.map((b) => b[2])) + LAMP_MARGIN, x0 + 1, w);
  const y1 = clamp(Math.max(...boxes.map((b) => b[3])) + LAMP_MARGIN, y0 + 1, h);
  post({ type: "aligned", job: m.job, ok: a.ok, score: a.score, reference: a.reference, H: a.H, lampCrop: [x0, y0, x1, y1] });
}

async function frame(m: Extract<ToWorker, { type: "frame" }>): Promise<void> {
  const j = job;
  if (!j || j.id !== m.job) return; // cancelled: skip what is left in the queue
  const rgba = new Uint8ClampedArray(m.det.data);
  if (m.t >= j.nextAlign) realign(j, halfGray(rgba, m.det.width, m.det.height), m.t);
  const dets = await need().detector.detect(rgba, m.det.width, m.det.height, 4, j.workSize[0]);
  if (job !== j) return;
  const scores = m.lamp
    ? (lampScores(new Uint8ClampedArray(m.lamp.data), m.lamp.width, m.lamp.height, 4, j.boxes, m.lamp.origin) as [number, number, number])
    : null;
  j.analyser.push(m.t, dets, scores);
  post({ type: "frame-done", job: m.job, index: m.index, detections: dets.length });
  if (m.live) {
    const groups = need().scene.c.tracking;
    const boxes = (j.analyser.lastFrame?.boxes ?? []).map(
      ([x1, y1, x2, y2, id, cls]): LiveBox => ({ id, cls, group: groupOf(id, groups), box: [x1, y1, x2, y2] }),
    );
    post({ type: "live", job: m.job, index: m.index, t: m.t, boxes, risk: j.analyser.lastRisk ?? 0 });
  }
}

/** A keyframe (pipeline.py analyse()): register it and, if that works, read the lamps where it puts them. */
function realign(j: RunningJob, gray: Uint8Array, t: number): void {
  const e = need();
  const a = alignToReference(gray, e.refs, j.workSize, e.scene.c.ref_size);
  if (a.ok) j.boxes = lampPatches(matInv(a.H), e.scene.c.signal_lamps);
  if (gray.length === j.keyframes[0].length) j.keyframes.push(gray);
  j.nextAlign = t + (t < REALIGN_EARLY[1] ? REALIGN_EARLY[0] : REALIGN_EVERY);
}

function finish(m: Extract<ToWorker, { type: "finish" }>): void {
  const j = job;
  if (!j || j.id !== m.job) return;
  if (j.keyframes.length > 2) {
    const e = need();
    const settled = alignToReference(medianGray(j.keyframes), e.refs, j.workSize, e.scene.c.ref_size);
    if (settled.ok) j.analyser.settle(settled);
  }
  const result = j.analyser.finish(m.duration, m.clip);
  job = null;
  post({ type: "result", job: m.job, result, groups: Object.keys(need().scene.c.tracking.GROUPS) });
}

async function handle(m: ToWorker): Promise<void> {
  switch (m.type) {
    case "init":
      return init(m.base, m.refs, m.allowWebGpu);
    case "align":
      return align(m);
    case "frame":
      return frame(m);
    case "finish":
      return finish(m);
  }
}

// One message at a time, in order: frames must reach the Analyser in time order, and one ONNX
// session runs one inference at a time. Cancel jumps the queue so that queued frames are skipped.
let queue: Promise<void> = Promise.resolve();
scope.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "cancel") {
    if (job?.id === m.job) job = null;
    return;
  }
  queue = queue.then(async () => {
    try {
      await handle(m);
    } catch (err) {
      const id = m.type === "init" ? null : m.job;
      if (id !== null && job?.id === id) job = null;
      post({ type: "error", job: id, message: err instanceof Error ? err.message : String(err) });
    }
  });
};
