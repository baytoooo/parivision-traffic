// Messages between the demo page (src/scripts/local_api.ts) and the pipeline's Web Worker
// (worker.ts). One job runs at a time; every job message carries its id so that late replies
// of a cancelled job can be told apart. Pixel buffers are transferred, not copied.

import type { PipelineResult } from "./types.ts";

/** Where the detector runs: WebGPU, or WebAssembly on the CPU. */
export type Backend = "webgpu" | "wasm";

/** RGBA pixels, row-major, as a canvas's getImageData gives them. */
export interface Pixels {
  data: ArrayBuffer;
  width: number;
  height: number;
}

export type ToWorker =
  /** Load scene.json, scene.bin.gz and model.onnx from `base` (the site's /pipeline/ URL) and
   * create the ONNX session. `refs` are the reference views as grey bytes at 1/4 size; WebGPU is
   * tried only when `allowWebGpu` (the page can force the WebAssembly path). */
  | { type: "init"; base: string; refs: Record<string, Uint8Array>; allowWebGpu: boolean }
  /** Start a job: align its first frame (grey, work size / 4) and set up an Analyser. */
  | { type: "align"; job: string; gray: Uint8Array; workSize: [number, number]; fps: number }
  /** One analysed frame: the whole frame for the detector (960 px wide) and the lamp crop at work
   * scale, whose top-left corner is at `origin` in work pixels. */
  | { type: "frame"; job: string; index: number; t: number; det: Pixels; lamp: (Pixels & { origin: [number, number] }) | null }
  /** No more frames: build trajectories, apply the rules, send the result. */
  | { type: "finish"; job: string; duration: number; clip: string }
  /** Drop the job; frames of it still in the queue are skipped. */
  | { type: "cancel"; job: string };

export type FromWorker =
  /** Downloads during init: `fraction` of the model's bytes received so far. */
  | { type: "loading"; fraction: number }
  /** The session is up; `input` is the detector's input [h, w] (scene.json detector.input). */
  | { type: "ready"; backend: Backend; input: [number, number] }
  /** The alignment of a job's first frame and the work-pixel rectangle [x0, y0, x1, y1) that holds
   * the three lamp patches, which is what the page crops for the signal reader. */
  | { type: "aligned"; job: string; ok: boolean; score: number; reference: string; H: number[]; lampCrop: [number, number, number, number] }
  | { type: "frame-done"; job: string; index: number; detections: number }
  /** `groups[i]` names the object group of track ids in [(i + 1) * 1e6, (i + 2) * 1e6). */
  | { type: "result"; job: string; result: PipelineResult; groups: string[] }
  /** `job` is null for a failed init. */
  | { type: "error"; job: string | null; message: string };
