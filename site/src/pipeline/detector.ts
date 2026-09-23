// The browser detector: public/pipeline/model.onnx, a YOLO26 export with the end-to-end head (no NMS).
// Ports what Ultralytics does around the model when detector.py calls YOLO(...).predict on an ONNX file:
// LetterBox(auto=False, center=True) + /255 (engine/predictor.py preprocess), the end2end branch of
// utils/nms.non_max_suppression (score > conf, class filter) and ops.scale_boxes with clipping
// (models/yolo/detect/predict.py). Works with onnxruntime-web and onnxruntime-node: the caller passes
// the module in.

import { roundHalfEven } from "./geometry.ts";
import type { Scene } from "./scene.ts";
import type { Detection } from "./types.ts";

export type Pixels = Uint8Array | Uint8ClampedArray;

/** The model input and where the frame sits in it (padding in input pixels, gain = input px per frame px). */
export interface Letterboxed {
  data: Float32Array;
  padTop: number;
  padLeft: number;
  gain: number;
}

/** The part of the onnxruntime API used here; onnxruntime-web and onnxruntime-node both provide it. */
export interface OrtTensor {
  readonly data: unknown;
  readonly dims: readonly number[];
  dispose?(): void;
}
export interface OrtSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}
export interface OrtModule {
  InferenceSession: { create(model: Uint8Array, options?: { executionProviders?: string[]; logSeverityLevel?: 0 | 1 | 2 | 3 | 4 }): Promise<OrtSession> };
  Tensor: new (type: "float32", data: Float32Array, dims: readonly number[]) => OrtTensor;
}

const PAD = 114;
// uint8 -> float32 / 255, as torch's im.float().div_(255) (checked bit for bit against torch).
const LUT = Float32Array.from({ length: 256 }, (_, v) => v / 255);

/**
 * LetterBox(new_shape=input, auto=False, center=True, scaleup=True) followed by HWC uint8 -> CHW float 0..1.
 * `pixels` is RGB or RGBA (alpha ignored), row-major, `width` x `height`; `input` is [h, w].
 * The data is written into `out` when given (length 3 * h * w), so a caller can reuse one buffer.
 */
export function letterbox(
  pixels: Pixels,
  width: number,
  height: number,
  channels: 3 | 4,
  input: [number, number],
  out?: Float32Array,
): Letterboxed {
  const [ih, iw] = input;
  const r = Math.min(ih / height, iw / width);
  // Python's round() on these (half to even); dh - 0.1 and dw - 0.1 are never halves.
  const nw = roundHalfEven(width * r);
  const nh = roundHalfEven(height * r);
  const padTop = roundHalfEven((ih - nh) / 2 - 0.1) || 0;
  const padLeft = roundHalfEven((iw - nw) / 2 - 0.1) || 0;

  let src: Pixels = pixels;
  let c: number = channels;
  if (nw !== width || nh !== height) {
    src = resizeLinear(pixels, width, height, channels, nw, nh);
    c = 3;
  }

  const plane = ih * iw;
  const data = out ?? new Float32Array(3 * plane);
  if (data.length !== 3 * plane) throw new Error(`letterbox: buffer holds ${data.length} values, need ${3 * plane}`);
  data.fill(LUT[PAD]);
  const g = plane, b = 2 * plane;
  for (let y = 0; y < nh; y++) {
    let s = y * nw * c;
    let d = (y + padTop) * iw + padLeft;
    for (let x = 0; x < nw; x++, s += c, d++) {
      data[d] = LUT[src[s]];
      data[g + d] = LUT[src[s + 1]];
      data[b + d] = LUT[src[s + 2]];
    }
  }
  return { data, padTop, padLeft, gain: r };
}

/**
 * cv2.resize(INTER_LINEAR) of 8-bit pixels to RGB: cv2's sample positions, edge clamping, 11-bit fixed-point
 * weights and the rounding of its vectorised vertical pass. Only used when the frame is not already at the
 * letterbox scale (the browser draws frames at 960 wide, so a 16:9 video never gets here). Against
 * Ultralytics' LetterBox on arm64 over 99.9% of values are exact when upscaling; OpenCV's platform HALs
 * (carotene there) round some downscales differently, by at most 1 grey level.
 */
function resizeLinear(src: Pixels, w: number, h: number, c: number, nw: number, nh: number): Uint8Array {
  const axis = (n: number, m: number) => {
    const i0 = new Int32Array(m), i1 = new Int32Array(m), a0 = new Int32Array(m), a1 = new Int32Array(m);
    const scale = n / m;
    for (let d = 0; d < m; d++) {
      let f = Math.fround((d + 0.5) * scale - 0.5);
      let i = Math.floor(f);
      f -= i;
      if (i < 0) {
        i = 0;
        f = 0;
      }
      if (i >= n - 1) {
        i = n - 1;
        f = 0;
      }
      i0[d] = i;
      i1[d] = Math.min(i + 1, n - 1);
      a0[d] = roundHalfEven((1 - f) * 2048);
      a1[d] = roundHalfEven(f * 2048);
    }
    return { i0, i1, a0, a1 };
  };
  const X = axis(w, nw), Y = axis(h, nh);
  const dst = new Uint8Array(nw * nh * 3);
  for (let y = 0; y < nh; y++) {
    const r0 = Y.i0[y] * w * c, r1 = Y.i1[y] * w * c, b0 = Y.a0[y], b1 = Y.a1[y];
    for (let x = 0; x < nw; x++) {
      const c0 = X.i0[x] * c, c1 = X.i1[x] * c, a0 = X.a0[x], a1 = X.a1[x];
      for (let k = 0; k < 3; k++) {
        const s0 = src[r0 + c0 + k] * a0 + src[r0 + c1 + k] * a1;
        const s1 = src[r1 + c0 + k] * a0 + src[r1 + c1 + k] * a1;
        // VResizeLinearVec_32s8u: v_mul_hi on the 16-bit (S >> 4) and beta, then (sum + 2) >> 2
        dst[(y * nw + x) * 3 + k] = ((((s0 >> 4) * b0) >> 16) + (((s1 >> 4) * b1) >> 16) + 2) >> 2;
      }
    }
  }
  return dst;
}

/**
 * The model's [N, 6] rows (x1, y1, x2, y2, score, class in input pixels) -> detections in work pixels.
 * Keeps rows with score > conf and, when `keep` is given, a class in it (nms.non_max_suppression, end2end);
 * maps boxes back to the imgW x imgH frame and clips them to it (ops.scale_boxes), then multiplies by
 * workScale (work width / frame width). Rows keep the model's order (highest score first).
 */
export function parseOutput(
  out: Float32Array,
  lb: Pick<Letterboxed, "padTop" | "padLeft" | "gain">,
  imgW: number,
  imgH: number,
  conf: number,
  keep: readonly number[] | null,
  workScale: number,
): Detection[] {
  const classes = keep ? new Set(keep) : null;
  // torch compares the float32 scores with conf cast to float32: a score of exactly float32(0.1) is dropped
  const thr = Math.fround(conf);
  const { padTop, padLeft, gain } = lb;
  const clip = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  const dets: Detection[] = [];
  for (let i = 0; i + 6 <= out.length; i += 6) {
    const score = out[i + 4], cls = out[i + 5];
    if (!(score > thr) || (classes && !classes.has(cls))) continue;
    dets.push({
      x1: clip((out[i] - padLeft) / gain, imgW) * workScale,
      y1: clip((out[i + 1] - padTop) / gain, imgH) * workScale,
      x2: clip((out[i + 2] - padLeft) / gain, imgW) * workScale,
      y2: clip((out[i + 3] - padTop) / gain, imgH) * workScale,
      conf: score,
      cls,
    });
  }
  return dets;
}

/** One ONNX session plus the scene's detector constants (scene.json `detector`). */
export class OnnxDetector {
  /** The execution provider the session was created with (the first of `providers` that worked). */
  readonly provider: string;
  /** Model input [h, w]. */
  readonly input: [number, number];
  private ort: OrtModule;
  private session: OrtSession;
  private conf: number;
  private keep: number[];
  private buf: Float32Array;
  private queue: Promise<unknown> = Promise.resolve();

  private constructor(ort: OrtModule, session: OrtSession, provider: string, scene: Scene) {
    this.ort = ort;
    this.session = session;
    this.provider = provider;
    this.input = scene.c.detector.input;
    this.conf = scene.c.detector.conf;
    this.keep = scene.c.detector.keep_classes;
    this.buf = new Float32Array(3 * this.input[0] * this.input[1]);
  }

  /** Tries the providers in order (e.g. ["webgpu", "wasm"] or ["cpu"]) and keeps the first that loads. */
  static async create(
    ort: OrtModule,
    modelBytes: ArrayBuffer | Uint8Array,
    scene: Scene,
    providers: string[],
  ): Promise<OnnxDetector> {
    if (!providers.length) throw new Error("OnnxDetector.create: no execution provider given");
    const bytes = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
    let error: unknown;
    for (let i = 0; i < providers.length; i++) {
      try {
        // A copy for every attempt but the last: onnxruntime-web's proxy worker takes (detaches) the buffer.
        const model = i < providers.length - 1 ? bytes.slice() : bytes;
        // 3 = errors only: ORT otherwise warns that it keeps shape ops on the CPU, which is intended
        const session = await ort.InferenceSession.create(model, { executionProviders: [providers[i]], logSeverityLevel: 3 });
        return new OnnxDetector(ort, session, providers[i], scene);
      } catch (e) {
        error = e;
      }
    }
    throw error;
  }

  /** Runs the model on a letterboxed input and returns its raw [N * 6] output. */
  infer(lb: Letterboxed): Promise<Float32Array> {
    return this.enqueue(() => this.run(lb));
  }

  /** Detections in work pixels (frame scaled to workWidth), with the scene's classes and confidence. */
  detect(pixels: Pixels, width: number, height: number, channels: 3 | 4, workWidth = 1920): Promise<Detection[]> {
    return this.enqueue(async () => {
      const lb = letterbox(pixels, width, height, channels, this.input, this.buf);
      const out = await this.run(lb);
      return parseOutput(out, lb, width, height, this.conf, this.keep, workWidth / width);
    });
  }

  async release(): Promise<void> {
    await this.session.release?.();
  }

  // One session run at a time (a session cannot run concurrently), which also guards the shared input buffer.
  private enqueue<T>(job: () => Promise<T>): Promise<T> {
    const p = this.queue.then(job, job);
    this.queue = p.catch(() => undefined);
    return p;
  }

  private async run(lb: Letterboxed): Promise<Float32Array> {
    const [h, w] = this.input;
    const feeds = { [this.session.inputNames[0]]: new this.ort.Tensor("float32", lb.data, [1, 3, h, w]) };
    const res = await this.session.run(feeds);
    const t = res[this.session.outputNames[0]];
    if (t.dims[t.dims.length - 1] !== 6) throw new Error(`detector output is [${t.dims}], expected [1, N, 6]`);
    const data = Float32Array.from(t.data as Float32Array);
    t.dispose?.();
    return data;
  }
}
