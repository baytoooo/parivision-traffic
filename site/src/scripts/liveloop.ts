// The parts of the live page's loop (live.ts) that need no browser, so that the Node tests can
// import them: frame pacing, the processing rate, the rolling risk window, the counts in view and
// the sizes frames are drawn at for the pipeline.

import { roundHalfEven } from "../pipeline/geometry.ts";
import type { LiveBox } from "../pipeline/messages.ts";

/** Least time between two frames sent to the worker, in ms: 5 frames per second at most, the rate
 * the demo analyses clips at. */
export const MIN_FRAME_MS = 200;

/**
 * When the next frame may go to the worker. One frame at a time: the next one leaves only after
 * the reply to the last, and no sooner than `gap` ms after the last one left. A slow device just
 * runs at a lower rate, with nothing queued in the worker.
 */
export class Pacer {
  readonly gap: number;
  private sentAt: number | null = null;
  private out = false;

  constructor(gap = MIN_FRAME_MS) {
    this.gap = gap;
  }

  /** Milliseconds to wait at `now` before the next frame may leave; Infinity while one is out. */
  wait(now: number): number {
    if (this.out) return Infinity;
    return this.sentAt === null ? 0 : Math.max(0, this.sentAt + this.gap - now);
  }

  /** A frame left at `now`. */
  sent(now: number): void {
    if (this.out) throw new Error("Pacer: the last frame has no reply yet");
    this.out = true;
    this.sentAt = now;
  }

  /** The reply to the frame that is out came back. */
  replied(): void {
    this.out = false;
  }
}

/** Frames per second over the frames analysed in the last `span` ms. */
export class RateMeter {
  readonly span: number;
  private times: number[] = [];

  constructor(span = 3000) {
    this.span = span;
  }

  tick(now: number): void {
    this.times.push(now);
    this.drop(now);
  }

  /** The rate at `now`; null while fewer than two frames fall in the window. */
  rate(now: number): number | null {
    this.drop(now);
    const n = this.times.length;
    const dt = n > 1 ? (this.times[n - 1] - this.times[0]) / 1000 : 0;
    return dt > 0 ? (n - 1) / dt : null;
  }

  private drop(now: number): void {
    while (this.times.length && this.times[0] < now - this.span) this.times.shift();
  }
}

/** The [t, value] points of the last `span` seconds up to the newest one: the live risk line. */
export class Rolling {
  readonly span: number;
  readonly points: [number, number][] = [];

  constructor(span = 60) {
    this.span = span;
  }

  push(t: number, v: number): void {
    this.points.push([t, v]);
    while (this.points[0][0] < t - this.span) this.points.shift();
  }

  /** Where the window starts: `span` s before the newest point, and not before 0. */
  get start(): number {
    const last = this.points.at(-1);
    return last ? Math.max(0, last[0] - this.span) : 0;
  }
}

/** COCO names of the classes the detector keeps (scene.json detector.keep_classes). */
export const CLASS_NAMES: Record<number, string> = {
  0: "person",
  1: "bicycle",
  2: "car",
  3: "motorcycle",
  5: "bus",
  7: "truck",
  14: "bird",
  15: "cat",
  16: "dog",
  17: "horse",
  18: "sheep",
  19: "cow",
};

/** The classes the page counts, in the order it shows them (the demo's per-class counts). */
export const COUNTED = ["car", "bus", "truck", "motorcycle", "person", "bicycle"] as const;
export type Counted = (typeof COUNTED)[number];

/** Tracked objects in a frame, per counted class. */
export function countInView(boxes: Pick<LiveBox, "cls">[]): Record<Counted, number> {
  const out = Object.fromEntries(COUNTED.map((name) => [name, 0])) as Record<Counted, number>;
  for (const b of boxes) {
    const name = CLASS_NAMES[b.cls] as Counted | undefined;
    if (name && name in out) out[name]++;
  }
  return out;
}

/**
 * The width a `w` x `h` frame is drawn at for the detector, whose input is `input` [h, w]: the input
 * width for 16:9 and wider frames, as the demo draws clips, and narrower for taller ones (a 4:3
 * camera, a phone held upright) so that the frame fits the input height and the letterbox needs no
 * resize in JavaScript.
 */
export function detectorWidth(w: number, h: number, input: [number, number]): number {
  return Math.max(1, Math.min(input[1], Math.floor((input[0] * w) / h)));
}

/** What a picture's size decides for the pipeline: its work frame [w, h] and the frame the
 * detector gets [w, h] (FrameReader's canvases). */
export interface FrameShape {
  work: [number, number];
  det: [number, number];
}

/** The FrameShape of a `w` x `h` picture, for a detector whose input is `input` [h, w] and a work
 * frame `workWidth` px wide. It depends on the picture's shape only, not on its size. */
export function frameShape(w: number, h: number, input: [number, number], workWidth: number): FrameShape {
  const dw = detectorWidth(w, h, input);
  return { work: [workWidth, roundHalfEven((h * workWidth) / w)], det: [dw, roundHalfEven((dw * h) / w)] };
}

/** Whether frames of two shapes go to the pipeline alike: a camera that only changes resolution
 * keeps its shape, and a run can go on over it. */
export function sameShape(a: FrameShape, b: FrameShape): boolean {
  return a.work[0] === b.work[0] && a.work[1] === b.work[1] && a.det[0] === b.det[0] && a.det[1] === b.det[1];
}
