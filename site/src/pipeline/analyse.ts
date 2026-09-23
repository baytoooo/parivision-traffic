// Part A end to end on frames pushed one at a time: the loop of pipeline.py analyse() (raw
// signal phase per frame from the lamp scores, MultiTracker -> collect, then build, fill_phases,
// Context and detect_from_context), plus what the demo server adds around it in demo/worker.py:
// the causal risk curve from an Anticipator fed the same detections (on_frame), detections per
// class per 5 s bin, and the signal as segments. It also keeps the tracker boxes of every frame
// so the page can draw them over the video. No DOM: worker.ts owns the detector and the frames.

import type { AlignResult } from "./align.ts";
import { diag } from "./geometry.ts";
import { Anticipator } from "./risk.ts";
import { Context, detectFromContext } from "./rules.ts";
import type { Scene, SceneConstants } from "./scene.ts";
import { fillPhases, phaseFromScores } from "./signal.ts";
import { collect, MultiTracker, type Track } from "./tracker.ts";
import { build, COCO, type Classes } from "./trajectories.ts";
import type { Detection, Evidence, OverlayFrame, PipelineResult, Seg } from "./types.ts";

const f32 = Math.fround;

// demo/worker.py on_frame: detections at this confidence or more are counted, per BIN_SEC bin
const COUNT_CONF = 0.35;
const BIN_SEC = 5;

/** Python's round(x, n), except that exact binary ties round away from zero. */
function round(x: number, n: number): number {
  return Number(x.toFixed(n));
}

export class Analyser {
  readonly fps: number;
  /** Work frame [w, h]: the clip scaled to 1920 px wide. */
  readonly workSize: [number, number];
  private scene: Scene;
  private tracker: MultiTracker;
  private tracks = new Map<number, Track>();
  private risk: Anticipator;
  private alignment: AlignResult | null = null;
  private sigT: number[] = [];
  private sigRaw: string[] = [];
  private riskCurve: [number, number][] = [];
  private frames: OverlayFrame[] = [];
  private perBin = new Map<number, Map<string, number>>();
  private framesInBin = new Map<number, number>();
  /** worker.py's names for the counted classes (detector.COCO_NAMES), by class id. */
  private counted: Map<number, string>;

  constructor(scene: Scene, fps: number, workSize: [number, number]) {
    this.scene = scene;
    this.fps = fps;
    this.workSize = workSize;
    const tracking = scene.c.tracking;
    this.tracker = new MultiTracker(fps, tracking.buffer_sec, tracking);
    // worker.py: Anticipator({"fps": 5.0, "width": w, "height": h}), the analysis rate and the work frame
    this.risk = new Anticipator({ fps, width: workSize[0], height: workSize[1] }, scene);
    const k = (scene.c as SceneConstants & { trajectories?: Classes }).trajectories ?? COCO;
    this.counted = new Map([
      [k.CAR, "car"],
      [k.BUS, "bus"],
      [k.TRUCK, "truck"],
      [k.MOTORCYCLE, "motorcycle"],
      [k.PERSON, "person"],
      [tracking.BICYCLE as number, "bicycle"],
    ]);
  }

  /** The first frame's registration (pipeline.py: align_best on the first sampled frame). */
  setAlignment(a: AlignResult): void {
    this.alignment = a;
    this.risk.setAlignment(a.H);
  }

  /**
   * One analysed frame, in time order: its detections in work pixels and the lamp scores read
   * with the lamp patches of the alignment (null when the lamps were not read: the phase is unknown).
   */
  push(t: number, dets: Detection[], lampScores: [number, number, number] | null): void {
    if (!this.alignment) throw new Error("Analyser: setAlignment() before push()");
    const shape: [number, number] = [this.workSize[1], this.workSize[0]];
    this.sigT.push(t);
    this.sigRaw.push(lampScores ? phaseFromScores(lampScores, this.scene.c.signal_min_contrast) : "unknown");

    const rows = this.tracker.update(dets, shape);
    collect(this.tracks, t, rows, this.scene.c.tracking);
    const r1 = (v: number) => round(v, 1);
    this.frames.push({ t, boxes: rows.map(([x1, y1, x2, y2, id, , cls]) => [r1(x1), r1(y1), r1(x2), r1(y2), Math.trunc(id), Math.trunc(cls)]) });

    // worker.py on_frame: the risk model sees the same detections in time order
    this.riskCurve.push([round(t, 2), round(this.risk.observe(dets, shape, t), 4)]);
    const b = Math.floor(t / BIN_SEC);
    this.framesInBin.set(b, (this.framesInBin.get(b) ?? 0) + 1);
    let row = this.perBin.get(b);
    if (!row) this.perBin.set(b, (row = new Map()));
    for (const d of dets) {
      const name = this.counted.get(d.cls);
      // det.conf is float32 and NumPy compares it with float32(0.35)
      if (name && f32(d.conf) >= f32(COUNT_CONF)) row.set(name, (row.get(name) ?? 0) + 1);
    }
  }

  /** Frames pushed so far. */
  get frameCount(): number {
    return this.sigT.length;
  }

  /** Events and everything the page draws. `duration` is the analysed length (pipeline.py `limit`). */
  finish(duration: number, clip: string): PipelineResult {
    const [w, h] = this.workSize;
    const ref = this.scene.c.ref_size;
    // an unreadable clip gets a plain rescale, as in analyse()
    const H = this.alignment?.H ?? diag(ref[0] / w, ref[1] / h);
    const trajectories = build(this.tracks.values(), H);
    const phases = fillPhases(this.sigRaw, this.sigT);
    const ctx = new Context(trajectories, this.sigT, phases, duration, this.scene);
    const { events, evidence } = detectFromContext(ctx, H);

    // worker.py: runs of one phase, the ends rounded to 0.1 s
    const signal: Seg[] = [];
    this.sigT.forEach((t, i) => {
      const last = signal[signal.length - 1];
      if (last && last[2] === phases[i]) last[1] = round(t, 1);
      else signal.push([round(t, 1), round(t, 1), phases[i]]);
    });

    const bins = [...this.perBin.keys()].sort((a, b) => a - b);
    const counts: Record<string, number[]> = { t: bins.map((b) => b * BIN_SEC) };
    for (const name of ["car", "bus", "truck", "motorcycle", "person", "bicycle"])
      counts[name] = bins.map((b) => round((this.perBin.get(b)!.get(name) ?? 0) / this.framesInBin.get(b)!, 2));

    return {
      clip,
      duration: round(duration, 2),
      events,
      evidence: evidence.map((e): Evidence => ({ label: e.label, start: round(e.start, 2), end: round(e.end, 2), actors: e.actors, note: e.note })),
      signal,
      risk: this.riskCurve,
      counts,
      aligned: !!this.alignment?.ok,
      overlay: { work: this.workSize, frames: this.frames },
      fps: this.fps,
    };
  }
}
