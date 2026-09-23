// Shared types of the in-browser pipeline. Pixel conventions are in README.md:
// "work" = frame scaled to 1920 px wide, "reference" = the 1920x1080 reference view.

export type Group = "vehicle" | "person" | "bicycle" | "animal";

/** One detector box in work pixels. */
export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
}

/** Tracker output row, as Python's MultiTracker.update: [x1, y1, x2, y2, track_id, score, cls] in work pixels. */
export type TrackRow = [number, number, number, number, number, number, number];

/** A tracked road user (trajectories.py Trajectory). All arrays have one entry per sample. */
export interface Trajectory {
  tid: number;
  group: Group;
  /** Majority COCO class over the track. */
  cls: number;
  t: number[];
  /** [x1, y1, x2, y2] in work pixels. */
  box: number[][];
  /** [x, y] in reference pixels, smoothed. */
  foot: number[][];
  /** [vx, vy] in reference pixels per second. */
  vel: number[][];
  /** Box height in reference pixels. */
  height: number[];
  conf: number[];
}

/** One actor-interval found by a rule (rules.py Evidence). */
export interface Evidence {
  label: string;
  start: number;
  end: number;
  actors: number[];
  note: string;
}

/** [start_sec, end_sec, label] */
export type Seg = [number, number, string];

/** Boxes drawn over the video while it plays: [x1, y1, x2, y2, track_id, cls] in work pixels. */
export interface OverlayFrame {
  t: number;
  boxes: [number, number, number, number, number, number][];
}

/** What the pipeline hands to the page; a superset of the site's ClipResult. */
export interface PipelineResult {
  clip: string;
  duration: number;
  events: Seg[];
  evidence: Evidence[];
  /** Signal phase as segments [start, end, phase]. */
  signal: Seg[];
  /** [t_sec, score] per analysed frame. */
  risk: [number, number][];
  /** Mean detections per frame per 5 s bin, by class name, plus "t". */
  counts: Record<string, number[]>;
  aligned: boolean;
  overlay: { work: [number, number]; frames: OverlayFrame[] };
  /** Frames per second the clip was analysed at. */
  fps: number;
}
