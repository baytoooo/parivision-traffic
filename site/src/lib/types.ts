// Shapes of the files in public/data. See BRIEF.md, "Data contract".

/** [start_sec, end_sec, label] */
export type Seg = [number, number, string];

/** A number from evaluate.py, or "TBD" (or null) while it is not known yet. */
export type Num = number | string | null | undefined;

export interface Clip {
  id: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  local_time: string;
  light: string;
  video: string;
  poster: string;
}

export interface Evidence {
  label: string;
  start: number;
  end: number;
  actors: number[];
  note: string;
  /** Optional. Where on the junction it happened; see ZONES in classes.ts. */
  zone?: string;
}

export interface Counts {
  t: number[];
  [objectClass: string]: number[];
}

export interface ClipResult {
  clip: string;
  duration: number;
  events: Seg[];
  labels?: Seg[];
  /** [t_sec, score] at 10 Hz */
  risk: [number, number][];
  signal?: Seg[];
  evidence?: Evidence[];
  counts?: Counts;
  /** Only in demo job results: URL of the annotated video, relative to the API. */
  video?: string;
}

export interface PRF {
  tp: Num;
  fp: Num;
  fn: Num;
  precision: Num;
  recall: Num;
  f1: Num;
}

export interface PerClass {
  "0.3": PRF;
  "0.5": PRF;
  "0.7": PRF;
  f1_mean: Num;
}

export interface Metrics {
  model_score: Num;
  part_a: {
    score_a: Num;
    classes?: string[];
    per_class: Record<string, PerClass>;
  };
  part_b: null | {
    score_b: Num;
    ap: Num;
    f1_alarm: Num;
    mtta_sec: Num;
    [k: string]: unknown;
  };
}

export interface Ablation {
  name: string;
  score_a: Num;
  /** Detector compute relative to the submitted configuration (input pixels x frames). */
  cost_x: Num;
  note: string;
  per_class?: Record<string, number>;
}

/** Wall-clock times from run_submission.py's log, per clip. */
export interface Runtime {
  machine: string;
  note: string;
  clips: { clip: string; duration: number; part_a_sec: number; part_b_sec: number; total_sec: number }[];
}

export interface Eda {
  clips: string[];
  density: Record<string, { t: number[]; vehicles: number[]; people: number[] }>;
  speeds: { sb: number[]; nb: number[] };
  signal_cycle: { green: Num; red: Num; cycle: Num };
  images: { heatmap_vehicle: string; heatmap_person: string; trajectories: string; directions: string };
  findings: string[];
}

export interface Example {
  label: string;
  clip: string;
  t: number;
  thumb: string;
  caption: string;
}

export interface Failure {
  clip: string;
  t: number;
  label: string;
  kind: string;
  caption: string;
}

export interface Member {
  name: string;
  role: string;
  did: string[];
  github: string;
  linkedin: string;
  portfolio: string;
  projects: { name: string; url: string }[];
}

export interface Team {
  team: string;
  university: string;
  members: Member[];
}

/** GET /api/samples */
export interface Sample {
  name: string;
  label: string;
  seconds: number;
}

/** GET /api/jobs/<id> */
export interface Job {
  status: "queued" | "running" | "done" | "error";
  progress: number;
  stage: string;
  eta_sec: number | null;
  error: string | null;
  result: ClipResult | null;
}
