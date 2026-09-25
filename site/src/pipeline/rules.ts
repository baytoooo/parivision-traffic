// Event rules on trajectories + scene layout (rules.py) and the final event list (events.py).
//
// Every rule returns Evidence (one per actor and interval); detectFromContext unions them per
// class into the final segments. The OpenCV rasters of rules.Context (masks, distance transforms,
// zones, failure_to_yield's dilated zebras) are read from the Scene, the thresholds (PARAMS) from
// scene.json `rules` and the class list, gaps and minimum lengths from scene.json `events`.
// Distances are in reference pixels. Boxes are float32 in Python, so box arithmetic goes through
// Math.fround as NumPy does it.

import { median, percentile, roundHalfEven, signedSide, warpPoints } from "./geometry.ts";
import { metresPerPx } from "./risk.ts";
import type { Mat3 } from "./geometry.ts";
import type { DistName, Scene, SceneConstants } from "./scene.ts";
import { finalize, runs } from "./segments.ts";
import { COCO, isPerson, isVehicle, speed } from "./trajectories.ts";
import type { Classes } from "./trajectories.ts";
import type { Evidence, Seg, Trajectory } from "./types.ts";

const f32 = Math.fround;

/** Python's a % b for floats: the result takes the sign of b. */
function pymod(a: number, b: number): number {
  const m = a % b;
  return m !== 0 && m < 0 !== b < 0 ? m + b : m;
}

const degrees = (rad: number) => rad * (180 / Math.PI);

/** Python f"{x:.1f}": like toFixed, but an exact tie rounds to even. */
function fixed1(x: number): string {
  const tie = Number.isInteger(x * 4) && !Number.isInteger(x * 2);
  return (tie ? roundHalfEven(x * 10) / 10 : x).toFixed(1);
}

/** np.searchsorted(a, v) (side "left") on a sorted array. */
function searchsorted(a: number[], v: number): number {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

type BoxBin = [number[], boolean][];

export class Context {
  readonly trajectories: Trajectory[];
  /** Sample times of the phase timeline. */
  readonly signalT: number[];
  /** "red" / "green" / "yellow" / "unknown" per sample. */
  readonly signalPhase: string[];
  readonly duration: number;
  readonly scene: Scene;
  /** rules.PARAMS */
  readonly params: Record<string, number>;
  readonly classes: Classes;
  readonly byId: Map<number, Trajectory>;
  /** Sampling step: 0.1 s at 10 fps, longer where frames were thinned. */
  readonly dt: number;
  readonly vehicles: Trajectory[];
  readonly twoWheelers: Trajectory[];
  readonly people: Trajectory[];
  private readonly boxesAt: Map<number, BoxBin>;
  private readonly speeds = new Map<Trajectory, number[]>();

  constructor(trajs: Trajectory[], signalT: number[], signalPhase: string[], duration: number, scene: Scene) {
    this.trajectories = trajs;
    this.signalT = signalT;
    this.signalPhase = signalPhase;
    this.duration = duration;
    this.scene = scene;
    this.params = scene.c.rules;
    this.classes = (scene.c as SceneConstants & { trajectories?: Classes }).trajectories ?? COCO;
    this.byId = new Map(trajs.map((tr) => [tr.tid, tr]));
    const frames = [...new Set(trajs.flatMap((tr) => tr.t))].sort((a, b) => a - b);
    this.dt = frames.length > 2 ? percentile(frames.slice(1).map((v, i) => v - frames[i]), 90) : 0.1;
    this.vehicles = trajs.filter((tr) => isVehicle(tr, this.classes));
    this.twoWheelers = trajs.filter((tr) => tr.group === "bicycle" || tr.cls === this.classes.MOTORCYCLE);
    this.boxesAt = indexBoxes([...this.vehicles, ...this.twoWheelers], this.classes);
    this.people = trajs.filter((tr) => isPerson(tr, this.classes) && !this.notAPedestrian(tr));
  }

  // -- lookups ---------------------------------------------------------
  /** Trajectory speed (px/s), computed once per trajectory. */
  speed(tr: Trajectory): number[] {
    let s = this.speeds.get(tr);
    if (!s) this.speeds.set(tr, (s = speed(tr)));
    return s;
  }

  /** ctx.sample(mask, pts): the named scene mask (road, walk, cw_<name>, cw_<name>_zone) at each point. */
  mask(name: string, pts: number[][]): number[] {
    return pts.map(([x, y]) => this.scene.mask(name, x, y));
  }

  /** ctx.sample(ctx.zones[name], pts) */
  zone(name: string, pts: number[][]): number[] {
    return pts.map(([x, y]) => this.scene.zone(name, x, y));
  }

  /** ctx.sample(ctx.<name>, pts) for road_dist, walk_dist, cw_dist. */
  dist(name: DistName, pts: number[][]): number[] {
    return pts.map(([x, y]) => this.scene.dist(name, x, y));
  }

  private signalIndex(t: number): number {
    return Math.min(Math.max(searchsorted(this.signalT, t), 0), this.signalT.length - 1);
  }

  phaseAt(t: number): string {
    if (this.signalT.length === 0) return "unknown";
    return this.signalPhase[this.signalIndex(t)];
  }

  /** A gap to bridge between samples: never shorter than 1.5 sampling steps. */
  gap(seconds: number): number {
    return Math.max(seconds, 1.5 * this.dt);
  }

  phasesAt(ts: number[]): string[] {
    return ts.map((t) => this.phaseAt(t));
  }

  /** Seconds the phase has been red at time t (0 if it is not red). */
  redSince(t: number): number {
    if (this.phaseAt(t) !== "red") return 0;
    let j = this.signalIndex(t);
    while (j > 0 && this.signalPhase[j - 1] === "red") j--;
    return t - this.signalT[j];
  }

  nextGreen(t: number): number {
    for (let i = 0; i < this.signalT.length; i++) if (this.signalT[i] > t && this.signalPhase[i] === "green") return this.signalT[i];
    return this.duration;
  }

  // -- riders and occupants --------------------------------------------
  /** Cyclists and scooter riders (person on a two-wheeler), and people seen through a car window. */
  notAPedestrian(person: Trajectory): boolean {
    const p = this.params;
    const sp = this.speed(person);
    if (median(sp) > p.rider_speed) return true;
    // the same test in the person's own scale: a moped rider far up the avenue is only ~40 px tall
    if (median(sp.map((v, i) => v / Math.max(person.height[i], 20))) > p.rider_rel_speed) return true;
    let inside = 0;
    person.t.forEach((t, i) => {
      const pb = person.box[i];
      const area = Math.max(1, f32(f32(pb[2] - pb[0]) * f32(pb[3] - pb[1])));
      for (const [b, isTwoWheeler] of this.boxesAt.get(roundHalfEven(t * 10)) ?? []) {
        const iw = Math.max(0, f32(Math.min(pb[2], b[2]) - Math.max(pb[0], b[0])));
        const ih = Math.max(0, f32(Math.min(pb[3], b[3]) - Math.max(pb[1], b[1])));
        const cover = f32(f32(iw * ih) / area);
        if (cover > f32(isTwoWheeler ? p.rider_overlap : 0.6)) {
          inside++;
          break;
        }
      }
    });
    return inside > 0.4 * person.t.length;
  }
}

/** Boxes by 0.1 s time bin, flagged when they belong to a bicycle or motorbike. */
function indexBoxes(trajectories: Trajectory[], k: Classes): Map<number, BoxBin> {
  const out = new Map<number, BoxBin>();
  for (const tr of trajectories) {
    const two = tr.group === "bicycle" || tr.cls === k.MOTORCYCLE;
    tr.t.forEach((t, i) => {
      const key = roundHalfEven(t * 10);
      let bin = out.get(key);
      if (!bin) out.set(key, (bin = []));
      bin.push([tr.box[i], two]);
    });
  }
  return out;
}

/** Bottom-centre of the box in reference pixels: the front bumper for traffic coming at the camera. */
function frontPoints(tr: Trajectory, H: Mat3): number[][] {
  return warpPoints(tr.box.map((b) => [f32(f32(b[0] + b[2]) / 2), b[3]]), H);
}

/** Five points on the lower part of each box, [N][5][x, y] in reference pixels. */
function footprintPoints(tr: Trajectory, H: Mat3): number[][][] {
  return tr.box.map((b) => {
    const h = f32(b[3] - b[1]);
    const w10 = f32(f32(0.1) * f32(b[2] - b[0]));
    const xs = [f32(b[0] + w10), f32(f32(b[0] + b[2]) / 2), f32(b[2] - w10)];
    const up = f32(b[3] - f32(f32(0.25) * h));
    return warpPoints([[xs[0], b[3]], [xs[1], b[3]], [xs[2], b[3]], [xs[0], up], [xs[2], up]], H);
  });
}

/** np.median(pts, axis=0)[axis] */
const medianAxis = (pts: number[][], axis: number) => median(pts.map((q) => q[axis]));

// ---------------------------------------------------------------------------
// pedestrians
// ---------------------------------------------------------------------------
export function jaywalking(ctx: Context): Evidence[] {
  const p = ctx.params;
  const out: Evidence[] = [];
  const [c0, cx, cy] = ctx.scene.c.person_height_px;
  for (const tr of ctx.people) {
    const walk = ctx.mask("walk", tr.foot);
    const kerb = ctx.dist("walk_dist", tr.foot);
    const cw = ctx.dist("cw_dist", tr.foot);
    const strict: boolean[] = [], loose: boolean[] = [];
    tr.foot.forEach(([x, y], i) => {
      // a box much shorter than a person standing there is cut off (legs hidden behind a car):
      // its bottom edge is not the feet, so those samples cannot put anyone on the road
      const expected = c0 + cx * x + cy * y;
      const whole = tr.height[i] >= p.jay_min_height * expected;
      const road = walk[i] > 0 && whole;
      const h = Math.max(tr.height[i], 20); // perspective: margins in units of the person's apparent height
      strict[i] = road && kerb[i] > p.jay_margin_kerb * h && cw[i] > p.jay_margin_cw * h;
      loose[i] = road && cw[i] > 2.0;
    });
    for (let [s, e] of runs(tr.t, strict, ctx.gap(p.jay_gap))) {
      if (e - s < p.jay_min_dur) continue;
      // boundaries from the enclosing loose run: the moment the foot left the kerb / zebra
      for (const [ls, le] of runs(tr.t, loose, ctx.gap(p.jay_gap))) {
        if (ls <= s && le >= e) {
          [s, e] = [ls, le];
          break;
        }
      }
      out.push({ label: "jaywalking", start: s, end: e, actors: [tr.tid], note: "" });
    }
  }
  return out;
}

export function failureToYield(ctx: Context, H: Mat3): Evidence[] {
  const p = ctx.params;
  const out: Evidence[] = [];
  for (const name of Object.keys(ctx.scene.c.crosswalks)) {
    const m = `cw_${name}`;
    const pedZone = `cw_${name}_zone`; // the zebra dilated by fty_cw_dilate px
    // who is out on this crossing (not waiting on the kerb at its end), indexed by time (0.1 s bins)
    const onCw = new Map<number, [number, number[]][]>();
    for (const ped of ctx.people) {
      const kerb = ctx.dist("road_dist", ped.foot);
      const zone = ctx.mask(pedZone, ped.foot);
      const sp = ctx.speed(ped);
      ped.t.forEach((t, i) => {
        const h = Math.max(ped.height[i], 20);
        const walking = sp[i] > p.fty_walk * h; // someone standing still beside the car's path is not being cut off
        if (!(zone[i] > 0 && kerb[i] > p.fty_kerb * h && walking)) return;
        const key = roundHalfEven(t * 10);
        let bin = onCw.get(key);
        if (!bin) onCw.set(key, (bin = []));
        bin.push([ped.tid, ped.foot[i]]);
      });
    }
    if (onCw.size === 0) continue;
    for (const veh of ctx.vehicles) {
      const vs = ctx.speed(veh);
      // scooters get walked along the zebras; only ones clearly being ridden count
      if (veh.cls === ctx.classes.MOTORCYCLE && median(vs) < p.fty_moto_speed) continue;
      const on = footprintPoints(veh, H).map((pts) => ctx.mask(m, pts).some((v) => v > 0));
      if (!on.some(Boolean)) continue;
      for (const [s_, e_] of runs(veh.t, on, ctx.gap(0.3))) {
        const sel = vs.filter((_, i) => veh.t[i] >= s_ && veh.t[i] <= e_);
        if (e_ - s_ < 0.2 || median(sel) < p.fty_min_speed) continue;
        const victims = new Set<number>();
        veh.t.forEach((t, i) => {
          if (!(t >= s_ - 0.3 && t <= e_)) return;
          for (const [tid, f] of onCw.get(roundHalfEven(t * 10)) ?? []) {
            const dx = f[0] - veh.foot[i][0], dy = f[1] - veh.foot[i][1];
            if (Math.sqrt(dx * dx + dy * dy) < p.fty_near_px) victims.add(tid);
          }
        });
        if (victims.size) {
          // the convention runs from the front entering the zebra to the rear leaving it
          out.push({
            label: "failure_to_yield",
            start: s_ - p.fty_pad_start,
            end: e_ + p.fty_pad_end,
            actors: [veh.tid, ...[...victims].sort((a, b) => a - b)],
            note: name,
          });
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// signal-related
// ---------------------------------------------------------------------------
/** [trajectory, crossing time] for SB vehicles crossing the stop line towards the camera. */
function stopLineCrossings(ctx: Context, H: Mat3): [Trajectory, number][] {
  const line = ctx.scene.c.stop_line_sb;
  const [[x1], [x2]] = line;
  const out: [Trajectory, number][] = [];
  for (const veh of ctx.vehicles) {
    const front = frontPoints(veh, H);
    const side = signedSide(front, line);
    for (let i = 1; i < side.length; i++) {
      if (side[i - 1] < 0 && 0 <= side[i] && x1 - 20 <= front[i][0] && front[i][0] <= x2 + 20 && veh.vel[i][1] > 0) {
        const a = side[i - 1] / (side[i - 1] - side[i]);
        out.push([veh, veh.t[i - 1] + a * (veh.t[i] - veh.t[i - 1])]);
        break;
      }
    }
  }
  return out;
}

export function redLight(ctx: Context, H: Mat3): Evidence[] {
  const p = ctx.params;
  const out: Evidence[] = [];
  for (const [veh, tc] of stopLineCrossings(ctx, H)) {
    if (ctx.redSince(tc) < p.red_settle || ctx.nextGreen(tc) - tc < p.red_before_green) continue;
    // end: the car leaves the junction or the frame (its track ends), even if it waits inside first
    const end = Math.min(veh.t[veh.t.length - 1], tc + p.red_max_len);
    out.push({ label: "red_light", start: tc, end, actors: [veh.tid], note: `red for ${fixed1(ctx.redSince(tc))}s` });
  }
  return out;
}

export function stopLine(ctx: Context, H: Mat3): Evidence[] {
  const p = ctx.params;
  const out: Evidence[] = [];
  const [[x1, y1], [x2, y2]] = ctx.scene.c.stop_line_sb;
  // Python writes np.hypot(645, 70): the length of the stop line
  const lineLength = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  for (const veh of ctx.vehicles) {
    const front = frontPoints(veh, H);
    const side = signedSide(front, ctx.scene.c.stop_line_sb);
    const inZone = ctx.zone("stop", front);
    const sp = ctx.speed(veh);
    const phases = ctx.phasesAt(veh.t);
    const flags = veh.t.map(
      (_, i) => side[i] > p.stop_line_margin * lineLength && inZone[i] > 0 && sp[i] < p.stop_speed && phases[i] === "red",
    );
    for (const [s, e] of runs(veh.t, flags, ctx.gap(0.4))) {
      if (e - s < p.stop_line_min) continue;
      const end = Math.min(ctx.nextGreen(s), veh.t[veh.t.length - 1]);
      out.push({ label: "stop_line", start: s, end, actors: [veh.tid], note: "" });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// stopped vehicles
// ---------------------------------------------------------------------------
/** A vehicle standing 10 s or more where traffic is supposed to flow (NB lanes, junction box), not in a signal queue. */
export function stoppedVehicle(ctx: Context): Evidence[] {
  const p = ctx.params;
  const spans: [number, number, number, number, number][] = []; // (start, end, x, y, tid)
  for (const veh of ctx.vehicles) {
    const onRoad = ctx.mask("road", veh.foot);
    const sp = ctx.speed(veh);
    for (const [s, e] of runs(veh.t, sp.map((v, i) => v < p.stop_speed && onRoad[i] > 0), ctx.gap(0.5))) {
      const sel = veh.foot.filter((_, i) => veh.t[i] >= s && veh.t[i] <= e);
      spans.push([s, e, medianAxis(sel, 0), medianAxis(sel, 1), veh.tid]);
    }
  }
  // link fragments of the same standing car (tracker id switches while it stands still)
  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3] || a[4] - b[4]);
  const linked: [number, number, number, number, number[]][] = [];
  for (const sp of spans) {
    const L = linked.find((L) => Math.abs(L[2] - sp[2]) < 20 && Math.abs(L[3] - sp[3]) < 20 && sp[0] - L[1] < 5.0);
    if (L) {
      L[1] = Math.max(L[1], sp[1]);
      L[4].push(sp[4]);
    } else linked.push([sp[0], sp[1], sp[2], sp[3], [sp[4]]]);
  }
  const out: Evidence[] = [];
  for (const [s, e, x, y, tids] of linked) {
    if (e - s < p.stopped_min) continue;
    const zone = (name: string) => ctx.zone(name, [[x, y]])[0] > 0;
    if (zone("bus")) continue;
    if (!(zone("nb") || zone("flow_box")) || x > p.stopped_ne_corner_x) continue;
    out.push({
      label: "stopped_vehicle",
      start: s,
      end: e,
      actors: [...new Set(tids)],
      note: zone("nb") ? "NB lane" : "junction box",
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// congestion
// ---------------------------------------------------------------------------
/** SB traffic standing still while it has green, carried on into the red until the box clears. */
export function congestion(ctx: Context): Evidence[] {
  const p = ctx.params;
  const step = 0.5;
  const ts = Array.from({ length: Math.max(0, Math.ceil(ctx.duration / step)) }, (_, i) => i * step);
  const approach = new Array<number>(ts.length).fill(0);
  const box = new Array<number>(ts.length).fill(0);
  for (const v of ctx.vehicles) {
    const idx = v.t.map((t) => roundHalfEven(t / step));
    const sp = ctx.speed(v);
    const inApp = ctx.zone("flow_approach", v.foot);
    const inFlowBox = ctx.zone("flow_box", v.foot);
    const inStop = ctx.zone("stop", v.foot);
    const hitApp = new Set<number>(), hitBox = new Set<number>();
    idx.forEach((k, i) => {
      if (!(k >= 0 && k < ts.length && sp[i] < p.cong_speed)) return;
      if (inApp[i] > 0) hitApp.add(k);
      if (inFlowBox[i] > 0 || inStop[i] > 0) hitBox.add(k);
    });
    for (const k of hitApp) approach[k] += 1;
    for (const k of hitBox) box[k] += 1;
  }
  const phase = ctx.phasesAt(ts);
  // seconds since the green started: the red-light queue needs a while to get going
  const intoGreen = ts.map((t) => t - greenStart(ctx, t));
  const jam = ts.map((_, i) => {
    const settled = phase[i] === "green" && intoGreen[i] >= p.cong_after_green;
    return (settled && approach[i] + box[i] >= p.cong_green_n) || box[i] >= p.cong_red_n;
  });
  // a jam has to start on green; standing in the box on red only continues one
  const out: Evidence[] = [];
  for (const [s, e] of runs(ts, jam, p.cong_gap)) {
    const ph = ctx.phaseAt(s);
    const startedGreen = (ph === "green" || ph === "yellow") && s - greenStart(ctx, s) >= p.cong_after_green;
    if (e - s >= p.cong_min && startedGreen) {
      let most = -Infinity;
      ts.forEach((t, i) => {
        if (t >= s && t <= e) most = Math.max(most, approach[i] + box[i]);
      });
      out.push({ label: "congestion", start: s, end: e, actors: [], note: `max ${Math.trunc(most)} standing` });
    }
  }
  return out;
}

/** Start of the green phase that is on (or last was on) at time t. */
function greenStart(ctx: Context, t: number): number {
  let j = -1;
  for (let i = 0; i < ctx.signalT.length; i++) if (ctx.signalT[i] <= t && ctx.signalPhase[i] === "green") j = i;
  if (j < 0) return -1e9;
  while (j > 0 && ctx.signalPhase[j - 1] === "green") j--;
  return ctx.signalT[j];
}

// ---------------------------------------------------------------------------
// direction
// ---------------------------------------------------------------------------
export function wrongWay(ctx: Context): Evidence[] {
  const p = ctx.params;
  const out: Evidence[] = [];
  const movers = [...ctx.vehicles, ...ctx.twoWheelers];
  for (const [name, { heading }] of Object.entries(ctx.scene.c.direction_zones)) {
    for (const tr of movers) {
      const inside = ctx.zone(name, tr.foot);
      if (!inside.some((v) => v > 0)) continue;
      const sp = ctx.speed(tr);
      const bad = tr.vel.map(([vx, vy], i) => {
        const diff = Math.abs(pymod(degrees(Math.atan2(vy, vx)) - heading + 180, 360) - 180);
        return inside[i] > 0 && sp[i] > p.ww_min_speed && diff > p.ww_angle;
      });
      for (const [s, e] of runs(tr.t, bad, ctx.gap(0.5))) {
        if (e - s >= p.ww_min_dur) out.push({ label: "wrong_way", start: s, end: e, actors: [tr.tid], note: name });
      }
    }
  }
  return out;
}

/** Southbound vehicles that turn round the median nose and leave northbound. */
export function uTurns(ctx: Context): Evidence[] {
  const p = ctx.params;
  const out: Evidence[] = [];
  const [nx, ny] = ctx.scene.c.median_nose;
  const sb = ctx.scene.c.direction_zones.sb.heading, nb = ctx.scene.c.direction_zones.nb.heading;
  const dev = (h: number, ref: number) => Math.abs(pymod(h - ref + 180, 360) - 180);
  for (const veh of ctx.vehicles) {
    const sp = ctx.speed(veh);
    const moving = sp.map((v) => v > 25);
    const mi = moving.flatMap((m, i) => (m ? [i] : []));
    if (mi.length < 15) continue;
    const t = mi.map((i) => veh.t[i]);
    const head = mi.map((i) => pymod(degrees(Math.atan2(veh.vel[i][1], veh.vel[i][0])), 360));
    const n = Math.max(5, Math.floor(head.length / 8));
    const first = head.slice(0, n), last = head.slice(-n);
    if (median(first.map((h) => dev(h, sb))) > p.ut_dir_tol || median(last.map((h) => dev(h, nb))) > p.ut_dir_tol) continue;
    const nearest = veh.foot.reduce((m, [x, y]) => Math.min(m, Math.sqrt((x - nx) * (x - nx) + (y - ny) * (y - ny))), Infinity);
    if (nearest > p.ut_nose_px) continue;
    const start = mi.slice(0, 10).map((i) => veh.foot[i]);
    const cameFromSb = ctx.zone("sb", start).some((v) => v > 0) || ctx.zone("stop", start).some((v) => v > 0);
    if (!cameFromSb) continue;
    const offSb = head.flatMap((h, i) => (dev(h, sb) > p.ut_turn_deg ? [i] : []));
    const onNb = head.flatMap((h, i) => (dev(h, nb) < p.ut_turn_deg ? [i] : []));
    if (!offSb.length || !onNb.length) continue;
    const s_ = t[offSb[0]];
    const j = onNb.find((i) => i >= offSb[0]);
    const e_ = j === undefined ? null : t[j];
    if (e_ === null || e_ - s_ < p.ut_min_dur) continue;
    out.push({ label: "illegal_u_turn", start: s_, end: e_, actors: [veh.tid], note: "U-turn round the median nose" });
  }
  return out;
}

// ---------------------------------------------------------------------------
// collisions
// ---------------------------------------------------------------------------
/** Python f"{x:.0f}" for x >= 0: like Math.round, but an exact tie rounds to even. */
const fixed0 = (x: number) => String(Number.isInteger(x * 2) && !Number.isInteger(x) ? roundHalfEven(x) : Math.round(x));

/** np.median over the rows of a column (axis 0) of 2-vectors. */
const medianVec = (v: number[][]) => [median(v.map((q) => q[0])), median(v.map((q) => q[1]))];

/**
 * rules.collisions: road users that meet at speed, both change velocity at the contact and then
 * stand together. `mpp(x, y)` is metres per pixel of the trajectories' plane; `p` is rules.CRASH.
 */
export function collisions(trajectories: Trajectory[], mpp: (x: number, y: number) => number,
  p: Record<string, number>, radius: Record<string, number>, k: Classes = COCO): Evidence[] {
  const step = 0.1;
  const tracks: { tr: Trajectory; keys: number[]; first: number[] }[] = [];
  for (const tr of trajectories) {
    if (tr.t.length < 5 || !(isVehicle(tr, k) || isPerson(tr, k) || tr.group === "bicycle")) continue;
    // np.unique(np.round(t / step).astype(int), return_index=True)
    const raw = tr.t.map((t) => roundHalfEven(t / step));
    const order = raw.map((_, i) => i).sort((a, b) => raw[a] - raw[b] || a - b);
    const keys: number[] = [], first: number[] = [];
    for (const i of order) if (!keys.length || raw[i] !== keys[keys.length - 1]) { keys.push(raw[i]); first.push(i); }
    tracks.push({ tr, keys, first });
  }
  const out: Evidence[] = [];
  const hyp = (x: number, y: number) => Math.sqrt(x * x + y * y);
  for (let i = 0; i < tracks.length; i++) {
    const { tr: A, keys: ka, first: fa } = tracks[i];
    for (let j = i + 1; j < tracks.length; j++) {
      const { tr: B, keys: kb, first: fb } = tracks[j];
      if (!(isVehicle(A, k) || isVehicle(B, k))) continue;
      const ia: number[] = [], ib: number[] = [], common: number[] = [];
      for (let x = 0, y = 0; x < ka.length && y < kb.length;) {
        if (ka[x] === kb[y]) { common.push(ka[x]); ia.push(fa[x++]); ib.push(fb[y++]); }
        else if (ka[x] < kb[y]) x++;
        else y++;
      }
      if (common.length < 10) continue;
      const pa = ia.map((q) => A.foot[q]), pb = ib.map((q) => B.foot[q]);
      if (Math.min(...pa.map((a, n) => Math.max(Math.abs(a[0] - pb[n][0]), Math.abs(a[1] - pb[n][1])))) > 400) continue;
      const m = pa.map((a, n) => mpp((a[0] + pb[n][0]) / 2, (a[1] + pb[n][1]) / 2));
      const d = pa.map((a, n) => hyp(a[0] - pb[n][0], a[1] - pb[n][1]) * m[n]);
      const va = ia.map((q, n) => [A.vel[q][0] * m[n], A.vel[q][1] * m[n]]);
      const vb = ib.map((q, n) => [B.vel[q][0] * m[n], B.vel[q][1] * m[n]]);
      const sa = ia.map((q, n) => hyp(A.vel[q][0], A.vel[q][1]) * m[n]);
      const sb = ib.map((q, n) => hyp(B.vel[q][0], B.vel[q][1]) * m[n]);
      const closing = pa.map((a, n) => {  // same operation order as NumPy: ((vA - vB) * m) * ((pA - pB) * m)
        const [av, bv] = [A.vel[ia[n]], B.vel[ib[n]]];
        const rx = (av[0] - bv[0]) * m[n] * ((a[0] - pb[n][0]) * m[n]);
        const ry = (av[1] - bv[1]) * m[n] * ((a[1] - pb[n][1]) * m[n]);
        return -(rx + ry) / Math.max(d[n], 1e-3);
      });
      const ts = common.map((c) => c * step);
      const reach = ((radius[String(A.cls)] ?? 1.0) + (radius[String(B.cls)] ?? 1.0)) * p.contact;
      const still = ts.map((_, n) => sa[n] < p.stop_speed && sb[n] < p.stop_speed && d[n] < p.near);
      const pick = (f: (t: number) => boolean) => ts.flatMap((t, q) => (f(t) ? [q] : []));
      for (let n = 0; n < ts.length; n++) {
        if (!(d[n] <= reach)) continue;
        const before = pick((t) => t >= ts[n] - 1.0 && t < ts[n]);
        const late = pick((t) => t >= ts[n] - 0.5 && t < ts[n]);
        if (before.length < 3 || Math.max(0.0, ...late.map((q) => closing[q])) < p.closing) continue;
        if (Math.max(median(before.map((q) => sa[q])), median(before.map((q) => sb[q]))) < p.pre_speed) continue;
        const after = pick((t) => t > ts[n] && t <= ts[n] + 0.7);
        if (after.length < 2) continue;
        const kick = [va, vb].map((v) => {
          const mean = [0, 1].map((c) => after.reduce((acc, q) => acc + v[q][c], 0) / after.length);
          const med = medianVec(before.map((q) => v[q]));
          return hyp(mean[0] - med[0], mean[1] - med[1]);
        });
        if (Math.min(...kick) < p.kick) continue;
        const rest = pick((t) => t > ts[n] && t <= ts[n] + p.stop_within).filter((q) => still[q]);
        if (!rest.length) continue;
        const tRest = ts[rest[0]];
        const win = pick((t) => t >= tRest && t <= tRest + p.stay);
        if (Math.max(...win.map((q) => ts[q])) - tRest < p.stay - 0.3) continue;
        if (win.filter((q) => still[q]).length / win.length < 0.8) continue;
        out.push({ label: "accident", start: ts[n], end: tRest + p.end_pad, actors: [A.tid, B.tid],
          note: `met at ${fixed0(Math.max(...before.map((q) => closing[q])))} m/s` });
        break;
      }
    }
  }
  return out;
}

export function accident(ctx: Context): Evidence[] {
  const scene = ctx.scene;
  return collisions(ctx.trajectories, (x, y) => metresPerPx(scene, x, y), scene.c.crash,
    scene.c.risk.RADIUS_M as Record<string, number>, ctx.classes);
}

/** events.detect_from_context: every rule's evidence, and the final [start, end, label] segments. */
export function detectFromContext(ctx: Context, H: Mat3): { events: Seg[]; evidence: Evidence[] } {
  const { enabled, shown, gap, min_len } = ctx.scene.c.events;
  let evidence: Evidence[] = [
    ...jaywalking(ctx),
    ...failureToYield(ctx, H),
    ...redLight(ctx, H),
    ...stopLine(ctx, H),
    ...congestion(ctx),
    ...stoppedVehicle(ctx),
    ...wrongWay(ctx),
    ...uTurns(ctx),
    ...accident(ctx),
  ];
  evidence = evidence.filter((ev) => shown.includes(ev.label));
  const perClass: Record<string, [number, number][]> = {};
  for (const ev of evidence) if (enabled.includes(ev.label)) (perClass[ev.label] ??= []).push([ev.start, ev.end]);
  return { events: finalize(perClass, ctx.duration, gap, min_len), evidence };
}
