// Ties one annotated video to its timeline, risk curve, signal lamp and event
// table. Used by the Results page (sample clips) and the Demo page (job result).
// A demo result also carries the tracker's boxes, which are drawn over the video.
import { className, zoneFor, ZONE_BY_KEY } from "../lib/classes";
import { fmtTime, iou } from "../lib/format";
import type { ClipResult, Evidence, Overlay, Seg } from "../lib/types";
import { RiskCurve } from "./riskcurve";
import { Timeline, type Mode } from "./timeline";

export interface PlayerSource {
  result: ClipResult;
  video: string;
  poster?: string;
  title?: string;
}

interface Row {
  seg: Seg;
  kind: "pred" | "label";
  ev: { actors: number[]; notes: string[]; zone: string };
  match: number | null;
  tr?: HTMLTableRowElement;
}

const $ = <T extends Element = HTMLElement>(root: Element, sel: string) => root.querySelector<T>(`[data-el="${sel}"]`);

// The colours of the rendered videos on the Results page (src/parivision/render.py GROUP_COLORS and
// CLASS_COLORS, BGR there): a box takes its group's colour, or its event's while it is in one.
const GROUP_RGB: Record<string, string> = {
  vehicle: "120, 200, 235",
  person: "140, 235, 140",
  bicycle: "250, 170, 230",
  animal: "255, 200, 90",
};
const CLASS_RGB: Record<string, string> = {
  accident: "230, 40, 40",
  near_miss: "255, 120, 60",
  red_light: "255, 50, 50",
  wrong_way: "200, 60, 200",
  illegal_u_turn: "180, 120, 220",
  stopped_vehicle: "255, 190, 0",
  jaywalking: "255, 220, 0",
  failure_to_yield: "255, 170, 60",
  illegal_turn: "255, 110, 180",
  solid_line_crossing: "80, 180, 255",
  stop_line: "255, 80, 80",
  congestion: "120, 120, 120",
  road_obstacle: "120, 200, 60",
  fire_smoke: "180, 30, 30",
};
const OTHER_RGB = "200, 200, 200";

/**
 * Draws a result's tracked boxes on a canvas laid over the video: the analysed frame nearest the
 * playhead, placed inside the picture as object-fit: contain shows it. An actor named in the
 * evidence of an event that is on at that moment gets the event's colour and its class name.
 */
class BoxOverlay {
  private canvas: HTMLCanvasElement;
  private video: HTMLVideoElement;
  private toggle: HTMLElement | null;
  private data: Overlay | null = null;
  private times: number[] = [];
  /** Per track id: when it is an actor of an event that is on, and which. */
  private roles = new Map<number, { s: number; e: number; label: string }[]>();
  private visible = true;
  private t = 0;
  private drawn = "";

  constructor(canvas: HTMLCanvasElement, video: HTMLVideoElement, toggle: HTMLElement | null, box: HTMLInputElement | null) {
    this.canvas = canvas;
    this.video = video;
    this.toggle = toggle;
    box?.addEventListener("change", () => {
      this.visible = box.checked;
      this.draw(this.t);
    });
    new ResizeObserver(() => this.draw(this.t)).observe(canvas);
    video.addEventListener("loadeddata", () => this.draw(this.t));
  }

  set(result: ClipResult) {
    const o = result.overlay;
    this.data = o && o.frames.length ? o : null;
    this.times = this.data ? this.data.frames.map((f) => f.t) : [];
    this.roles.clear();
    const events = result.events ?? [];
    for (const ev of result.evidence ?? []) {
      // only the part of the evidence inside a final event of its class
      for (const [s, e, label] of events) {
        if (label !== ev.label) continue;
        const a = Math.max(s, ev.start), b = Math.min(e, ev.end);
        if (a > b) continue;
        for (const id of ev.actors ?? []) {
          const list = this.roles.get(id) ?? [];
          list.push({ s: a, e: b, label });
          this.roles.set(id, list);
        }
      }
    }
    this.canvas.hidden = !this.data;
    if (this.toggle) this.toggle.hidden = !this.data;
    this.drawn = "";
    this.draw(this.t);
  }

  /** Index of the analysed frame nearest t, or -1 when none is within one frame step. */
  private nearest(t: number): number {
    const ts = this.times;
    if (!ts.length) return -1;
    let lo = 0, hi = ts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ts[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    const i = lo > 0 && Math.abs(ts[lo - 1] - t) <= Math.abs(ts[lo] - t) ? lo - 1 : lo;
    const step = ts.length > 1 ? (ts[ts.length - 1] - ts[0]) / (ts.length - 1) : 0.2;
    return Math.abs(ts[i] - t) <= step * 0.75 ? i : -1;
  }

  draw(t: number) {
    this.t = t;
    const c = this.canvas;
    if (!this.data) return;
    const dpr = window.devicePixelRatio || 1;
    const cr = c.getBoundingClientRect();
    const W = Math.round(cr.width * dpr), H = Math.round(cr.height * dpr);
    // no boxes over a video that shows no frame yet
    const i = this.visible && this.video.readyState >= 2 ? this.nearest(t) : -1;
    const key = `${i} ${W}x${H} ${this.video.videoWidth}`;
    if (key === this.drawn) return;
    this.drawn = key;
    if (c.width !== W || c.height !== H) {
      c.width = W;
      c.height = H;
    }
    const g = c.getContext("2d");
    if (!g) return;
    g.clearRect(0, 0, W, H);
    if (i < 0 || !W || !H) return;

    // the picture inside the video element (object-fit: contain), in canvas pixels
    const vr = this.video.getBoundingClientRect();
    const [ww, wh] = this.data.work;
    const vw = this.video.videoWidth || ww, vh = this.video.videoHeight || wh;
    const scale = Math.min(vr.width / vw, vr.height / vh);
    const pw = vw * scale, ph = vh * scale;
    const ox = (vr.left - cr.left + (vr.width - pw) / 2) * dpr, oy = (vr.top - cr.top + (vr.height - ph) / 2) * dpr;
    const sx = (pw * dpr) / ww, sy = (ph * dpr) / wh;

    const frame = this.data.frames[i];
    const font = `600 ${Math.round(11 * dpr)}px "Overpass Mono", ui-monospace, monospace`;
    const actors: [number[], string[]][] = [];
    g.lineJoin = "round";
    for (const b of frame.boxes) {
      const labels = [...new Set((this.roles.get(b[4]) ?? []).filter((r) => frame.t >= r.s - 1e-3 && frame.t <= r.e + 1e-3).map((r) => r.label))];
      if (labels.length) {
        actors.push([b, labels]);
        continue;
      }
      g.strokeStyle = `rgba(${GROUP_RGB[this.data.groups[Math.floor(b[4] / 1_000_000) - 1]] ?? OTHER_RGB}, 0.85)`;
      g.lineWidth = 1.25 * dpr;
      g.strokeRect(ox + b[0] * sx, oy + b[1] * sy, (b[2] - b[0]) * sx, (b[3] - b[1]) * sy);
    }
    // actors last, so that their labels sit on top
    g.font = font;
    g.textBaseline = "bottom";
    for (const [b, labels] of actors) {
      const rgb = CLASS_RGB[labels[0]] ?? OTHER_RGB;
      const x = ox + b[0] * sx, y = oy + b[1] * sy;
      g.strokeStyle = `rgb(${rgb})`;
      g.lineWidth = 2.5 * dpr;
      g.strokeRect(x, y, (b[2] - b[0]) * sx, (b[3] - b[1]) * sy);
      const text = labels.map(className).join(", ");
      const pad = 4 * dpr, h = 16 * dpr;
      const tw = g.measureText(text).width + 2 * pad;
      const tx = Math.min(Math.max(x - 1 * dpr, 0), Math.max(0, W - tw));
      const ty = y - h >= 0 ? y - h : y;
      g.fillStyle = `rgb(${rgb})`;
      g.fillRect(tx, ty, tw, h);
      g.fillStyle = "#151618";
      g.fillText(text, tx + pad, ty + h - 3 * dpr);
    }
  }
}

export class Player {
  private root: HTMLElement;
  readonly video: HTMLVideoElement;
  private timeline: Timeline;
  private risk: RiskCurve;
  private src!: PlayerSource;
  private mode: Mode = "pred";
  private rows: Row[] = [];
  private raf = 0;
  private theta: number;
  private lastPhase = "";
  private overlay: BoxOverlay;

  constructor(root: HTMLElement, src: PlayerSource, opts: { theta: number; mergeGap: number }) {
    this.root = root;
    this.theta = opts.theta;
    this.video = $<HTMLVideoElement>(root, "video")!;
    this.overlay = new BoxOverlay($<HTMLCanvasElement>(root, "overlay")!, this.video, $(root, "boxes-toggle"), $<HTMLInputElement>(root, "boxes"));
    const seek = (t: number) => this.seek(t);
    this.timeline = new Timeline($(root, "timeline")!, this.tlData(src.result), seek);
    this.risk = new RiskCurve($(root, "risk")!, { duration: src.result.duration, risk: src.result.risk ?? [] }, { ...opts, onSeek: seek });
    this.bindVideo();
    this.bindControls();
    this.load(src);
  }

  private tlData(r: ClipResult) {
    return { duration: r.duration, events: r.events ?? [], labels: r.labels, signal: r.signal };
  }

  load(src: PlayerSource, startAt = 0) {
    this.src = src;
    const r = src.result;
    const hasLabels = Array.isArray(r.labels) && r.labels.length > 0;
    const modeBox = $(this.root, "mode");
    if (modeBox) modeBox.hidden = !hasLabels;
    if (!hasLabels) this.mode = "pred";
    else {
      const checked = this.root.querySelector<HTMLInputElement>('[data-el="mode"] input:checked');
      this.mode = (checked?.value as Mode) ?? "pred";
    }
    this.timeline.setData(this.tlData(r));
    this.timeline.setMode(this.mode);
    this.risk.setData({ duration: r.duration, risk: r.risk ?? [] });

    const v = this.video;
    if (v.getAttribute("src") !== src.video) {
      v.pause();
      if (src.poster) v.poster = src.poster;
      else v.removeAttribute("poster");
      v.setAttribute("src", src.video);
      v.preload = "metadata";
      v.load();
    }
    const dur = $(this.root, "dur");
    if (dur) dur.textContent = fmtTime(r.duration);
    const title = $(this.root, "title");
    if (title && src.title) title.textContent = src.title;

    this.buildRows();
    this.renderSummary();
    this.overlay.set(r);
    this.update(startAt);
    if (startAt > 0) this.seek(startAt);
  }

  /** Seek the video (and everything synced to it). */
  seek(t: number, play = false) {
    const D = this.src.result.duration;
    t = Math.max(0, Math.min(D, t));
    this.update(t);
    const v = this.video;
    const go = () => {
      const max = Number.isFinite(v.duration) ? Math.max(0, v.duration - 0.05) : t;
      try {
        v.currentTime = Math.min(t, max);
      } catch {
        /* not seekable yet */
      }
      if (play) v.play().catch(() => undefined);
    };
    if (v.readyState >= 1) go();
    else {
      v.addEventListener("loadedmetadata", go, { once: true });
      if (v.preload === "none") {
        v.preload = "metadata";
        v.load();
      }
    }
  }

  focusVideo() {
    this.video.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  }

  private bindVideo() {
    const v = this.video;
    const tick = () => {
      this.update(v.currentTime);
      if (!v.paused && !v.ended) this.raf = requestAnimationFrame(tick);
    };
    v.addEventListener("play", () => {
      cancelAnimationFrame(this.raf);
      this.raf = requestAnimationFrame(tick);
    });
    v.addEventListener("pause", () => cancelAnimationFrame(this.raf));
    v.addEventListener("seeked", () => this.update(v.currentTime));
    v.addEventListener("timeupdate", () => {
      if (v.paused) this.update(v.currentTime);
    });
    v.addEventListener("error", () => {
      const err = $(this.root, "video-error");
      if (err) err.hidden = false;
    });
    v.addEventListener("loadeddata", () => {
      const err = $(this.root, "video-error");
      if (err) err.hidden = true;
    });
  }

  private bindControls() {
    const modeBox = $(this.root, "mode");
    modeBox?.addEventListener("change", (e) => {
      const val = (e.target as HTMLInputElement).value as Mode;
      this.mode = val;
      this.timeline.setMode(val);
      this.buildRows();
      this.update(this.video.currentTime || 0);
    });
    const all = $<HTMLInputElement>(this.root, "all");
    all?.addEventListener("change", () => this.timeline.setShowAll(all.checked));

    const tbody = $(this.root, "rows");
    tbody?.addEventListener("click", (e) => {
      const btn = (e.target as Element).closest<HTMLButtonElement>("button[data-t]");
      if (!btn) return;
      this.seek(Number(btn.dataset.t), true);
      this.focusVideo();
    });
  }

  private evidenceFor(seg: Seg): Row["ev"] {
    const ev: Evidence[] = (this.src.result.evidence ?? []).filter(
      (e) => e.label === seg[2] && e.end >= seg[0] - 0.05 && e.start <= seg[1] + 0.05,
    );
    const actors = [...new Set(ev.flatMap((e) => e.actors ?? []))];
    const notes = [...new Set(ev.map((e) => e.note).filter(Boolean))];
    const zone = zoneFor(seg[2], ev.find((e) => e.zone)?.zone);
    return { actors, notes, zone };
  }

  private buildRows() {
    const r = this.src.result;
    const labels = r.labels ?? [];
    const hasLabels = labels.length > 0;
    const showLabels = this.mode === "labels";
    const main = showLabels ? labels : r.events ?? [];
    const other = showLabels ? r.events ?? [] : labels;
    this.rows = main
      .slice()
      .sort((a, b) => a[0] - b[0])
      .map((seg) => {
        let best = 0;
        for (const o of other) if (o[2] === seg[2]) best = Math.max(best, iou([seg[0], seg[1]], [o[0], o[1]]));
        return { seg, kind: showLabels ? "label" : "pred", ev: this.evidenceFor(seg), match: hasLabels ? best : null } as Row;
      });

    const head = $(this.root, "match-head");
    if (head) {
      head.hidden = !hasLabels;
      head.textContent = showLabels ? "Best IoU with a prediction" : "Best IoU with a label";
    }
    const cap = $(this.root, "table-cap");
    if (cap) cap.textContent = showLabels ? `Our labels (${this.rows.length})` : `Predicted events (${this.rows.length})`;

    const tbody = $(this.root, "rows")!;
    tbody.textContent = "";
    if (!this.rows.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.className = "empty";
      td.textContent = "No events.";
      tr.append(td);
      tbody.append(tr);
      return;
    }
    const frag = document.createDocumentFragment();
    for (const row of this.rows) {
      const [a, b, l] = row.seg;
      const tr = document.createElement("tr");
      const cell = (text: string, cls = "") => {
        const td = document.createElement("td");
        if (cls) td.className = cls;
        td.textContent = text;
        tr.append(td);
        return td;
      };
      const tdT = document.createElement("td");
      tdT.className = "t";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "seek";
      btn.dataset.t = String(a);
      btn.setAttribute("aria-label", `Play ${className(l)} from ${fmtTime(a)}`);
      btn.innerHTML = `<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2 1 L9 5 L2 9 Z"/></svg><span>${fmtTime(a)}</span>`;
      tdT.append(btn);
      tr.append(tdT);
      cell(fmtTime(b), "t");
      cell(`${(b - a).toFixed(1)} s`, "num");
      const tdC = cell("", "cls");
      tdC.innerHTML = `<span class="cls-name">${className(l)}</span>`;
      cell(ZONE_BY_KEY[row.ev.zone]?.name ?? "", "zone");
      const note = row.ev.notes.join("; ");
      const actors = row.ev.actors.length ? `#${row.ev.actors.join(", #")}` : "";
      const tdN = cell("", "note");
      tdN.innerHTML = `${note ? `<span>${escapeHtml(note)}</span>` : ""}${actors ? `<span class="actors">${actors}</span>` : ""}`;
      if (row.match !== null) {
        const td = cell(row.match > 0 ? row.match.toFixed(2) : row.kind === "pred" ? "no label" : "missed", "num match");
        if (row.match < 0.3) td.classList.add("weak");
        if (row.match <= 0) td.classList.add("nomatch");
      }
      row.tr = tr;
      frag.append(tr);
    }
    tbody.append(frag);
  }

  private renderSummary() {
    const r = this.src.result;
    const ev = r.events ?? [];
    const classes = new Set(ev.map((e) => e[2]));
    let maxV = 0;
    for (const [, v] of r.risk ?? []) maxV = Math.max(maxV, v);
    const alarms = this.risk.alarms().length;
    const el = $(this.root, "summary");
    if (el)
      el.innerHTML = [
        `<span><b>${ev.length}</b> events</span>`,
        `<span><b>${classes.size}</b> classes</span>`,
        r.labels?.length ? `<span><b>${r.labels.length}</b> labels</span>` : "",
        `<span>risk max <b>${maxV.toFixed(2)}</b></span>`,
        `<span><b>${alarms}</b> alarm${alarms === 1 ? "" : "s"} at ${this.theta}</span>`,
      ].join("");
  }

  private phaseAt(t: number): string {
    for (const [a, b, l] of this.src.result.signal ?? []) if (t >= a && t <= b) return l;
    return "unknown";
  }

  private update(t: number) {
    this.timeline.setTime(t);
    this.risk.setTime(t);
    this.overlay.draw(t);
    const time = $(this.root, "time");
    if (time) time.textContent = fmtTime(t);

    const phase = this.src.result.signal?.length ? this.phaseAt(t) : "none";
    if (phase !== this.lastPhase) {
      this.lastPhase = phase;
      const lamp = $(this.root, "lamp");
      if (lamp) lamp.dataset.phase = phase;
      const txt = $(this.root, "phase");
      if (txt) txt.textContent = phase === "none" ? "no signal data" : phase;
    }
    const live: string[] = [];
    for (const row of this.rows) {
      const on = t >= row.seg[0] && t <= row.seg[1];
      row.tr?.classList.toggle("live", on);
      if (on && !live.includes(row.seg[2])) live.push(row.seg[2]);
    }
    const now = $(this.root, "now");
    if (now) {
      const text = live.length ? live.map(className).join(", ") : "none";
      if (now.textContent !== text) now.textContent = text;
    }
  }
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
