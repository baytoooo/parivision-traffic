// Ties one annotated video to its timeline, risk curve, signal lamp and event
// table. Used by the Results page (sample clips) and the Demo page (job result).
import { className, zoneFor, ZONE_BY_KEY } from "../lib/classes";
import { fmtTime, iou } from "../lib/format";
import type { ClipResult, Evidence, Seg } from "../lib/types";
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

  constructor(root: HTMLElement, src: PlayerSource, opts: { theta: number; mergeGap: number }) {
    this.root = root;
    this.theta = opts.theta;
    this.video = $<HTMLVideoElement>(root, "video")!;
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
