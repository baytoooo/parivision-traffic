// Event timeline: one lane per class, the signal phase on top, a playhead,
// click or drag to seek, and keyboard access (a slider for position and
// arrow-key navigation between segments).
import { CLASSES, CLASS_BY_KEY, className } from "../lib/classes";
import { fmtTime } from "../lib/format";
import type { Seg } from "../lib/types";
import { clear, s, tickStep, uid } from "./svg";

export type Mode = "pred" | "labels" | "both";

export interface TimelineData {
  duration: number;
  events: Seg[];
  labels?: Seg[];
  signal?: Seg[];
}

/** Width of the label column; the risk curve uses the same so the playheads line up. */
export function gutterFor(width: number): number {
  return width < 560 ? 50 : 150;
}

interface Item {
  seg: Seg;
  kind: "pred" | "label";
  lane: number;
  el: SVGRectElement;
}

const SIG_CLASS: Record<string, string> = { red: "sig-red", yellow: "sig-amber", green: "sig-green" };

export class Timeline {
  private host: HTMLElement;
  private svg: SVGSVGElement;
  private tip: HTMLDivElement;
  private data: TimelineData;
  private onSeek: (t: number) => void;
  private mode: Mode = "pred";
  private showAll = false;
  private t = 0;
  private items: Item[] = [];
  private focusIdx = -1;
  private g = { W: 0, gutter: 150, trackW: 1, laneTop: 0, laneH: 24, lanes: [] as string[], axisTop: 0, H: 0 };
  private playhead?: SVGGElement;
  private focusRing?: SVGRectElement;
  private scrub?: SVGRectElement;
  private hoverLine?: SVGLineElement;
  private dragging = false;
  private descId = uid("tl-desc");

  constructor(host: HTMLElement, data: TimelineData, onSeek: (t: number) => void) {
    this.host = host;
    this.data = data;
    this.onSeek = onSeek;
    this.host.classList.add("tl");
    this.svg = s("svg", { class: "tl-svg", role: "group", "aria-label": "Event timeline", "aria-describedby": this.descId });
    const desc = document.createElement("p");
    desc.id = this.descId;
    desc.className = "visually-hidden";
    desc.textContent =
      "Tab to the time axis and use the arrow keys to move the playhead, Shift with an arrow for 10 seconds. Tab again to reach the events: arrow keys move between them, Enter plays from the start of the focused event.";
    this.tip = document.createElement("div");
    this.tip.className = "tip";
    this.tip.setAttribute("aria-hidden", "true");
    this.host.append(desc, this.svg, this.tip);
    this.bind();
    new ResizeObserver(() => {
      if (Math.abs(this.host.clientWidth - this.g.W) > 1) this.render();
    }).observe(this.host);
    this.render();
  }

  setData(data: TimelineData) {
    this.data = data;
    this.focusIdx = -1;
    this.render();
  }

  setMode(mode: Mode) {
    this.mode = mode;
    this.render();
  }

  setShowAll(v: boolean) {
    this.showAll = v;
    this.render();
  }

  private x(t: number) {
    return this.g.gutter + (Math.max(0, Math.min(this.data.duration, t)) / this.data.duration) * this.g.trackW;
  }

  private tAt(clientX: number) {
    const r = this.svg.getBoundingClientRect();
    const px = clientX - r.left;
    return Math.max(0, Math.min(this.data.duration, ((px - this.g.gutter) / this.g.trackW) * this.data.duration));
  }

  lanes(): string[] {
    const present = new Set<string>();
    for (const e of this.data.events) present.add(e[2]);
    for (const e of this.data.labels ?? []) present.add(e[2]);
    const known = CLASSES.map((c) => c.key);
    const extra = [...present].filter((k) => !known.includes(k));
    const base = this.showAll ? known : known.filter((k) => present.has(k));
    return [...base, ...extra];
  }

  render() {
    const W = Math.max(280, Math.floor(this.host.clientWidth));
    const narrow = W < 560;
    const gutter = gutterFor(W);
    const lanes = this.lanes();
    const laneH = narrow ? 24 : 24;
    const sigTop = 4;
    const sigH = 10;
    const laneTop = sigTop + sigH + 10;
    const lanesH = Math.max(1, lanes.length) * laneH;
    const axisTop = laneTop + lanesH + 2;
    const H = axisTop + 26;
    const trackW = W - gutter - 8;
    this.g = { W, gutter, trackW, laneTop, laneH, lanes, axisTop, H };

    const svg = this.svg;
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", String(W));
    svg.setAttribute("height", String(H));

    // lanes
    const gl = s("g", { class: "tl-lanes" }, svg);
    s("text", { x: gutter - 10, y: sigTop + sigH - 1, "text-anchor": "end", class: "tl-lab tl-lab-sig" }, gl).textContent = narrow ? "SIG" : "Signal";
    lanes.forEach((key, i) => {
      const y = laneTop + i * laneH;
      s("rect", { x: gutter, y, width: trackW, height: laneH, class: i % 2 ? "tl-band odd" : "tl-band" }, gl);
      const meta = CLASS_BY_KEY[key];
      const t = s("text", {
        x: gutter - 10,
        y: y + laneH / 2 + 4,
        "text-anchor": "end",
        class: `tl-lab${meta && meta.status !== "on" ? " dim" : ""}`,
      }, gl);
      t.textContent = narrow ? (meta?.code ?? key.slice(0, 4).toUpperCase()) : className(key);
    });
    if (!lanes.length) {
      s("text", { x: gutter + 10, y: laneTop + laneH / 2 + 4, class: "tl-lab" }, gl).textContent = "No events in this clip.";
    }
    s("line", { x1: gutter, x2: gutter, y1: sigTop, y2: axisTop, class: "tl-edge" }, gl);

    // signal phase strip
    const gs = s("g", { class: "tl-signal" }, svg);
    s("rect", { x: gutter, y: sigTop, width: trackW, height: sigH, class: "tl-sig-bg" }, gs);
    for (const [a, b, l] of this.data.signal ?? []) {
      const x0 = this.x(a);
      const w = Math.max(0.8, this.x(b) - x0);
      s("rect", { x: x0, y: sigTop, width: w, height: sigH, class: `tl-sig ${SIG_CLASS[l] ?? "sig-off"}` }, gs);
    }

    // axis
    const ga = s("g", { class: "tl-axis" }, svg);
    const step = tickStep(this.data.duration, trackW, narrow ? 56 : 70);
    for (let t = 0; t <= this.data.duration + 1e-6; t += step) {
      const x = this.x(t);
      s("line", { x1: x, x2: x, y1: laneTop, y2: axisTop + 4, class: "tl-grid" }, ga);
      const tx = s("text", { x, y: axisTop + 17, "text-anchor": t === 0 ? "start" : "middle" }, ga);
      tx.textContent = fmtTime(t, 0);
    }
    this.scrub = s("rect", {
      x: gutter, y: axisTop, width: trackW, height: 24,
      class: "tl-scrub", tabindex: 0, role: "slider",
      "aria-label": "Playback position",
      "aria-valuemin": 0, "aria-valuemax": this.data.duration.toFixed(1),
    }, svg);

    // segments
    const gseg = s("g", { class: "tl-segs" }, svg);
    this.items = [];
    const both = this.mode === "both";
    const addSegs = (segs: Seg[] | undefined, kind: "pred" | "label") => {
      for (const seg of segs ?? []) {
        const lane = lanes.indexOf(seg[2]);
        if (lane < 0) continue;
        const y0 = laneTop + lane * laneH;
        let y = y0 + 5;
        let h = laneH - 10;
        if (both) {
          y = kind === "pred" ? y0 + 4 : y0 + laneH - 10;
          h = kind === "pred" ? 9 : 6;
        }
        const x0 = this.x(seg[0]);
        const w = Math.max(2.5, this.x(seg[1]) - x0);
        const el = s("rect", {
          x: x0, y, width: w, height: h, rx: 1,
          class: `tl-seg ${kind === "pred" ? "tl-pred" : "tl-label"}`,
          tabindex: -1, role: "button",
          "aria-label": `${className(seg[2])}, ${fmtTime(seg[0])} to ${fmtTime(seg[1])}, ${kind === "pred" ? "prediction" : "our label"}`,
        }, gseg);
        this.items.push({ seg, kind, lane, el });
      }
    };
    if (this.mode !== "labels") addSegs(this.data.events, "pred");
    if (this.mode !== "pred") addSegs(this.data.labels, "label");
    this.items.sort((a, b) => a.lane - b.lane || a.seg[0] - b.seg[0] || (a.kind === "pred" ? -1 : 1));
    this.items.forEach((it, i) => (it.el.dataset.idx = String(i)));
    if (this.items.length) {
      if (this.focusIdx < 0 || this.focusIdx >= this.items.length) this.focusIdx = 0;
      this.items[this.focusIdx].el.setAttribute("tabindex", "0");
    }

    // hover line, playhead, focus ring
    this.hoverLine = s("line", { y1: sigTop, y2: axisTop, class: "tl-hover", visibility: "hidden" }, svg);
    this.playhead = s("g", { class: "tl-playhead" }, svg);
    s("line", { x1: 0, x2: 0, y1: sigTop - 2, y2: axisTop + 6, class: "tl-ph-line" }, this.playhead);
    s("path", { d: "M-5,-2 L5,-2 L0,5 Z", class: "tl-ph-head" }, this.playhead);
    s("rect", { x: -16, y: axisTop + 5, width: 32, height: 3, class: "tl-ph-foot" }, this.playhead);
    this.focusRing = s("rect", { class: "tl-focus", rx: 2 }, svg);
    this.setTime(this.t);
  }

  setTime(t: number) {
    this.t = t;
    if (!this.playhead) return;
    const x = this.x(t);
    this.playhead.setAttribute("transform", `translate(${x.toFixed(1)},0)`);
    if (this.scrub) {
      this.scrub.setAttribute("aria-valuenow", t.toFixed(1));
      this.scrub.setAttribute("aria-valuetext", `${fmtTime(t)} of ${fmtTime(this.data.duration)}`);
    }
    for (const it of this.items) it.el.classList.toggle("live", t >= it.seg[0] && t <= it.seg[1]);
  }

  private showTip(html: string, x: number, y: number) {
    this.tip.innerHTML = html;
    const W = this.g.W;
    const cx = Math.max(70, Math.min(W - 70, x));
    this.tip.style.left = `${cx}px`;
    this.tip.style.top = `${y}px`;
    this.tip.classList.add("show");
  }

  private hideTip() {
    this.tip.classList.remove("show");
  }

  private ring(el: SVGGraphicsElement | null) {
    if (!this.focusRing) return;
    if (!el) {
      this.focusRing.style.display = "none";
      return;
    }
    const b = el.getBBox();
    this.focusRing.setAttribute("x", String(b.x - 3));
    this.focusRing.setAttribute("y", String(b.y - 3));
    this.focusRing.setAttribute("width", String(b.width + 6));
    this.focusRing.setAttribute("height", String(b.height + 6));
    this.focusRing.style.display = "block";
  }

  private segTip(it: Item) {
    const [a, b, l] = it.seg;
    const x = (this.x(a) + this.x(b)) / 2;
    const y = this.g.laneTop + it.lane * this.g.laneH;
    this.showTip(
      `${className(l)}${it.kind === "label" ? " (label)" : ""}<br>${fmtTime(a)} to ${fmtTime(b)}, ${(b - a).toFixed(1)} s`,
      x,
      y,
    );
  }

  private focusItem(i: number) {
    if (!this.items.length) return;
    i = Math.max(0, Math.min(this.items.length - 1, i));
    this.items[this.focusIdx]?.el.setAttribute("tabindex", "-1");
    this.focusIdx = i;
    const it = this.items[i];
    it.el.setAttribute("tabindex", "0");
    it.el.focus();
  }

  private bind() {
    const svg = this.svg;
    let pending: number | null = null;
    let lastT = 0;
    const seekSoon = (t: number) => {
      lastT = t;
      this.setTime(t);
      if (pending === null)
        pending = requestAnimationFrame(() => {
          pending = null;
          this.onSeek(lastT);
        });
    };

    svg.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      const target = e.target as Element;
      const idx = (target as SVGElement).dataset?.idx;
      if (idx !== undefined) {
        const it = this.items[Number(idx)];
        this.onSeek(it.seg[0]);
        this.setTime(it.seg[0]);
        this.items[this.focusIdx]?.el.setAttribute("tabindex", "-1");
        this.focusIdx = Number(idx);
        it.el.setAttribute("tabindex", "0");
        return;
      }
      const r = svg.getBoundingClientRect();
      if (e.clientX - r.left < this.g.gutter - 4) return;
      this.dragging = true;
      svg.setPointerCapture(e.pointerId);
      seekSoon(this.tAt(e.clientX));
      e.preventDefault();
    });
    svg.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const px = e.clientX - r.left;
      if (this.dragging) {
        seekSoon(this.tAt(e.clientX));
        return;
      }
      if (e.pointerType !== "mouse") return;
      const idx = (e.target as SVGElement).dataset?.idx;
      if (px < this.g.gutter || px > this.g.gutter + this.g.trackW) {
        this.hoverLine?.setAttribute("visibility", "hidden");
        this.hideTip();
        return;
      }
      this.hoverLine?.setAttribute("x1", String(px));
      this.hoverLine?.setAttribute("x2", String(px));
      this.hoverLine?.setAttribute("visibility", "visible");
      if (idx !== undefined) this.segTip(this.items[Number(idx)]);
      else this.showTip(fmtTime(this.tAt(e.clientX)), px, this.g.laneTop);
    });
    const end = (e: PointerEvent) => {
      if (this.dragging) {
        this.dragging = false;
        try {
          svg.releasePointerCapture(e.pointerId);
        } catch {
          /* already released */
        }
      }
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.addEventListener("pointerleave", () => {
      this.hoverLine?.setAttribute("visibility", "hidden");
      if (document.activeElement && !svg.contains(document.activeElement)) this.hideTip();
      else if (!document.activeElement || document.activeElement === document.body) this.hideTip();
    });

    svg.addEventListener("focusin", (e) => {
      const el = e.target as SVGGraphicsElement;
      this.ring(el);
      const idx = (el as SVGElement).dataset?.idx;
      if (idx !== undefined) this.segTip(this.items[Number(idx)]);
      else this.hideTip();
    });
    svg.addEventListener("focusout", () => {
      this.ring(null);
      this.hideTip();
    });

    svg.addEventListener("keydown", (e) => {
      const el = e.target as SVGElement;
      const D = this.data.duration;
      if (el === this.scrub) {
        const big = e.shiftKey ? 10 : 1;
        let t = this.t;
        switch (e.key) {
          case "ArrowRight":
          case "ArrowUp":
            t += big;
            break;
          case "ArrowLeft":
          case "ArrowDown":
            t -= big;
            break;
          case "PageUp":
            t += 30;
            break;
          case "PageDown":
            t -= 30;
            break;
          case "Home":
            t = 0;
            break;
          case "End":
            t = D;
            break;
          default:
            return;
        }
        e.preventDefault();
        t = Math.max(0, Math.min(D, t));
        this.setTime(t);
        this.onSeek(t);
        return;
      }
      const idxS = el.dataset?.idx;
      if (idxS === undefined) return;
      const i = Number(idxS);
      const it = this.items[i];
      const inLane = (lane: number) => this.items.map((x, j) => [x, j] as const).filter(([x]) => x.lane === lane);
      switch (e.key) {
        case "ArrowRight":
          if (this.items[i + 1]?.lane === it.lane) this.focusItem(i + 1);
          break;
        case "ArrowLeft":
          if (this.items[i - 1]?.lane === it.lane) this.focusItem(i - 1);
          break;
        case "ArrowDown":
        case "ArrowUp": {
          const dir = e.key === "ArrowDown" ? 1 : -1;
          for (let lane = it.lane + dir; lane >= 0 && lane < this.g.lanes.length; lane += dir) {
            const cands = inLane(lane);
            if (!cands.length) continue;
            let best = cands[0];
            for (const c of cands) if (Math.abs(c[0].seg[0] - it.seg[0]) < Math.abs(best[0].seg[0] - it.seg[0])) best = c;
            this.focusItem(best[1]);
            break;
          }
          break;
        }
        case "Home": {
          const c = inLane(it.lane);
          this.focusItem(c[0][1]);
          break;
        }
        case "End": {
          const c = inLane(it.lane);
          this.focusItem(c[c.length - 1][1]);
          break;
        }
        case "Enter":
        case " ":
          this.setTime(it.seg[0]);
          this.onSeek(it.seg[0]);
          break;
        default:
          return;
      }
      e.preventDefault();
    });
  }
}
