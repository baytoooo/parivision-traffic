// Part B risk curve under the timeline, same x scale, synced playhead.
import { alarmRuns, fmtTime } from "../lib/format";
import { clear, s, tickStep, uid } from "./svg";
import { gutterFor } from "./timeline";

export interface RiskData {
  duration: number;
  risk: [number, number][];
}

export class RiskCurve {
  private host: HTMLElement;
  private svg: SVGSVGElement;
  private tip: HTMLDivElement;
  private data: RiskData;
  private theta: number;
  private mergeGap: number;
  private onSeek: (t: number) => void;
  private g = { W: 0, gutter: 150, trackW: 1, top: 8, plotH: 80, H: 0 };
  private playhead?: SVGLineElement;
  private hover?: SVGLineElement;
  private dot?: SVGCircleElement;
  private dragging = false;
  private clipId = uid("risk-clip");

  constructor(host: HTMLElement, data: RiskData, opts: { theta: number; mergeGap: number; onSeek: (t: number) => void }) {
    this.host = host;
    this.data = data;
    this.theta = opts.theta;
    this.mergeGap = opts.mergeGap;
    this.onSeek = opts.onSeek;
    this.host.classList.add("risk");
    this.svg = s("svg", { class: "risk-svg", role: "img" });
    this.tip = document.createElement("div");
    this.tip.className = "tip";
    this.tip.setAttribute("aria-hidden", "true");
    this.host.append(this.svg, this.tip);
    this.bind();
    new ResizeObserver(() => {
      if (Math.abs(this.host.clientWidth - this.g.W) > 1) this.render();
    }).observe(this.host);
    this.render();
  }

  setData(data: RiskData) {
    this.data = data;
    this.render();
  }

  alarms() {
    return alarmRuns(this.data.risk, this.theta, this.mergeGap);
  }

  private x(t: number) {
    return this.g.gutter + (Math.max(0, Math.min(this.data.duration, t)) / this.data.duration) * this.g.trackW;
  }

  private y(v: number) {
    return this.g.top + (1 - v) * this.g.plotH;
  }

  private valueAt(t: number): number {
    const r = this.data.risk;
    if (!r.length) return NaN;
    let lo = 0;
    let hi = r.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (r[mid][0] < t) lo = mid + 1;
      else hi = mid;
    }
    return r[lo][1];
  }

  render() {
    const W = Math.max(280, Math.floor(this.host.clientWidth));
    const narrow = W < 560;
    const gutter = gutterFor(W);
    const plotH = narrow ? 70 : 88;
    const top = 8;
    const H = top + plotH + 24;
    const trackW = W - gutter - 8;
    this.g = { W, gutter, trackW, top, plotH, H };
    const svg = this.svg;
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", String(W));
    svg.setAttribute("height", String(H));

    const risk = this.data.risk;
    let maxV = 0;
    let maxT = 0;
    for (const [t, v] of risk) if (v > maxV) (maxV = v), (maxT = t);
    const alarms = this.alarms();
    svg.setAttribute(
      "aria-label",
      risk.length
        ? `Risk curve. Highest value ${maxV.toFixed(2)} at ${fmtTime(maxT)}. ${alarms.length} alarm${alarms.length === 1 ? "" : "s"} at threshold ${this.theta}.`
        : "No risk curve for this clip.",
    );

    // frame
    s("rect", { x: gutter, y: top, width: trackW, height: plotH, class: "risk-bg" }, svg);
    for (const v of [0, 0.5, 1]) {
      s("line", { x1: gutter, x2: gutter + trackW, y1: this.y(v), y2: this.y(v), class: v === this.theta ? "risk-theta" : "risk-grid" }, svg);
      s("text", { x: gutter - 8, y: this.y(v) + 3.5, "text-anchor": "end" }, svg).textContent = v.toFixed(1);
    }
    if (!narrow) {
      s("text", { x: 0, y: top + plotH / 2 + 4, class: "risk-title" }, svg).textContent = "Risk";
      s("text", { x: gutter + trackW - 4, y: this.y(this.theta) - 5, "text-anchor": "end", class: "risk-theta-lab" }, svg).textContent = `alarm at ${this.theta}`;
    }

    // axis
    const step = tickStep(this.data.duration, trackW, narrow ? 56 : 70);
    for (let t = 0; t <= this.data.duration + 1e-6; t += step) {
      const x = this.x(t);
      s("line", { x1: x, x2: x, y1: top + plotH, y2: top + plotH + 4, class: "risk-grid" }, svg);
      s("text", { x, y: top + plotH + 16, "text-anchor": t === 0 ? "start" : "middle" }, svg).textContent = fmtTime(t, 0);
    }

    if (risk.length) {
      // one max per pixel column so short peaks survive
      const cols = Math.max(1, Math.floor(trackW));
      const colMax = new Float32Array(cols).fill(-1);
      for (const [t, v] of risk) {
        const c = Math.min(cols - 1, Math.max(0, Math.floor((t / this.data.duration) * cols)));
        if (v > colMax[c]) colMax[c] = v;
      }
      let line = "";
      let first = -1;
      let last = -1;
      for (let c = 0; c < cols; c++) {
        if (colMax[c] < 0) continue;
        const X = (gutter + c + 0.5).toFixed(1);
        const Y = this.y(colMax[c]).toFixed(1);
        line += line ? ` L${X},${Y}` : `M${X},${Y}`;
        if (first < 0) first = c;
        last = c;
      }
      const base = this.y(0).toFixed(1);
      const area = `${line} L${(gutter + last + 0.5).toFixed(1)},${base} L${(gutter + first + 0.5).toFixed(1)},${base} Z`;
      const defs = s("defs", {}, svg);
      const cp = s("clipPath", { id: this.clipId }, defs);
      s("rect", { x: gutter, y: 0, width: trackW, height: this.y(this.theta) }, cp);
      s("path", { d: area, class: "risk-area" }, svg);
      s("path", { d: line, class: "risk-line" }, svg);
      const hot = s("g", { "clip-path": `url(#${this.clipId})` }, svg);
      s("path", { d: area, class: "risk-area hot" }, hot);
      s("path", { d: line, class: "risk-line hot" }, hot);
      for (const [a] of alarms) {
        const x = this.x(a);
        s("path", { d: `M${x - 4},${top} L${x + 4},${top} L${x},${top + 7} Z`, class: "risk-alarm" }, svg);
      }
    } else {
      s("text", { x: gutter + 10, y: top + plotH / 2, class: "risk-empty" }, svg).textContent = "No risk curve for this clip.";
    }

    this.hover = s("line", { y1: top, y2: top + plotH, class: "tl-hover", visibility: "hidden" }, svg);
    this.dot = s("circle", { r: 3.5, class: "risk-dot", visibility: "hidden" }, svg);
    this.playhead = s("line", { y1: top - 2, y2: top + plotH + 4, class: "tl-ph-line" }, svg);
    this.setTime(this.t);
  }

  private t = 0;
  setTime(t: number) {
    this.t = t;
    if (!this.playhead) return;
    const x = this.x(t).toFixed(1);
    this.playhead.setAttribute("x1", x);
    this.playhead.setAttribute("x2", x);
  }

  private tAt(clientX: number) {
    const r = this.svg.getBoundingClientRect();
    return Math.max(0, Math.min(this.data.duration, ((clientX - r.left - this.g.gutter) / this.g.trackW) * this.data.duration));
  }

  private bind() {
    const svg = this.svg;
    svg.addEventListener("pointerdown", (e) => {
      const r = svg.getBoundingClientRect();
      if (e.clientX - r.left < this.g.gutter - 4 || e.button !== 0) return;
      this.dragging = true;
      svg.setPointerCapture(e.pointerId);
      const t = this.tAt(e.clientX);
      this.setTime(t);
      this.onSeek(t);
      e.preventDefault();
    });
    svg.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const px = e.clientX - r.left;
      const t = this.tAt(e.clientX);
      if (this.dragging) {
        this.setTime(t);
        this.onSeek(t);
        return;
      }
      if (e.pointerType !== "mouse" || px < this.g.gutter || px > this.g.gutter + this.g.trackW) {
        this.hover?.setAttribute("visibility", "hidden");
        this.dot?.setAttribute("visibility", "hidden");
        this.tip.classList.remove("show");
        return;
      }
      const v = this.valueAt(t);
      this.hover?.setAttribute("x1", String(px));
      this.hover?.setAttribute("x2", String(px));
      this.hover?.setAttribute("visibility", "visible");
      if (Number.isFinite(v)) {
        this.dot?.setAttribute("cx", String(px));
        this.dot?.setAttribute("cy", String(this.y(v)));
        this.dot?.setAttribute("visibility", "visible");
      }
      this.tip.textContent = `${fmtTime(t)}  risk ${Number.isFinite(v) ? v.toFixed(3) : "-"}`;
      this.tip.style.left = `${Math.max(70, Math.min(this.g.W - 70, px))}px`;
      this.tip.style.top = `${this.g.top}px`;
      this.tip.classList.add("show");
    });
    const end = (e: PointerEvent) => {
      this.dragging = false;
      try {
        svg.releasePointerCapture(e.pointerId);
      } catch {
        /* not captured */
      }
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.addEventListener("pointerleave", () => {
      this.hover?.setAttribute("visibility", "hidden");
      this.dot?.setAttribute("visibility", "hidden");
      this.tip.classList.remove("show");
    });
  }
}
