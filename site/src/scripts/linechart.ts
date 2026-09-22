// Small multi-series line chart: hover crosshair with values, legend toggles.
import { clear, niceMax, s, tickStep } from "./svg";

export interface Series {
  key: string;
  label: string;
  values: number[];
  color: string;
  dash?: string;
  width?: number;
}

export interface LineChartOpts {
  x: number[];
  series: Series[];
  height?: number;
  xFormat?: (v: number) => string;
  xTickStep?: number;
  yLabel?: string;
  ariaLabel: string;
  legend?: boolean;
  area?: string; // key of a series to fill under
}

export class LineChart {
  private host: HTMLElement;
  private svg: SVGSVGElement;
  private tip: HTMLDivElement;
  private legendEl: HTMLUListElement | null = null;
  private opts: LineChartOpts;
  private hidden = new Set<string>();
  private g = { W: 0, H: 0, l: 38, r: 10, t: 12, b: 26, yMax: 1 };
  private cross?: SVGLineElement;

  constructor(host: HTMLElement, opts: LineChartOpts) {
    this.host = host;
    this.opts = opts;
    host.classList.add("chart");
    this.svg = s("svg", { role: "img", "aria-label": opts.ariaLabel });
    this.tip = document.createElement("div");
    this.tip.className = "tip";
    this.tip.setAttribute("aria-hidden", "true");
    host.append(this.svg, this.tip);
    if (opts.legend !== false) {
      this.legendEl = document.createElement("ul");
      this.legendEl.className = "legend";
      host.append(this.legendEl);
    }
    this.bind();
    new ResizeObserver(() => {
      if (Math.abs(host.clientWidth - this.g.W) > 1) this.render();
    }).observe(host);
    this.render();
  }

  setData(x: number[], series: Series[], ariaLabel?: string) {
    this.opts = { ...this.opts, x, series, ariaLabel: ariaLabel ?? this.opts.ariaLabel };
    this.svg.setAttribute("aria-label", this.opts.ariaLabel);
    this.render();
  }

  private px(v: number) {
    const { x } = this.opts;
    const x0 = x[0] ?? 0;
    const x1 = x[x.length - 1] ?? 1;
    return this.g.l + ((v - x0) / (x1 - x0 || 1)) * (this.g.W - this.g.l - this.g.r);
  }

  private py(v: number) {
    return this.g.t + (1 - v / this.g.yMax) * (this.g.H - this.g.t - this.g.b);
  }

  render() {
    const W = Math.max(260, Math.floor(this.host.clientWidth));
    const H = this.opts.height ?? (W < 560 ? 200 : 260);
    const vis = this.opts.series.filter((se) => !this.hidden.has(se.key));
    let max = 0;
    for (const se of vis) for (const v of se.values) if (v > max) max = v;
    this.g = { ...this.g, W, H, yMax: niceMax(max * 1.05 || 1) };
    const svg = this.svg;
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", String(W));
    svg.setAttribute("height", String(H));
    const { l, r, t, b, yMax } = this.g;
    const x = this.opts.x;

    // y grid: 4 or 5 steps, whichever gives round numbers
    const nY = Number.isInteger(yMax / 4) ? 4 : Number.isInteger(yMax / 5) ? 5 : 4;
    for (let i = 0; i <= nY; i++) {
      const v = (yMax / nY) * i;
      const y = this.py(v);
      s("line", { x1: l, x2: W - r, y1: y, y2: y, class: "c-grid" }, svg);
      s("text", { x: l - 6, y: y + 3.5, "text-anchor": "end" }, svg).textContent = Number.isInteger(v) ? String(v) : v.toFixed(1);
    }
    if (this.opts.yLabel) s("text", { x: l, y: t - 2, class: "c-ylab" }, svg).textContent = this.opts.yLabel;
    // x ticks
    if (x.length > 1) {
      const span = x[x.length - 1] - x[0];
      let step = this.opts.xTickStep ?? tickStep(span, W - l - r, 64);
      while ((step / (span || 1)) * (W - l - r) < 58) step *= 2;
      const fmt = this.opts.xFormat ?? ((v: number) => String(v));
      for (let v = Math.ceil(x[0] / step) * step; v <= x[x.length - 1] + 1e-9; v += step) {
        const X = this.px(v);
        s("line", { x1: X, x2: X, y1: H - b, y2: H - b + 4, class: "c-grid" }, svg);
        s("text", { x: X, y: H - b + 16, "text-anchor": "middle" }, svg).textContent = fmt(v);
      }
    }
    s("line", { x1: l, x2: W - r, y1: H - b, y2: H - b, class: "c-axis" }, svg);

    for (const se of vis) {
      let d = "";
      se.values.forEach((v, i) => {
        if (!Number.isFinite(v) || x[i] === undefined) return;
        d += `${d ? " L" : "M"}${this.px(x[i]).toFixed(1)},${this.py(v).toFixed(1)}`;
      });
      if (this.opts.area === se.key && d) {
        const first = this.px(x[0]).toFixed(1);
        const last = this.px(x[Math.min(x.length, se.values.length) - 1]).toFixed(1);
        s("path", { d: `${d} L${last},${this.py(0)} L${first},${this.py(0)} Z`, fill: se.color, opacity: 0.12 }, svg);
      }
      s("path", { d, fill: "none", stroke: se.color, "stroke-width": se.width ?? 1.8, "stroke-dasharray": se.dash, "stroke-linejoin": "round" }, svg);
    }
    this.cross = s("line", { y1: t, y2: H - b, class: "c-cross", visibility: "hidden" }, svg);
    this.renderLegend();
  }

  private renderLegend() {
    if (!this.legendEl) return;
    this.legendEl.textContent = "";
    for (const se of this.opts.series) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.setAttribute("aria-pressed", String(!this.hidden.has(se.key)));
      btn.innerHTML = `<span class="sw" style="background:${se.color}"></span>${se.label}`;
      btn.addEventListener("click", () => {
        if (this.hidden.has(se.key)) this.hidden.delete(se.key);
        else if (this.hidden.size < this.opts.series.length - 1) this.hidden.add(se.key);
        this.render();
        (this.legendEl?.querySelectorAll("button")[this.opts.series.indexOf(se)] as HTMLButtonElement | undefined)?.focus();
      });
      li.append(btn);
      this.legendEl.append(li);
    }
  }

  private bind() {
    const svg = this.svg;
    const move = (clientX: number) => {
      const x = this.opts.x;
      if (x.length < 2) return;
      const rect = svg.getBoundingClientRect();
      const px = clientX - rect.left;
      if (px < this.g.l || px > this.g.W - this.g.r) return this.hide();
      let best = 0;
      for (let i = 1; i < x.length; i++) if (Math.abs(this.px(x[i]) - px) < Math.abs(this.px(x[best]) - px)) best = i;
      const X = this.px(x[best]);
      this.cross?.setAttribute("x1", String(X));
      this.cross?.setAttribute("x2", String(X));
      this.cross?.setAttribute("visibility", "visible");
      const fmt = this.opts.xFormat ?? ((v: number) => String(v));
      const rows = this.opts.series
        .filter((se) => !this.hidden.has(se.key))
        .map((se) => `<span style="color:${se.color}">${se.label}</span> ${Number.isFinite(se.values[best]) ? se.values[best] : "-"}`);
      this.tip.innerHTML = `${fmt(x[best])}<br>${rows.join("<br>")}`;
      this.tip.style.left = `${Math.max(60, Math.min(this.g.W - 60, X))}px`;
      this.tip.style.top = `${this.g.t + 4}px`;
      this.tip.style.transform = "translate(-50%, 0)";
      this.tip.classList.add("show");
    };
    svg.addEventListener("pointermove", (e) => move(e.clientX));
    svg.addEventListener("pointerdown", (e) => move(e.clientX));
    svg.addEventListener("pointerleave", () => this.hide());
  }

  private hide() {
    this.cross?.setAttribute("visibility", "hidden");
    this.tip.classList.remove("show");
  }
}
