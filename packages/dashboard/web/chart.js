// Small canvas time-series chart: one y-axis, 2px lines, gaps for missing values,
// a crosshair that snaps to the nearest sample and a tooltip listing every series.

const tooltip = () => document.getElementById("tooltip");

export function cssColor(value) {
  // "var(--s1)" -> resolved colour; anything else is returned as is
  const m = /^var\((--[\w-]+)\)$/.exec(value);
  return m ? getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim() : value;
}

export function fmt(v, digits = 2) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(Math.min(digits, 1));
  return v.toFixed(digits);
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

// index of the last sample with t <= x (binary search), -1 if none
function lastAtOrBefore(ts, x) {
  let lo = 0, hi = ts.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= x) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

export class TimeChart {
  /**
   * opts.series: [{label, color, get: () => ({t: [...], v: [[...]]} | undefined), col, scale}]
   * opts.title, opts.unit, opts.digits, opts.minSpan (smallest y range shown)
   */
  constructor(container, opts) {
    this.el = container;
    this.opts = { digits: 2, minSpan: 1e-3, ...opts };
    this.canvas = document.createElement("canvas");
    this.el.appendChild(this.canvas);
    this.titleEl = document.createElement("div");
    this.titleEl.className = "chart-title";
    this.titleText = document.createElement("span");
    this.nowText = document.createElement("span");
    this.nowText.className = "now";
    this.titleEl.append(this.titleText, this.nowText);
    this.el.appendChild(this.titleEl);
    this.setTitle(this.opts.title || "");
    this.hoverX = null;
    this.pad = { l: 44, r: 8, t: 18, b: 18 };
    this.canvas.addEventListener("pointermove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hoverX = e.clientX - r.left;
      this.hoverClient = [e.clientX, e.clientY];
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.hoverX = null;
      tooltip().hidden = true;
    });
  }

  setTitle(text) { this.titleText.textContent = text; }
  setSeries(series) { this.opts.series = series; }

  // value of series s at sample i, in display units (null if missing)
  val(s, data, i) {
    const row = data.v[i];
    const x = row ? row[s.col || 0] : null;
    return x === null || x === undefined ? null : x * (s.scale || 1);
  }

  draw(now, windowS) {
    const { canvas } = this;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const { l, r, t, b } = this.pad;
    const pw = w - l - r, ph = h - t - b;
    const t0 = now - windowS;
    const series = this.opts.series || [];

    // visible y range
    let lo = Infinity, hi = -Infinity;
    const datas = series.map((s) => s.get());
    series.forEach((s, k) => {
      const d = datas[k];
      if (!d || !d.t.length) return;
      for (let i = Math.max(0, lastAtOrBefore(d.t, t0)); i < d.t.length; i++) {
        const y = this.val(s, d, i);
        if (y === null) continue;
        if (y < lo) lo = y;
        if (y > hi) hi = y;
      }
    });
    const empty = !Number.isFinite(lo);
    if (empty) { lo = 0; hi = 1; }
    if (hi - lo < this.opts.minSpan) {
      const mid = (hi + lo) / 2;
      lo = mid - this.opts.minSpan / 2;
      hi = mid + this.opts.minSpan / 2;
    }
    const step = niceStep(hi - lo, 3);
    lo = Math.floor(lo / step) * step;
    hi = Math.ceil(hi / step) * step;
    if (hi === lo) hi = lo + step;

    const X = (tt) => l + ((tt - t0) / windowS) * pw;
    const Y = (v) => t + (1 - (v - lo) / (hi - lo)) * ph;

    const grid = cssColor("var(--grid)"), axis = cssColor("var(--axis)"), muted = cssColor("var(--muted)");
    ctx.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.lineWidth = 1;
    // y grid + labels
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const digits = Math.max(0, -Math.floor(Math.log10(step)));
    for (let v = lo; v <= hi + step / 2; v += step) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(w - r, y); ctx.stroke();
      ctx.fillText(v.toFixed(Math.min(digits, 4)), l - 6, y);
    }
    // x labels: seconds ago
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xs = windowS <= 10 ? 2 : windowS <= 30 ? 10 : 20;
    for (let s = 0; s <= windowS; s += xs) {
      const x = X(now - s);
      ctx.fillText(s === 0 ? "now" : `-${s}s`, Math.min(Math.max(x, l + 10), w - r - 12), h - b + 4);
    }
    ctx.strokeStyle = axis;
    ctx.beginPath(); ctx.moveTo(l, t + ph + 0.5); ctx.lineTo(w - r, t + ph + 0.5); ctx.stroke();

    if (empty) {
      ctx.fillStyle = muted;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("no data", l + pw / 2, t + ph / 2);
    }

    // lines (later series drawn underneath so measured stays on top)
    ctx.save();
    ctx.beginPath(); ctx.rect(l, t - 2, pw, ph + 4); ctx.clip();
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (let k = series.length - 1; k >= 0; k--) {
      const s = series[k], d = datas[k];
      if (!d || !d.t.length) continue;
      ctx.strokeStyle = cssColor(s.color);
      ctx.beginPath();
      let pen = false;
      for (let i = Math.max(0, lastAtOrBefore(d.t, t0)); i < d.t.length; i++) {
        const y = this.val(s, d, i);
        if (y === null) { pen = false; continue; }
        const px = X(d.t[i]), py = Y(y);
        if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
      }
      ctx.stroke();
    }
    ctx.restore();

    // current value of the first series in the title
    const d0 = datas[0];
    const s0 = series[0];
    const last = d0 && d0.t.length ? this.val(s0, d0, d0.t.length - 1) : null;
    this.nowText.textContent = last === null ? "" : `${fmt(last, this.opts.digits)}${this.opts.unit ? " " + this.opts.unit : ""}`;

    this.drawHover(ctx, { X, Y, t0, now, l, r, t, ph, w, windowS, datas, series });
  }

  drawHover(ctx, g) {
    const tip = tooltip();
    if (this.hoverX === null || this.hoverX < g.l || this.hoverX > g.w - g.r) {
      if (this.hoverX !== null) tip.hidden = true;
      return;
    }
    const tt = g.t0 + ((this.hoverX - g.l) / (g.w - g.l - g.r)) * g.windowS;
    const rows = [];
    let snapT = null;
    g.series.forEach((s, k) => {
      const d = g.datas[k];
      if (!d || !d.t.length) { rows.push({ s, v: null }); return; }
      let i = lastAtOrBefore(d.t, tt);
      if (i < d.t.length - 1 && (i < 0 || Math.abs(d.t[i + 1] - tt) < Math.abs(d.t[i] - tt))) i += 1;
      if (i < 0 || Math.abs(d.t[i] - tt) > 0.5) { rows.push({ s, v: null }); return; }
      const v = this.val(s, d, i);
      if (k === 0 || snapT === null) snapT = d.t[i];
      rows.push({ s, v, ti: d.t[i] });
    });
    const x = Math.round(g.X(snapT ?? tt)) + 0.5;
    ctx.strokeStyle = cssColor("var(--axis)");
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, g.t); ctx.lineTo(x, g.t + g.ph); ctx.stroke();
    const surface = cssColor("var(--surface)");
    rows.forEach(({ s, v, ti }) => {
      if (v === null) return;
      ctx.fillStyle = cssColor(s.color);
      ctx.strokeStyle = surface;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(g.X(ti), g.Y(v), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    });

    tip.replaceChildren();
    const head = document.createElement("div");
    head.className = "t";
    head.textContent = `${this.titleText.textContent} · ${(g.now - (snapT ?? tt)).toFixed(1)} s ago`;
    tip.appendChild(head);
    rows.forEach(({ s, v }) => {
      const row = document.createElement("div");
      row.className = "row";
      const key = document.createElement("i");
      key.className = "k";
      key.style.background = cssColor(s.color);
      const val = document.createElement("b");
      val.textContent = `${fmt(v, this.opts.digits)}${this.opts.unit && v !== null ? " " + this.opts.unit : ""}`;
      const lab = document.createElement("span");
      lab.textContent = s.label;
      row.append(key, val, lab);
      tip.appendChild(row);
    });
    tip.hidden = false;
    const [cx, cy] = this.hoverClient;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = `${Math.min(cx + 14, window.innerWidth - tw - 8)}px`;
    tip.style.top = `${Math.max(8, cy - th - 10)}px`;
  }
}
