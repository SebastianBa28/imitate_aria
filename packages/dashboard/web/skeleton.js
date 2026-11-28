// Robot skeletons: forward kinematics from the URDF chains in /api/config (kinematics) and a small
// orthographic 3D view on a 2D canvas. Drag to rotate, double-click to reset. No libraries, no WebGL:
// a few dozen line segments, redrawn only when the data or the view changes.
import { cssColor } from "./chart.js";

// --- kinematics -----------------------------------------------------------------------------
// 4x4 matrices as flat row-major arrays of 16
const mul = (a, b) => {
  const o = new Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    o[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
  }
  return o;
};
const flat = (m) => m.flat();
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function axisAngle([x, y, z], a) {
  const c = Math.cos(a), s = Math.sin(a), C = 1 - c;
  return [
    c + x * x * C, x * y * C - z * s, x * z * C + y * s, 0,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s, 0,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C, 0,
    0, 0, 0, 1,
  ];
}

/** Prepare a chain from /api/config ([{type, origin (4x4 rows), axis, q}]) once. */
export function prepareChain(chain) {
  return chain.map((j) => ({ ...j, origin: flat(j.origin) }));
}

/**
 * Frames along a chain: T_base_x (flat 4x4) at each joint (after its origin, before its rotation, i.e.
 * where the joint sits), plus the last frame after the final joint. q: joint vector indexed by j.q.
 * Returns null if a needed joint value is missing.
 */
export function chainFrames(chain, q) {
  let T = IDENTITY;
  const out = [];
  for (const j of chain) {
    T = mul(T, j.origin);
    out.push(T);
    if (j.type === "revolute") {
      const v = q[j.q];
      if (v === null || v === undefined || !Number.isFinite(v)) return null;
      T = mul(T, axisAngle(j.axis, v));
    }
  }
  out.push(T);
  return out;
}

export const pos = (T) => [T[3], T[7], T[11]];
export const zAxis = (T) => [T[2], T[6], T[10]];

// --- view -----------------------------------------------------------------------------------
export class SkeletonView {
  /**
   * opts.bounds: [[xmin, xmax], [ymin, ymax], [zmin, zmax]] kept in view (fixed, so the picture doesn't jump)
   * opts.yaw, opts.pitch: default view angles (rad). z is drawn up.
   */
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.opts = opts;
    this.yaw = opts.yaw;
    this.pitch = opts.pitch;
    this.dirty = true;
    this.lastKey = null;
    let drag = null;
    canvas.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.yaw = drag.yaw - (e.clientX - drag.x) * 0.01;
      this.pitch = Math.max(-1.5, Math.min(1.5, drag.pitch + (e.clientY - drag.y) * 0.01));
      this.dirty = true;
    });
    const end = () => { drag = null; };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("dblclick", () => { this.yaw = opts.yaw; this.pitch = opts.pitch; this.dirty = true; });
    new ResizeObserver(() => { this.dirty = true; }).observe(canvas);
  }

  /** True if the scene must be redrawn for this data key (cheap check before building a scene). */
  needs(key) {
    return this.dirty || key !== this.lastKey;
  }

  // view-plane coordinates (right, up) in metres, before scaling
  rotate(p) {
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
    const x1 = c * p[0] + s * p[1], y1 = -s * p[0] + c * p[1];   // yaw about z; y1 points away from the viewer
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [x1, -sp * y1 + cp * p[2]];                           // pitch: look down onto the scene
  }

  project(p) {
    const [u, v] = this.rotate(p);
    return [this.w / 2 + (u - this.mid[0]) * this.scale, this.h / 2 - (v - this.mid[1]) * this.scale];
  }

  // fit the bounds box, as seen from the current angle, into the canvas
  fit() {
    const b = this.opts.bounds;
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (let i = 0; i < 8; i++) {
      const [u, v] = this.rotate([b[0][i & 1], b[1][(i >> 1) & 1], b[2][(i >> 2) & 1]]);
      u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
    }
    this.mid = [(u0 + u1) / 2, (v0 + v1) / 2];
    this.scale = 0.94 * Math.min(this.w / (u1 - u0), this.h / (v1 - v0));
  }

  /**
   * scene: {
   *   lines: [{pts: [[x,y,z]...], color, width, dash, alpha}],
   *   dots: [{p, color, r, ring, label, labelColor}],
   * }
   */
  draw(scene, key) {
    const { canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    this.w = w; this.h = h;
    this.fit();

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (const l of scene.lines) {
      if (!l.pts || l.pts.length < 2) continue;
      ctx.globalAlpha = l.alpha ?? 1;
      ctx.strokeStyle = cssColor(l.color);
      ctx.lineWidth = l.width || 1.5;
      ctx.setLineDash(l.dash || []);
      ctx.beginPath();
      l.pts.forEach((p, i) => {
        const [x, y] = this.project(p);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.font = "12px system-ui, sans-serif";
    for (const d of scene.dots) {
      if (!d.p) continue;
      const [x, y] = this.project(d.p);
      ctx.beginPath();
      ctx.arc(x, y, d.r || 3, 0, 2 * Math.PI);
      if (d.ring) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = cssColor(d.color);
        ctx.stroke();
      } else {
        ctx.fillStyle = cssColor(d.color);
        ctx.fill();
      }
      if (d.label) {
        ctx.fillStyle = cssColor(d.labelColor || "var(--ink)");
        ctx.textBaseline = "middle";
        const right = x < w - 110;
        ctx.textAlign = right ? "left" : "right";
        ctx.fillText(d.label, x + (right ? 1 : -1) * ((d.r || 3) + 5), y - (d.labelDy || 0));
      }
    }
    this.dirty = false;
    this.lastKey = key;
  }
}

// --- scene helpers --------------------------------------------------------------------------
/** The 12 edges of an axis-aligned box as polylines (for the workspace). */
export function boxLines({ x, y, z }) {
  const P = (i, j, k) => [x[i], y[j], z[k]];
  return [
    [P(0, 0, 0), P(1, 0, 0), P(1, 1, 0), P(0, 1, 0), P(0, 0, 0)],
    [P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1), P(0, 0, 1)],
    [P(0, 0, 0), P(0, 0, 1)], [P(1, 0, 0), P(1, 0, 1)], [P(1, 1, 0), P(1, 1, 1)], [P(0, 1, 0), P(0, 1, 1)],
  ];
}

/** Floor grid lines on z = 0. */
export function gridLines([x0, x1], [y0, y1], step) {
  const out = [];
  for (let x = Math.ceil(x0 / step) * step; x <= x1 + 1e-9; x += step) out.push([[x, y0, 0], [x, y1, 0]]);
  for (let y = Math.ceil(y0 / step) * step; y <= y1 + 1e-9; y += step) out.push([[x0, y, 0], [x1, y, 0]]);
  return out;
}

export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
