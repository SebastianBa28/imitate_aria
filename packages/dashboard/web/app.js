// Station dashboard. Talks to deploy.station.server:
//   GET /api/config, POST /api/command, WS /ws (JSON snapshot/tick + binary JPEG: 1 byte channel, then the image)
import { TimeChart, cssColor, fmt } from "./chart.js";

const $ = (id) => document.getElementById(id);
const DEG = 180 / Math.PI;
const CHANNELS = ["color", "depth"];          // byte 0 of a binary message
const FRESH_S = 1.0;                           // markers older than this are drawn faded
// name -> [title, unit, smallest y range shown (keeps noise from filling the plot)]
const METRIC_NAMES = {
  latency_ms: ["Policy latency", "ms", 10],
  loop_hz: ["Loop rate", "Hz", 4],
  "obs_age_ms.hand": ["Hand obs. age", "ms", 20],
  "obs_age_ms.arm": ["Arm obs. age", "ms", 20],
  "obs_age_ms.color": ["Camera obs. age", "ms", 20],
  hand_clip_deg: ["Hand clipped", "deg", 2],
  arm_clip_mm: ["Arm clipped", "mm", 2],
};
const METRIC_ORDER = Object.keys(METRIC_NAMES);

const state = {
  config: null,
  series: new Map(),         // name -> {t: [], v: [[...]]}
  meta: {},
  policy: {},
  health: {},
  clockOffset: 0,            // server time - browser time (s)
  lastTick: 0,               // browser time of the last tick (ms)
  connected: false,
  windowS: 30,
  paused: false,
  pausedAt: null,
  joint: 0,
  cam: { view: "color", frames: { color: [], depth: [] }, urls: {} },
  charts: { metrics: new Map() },
  events: [],
};

const serverNow = () => Date.now() / 1000 + state.clockOffset;

// --- series store ---------------------------------------------------------------------------
function addSeries(batch) {
  const keep = state.config ? state.config.history_s : 60;
  for (const [name, { t, v }] of Object.entries(batch)) {
    let s = state.series.get(name);
    if (!s) { s = { t: [], v: [] }; state.series.set(name, s); }
    for (let i = 0; i < t.length; i++) { s.t.push(t[i]); s.v.push(v[i]); }
    const cutoff = t[t.length - 1] - keep;
    let drop = 0;
    while (drop < s.t.length && s.t[drop] < cutoff) drop++;
    if (drop > 0) { s.t.splice(0, drop); s.v.splice(0, drop); }
  }
}
const series = (name) => state.series.get(name);
function latest(name) {
  const s = series(name);
  if (!s || !s.t.length) return null;
  return { t: s.t[s.t.length - 1], v: s.v[s.v.length - 1] };
}

// --- commands -------------------------------------------------------------------------------
async function send(target, cmd) {
  const out = $("cmd-result");
  try {
    const res = await fetch("/api/command", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ target, cmd }),
    });
    const body = await res.json().catch(() => ({ ok: false, message: `HTTP ${res.status}` }));
    out.dataset.ok = String(!!body.ok);
    out.textContent = `${target} ${cmd}: ${body.message}`;
    return body;
  } catch (e) {
    out.dataset.ok = "false";
    out.textContent = `${target} ${cmd}: station unreachable (${e.message})`;
    return { ok: false };
  }
}

function estop() {
  send("all", "estop");
}

function setupControls() {
  $("estop").addEventListener("click", estop);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !e.repeat) { e.preventDefault(); estop(); }
  });
  document.querySelectorAll("[data-policy]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cmd = btn.dataset.policy;
      if (cmd === "live" && !confirm("Switch to LIVE?\n\nOn the next start, the policy's actions are sent to the robot and it moves.")) return;
      if ((cmd === "start" || cmd === "home") && state.policy.mode === "live" &&
          !confirm(`${cmd === "start" ? "Start the policy" : "Move to home"} in LIVE mode? The robot will move.`)) return;
      send("policy", cmd);
    });
  });
  document.querySelectorAll("[data-hand]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cmd = btn.dataset.hand;
      if (cmd === "estop" && !confirm("Hand E-stop releases torque and disconnects the hand. hand_node must be restarted to use it again. Continue?")) return;
      send("hand", cmd);
    });
  });
  document.querySelectorAll("[data-cam]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.cam.view = btn.dataset.cam;
      document.querySelector(".cams").dataset.view = state.cam.view;
      document.querySelectorAll("[data-cam]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      try { localStorage.setItem("cam-view", state.cam.view); } catch { /* storage unavailable */ }
    });
  });
  document.querySelectorAll("[data-window]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.windowS = Number(btn.dataset.window);
      document.querySelectorAll("[data-window]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    });
  });
  $("pause").addEventListener("click", () => {
    state.paused = !state.paused;
    state.pausedAt = state.paused ? serverNow() : null;
    $("pause").setAttribute("aria-pressed", String(state.paused));
    $("pause").textContent = state.paused ? "Resume" : "Pause";
  });
  $("events-clear").addEventListener("click", () => { state.events = []; renderEvents(); });
  try {
    const v = localStorage.getItem("cam-view");
    if (v) document.querySelector(`[data-cam="${v}"]`)?.click();
  } catch { /* storage unavailable */ }
}

// --- policy panel ---------------------------------------------------------------------------
const POLICY_TONE = { idle: "muted", running: "good", manual: "good", homing: "warning", fault: "critical" };

function renderPolicy() {
  const p = state.policy || {};
  const known = !!p.status && state.connected;
  const status = p.status || "unknown";
  const pill = $("policy-pill");
  pill.dataset.tone = known ? POLICY_TONE[status] || "muted" : "muted";
  pill.querySelector(".label").textContent = p.status ? `Policy: ${status}` : "Policy: no state yet";
  const mode = $("mode-pill");
  mode.dataset.mode = p.mode || "unknown";
  mode.textContent = p.mode === "live" ? "LIVE" : p.mode === "preview" ? "Preview (robot does not move)" : "Mode: –";
  $("live-banner").hidden = p.mode !== "live";

  const st = $("p-status");
  st.replaceChildren();
  const badge = document.createElement("span");
  badge.className = "status";
  badge.dataset.s = status;
  badge.textContent = p.mode ? `${status} · ${p.mode}` : status;
  st.appendChild(badge);
  $("p-steps").textContent = p.steps ?? "–";
  $("p-plugin").textContent = p.plugin || state.config?.policy.plugin || "–";
  $("p-ckpt").textContent = state.config?.policy.checkpoint || "none";
  $("p-home").textContent = p.home_configured === undefined ? "–" : p.home_configured ? "configured" : "not set (hand only)";

  const msg = $("p-message");
  msg.hidden = !p.message;
  msg.textContent = p.message || "";
  const refused = !!p.message && p.message.includes(": ") && !p.message.endsWith(": ok") && !p.message.startsWith("home:");
  msg.dataset.level = status === "fault" ? "error" : refused ? "warn" : "info";

  const idle = status === "idle", fault = status === "fault";
  const moving = status === "running" || status === "homing" || status === "manual";
  const enable = { start: idle || fault, home: idle || fault, stop: moving, reset: known, preview: idle, live: idle };
  document.querySelectorAll("[data-policy]").forEach((b) => {
    const cmd = b.dataset.policy;
    b.disabled = !known || !enable[cmd];
    if (cmd === "preview" || cmd === "live") b.setAttribute("aria-pressed", String(p.mode === cmd));
  });
  $("mode-hint").textContent = fault ? "Fault: press Reset before changing mode." :
    status === "manual" ? "Exit manual control to change mode." :
    moving ? "Stop the policy to change mode." : "Mode only changes while the policy is idle.";
  document.querySelectorAll("[data-hand]").forEach((b) => { b.disabled = !state.connected; });
}

// --- manual control -------------------------------------------------------------------------
const manual = { pending: {}, lastInput: 0, timer: null, jogStep: 0.01, wasActive: false };

async function sendManual(body) {
  try {
    const res = await fetch("/api/manual", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      $("cmd-result").dataset.ok = "false";
      $("cmd-result").textContent = `manual: ${b.message || res.status}`;
    }
  } catch (e) {
    $("cmd-result").dataset.ok = "false";
    $("cmd-result").textContent = `manual: station unreachable (${e.message})`;
  }
}

function flushSliders() {
  manual.timer = null;
  const targets = manual.pending;
  manual.pending = {};
  if (Object.keys(targets).length) sendManual({ hand: targets });
}

function buildManual() {
  const c = state.config;
  const box = $("sliders");
  manual.sliders = c.joint_names.map((name, j) => {
    const row = document.createElement("div");
    row.className = "slider";
    const id = `sl-${name}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = name;
    const input = document.createElement("input");
    input.type = "range";
    input.id = id;
    input.min = (c.joint_lower[j] * DEG).toFixed(1);
    input.max = (c.joint_upper[j] * DEG).toFixed(1);
    input.step = "0.5";
    input.value = (c.hand_neutral[j] * DEG).toFixed(1);
    const out = document.createElement("output");
    out.htmlFor = id;
    out.textContent = `${Number(input.value).toFixed(1)}°`;
    input.addEventListener("input", () => {
      out.textContent = `${Number(input.value).toFixed(1)}°`;
      manual.pending[name] = Number(input.value) / DEG;
      manual.lastInput = Date.now();
      if (!manual.timer) manual.timer = setTimeout(flushSliders, 33);  // at most ~30 messages/s
    });
    row.append(label, input, out);
    box.appendChild(row);
    return { input, out };
  });

  $("manual-enter").addEventListener("click", () => {
    if (state.policy.mode === "live" &&
        !confirm("Enter manual control in LIVE mode?\n\nSliders and jog buttons will move the robot.")) return;
    send("policy", "manual");
  });
  $("manual-exit").addEventListener("click", () => send("policy", "stop"));
  $("hand-neutral").addEventListener("click", () => sendManual({ hand_neutral: true }));
  document.querySelectorAll("[data-jog-step]").forEach((btn) => {
    btn.addEventListener("click", () => {
      manual.jogStep = Number(btn.dataset.jogStep);
      document.querySelectorAll("[data-jog-step]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    });
  });
  const maxJog = c.manual.max_jog_m ?? 0.05;
  document.querySelectorAll("[data-jog-step]").forEach((b) => { if (Number(b.dataset.jogStep) > maxJog) b.remove(); });
  document.querySelectorAll("[data-jog]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const j = btn.dataset.jog;
      if (j === "hold") { sendManual({ arm_hold: true }); return; }
      const d = [0, 0, 0];
      d["xyz".indexOf(j[0])] = (j[1] === "+" ? 1 : -1) * manual.jogStep;
      sendManual({ arm_jog: d });
    });
  });
}

function renderManual() {
  const p = state.policy || {};
  const active = p.status === "manual" && state.connected;
  const c = state.config;
  const card = $("manual-card");
  card.dataset.active = String(active);
  const st = $("manual-state");
  st.dataset.s = active ? "manual" : "idle";
  st.textContent = active ? (p.mode === "live" ? "on · LIVE: the robot moves" : "on · preview: the robot does not move") : "off";
  $("manual-enter").disabled = !(state.connected && (p.status === "idle" || p.status === "fault"));
  $("manual-exit").disabled = !active;
  const hz = c.policy.hz, m = c.manual || {};
  $("manual-hint").textContent = active
    ? `Hand: ${m.hand_max_step_deg == null ? "sliders go straight to the hand" : `up to ${(m.hand_max_step_deg * hz).toFixed(0)}°/s per joint`}. Arm: up to ${((m.arm_max_step_m ?? 0.002) * hz * 100).toFixed(1)} cm/s. Clamped to the joint ranges and the arm workspace; the arm keeps its orientation.`
    : p.status === "running" || p.status === "homing" ? "Stop the policy to use manual control."
    : "Targets start at the robot's current pose, so nothing jumps when you enter. Choose Preview or Live first: in Preview the robot does not move.";
  manual.sliders.forEach(({ input }) => { input.disabled = !active; });
  card.querySelectorAll("[data-jog], #hand-neutral").forEach((b) => {
    b.disabled = !active || (b.dataset.jog && !(p.uses_arm ?? c.policy.uses_arm));
  });

  const tgt = active ? p.manual : null;
  // follow the node's targets (entering manual, Neutral pose) unless the user is dragging
  if (tgt && tgt.hand && (!manual.wasActive || Date.now() - manual.lastInput > 600) && !manual.timer) {
    tgt.hand.forEach((rad, j) => {
      const { input, out } = manual.sliders[j];
      if (document.activeElement === input && Date.now() - manual.lastInput < 2000) return;
      input.value = (rad * DEG).toFixed(1);
      out.textContent = `${(rad * DEG).toFixed(1)}°`;
    });
  }
  manual.wasActive = active;

  const meas = latest("arm.pos");
  ["x", "y", "z"].forEach((ax, i) => {
    $(`jog-t-${ax}`).textContent = tgt && tgt.arm_pos ? fmt(tgt.arm_pos[i], 3) : "–";
    $(`jog-m-${ax}`).textContent = meas ? fmt(meas.v[i], 3) : "–";
  });
  const dist = tgt && tgt.arm_pos && meas ? Math.hypot(...tgt.arm_pos.map((v, i) => v - meas.v[i])) : null;
  $("jog-dist").textContent = dist === null ? "" :
    dist < 0.002 ? "Arm is at the target." :
    p.mode === "live" ? `Arm is ${(dist * 100).toFixed(1)} cm from the target and moving.` :
    `Target is ${(dist * 100).toFixed(1)} cm from the arm (preview: the arm does not follow).`;
}

// --- connection -----------------------------------------------------------------------------
function renderConn() {
  const el = $("conn");
  const stalled = state.connected && Date.now() - state.lastTick > 1500;
  el.dataset.tone = !state.connected ? "critical" : stalled ? "warning" : "good";
  el.querySelector(".label").textContent = !state.connected ? "Disconnected, retrying…" : stalled ? "Connected, no updates" : "Connected";
  document.body.classList.toggle("offline", !state.connected);
}

// --- health ---------------------------------------------------------------------------------
function renderHealth() {
  const body = $("health-body");
  const expected = state.config ? Object.keys(state.config.expected_hz) : [];
  const topics = Object.keys(state.health).sort((a, b) => {
    const ia = expected.indexOf(a), ib = expected.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const rows = topics.map((topic) => {
    const h = state.health[topic];
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    topic.split(/(?<=\/)/).forEach((part, i) => {  // allow line breaks only after a slash
      if (i) name.appendChild(document.createElement("wbr"));
      name.appendChild(document.createTextNode(part));
    });
    // command topics have no expected rate: silence is normal while nothing is commanding
    const status = !h.expected_hz && h.status === "missing" ? "quiet" : h.status;
    const st = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "status";
    badge.dataset.s = status === "quiet" ? "idle" : status;
    badge.textContent = status;
    st.appendChild(badge);
    const rate = document.createElement("td");
    rate.className = "r num";
    rate.textContent = h.hz === null ? "–" : h.expected_hz ? `${fmt(h.hz, 1)} / ${h.expected_hz} Hz` : `${fmt(h.hz, 1)} Hz`;
    const age = document.createElement("td");
    age.className = "r num";
    age.textContent = h.age_s === null ? "–" : h.age_s < 10 ? `${h.age_s.toFixed(2)} s` : `${Math.round(h.age_s)} s`;
    tr.title = h.expected_hz ? "Expected rate from station.yaml" : "Command topic: quiet while idle is normal";
    tr.append(name, st, rate, age);
    return tr;
  });
  body.replaceChildren(...rows);
}

// --- hand joints ----------------------------------------------------------------------------
function buildJoints() {
  const { joint_names: names, joint_lower: lo, joint_upper: hi } = state.config;
  const box = $("joints");
  state.jointEls = names.map((name, j) => {
    const row = document.createElement("button");
    row.className = "joint";
    row.setAttribute("role", "option");
    const lod = lo[j] * DEG, hid = hi[j] * DEG;
    row.title = `${name}: range ${lod.toFixed(0)}° to ${hid.toFixed(0)}°`;
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = name;
    const track = document.createElement("span");
    track.className = "track";
    const zero = document.createElement("span");
    zero.className = "zero";
    zero.style.left = `${(-lod / (hid - lod)) * 100}%`;
    const m = document.createElement("span"); m.className = "m";
    const c = document.createElement("span"); c.className = "c";
    const p = document.createElement("span"); p.className = "p";
    track.append(zero, c, m, p);
    const val = document.createElement("span");
    val.className = "val";
    val.textContent = "–";
    row.append(label, track, val);
    row.addEventListener("click", () => selectJoint(j));
    box.appendChild(row);
    return { row, m, c, p, val, lod, hid };
  });
  selectJoint(0);
}

function selectJoint(j) {
  state.joint = j;
  state.jointEls.forEach((e, k) => e.row.setAttribute("aria-selected", String(k === j)));
  const name = state.config.joint_names[j];
  $("joint-plot-title").textContent = `Joint ${name} (deg)`;
  state.charts.joint.setTitle(name);
  state.charts.joint.setSeries(jointSeries(j));
}

function placeMarker(el, rad, lod, hid, fresh) {
  if (rad === null || rad === undefined) { el.hidden = true; return; }
  el.hidden = false;
  const pct = Math.min(1, Math.max(0, (rad * DEG - lod) / (hid - lod)));
  el.style.left = `${pct * 100}%`;
  el.style.opacity = fresh ? 1 : 0.3;
}

function renderJoints() {
  if (!state.jointEls) return;
  const now = serverNow();
  const q = latest("hand.q"), c = latest("hand.cmd"), p = latest("policy.hand");
  state.jointEls.forEach((e, j) => {
    placeMarker(e.m, q?.v[j], e.lod, e.hid, q && now - q.t < FRESH_S);
    placeMarker(e.c, c?.v[j], e.lod, e.hid, c && now - c.t < FRESH_S);
    placeMarker(e.p, p?.v[j], e.lod, e.hid, p && now - p.t < FRESH_S);
    const v = q?.v[j];
    e.val.textContent = v === null || v === undefined ? "–" : `${(v * DEG).toFixed(1)}°`;
  });
}

// --- arm ------------------------------------------------------------------------------------
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

function marker(ctx, kind, x, y, color, fresh) {
  ctx.globalAlpha = fresh ? 1 : 0.35;
  const surface = cssColor("var(--surface)");
  ctx.lineWidth = 2;
  if (kind === "m") {
    ctx.fillStyle = color; ctx.strokeStyle = surface;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  } else if (kind === "c") {
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x - 7, y); ctx.lineTo(x + 7, y); ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7); ctx.stroke();
  } else {
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function renderArm() {
  const ws = state.config.arm.workspace;
  const now = serverNow();
  const pts = [["m", "arm.pos", "var(--s1)"], ["c", "arm.cmd", "var(--s2)"], ["p", "policy.arm", "var(--s3)"]]
    .map(([kind, name, color]) => ({ kind, color: cssColor(color), l: latest(name) }));
  const ink2 = cssColor("var(--ink-2)"), muted = cssColor("var(--muted)"), grid = cssColor("var(--grid)"), axis = cssColor("var(--axis)");
  const font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";

  // top view: +x (forward) points up the screen, +y to the left, robot base at the origin
  {
    const [ctx, w, h] = fitCanvas($("arm-xy"));
    const xr = [Math.min(0, ws.x[0]) - 0.05, ws.x[1] + 0.05];
    const yspan = Math.max(Math.abs(ws.y[0]), Math.abs(ws.y[1])) + 0.05;
    const yr = [-yspan, yspan];
    const s = Math.min((w - 20) / (yr[1] - yr[0]), (h - 28) / (xr[1] - xr[0]));
    const cx = w / 2, by = h - 18 - (0 - xr[0]) * s;
    const P = (x, y) => [cx - y * s, by - x * s];
    ctx.font = font;
    // 10 cm grid
    ctx.strokeStyle = grid; ctx.lineWidth = 1;
    for (let x = Math.ceil(xr[0] * 10) / 10; x <= xr[1]; x += 0.1) {
      const [x0, y0] = P(x, yr[0]), [x1] = P(x, yr[1]);
      ctx.beginPath(); ctx.moveTo(x1, Math.round(y0) + 0.5); ctx.lineTo(x0, Math.round(y0) + 0.5); ctx.stroke();
    }
    for (let y = Math.ceil(yr[0] * 10) / 10; y <= yr[1] + 1e-9; y += 0.1) {
      const [x0, y0] = P(xr[0], y), [, y1] = P(xr[1], y);
      ctx.beginPath(); ctx.moveTo(Math.round(x0) + 0.5, y0); ctx.lineTo(Math.round(x0) + 0.5, y1); ctx.stroke();
    }
    // workspace box
    const [ax, ay] = P(ws.x[1], ws.y[1]), [bx, byy] = P(ws.x[0], ws.y[0]);
    ctx.strokeStyle = axis; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.strokeRect(ax, ay, bx - ax, byy - ay);
    ctx.setLineDash([]);
    ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText("workspace", ax + 4, ay - 2);
    // base + axes
    const [ox, oy] = P(0, 0);
    ctx.fillStyle = ink2;
    ctx.beginPath(); ctx.arc(ox, oy, 5, 0, Math.PI * 2); ctx.fill();
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("base", ox, oy + 7);
    ctx.strokeStyle = ink2; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy - 26); ctx.moveTo(ox, oy); ctx.lineTo(ox - 26, oy); ctx.stroke();
    ctx.textBaseline = "bottom"; ctx.fillText("+x", ox, oy - 27);
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText("+y", ox - 28, oy);
    // measured trail, last 10 s
    const trail = series("arm.pos");
    if (trail && trail.t.length) {
      ctx.strokeStyle = pts[0].color; ctx.globalAlpha = 0.35; ctx.lineWidth = 2;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < trail.t.length; i++) {
        if (trail.t[i] < now - 10) continue;
        const v = trail.v[i];
        if (v[0] === null || v[1] === null) { pen = false; continue; }
        const [x, y] = P(v[0], v[1]);
        if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    for (const pt of [...pts].reverse()) {
      if (!pt.l || pt.l.v[0] === null) continue;
      const [x, y] = P(pt.l.v[0], pt.l.v[1]);
      marker(ctx, pt.kind, x, y, pt.color, now - pt.l.t < FRESH_S);
    }
    if (!pts[0].l) {
      ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(state.config.policy.uses_arm ? "no arm pose yet" : "arm not used by this policy", w / 2, h / 2);
    }
  }

  // side view: height inside the workspace z range
  {
    const [ctx, w, h] = fitCanvas($("arm-z"));
    const zr = [Math.min(0, ws.z[0]), ws.z[1] + 0.05];
    const Y = (z) => h - 18 - ((z - zr[0]) / (zr[1] - zr[0])) * (h - 36);
    const x = w / 2 + 6;
    ctx.font = font;
    ctx.strokeStyle = grid; ctx.lineWidth = 4; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x, Y(zr[0])); ctx.lineTo(x, Y(zr[1])); ctx.stroke();
    ctx.strokeStyle = axis; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.lineCap = "butt";
    ctx.strokeRect(x - 10, Y(ws.z[1]), 20, Y(ws.z[0]) - Y(ws.z[1]));
    ctx.setLineDash([]);
    ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const z of [ws.z[0], ws.z[1]]) ctx.fillText(z.toFixed(2), x - 13, Y(z));
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("z (m)", w / 2, h - 14);
    for (const pt of [...pts].reverse()) {
      if (!pt.l || pt.l.v[2] === null) continue;
      marker(ctx, pt.kind, x, Y(pt.l.v[2]), pt.color, now - pt.l.t < FRESH_S);
    }
  }

  for (const [key, name] of [["m", "arm.pos"], ["c", "arm.cmd"], ["p", "policy.arm"]]) {
    const l = latest(name);
    ["x", "y", "z"].forEach((ax, i) => { $(`arm-${key}-${ax}`).textContent = l ? fmt(l.v[i], 3) : "–"; });
  }
}

// --- charts ---------------------------------------------------------------------------------
function jointSeries(j) {
  return [
    { label: "Measured", color: "var(--s1)", get: () => series("hand.q"), col: j, scale: DEG },
    { label: "Commanded", color: "var(--s2)", get: () => series("hand.cmd"), col: j, scale: DEG },
    { label: "Policy target", color: "var(--s3)", get: () => series("policy.hand"), col: j, scale: DEG },
  ];
}

function buildCharts() {
  state.charts.joint = new TimeChart($("chart-joint"), { unit: "°", digits: 1, minSpan: 2, series: jointSeries(0) });
  state.charts.arm = ["x", "y", "z"].map((ax, i) => new TimeChart($(`chart-arm-${ax}`), {
    title: ax, unit: "m", digits: 3, minSpan: 0.01,
    series: [
      { label: "Measured", color: "var(--s1)", get: () => series("arm.pos"), col: i },
      { label: "Commanded", color: "var(--s2)", get: () => series("arm.cmd"), col: i },
      { label: "Policy target", color: "var(--s3)", get: () => series("policy.arm"), col: i },
    ],
  }));
}

function syncMetricCharts() {
  const names = Object.keys(state.meta).filter((k) => k.startsWith("metric.")).map((k) => k.slice(7));
  names.sort((a, b) => {
    const ia = METRIC_ORDER.indexOf(a), ib = METRIC_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const box = $("metrics");
  for (const name of names) {
    if (state.charts.metrics.has(name)) continue;
    const [title, unit, minSpan] = METRIC_NAMES[name] || [name.startsWith("policy.") ? `policy: ${name.slice(7)}` : name, "", 1];
    const el = document.createElement("div");
    el.className = "chart";
    const chart = new TimeChart(el, {
      title, unit, digits: 1, minSpan,
      series: [{ label: title, color: "var(--s1)", get: () => series(`metric.${name}`) }],
    });
    state.charts.metrics.set(name, { chart, el });
  }
  // keep DOM order stable and sorted
  const els = names.map((n) => state.charts.metrics.get(n).el);
  if (els.some((el, i) => box.children[i] !== el)) box.replaceChildren(...els);
  $("metrics-empty").hidden = names.length > 0;
}

function drawCharts() {
  const now = state.paused ? state.pausedAt : serverNow();
  const w = state.windowS;
  state.charts.joint.draw(now, w);
  state.charts.arm.forEach((c) => c.draw(now, w));
  state.charts.metrics.forEach(({ chart }) => chart.draw(now, w));
}

// --- camera ---------------------------------------------------------------------------------
function onFrame(buf) {
  const bytes = new Uint8Array(buf);
  const ch = CHANNELS[bytes[0]];
  if (!ch) return;
  const blob = new Blob([bytes.subarray(1)], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  const img = $(`cam-${ch}`);
  const old = state.cam.urls[ch];
  img.onload = () => { if (old) URL.revokeObjectURL(old); };
  img.src = url;
  state.cam.urls[ch] = url;
  const f = state.cam.frames[ch];
  f.push(performance.now());
  while (f.length && f[0] < performance.now() - 2000) f.shift();
}

function renderCameraRates() {
  for (const ch of CHANNELS) {
    const f = state.cam.frames[ch];
    while (f.length && f[0] < performance.now() - 2000) f.shift();
    const el = $(`cam-${ch}-rate`);
    if (!state.cam.urls[ch]) el.textContent = "no frames";
    else if (!f.length) el.textContent = "stale (no frame for 2 s)";
    else el.textContent = `${(f.length / 2).toFixed(0)} fps shown`;
  }
}

// --- events ---------------------------------------------------------------------------------
function renderEvents() {
  const list = $("events");
  const items = state.events.slice(-300).reverse().map((e) => {
    const li = document.createElement("li");
    li.dataset.level = e.level;
    const time = document.createElement("span");
    time.className = "time";
    time.textContent = new Date((e.t - state.clockOffset) * 1000).toLocaleTimeString([], { hour12: false });
    const lvl = document.createElement("span");
    lvl.className = "lvl";
    lvl.dataset.level = e.level;
    lvl.textContent = e.level;
    const src = document.createElement("span");
    src.className = "src";
    src.textContent = e.source;
    const msg = document.createElement("span");
    msg.className = "msg";
    msg.textContent = e.text;
    li.append(time, lvl, src, msg);
    return li;
  });
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "No events.";
    items.push(li);
  }
  list.replaceChildren(...items);
}

// --- websocket ------------------------------------------------------------------------------
let retryMs = 500;
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { retryMs = 500; };
  ws.onmessage = (msg) => {
    if (typeof msg.data !== "string") { onFrame(msg.data); return; }
    const m = JSON.parse(msg.data);
    if (m.type === "snapshot") {
      state.series.clear();
      state.meta = m.series_meta;
      addSeries(m.series);
      state.connected = true;
      state.lastTick = Date.now();
      syncMetricCharts();
      return;
    }
    if (m.type !== "tick") return;
    state.clockOffset = m.now - Date.now() / 1000;
    state.lastTick = Date.now();
    state.connected = true;
    state.meta = m.series_meta;
    addSeries(m.series);
    state.health = m.health;
    state.policy = m.policy || {};
    if (m.events.length) {
      state.events.push(...m.events);
      if (state.events.length > 500) state.events.splice(0, state.events.length - 500);
      renderEvents();
    }
    syncMetricCharts();
  };
  ws.onclose = () => {
    state.connected = false;
    renderConn();
    renderPolicy();
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 5000);
  };
  ws.onerror = () => ws.close();
}

// --- main loop ------------------------------------------------------------------------------
let lastDom = 0;
function frame(ts) {
  if (state.config) {
    if (ts - lastDom > 66) {  // ~15 Hz for DOM-heavy panels
      lastDom = ts;
      renderConn();
      renderPolicy();
      renderManual();
      renderJoints();
      renderHealth();
      renderCameraRates();
    }
    renderArm();
    drawCharts();
  }
  requestAnimationFrame(frame);
}

async function main() {
  setupControls();
  renderEvents();
  for (;;) {
    try {
      const res = await fetch("/api/config");
      state.config = await res.json();
      break;
    } catch {
      renderConn();
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  const c = state.config;
  $("policy-name").textContent = `${c.policy.plugin} @ ${c.policy.hz} Hz`;
  const [near, far] = c.camera.depth_range_mm;
  $("depth-near").textContent = `${(near / 1000).toFixed(2)} m`;
  $("depth-far").textContent = `${(far / 1000).toFixed(1)} m`;
  buildCharts();
  buildJoints();
  buildManual();
  connect();
  requestAnimationFrame(frame);
}

main();
