// Station dashboard. Talks to deploy.station.server:
//   GET /api/config, POST /api/command, POST /api/process, WS /ws (JSON snapshot/tick + binary JPEG: 1 byte channel, then the image)
import { TimeChart, cssColor, fmt } from "./chart.js";
import { SkeletonView, boxLines, chainFrames, dist, gridLines, pos, prepareChain, zAxis } from "./skeleton.js";

const $ = (id) => document.getElementById(id);
const DEG = 180 / Math.PI;
const CHANNELS = ["color", "depth", "oak_color", "oak_depth"];   // byte 0 of a binary message (server CHANNELS)
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
  cam: { view: "color", which: "d435", frames: Object.fromEntries(CHANNELS.map((c) => [c, []])), urls: {} },
  charts: { metrics: new Map() },
  events: [],
  procs: [],                 // supervisor status rows (see deploy/station/supervisor.py)
  logs: new Map(),           // process name -> [[id, t, text], ...]
  logSel: null,
  logDirty: true,
  procPending: new Set(),    // process names (or "all") with a request in flight
  robot: null,               // skeleton views, chains and derived-series cursors (buildRobot)
};
const LOG_KEEP = 500;

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
  document.querySelectorAll("[data-cam-which]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.cam.which = btn.dataset.camWhich;
      document.querySelector(".cams").dataset.which = state.cam.which;
      document.querySelectorAll("[data-cam-which]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      try { localStorage.setItem("cam-which", state.cam.which); } catch { /* storage unavailable */ }
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
    const w = localStorage.getItem("cam-which");
    if (w) document.querySelector(`[data-cam-which="${w}"]`)?.click();
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
    status === "manual" ? "Manual mode: press Stop to change mode." :
    moving ? "Stop the policy to change mode." : "Mode only changes while the policy is idle.";
  document.querySelectorAll("[data-hand]").forEach((b) => { b.disabled = !state.connected; });
}

// --- processes ------------------------------------------------------------------------------
const POLICY_ACTIVE = ["running", "manual", "homing"];

async function sendProcess(name, action) {
  const out = $("proc-result");
  const touchesPolicy = name === "policy" || name === "all";
  if (action !== "start" && touchesPolicy && POLICY_ACTIVE.includes(state.policy.status) &&
      !confirm(`The policy is ${state.policy.status}. Stopping policy_node ends it abruptly. Continue?`)) return;
  state.procPending.add(name);
  out.dataset.ok = "true";
  out.textContent = `${action} ${name}…`;
  const post = async (confirmed) => {
    const res = await fetch("/api/process", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, action, confirmed }),
    });
    return res.json().catch(() => ({ ok: false, message: `HTTP ${res.status}` }));
  };
  try {
    // the server decides what needs confirming (real hardware that moves) and answers 409 + the warning
    let body = await post(false);
    if (body.confirm) {
      if (!confirm(`${action[0].toUpperCase() + action.slice(1)} ${name} on the real robot?\n\n${body.confirm}`)) {
        out.dataset.ok = "true";
        out.textContent = `${action} ${name}: cancelled`;
        return;
      }
      body = await post(true);
    }
    out.dataset.ok = String(!!body.ok);
    out.textContent = body.message;
  } catch (e) {
    out.dataset.ok = "false";
    out.textContent = `${action} ${name}: station unreachable (${e.message})`;
  } finally {
    state.procPending.delete(name);
  }
}

function selectLog(name) {
  state.logSel = name;
  state.logDirty = true;
  for (const [n, el] of Object.entries(state.procEls)) el.row.setAttribute("aria-selected", String(n === name));
}

function buildProcesses() {
  const pc = state.config.processes;
  $("proc-profile").textContent = `(${pc.profile} profile, ROS domain ${pc.domain})`;
  $("proc-card").dataset.profile = pc.profile;
  $("proc-refusal").hidden = !pc.refusal;
  $("proc-refusal").textContent = pc.refusal || "";
  state.procEls = {};
  const rows = pc.components.map((c) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.innerHTML = `<span class="pname"></span><span class="plabel"></span>`;
    name.querySelector(".pname").textContent = c.title;
    name.querySelector(".plabel").textContent = c.label;
    const st = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "status";
    st.appendChild(badge);
    const detail = document.createElement("td");
    detail.className = "detail";
    const act = document.createElement("td");
    const box = document.createElement("div");
    box.className = "actions";
    const buttons = {};
    for (const a of c.managed === false ? [] : ["start", "stop", "restart"]) {
      const b = document.createElement("button");
      b.className = "small";
      b.textContent = a[0].toUpperCase() + a.slice(1);
      b.addEventListener("click", (e) => { e.stopPropagation(); sendProcess(c.name, a); });
      box.appendChild(b);
      buttons[a] = b;
    }
    if (c.managed === false) {
      box.innerHTML = `<span class="muted">runs on its own machine</span>`;
    }
    act.appendChild(box);
    row.append(name, st, detail, act);
    row.addEventListener("click", () => selectLog(c.name));
    state.procEls[c.name] = { row, badge, detail, buttons, comp: c };
    return row;
  });
  $("proc-body").replaceChildren(...rows);
  $("proc-start-all").addEventListener("click", () => sendProcess("all", "start"));
  $("proc-stop-all").addEventListener("click", () => sendProcess("all", "stop"));
  const first = pc.components.find((c) => c.managed !== false) || pc.components[0];
  if (first) selectLog(first.name);
}

function duration(s) {
  if (s < 60) return `${Math.floor(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.floor(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}
const clock = (t) => new Date((t - state.clockOffset) * 1000).toLocaleTimeString([], { hour12: false });

function procView(p, c) {
  if (!state.connected || !p) return { s: "unknown", label: "unknown", detail: "" };
  if (p.managed === false) {
    return p.external ? { s: "running", label: "live", detail: `${p.topic} is live` }
      : { s: "missing", label: "no data", detail: `nothing on ${p.topic}: start ${c ? c.label : "it"}` };
  }
  if (c && c.missing && c.missing.length && p.state !== "running" && !p.external) {
    return { s: "fault", label: "unavailable", detail: `needs ${c.missing.join(", ")}: use the hw env` };
  }
  if (p.state === "running") return { s: "running", label: "running", detail: `pid ${p.pid} · up ${duration(p.uptime_s)}` };
  if (p.state === "stopping") return { s: "stopping", label: "stopping…", detail: `pid ${p.pid}` };
  if (p.external) return { s: "external", label: "external", detail: `${p.topic} is live` };
  if (p.state === "exited") {
    return { s: p.returncode ? "fault" : "idle", label: p.returncode ? `crashed (code ${p.returncode})` : "exited (code 0)",
             detail: p.ended_at ? `at ${clock(p.ended_at)}` : "" };
  }
  return { s: "stopped", label: "stopped", detail: p.ended_at ? `since ${clock(p.ended_at)}` : "" };
}

function renderProcesses() {
  if (!state.procEls) return;
  const refused = !!state.config.processes.refusal;
  const byName = Object.fromEntries(state.procs.map((p) => [p.name, p]));
  const allBusy = state.procPending.has("all");
  let canStart = false, canStop = false;
  for (const [name, el] of Object.entries(state.procEls)) {
    const p = byName[name];
    const v = procView(p, el.comp);
    el.badge.dataset.s = v.s;
    el.badge.textContent = v.label;
    el.detail.textContent = v.detail;
    if (el.comp.managed === false) continue;
    const missing = !!(el.comp.missing && el.comp.missing.length);
    const busy = allBusy || state.procPending.has(name) || !state.connected || !p;
    const running = p && p.state === "running";
    const startable = p && !running && p.state !== "stopping" && !p.external && !refused && !missing;
    el.buttons.start.disabled = busy || !startable;
    el.buttons.stop.disabled = busy || !running;
    el.buttons.restart.disabled = busy || !running || refused;
    canStart ||= !!startable;
    canStop ||= !!running;
  }
  $("proc-start-all").disabled = allBusy || !state.connected || !canStart;
  $("proc-stop-all").disabled = allBusy || !state.connected || !canStop;
  renderLog();
}

function addLogs(batch) {
  for (const [name, lines] of Object.entries(batch || {})) {
    let l = state.logs.get(name);
    if (!l) { l = []; state.logs.set(name, l); }
    l.push(...lines);
    if (l.length > LOG_KEEP) l.splice(0, l.length - LOG_KEEP);
    if (name === state.logSel) state.logDirty = true;
  }
}

function renderLog() {
  if (!state.logDirty) return;
  state.logDirty = false;
  const pre = $("proc-log");
  const c = state.config.processes.components.find((x) => x.name === state.logSel);
  $("proc-log-h").textContent = c ? `Log: ${c.title}` : "Log";
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
  const lines = state.logs.get(state.logSel) || [];
  pre.textContent = lines.length ? lines.map(([, t, text]) => `${clock(t)}  ${text}`).join("\n")
    : "No output yet. Only processes started from this dashboard have a log here.";
  if (atBottom) pre.scrollTop = pre.scrollHeight;   // follow, unless scrolled up to read
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

// --- charts ---------------------------------------------------------------------------------
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
  state.charts.metrics.forEach(({ chart }) => chart.draw(now, w));
}


// --- robot skeletons ------------------------------------------------------------------------
const FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
const TRAIL_S = 3;          // end-effector trails
const LAG_SEARCH_S = 3;     // how far back the lag estimate looks for the command matching the arm's position
const PREVIEW_MAX_AGE_S = 1;

// last sample at or before t (binary search), or null
function sampleAt(s, t) {
  if (!s || !s.t.length) return null;
  let lo = 0, hi = s.t.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (s.t[mid] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans < 0 ? null : { t: s.t[ans], v: s.v[ans] };
}

function trail(name, sinceT) {
  const s = series(name);
  if (!s) return [];
  const out = [];
  for (let i = s.t.length - 1; i >= 0 && s.t[i] >= sinceT; i--) if (s.v[i][0] !== null) out.push(s.v[i]);
  return out.reverse();
}

const finite = (v) => v && v.every((x) => x !== null && Number.isFinite(x));

function armFrames(q) {
  return q ? chainFrames(state.robot.armChain, q) : null;
}

// fingertip positions (5 x [x,y,z]) and the polyline of each finger, palm frame; null if q is incomplete
function handPose(q) {
  if (!q) return null;
  const fingers = {};
  for (const f of FINGERS) {
    const frames = chainFrames(state.robot.fingers[f], q);
    if (!frames) return null;
    fingers[f] = frames.slice(0, -1).map(pos);   // joint locations, ending at the fixed fingertip frame
  }
  return { fingers, tips: FINGERS.map((f) => fingers[f][fingers[f].length - 1]) };
}

function pushDerived(name, t, v) {
  let s = state.series.get(name);
  if (!s) { s = { t: [], v: [] }; state.series.set(name, s); }
  s.t.push(t); s.v.push([v]);
  const keep = state.config ? state.config.history_s : 60;
  let drop = 0;
  while (drop < s.t.length && s.t[drop] < t - keep) drop++;
  if (drop) { s.t.splice(0, drop); s.v.splice(0, drop); }
}

// tracking errors as series, sampled at each measurement against the command in force at that moment
function updateDerived() {
  const r = state.robot;
  if (!r) return;
  const meas = series("arm.pos"), cmd = series("arm.cmd");
  if (meas && cmd) {
    for (let i = 0; i < meas.t.length; i++) {
      if (meas.t[i] <= r.cursor.arm) continue;
      const c = sampleAt(cmd, meas.t[i]);
      if (c && finite(c.v) && finite(meas.v[i])) pushDerived("derived.arm_err", meas.t[i], dist(c.v, meas.v[i]));
      r.cursor.arm = meas.t[i];
    }
  }
  const hq = series("hand.q"), hc = series("hand.cmd");
  if (hq && hc) {
    let lastCmd = null, lastPose = null;
    for (let i = 0; i < hq.t.length; i++) {
      if (hq.t[i] <= r.cursor.hand) continue;
      r.cursor.hand = hq.t[i];
      const c = sampleAt(hc, hq.t[i]);
      if (!c) continue;
      if (!lastCmd || c.t !== lastCmd.t) { lastCmd = c; lastPose = handPose(c.v); }   // FK once per command
      const a = handPose(hq.v[i]);
      if (!a || !lastPose) continue;
      pushDerived("derived.hand_err", hq.t[i], a.tips.reduce((sum, p, k) => sum + dist(p, lastPose.tips[k]), 0) / 5);
    }
  }
}

function resetDerived() {
  if (!state.robot) return;
  state.robot.cursor = { arm: -Infinity, hand: -Infinity };
  state.series.delete("derived.arm_err");
  state.series.delete("derived.hand_err");
}

// how long ago the command was where the arm is now; null if not moving or no match
function armLag(measured, tNow) {
  const cmd = series("arm.cmd");
  if (!cmd || !measured) return null;
  let best = null, bestD = Infinity, moved = 0, prev = null;
  for (let i = cmd.t.length - 1; i >= 0 && cmd.t[i] >= tNow - LAG_SEARCH_S; i--) {
    const v = cmd.v[i];
    if (!finite(v)) continue;
    if (prev) moved += dist(prev, v);
    prev = v;
    const d = dist(v, measured);
    if (d < bestD) { bestD = d; best = cmd.t[i]; }
  }
  if (best === null || moved < 0.003) return null;             // commands not moving: nothing to lag behind
  if (bestD > 0.003) return { behind: true };                     // further behind than the search window or off-path
  return { s: Math.max(0, tNow - best) };
}

function buildRobot() {
  const k = state.config.kinematics;
  if (!k) return;
  const fingers = Object.fromEntries(FINGERS.map((f) => [f, prepareChain(k.hand.fingers[f])]));
  state.robot = { armChain: prepareChain(k.arm.chain), fingers, cursor: { arm: -Infinity, hand: -Infinity } };

  // hand bounds: open (zero), neutral, closed (upper limits) and halfway poses, plus a margin. Not the lower
  // limits: they bend every finger backwards at once, which would leave the usual poses small in the view.
  const lo = state.config.joint_lower, hi = state.config.joint_upper;
  const pts = [];
  for (const q of [hi, state.config.hand_neutral, lo.map((x, i) => (x + hi[i]) / 2), lo.map(() => 0)]) {
    const p = handPose(q);
    if (p) for (const f of FINGERS) pts.push(...p.fingers[f]);
  }
  pts.push([0, 0, 0]);
  const hb = [0, 1, 2].map((a) => {
    const v = pts.map((p) => p[a]);
    const mn = Math.min(...v), mx = Math.max(...v), m = (mx - mn) * 0.08 + 0.005;
    return [mn - m, mx + m];
  });
  state.robot.hand = new SkeletonView($("hand-view"), { bounds: hb, yaw: Math.PI + 0.3, pitch: 0.15 });

  const ws = state.config.arm.workspace;
  const ab = [[Math.min(-0.1, ws.x[0]), Math.max(0.75, ws.x[1])], [Math.min(-0.4, ws.y[0]), Math.max(0.4, ws.y[1])],
              [0, Math.max(0.85, ws.z[1])]];
  state.robot.armBounds = ab;
  state.robot.arm = new SkeletonView($("arm-view"), { bounds: ab, yaw: -0.55, pitch: 0.35 });
  state.robot.grid = gridLines([-0.2, ab[0][1]], ab[1], 0.1);
  state.robot.workspace = boxLines(ws);

  state.robot.armChart = new TimeChart($("arm-err-chart"), {
    title: "Cmd − measured", unit: "mm", digits: 1, minSpan: 5,
    series: [{ label: "cmd − measured", color: "var(--s2)", scale: 1000, get: () => series("derived.arm_err") }],
  });
  state.robot.handChart = new TimeChart($("hand-err-chart"), {
    title: "Fingertip error (mean)", unit: "mm", digits: 1, minSpan: 5,
    series: [{ label: "fingertip error", color: "var(--s2)", scale: 1000, get: () => series("derived.hand_err") }],
  });
}

const ageText = (t) => {
  if (t === null || t === undefined) return "never";
  const a = serverNow() - t;
  return a < 1 ? `${Math.round(a * 1000)} ms ago` : a < 120 ? `${a.toFixed(1)} s ago` : `${Math.round(a / 60)} min ago`;
};
const lastT = (name) => latest(name)?.t ?? null;

function drawArm() {
  const r = state.robot;
  const q = latest("arm.q"), m = latest("arm.pos"), c = latest("arm.cmd"), p = latest("policy.arm");
  const now = serverNow();
  const previewOn = state.policy.mode === "preview" && p && now - p.t < PREVIEW_MAX_AGE_S;
  const key = [q?.t, m?.t, c?.t, previewOn && p.t, state.connected, Math.floor(now)].join();
  if (!r.arm.needs(key)) return;

  const lines = [
    ...r.grid.map((pts) => ({ pts, color: "var(--grid)", width: 1 })),
    ...r.workspace.map((pts) => ({ pts, color: "var(--axis)", width: 1, dash: [4, 4] })),
  ];
  const dots = [{ p: [0, 0, 0], color: "var(--muted)", r: 4, label: "base", labelColor: "var(--muted)" }];
  const frames = armFrames(q && q.v);
  let ee = null;
  if (frames) {
    const pts = [];
    for (const T of frames) {
      const pt = pos(T);
      if (!pts.length || dist(pts[pts.length - 1], pt) > 1e-6) pts.push(pt);
    }
    lines.push({ pts, color: "var(--s1)", width: 5, alpha: 0.85 });
    for (const pt of pts.slice(1, -1)) dots.push({ p: pt, color: "var(--surface)", r: 2.5 });
    const T = frames[frames.length - 1];
    ee = pos(T);
    const z = zAxis(T);
    lines.push({ pts: [ee, [ee[0] + 0.06 * z[0], ee[1] + 0.06 * z[1], ee[2] + 0.06 * z[2]]], color: "var(--s1)", width: 2 });
  } else if (m && finite(m.v)) {
    ee = m.v;   // no joint angles: the end effector alone
  }
  lines.push({ pts: trail("arm.pos", now - TRAIL_S), color: "var(--s1)", width: 1.5, alpha: 0.6 });
  if (c && finite(c.v)) {
    lines.push({ pts: trail("arm.cmd", now - TRAIL_S), color: "var(--s2)", width: 1.5, alpha: 0.7, dash: [3, 3] });
    if (ee) lines.push({ pts: [ee, c.v], color: "var(--s2)", width: 1.5, dash: [2, 3] });
  }
  if (ee) dots.push({ p: ee, color: "var(--s1)", r: 5, label: frames ? "end effector" : "end effector (no joint data)", labelDy: 10 });
  if (c && finite(c.v)) dots.push({ p: c.v, color: "var(--s2)", r: 6, ring: true, label: "commanded", labelDy: -10 });
  if (previewOn && finite(p.v)) dots.push({ p: p.v, color: "var(--s3)", r: 6, ring: true, label: "policy", labelDy: -10 });
  r.arm.draw({ lines, dots }, key);

  // readouts
  const measured = m && finite(m.v) ? m.v : ee;
  $("arm-err").textContent = c && finite(c.v) && measured ? `${(dist(c.v, measured) * 1000).toFixed(1)} mm` : "–";
  const lag = c && measured ? armLag(measured, m ? m.t : now) : null;
  $("arm-lag").textContent = !c ? "–" : !lag ? "settled" : lag.behind ? "> 3 s / off path" : `${Math.round(lag.s * 1000)} ms`;
  $("arm-cmd-age").textContent = ageText(c?.t);
  $("arm-hint").textContent = !q && !m ? "No arm data: is the arm controller (or the simulated arm) running?"
    : !q ? `No joint angles on ${state.config.kinematics.arm.topic}: showing the end effector only.`
    : !c ? "No commands yet: they are sent in Live (policy) or manual mode. Drag to rotate, double-click to reset."
    : "Drag to rotate, double-click to reset.";
}

function drawHand() {
  const r = state.robot;
  const a = latest("hand.q"), c = latest("hand.cmd"), p = latest("policy.hand");
  const now = serverNow();
  const previewOn = state.policy.mode === "preview" && p && now - p.t < PREVIEW_MAX_AGE_S;
  const key = [a?.t, c?.t, previewOn && p.t, state.connected, Math.floor(now)].join();
  if (!r.hand.needs(key)) return;

  const lines = [], dots = [];
  const add = (pose, color, width, alpha, dash, tipLabels) => {
    if (!pose) return;
    const f = pose.fingers;
    const palm = [[0, 0, 0], f.thumb[0], f.index[0], f.middle[0], f.ring[0], f.pinky[0], [0, 0, 0]];
    lines.push({ pts: palm, color, width: Math.max(1, width - 1.5), alpha: alpha * 0.7, dash });
    for (const name of FINGERS) {
      lines.push({ pts: f[name], color, width, alpha, dash });
      const tip = f[name][f[name].length - 1];
      dots.push({ p: tip, color, r: width > 2.5 ? 3.5 : 2.5, label: tipLabels && name === "thumb" ? "thumb" : "",
                  labelColor: "var(--muted)" });
    }
  };
  const cmdPose = c ? handPose(c.v) : null, achieved = a ? handPose(a.v) : null;
  add(previewOn ? handPose(p.v) : null, "var(--s3)", 2, 0.9, [4, 3], false);
  add(cmdPose, "var(--s2)", 2, 0.9, [], false);
  add(achieved, "var(--s1)", 3.5, 1, [], true);
  r.hand.draw({ lines, dots }, key);

  if (achieved && cmdPose) {
    const d = achieved.tips.map((t, k) => dist(t, cmdPose.tips[k]));
    $("hand-err").textContent = `${(d.reduce((x, y) => x + y, 0) / 5 * 1000).toFixed(1)} mm (max ${(Math.max(...d) * 1000).toFixed(1)})`;
    let worst = -1, worstErr = 0;
    a.v.forEach((x, i) => {
      const y = c.v[i];
      if (x !== null && y !== null && Math.abs(x - y) > worstErr) { worstErr = Math.abs(x - y); worst = i; }
    });
    $("hand-jerr").textContent = worst < 0 ? "–" : `${(worstErr * DEG).toFixed(1)}° ${state.config.joint_names[worst]}`;
  } else {
    $("hand-err").textContent = "–";
    $("hand-jerr").textContent = "–";
  }
  $("hand-cmd-age").textContent = ageText(c?.t);
}

function drawRobot() {
  if (!state.robot) return;
  drawArm();
  drawHand();
  const now = state.paused ? state.pausedAt : serverNow();
  state.robot.armChart.draw(now, 10);
  state.robot.handChart.draw(now, 10);
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
      resetDerived();
      updateDerived();
      state.connected = true;
      state.lastTick = Date.now();
      state.procs = m.processes || [];
      state.logs.clear();
      addLogs(m.logs);
      state.logDirty = true;
      syncMetricCharts();
      return;
    }
    if (m.type !== "tick") return;
    state.clockOffset = m.now - Date.now() / 1000;
    state.lastTick = Date.now();
    state.connected = true;
    state.meta = m.series_meta;
    addSeries(m.series);
    updateDerived();
    state.health = m.health;
    state.policy = m.policy || {};
    state.procs = m.processes || [];
    addLogs(m.logs);
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
      renderProcesses();
      renderPolicy();
      renderHealth();
      renderCameraRates();
    }
    drawCharts();
    drawRobot();
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
  document.querySelectorAll(".depth-near").forEach((el) => { el.textContent = `${(near / 1000).toFixed(2)} m`; });
  document.querySelectorAll(".depth-far").forEach((el) => { el.textContent = `${(far / 1000).toFixed(1)} m`; });
  if (!c.oak) {   // no OAK in this profile (IMITATE_STATION_OAK=off): D435 only
    $("cam-which").hidden = true;
    document.querySelector(".cams").dataset.which = "d435";
  }
  buildProcesses();
  buildRobot();
  connect();
  requestAnimationFrame(frame);
}

main();
