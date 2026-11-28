"""Dashboard server: one process with the ROS bridge (background thread) and a FastAPI app.

    python -m deploy.station.server            # http://127.0.0.1:8000 (station.host/port in station.yaml)
    pixi run station / pixi run -e hw station-real / station-real-arm   (profile via IMITATE_STATION_*)

API
  GET  /api/config     joint names, series list, expected rates
  POST /api/command    {"target": "policy", "cmd": start|stop|reset|home|live|preview}
                       {"target": "hand", "cmd": estop|torque_on|torque_off}
                       {"target": "all", "cmd": "estop"}   policy stop + hand e-stop
  POST /api/manual     manual targets while the policy node is in "manual" (deploy/manual.py)
  POST /api/process    {"name": arm|hand|camera|policy|all, "action": start|stop|restart, "confirmed": bool}
                       starting a component with a `confirm` warning (the real hand) needs confirmed=true,
                       else 409 {"ok": false, "confirm": "<warning>"}
  GET  /api/processes  state of the processes the dashboard can start (deploy/station/supervisor.py)
  WS   /ws             text JSON: {"type": "snapshot"|"tick", ...}; binary: 1 byte channel (0 colour,
                       1 depth, 2 OAK colour, 3 OAK depth) + JPEG

Processes: the station starts and stops one profile's nodes itself (deploy.components; IMITATE_STATION_PROFILE
= mock (default) | real, IMITATE_STATION_HAND / _CAMERA = mock|real and _OAK = mock|real|off override it). They stop with the station.
A node whose topic is already live (e.g. started by `pixi run mock`) shows as "external" and is not started
twice. In the real profile the arm (Franka controller on the NUC) is shown, never started. The mock profile
is refused on the robot's domain; hardware nodes are refused when their driver module is missing (default env).
"""
import asyncio
import contextlib
import os
import threading
import time
from pathlib import Path

import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

from deploy import config as cfg
from deploy.components import LIVE_AGE_S, components, domain_refusal, is_live, monitored, station_selection
from deploy.station.bridge import EventLog, StationBridge
from deploy.station.buffers import SeriesStore
from deploy.station.camera import DEPTH_RANGE_MM, CameraEncoder
from deploy.station.monitor import HealthBoard
from deploy.station.supervisor import Spec, Supervisor
from imitate_common import contract, hand
from imitate_common.arm_kinematics import Panda
from imitate_common.orca_kinematics import FINGERS, OrcaHand
from imitate_common.urdf import chain_json

CONF = cfg.load()
SC = CONF["station"]
WEB_DIR = cfg.DEPLOY_ROOT.parent / "dashboard" / "web"   # static page, no build step
ROBOT_ASSETS = cfg.DEPLOY_ROOT / "assets" / "robot"
CHANNELS = {"color": 0, "depth": 1, "oak_color": 2, "oak_depth": 3}   # byte 0 of a binary WS message
SNAPSHOT_POINTS, TICK_POINTS = 1500, 60
PROFILE, HAND, CAMERA, OAK = station_selection(os.environ)   # processes the dashboard can start
DOMAIN = os.environ.get("ROS_DOMAIN_ID", "0")
COMPONENTS = {c.name: c for c in components(PROFILE, hand=HAND, camera=CAMERA, oak=OAK)}
MONITORED = {c.name: c for c in monitored(PROFILE)}       # shown, never started (the NUC's arm controller)
LOG_SNAPSHOT_LINES, LOG_TICK_LINES = 300, 200
# health lights: drop the OAK's topics when it is switched off (IMITATE_STATION_OAK=off), not "missing" forever
EXPECTED_HZ = {t: hz for t, hz in SC["expected_hz"].items()
               if "oak" in COMPONENTS or t not in (contract.OAK_COLOR, contract.OAK_DEPTH)}


def kinematics():
    """Joint chains for the dashboard's skeleton views (FK runs in the browser, web/skeleton.js)."""
    orca = OrcaHand()
    fingers = {}
    for f in FINGERS:
        joints = chain_json(orca.chain[f])
        for j in joints:
            j["q"] = orca.joint_index.get(j["name"], -1) if j["type"] == "revolute" else -1
        fingers[f] = joints
    arm = chain_json(Panda().chain)   # revolute joints are contract.ARM_JOINT_NAMES, in order (checked by Panda)
    k = 0
    for j in arm:
        j["q"] = k if j["type"] == "revolute" else -1
        k += j["type"] == "revolute"
    return {"arm": {"chain": arm, "joint_names": list(contract.ARM_JOINT_NAMES), "frame": contract.ARM_FRAME,
                    "topic": contract.ARM_JOINT_STATE},
            "hand": {"fingers": fingers, "frame": "palm"}}


KINEMATICS = kinematics()


class State:
    store = SeriesStore(SC["history_s"])
    health = HealthBoard(EXPECTED_HZ)
    events = EventLog()
    bridge = None
    camera = None
    executor = None
    supervisor = None


S = State()


def process_guard(name):
    """Why the dashboard must not start this process now, or None."""
    refusal = domain_refusal(PROFILE, DOMAIN)
    if refusal:
        return refusal
    c = COMPONENTS[name]
    missing = c.missing_modules()
    if missing:
        return (f"{c.label} needs {', '.join(missing)}, which this environment lacks: start the station in the "
                "hw env (`pixi run -e hw station-real`)")
    if is_live(S.health.snapshot().get(c.topic)) and not recently_ours(name):
        return f"{c.topic} is already published: {c.label} runs outside the dashboard (stop it there first)"
    return None


def recently_ours(name):
    """Our own process ended so recently that its last messages still make its topic look live."""
    ended = S.supervisor.procs[name].ended_at
    return ended is not None and time.time() - ended < LIVE_AGE_S + 0.5


def process_status():
    health = S.health.snapshot()
    out = S.supervisor.status()
    for p in out:
        c = COMPONENTS[p["name"]]
        ours = p["state"] in ("running", "stopping") or recently_ours(p["name"])
        p["external"] = not ours and is_live(health.get(c.topic))
        p["topic"] = c.topic
        p["managed"] = True
    for c in MONITORED.values():
        out.append({"name": c.name, "title": c.title, "state": "stopped", "managed": False, "pid": None,
                    "returncode": None, "started_at": None, "ended_at": None, "uptime_s": None,
                    "external": is_live(health.get(c.topic)), "topic": c.topic})
    return out


def needs_confirmation(name, action):
    """Warnings the user must accept before this start/restart, joined, or "" (stops never need one)."""
    if action not in ("start", "restart"):
        return ""
    names = list(COMPONENTS) if name == "all" else [name]
    running = {p["name"] for p in S.supervisor.status() if p["state"] in ("running", "stopping")}
    if action == "start":
        names = [n for n in names if n not in running]
    return " ".join(dict.fromkeys(COMPONENTS[n].confirm for n in names if n in COMPONENTS and COMPONENTS[n].confirm))


def start_supervisor():
    for c in [*COMPONENTS.values(), *MONITORED.values()]:
        S.health.watch(c.topic)   # the guard needs them even if station.yaml doesn't list them
    specs = {c.name: Spec(c.argv(), title=c.title, cwd=str(cfg.DEPLOY_ROOT)) for c in COMPONENTS.values()}
    S.supervisor = Supervisor(specs, guard=process_guard,
                              on_event=lambda level, text: S.events.add(level, "processes", text))


def start_ros():
    rclpy.init()
    S.bridge = StationBridge(S.store, S.health, S.events)
    S.camera = CameraEncoder(S.bridge, quality=SC["jpeg_quality"])
    # single-threaded on purpose: rclpy's MultiThreadedExecutor drops best-effort camera frames
    # (measured 1-8 Hz vs 30 Hz). Service calls block in worker threads, not in this executor.
    S.executor = SingleThreadedExecutor()
    S.executor.add_node(S.bridge)
    threading.Thread(target=spin, daemon=True).start()
    S.events.add("info", "station", "station started")


def spin():
    try:
        S.executor.spin()
    except ExternalShutdownException:
        pass   # Ctrl-C: rclpy's own signal handler shut ROS down; the lifespan exit does the rest


def stop_ros():
    if S.executor:
        S.executor.shutdown(timeout_sec=2.0)
    if S.bridge:
        S.bridge.destroy_node()
    rclpy.try_shutdown()


@contextlib.asynccontextmanager
async def lifespan(app):
    start_ros()
    start_supervisor()
    yield
    ok, msg = await asyncio.get_running_loop().run_in_executor(None, S.supervisor.shutdown)
    print(f"station: {msg}", flush=True)
    stop_ros()


app = FastAPI(title="imitate_aria station", lifespan=lifespan)


@app.get("/api/config")
def get_config():
    return {
        "joint_names": list(hand.JOINT_NAMES),
        "joint_lower": hand.LOWER_RAD.tolist(),
        "joint_upper": hand.UPPER_RAD.tolist(),
        "series": S.store.describe(),
        "expected_hz": EXPECTED_HZ,
        "history_s": SC["history_s"],
        "stream_hz": SC["stream_hz"],
        "policy": {"plugin": CONF["policy"]["plugin"], "hz": CONF["policy"]["hz"], "checkpoint": CONF["policy"].get("checkpoint"),
                   "uses_arm": CONF["policy"].get("uses_arm", True)},
        "arm": {"workspace": CONF["arm"]["workspace"], "home": CONF["arm"].get("home")},
        "manual": CONF.get("manual", {}),
        "hand_neutral": hand.NEUTRAL_RAD.tolist(),
        "camera": {"width": CONF["camera"]["width"], "height": CONF["camera"]["height"], "fps": CONF["camera"]["fps"],
                   "stream_hz": SC["camera_hz"], "depth_range_mm": list(DEPTH_RANGE_MM)},
        "oak": ({"width": CONF["oak"]["width"], "height": CONF["oak"]["height"], "fps": CONF["oak"]["fps"]}
                if "oak" in COMPONENTS else None),
        "channels": list(CHANNELS),
        "kinematics": KINEMATICS,
        "robot_assets": ROBOT_ASSETS.exists(),
        "processes": {"profile": PROFILE, "domain": DOMAIN, "refusal": domain_refusal(PROFILE, DOMAIN),
                      "components": [{"name": c.name, "title": c.title, "label": c.label, "topic": c.topic,
                                      "managed": c.managed, "confirm": c.confirm, "missing": c.missing_modules()}
                                     for c in [*MONITORED.values(), *COMPONENTS.values()]]},
    }


class Command(BaseModel):
    target: str
    cmd: str


@app.post("/api/command")
async def command(c: Command):
    loop = asyncio.get_running_loop()
    b = S.bridge
    try:
        if c.target == "policy":
            ok, msg = b.send_policy(c.cmd)
        elif c.target == "hand" and c.cmd == "estop":
            ok, msg = await loop.run_in_executor(None, b.call_setbool, b.estop_client, True)
        elif c.target == "hand" and c.cmd in ("torque_on", "torque_off"):
            ok, msg = await loop.run_in_executor(None, b.call_setbool, b.torque_client, c.cmd == "torque_on")
        elif c.target == "all" and c.cmd == "estop":
            ok1, m1 = b.send_policy("stop")
            ok2, m2 = await loop.run_in_executor(None, b.call_setbool, b.estop_client, True)
            ok, msg = ok1 and ok2, f"policy: {m1}; hand: {m2}"
        else:
            return JSONResponse({"ok": False, "message": f"unknown command {c.target}/{c.cmd}"}, status_code=400)
    except ValueError as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=400)
    S.events.add("info" if ok else "error", "dashboard", f"{c.target} {c.cmd}: {msg}")
    return {"ok": ok, "message": msg}


class ProcessCommand(BaseModel):
    name: str
    action: str
    confirmed: bool = False   # the user accepted the components' `confirm` warnings


@app.get("/api/processes")
def get_processes():
    return process_status()


@app.post("/api/process")
async def process_command(c: ProcessCommand):
    sv = S.supervisor
    if c.name in MONITORED:
        return JSONResponse({"ok": False, "message": f"{MONITORED[c.name].title} runs on its own machine "
                             f"({MONITORED[c.name].label}); the dashboard only shows it"}, status_code=400)
    warning = needs_confirmation(c.name, c.action) if c.name == "all" or c.name in COMPONENTS else ""
    if warning and not c.confirmed:
        return JSONResponse({"ok": False, "confirm": warning,
                             "message": f"{c.action} {c.name} needs confirmation: {warning}"}, status_code=409)
    if c.name == "all":
        fn = {"start": sv.start_all, "stop": sv.stop_all}.get(c.action)
    else:
        fn = {"start": sv.start, "stop": sv.stop, "restart": sv.restart}.get(c.action)
        fn = fn and (lambda f=fn: f(c.name))
    if fn is None:
        return JSONResponse({"ok": False, "message": f"unknown action {c.name}/{c.action}"}, status_code=400)
    try:
        ok, msg = await asyncio.get_running_loop().run_in_executor(None, fn)   # stop can take up to 10 s
    except ValueError as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=400)
    if not ok:
        S.events.add("warn", "dashboard", f"{c.action} {c.name}: {msg}")
    return {"ok": ok, "message": msg}


@app.post("/api/manual")
def manual(targets: dict):
    """Manual targets while the policy node is in "manual" (format: deploy/manual.py). Not logged per call."""
    ok, msg = S.bridge.send_manual(targets)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    loop = asyncio.get_running_loop()
    cursors, log_cursors, last_event = {}, {}, -1
    sent_seq = {ch: None for ch in CHANNELS}
    period, cam_period = 1.0 / SC["stream_hz"], 1.0 / SC["camera_hz"]
    try:
        await socket.send_json({"type": "snapshot", "series_meta": S.store.describe(),
                                "series": S.store.since(cursors, SNAPSHOT_POINTS),
                                "processes": process_status(),
                                "logs": S.supervisor.logs_since(log_cursors, LOG_SNAPSHOT_LINES)})
        next_cam = 0.0
        while True:
            t0 = time.monotonic()
            events = S.events.since(last_event)
            if events:
                last_event = events[-1]["id"]
            await socket.send_json({
                "type": "tick", "now": time.time(),
                "series": S.store.since(cursors, TICK_POINTS),
                "series_meta": S.store.describe(),
                "health": S.health.snapshot(),
                "policy": S.bridge.policy_state,
                "events": events,
                "processes": process_status(),
                "logs": S.supervisor.logs_since(log_cursors, LOG_TICK_LINES),
            })
            if t0 >= next_cam:
                next_cam = t0 + cam_period
                for ch, code in CHANNELS.items():
                    seq, jpg = await loop.run_in_executor(None, S.camera.jpeg, ch)
                    if seq is not None and seq != sent_seq[ch]:
                        sent_seq[ch] = seq
                        await socket.send_bytes(bytes([code]) + jpg)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away


if ROBOT_ASSETS.exists():
    app.mount("/robot", StaticFiles(directory=ROBOT_ASSETS), name="robot")
if (WEB_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    @app.get("/")
    def no_page():
        return HTMLResponse(f"<h1>Dashboard page not found</h1><p>Expected {WEB_DIR / 'index.html'}. "
                            "The API is up: <a href='/api/config'>/api/config</a>.</p>")


def main():
    uvicorn.run(app, host=SC["host"], port=SC["port"], log_level="warning")


if __name__ == "__main__":
    main()
