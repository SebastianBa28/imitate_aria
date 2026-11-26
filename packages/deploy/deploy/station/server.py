"""Dashboard server: one process with the ROS bridge (background thread) and a FastAPI app.

    python -m deploy.station.server            # http://127.0.0.1:8000 (station.host/port in station.yaml)

API
  GET  /api/config     joint names, series list, expected rates
  POST /api/command    {"target": "policy", "cmd": start|stop|reset|home|live|preview}
                       {"target": "hand", "cmd": estop|torque_on|torque_off}
                       {"target": "all", "cmd": "estop"}   policy stop + hand e-stop
  WS   /ws             text JSON: {"type": "snapshot"|"tick", ...}; binary: 1 byte channel (0 colour,
                       1 depth) + JPEG
"""
import asyncio
import contextlib
import threading
import time
from pathlib import Path

import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rclpy.executors import SingleThreadedExecutor

from deploy import config as cfg
from deploy.station.bridge import EventLog, StationBridge
from deploy.station.buffers import SeriesStore
from deploy.station.camera import DEPTH_RANGE_MM, CameraEncoder
from deploy.station.monitor import HealthBoard
from imitate_common import hand

CONF = cfg.load()
SC = CONF["station"]
WEB_DIR = cfg.DEPLOY_ROOT.parent / "dashboard" / "web"   # static page, no build step
ROBOT_ASSETS = cfg.DEPLOY_ROOT / "assets" / "robot"
CHANNELS = {"color": 0, "depth": 1}
SNAPSHOT_POINTS, TICK_POINTS = 1500, 60


class State:
    store = SeriesStore(SC["history_s"])
    health = HealthBoard(SC["expected_hz"])
    events = EventLog()
    bridge = None
    camera = None
    executor = None


S = State()


def start_ros():
    rclpy.init()
    S.bridge = StationBridge(S.store, S.health, S.events)
    S.camera = CameraEncoder(S.bridge, quality=SC["jpeg_quality"])
    # single-threaded on purpose: rclpy's MultiThreadedExecutor drops best-effort camera frames
    # (measured 1-8 Hz vs 30 Hz). Service calls block in worker threads, not in this executor.
    S.executor = SingleThreadedExecutor()
    S.executor.add_node(S.bridge)
    threading.Thread(target=S.executor.spin, daemon=True).start()
    S.events.add("info", "station", "station started")


def stop_ros():
    if S.executor:
        S.executor.shutdown(timeout_sec=2.0)
    if S.bridge:
        S.bridge.destroy_node()
    rclpy.try_shutdown()


@contextlib.asynccontextmanager
async def lifespan(app):
    start_ros()
    yield
    stop_ros()


app = FastAPI(title="imitate_aria station", lifespan=lifespan)


@app.get("/api/config")
def get_config():
    return {
        "joint_names": list(hand.JOINT_NAMES),
        "joint_lower": hand.LOWER_RAD.tolist(),
        "joint_upper": hand.UPPER_RAD.tolist(),
        "series": S.store.describe(),
        "expected_hz": SC["expected_hz"],
        "history_s": SC["history_s"],
        "stream_hz": SC["stream_hz"],
        "policy": {"plugin": CONF["policy"]["plugin"], "hz": CONF["policy"]["hz"], "checkpoint": CONF["policy"].get("checkpoint"),
                   "uses_arm": CONF["policy"].get("uses_arm", True)},
        "arm": {"workspace": CONF["arm"]["workspace"], "home": CONF["arm"].get("home")},
        "manual": CONF.get("manual", {}),
        "hand_neutral": hand.NEUTRAL_RAD.tolist(),
        "camera": {"width": CONF["camera"]["width"], "height": CONF["camera"]["height"], "fps": CONF["camera"]["fps"],
                   "stream_hz": SC["camera_hz"], "depth_range_mm": list(DEPTH_RANGE_MM)},
        "robot_assets": ROBOT_ASSETS.exists(),
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


@app.post("/api/manual")
def manual(targets: dict):
    """Manual targets while the policy node is in "manual" (format: deploy/manual.py). Not logged per call."""
    ok, msg = S.bridge.send_manual(targets)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    loop = asyncio.get_running_loop()
    cursors, last_event = {}, -1
    sent_seq = {ch: None for ch in CHANNELS}
    period, cam_period = 1.0 / SC["stream_hz"], 1.0 / SC["camera_hz"]
    try:
        await socket.send_json({"type": "snapshot", "series_meta": S.store.describe(),
                                "series": S.store.since(cursors, SNAPSHOT_POINTS)})
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
