"""The station's ROS side: subscribes to everything the dashboard shows and sends its commands."""
import concurrent.futures
import json
import threading
import time
from collections import deque

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

from deploy.ros_util import LATCHED_QOS, SENSOR_QOS, pose_from_msg
from imitate_common import contract, hand

POLICY_COMMANDS = ("start", "stop", "reset", "home", "manual", "live", "preview")


class EventLog:
    def __init__(self, maxlen=500):
        self._events = deque(maxlen=maxlen)
        self._next = 0
        self._lock = threading.Lock()

    def add(self, level, source, text):
        with self._lock:
            self._events.append({"id": self._next, "t": time.time(), "level": level, "source": source, "text": text})
            self._next += 1

    def since(self, last_id):
        with self._lock:
            return [e for e in self._events if e["id"] > last_id]


class StationBridge(Node):
    def __init__(self, store, health, events):
        super().__init__("station")
        self.store, self.health, self.events = store, health, events
        names = list(hand.JOINT_NAMES)
        for key, unit in (("hand.q", "rad"), ("hand.qd", "rad/s"), ("hand.cmd", "rad"), ("policy.hand", "rad")):
            store.define(key, hand.NUM_JOINTS, names, unit)
        for key in ("arm.pos", "arm.cmd", "policy.arm"):
            store.define(key, 3, ["x", "y", "z"], "m")
        for key in ("arm.quat", "arm.cmd_quat"):
            store.define(key, 4, ["qx", "qy", "qz", "qw"], "")

        for topic in (contract.HAND_COMMAND, contract.ARM_POSE_CMD, contract.POLICY_ACTION_HAND,
                      contract.POLICY_ACTION_ARM, contract.POLICY_STATE):
            health.watch(topic)

        self.policy_state = {}
        self.camera = {"color": None, "depth": None}   # (msg, seq)
        self._seq = {"color": 0, "depth": 0}
        self.camera_k = None
        self._lock = threading.Lock()

        self.create_subscription(JointState, contract.HAND_STATE, self.on_hand_state, 10)
        self.create_subscription(JointState, contract.HAND_COMMAND, self.joints_to("hand.cmd", contract.HAND_COMMAND), 10)
        self.create_subscription(JointState, contract.POLICY_ACTION_HAND,
                                 self.joints_to("policy.hand", contract.POLICY_ACTION_HAND), 10)
        self.create_subscription(PoseStamped, contract.ARM_POSE, self.on_arm, 10)
        self.create_subscription(PoseStamped, contract.ARM_POSE_CMD, self.on_arm_cmd, 10)
        self.create_subscription(PoseStamped, contract.POLICY_ACTION_ARM, self.on_policy_arm, 10)
        self.create_subscription(String, contract.POLICY_METRICS, self.on_metrics, 10)
        self.create_subscription(String, contract.POLICY_STATE, self.on_policy_state, LATCHED_QOS)
        self.create_subscription(Image, contract.CAMERA_COLOR, self.image_to("color", contract.CAMERA_COLOR), SENSOR_QOS)
        self.create_subscription(Image, contract.CAMERA_DEPTH, self.image_to("depth", contract.CAMERA_DEPTH), SENSOR_QOS)
        self.create_subscription(CameraInfo, contract.CAMERA_INFO, self.on_info, LATCHED_QOS)

        self.policy_cmd_pub = self.create_publisher(String, contract.POLICY_COMMAND, 10)
        self.manual_pub = self.create_publisher(String, contract.POLICY_MANUAL, 10)
        self.estop_client = self.create_client(SetBool, contract.HAND_ESTOP)
        self.torque_client = self.create_client(SetBool, contract.HAND_TORQUE)

    # --- subscriptions --------------------------------------------------------------------
    def on_hand_state(self, msg):
        t = time.time()
        self.health.tick(contract.HAND_STATE, t)
        self.store.add("hand.q", t, hand.reorder(msg.name, msg.position))
        if len(msg.velocity) == len(msg.name):
            self.store.add("hand.qd", t, hand.reorder(msg.name, msg.velocity))

    def joints_to(self, key, topic):
        def cb(msg):
            t = time.time()
            self.health.tick(topic, t)
            self.store.add(key, t, hand.reorder(msg.name, msg.position))
        return cb

    def on_arm(self, msg):
        t = time.time()
        self.health.tick(contract.ARM_POSE, t)
        pos, quat = pose_from_msg(msg)
        self.store.add("arm.pos", t, pos)
        self.store.add("arm.quat", t, quat)

    def on_arm_cmd(self, msg):
        t = time.time()
        self.health.tick(contract.ARM_POSE_CMD, t)
        pos, quat = pose_from_msg(msg)
        self.store.add("arm.cmd", t, pos)
        self.store.add("arm.cmd_quat", t, quat)

    def on_policy_arm(self, msg):
        t = time.time()
        self.health.tick(contract.POLICY_ACTION_ARM, t)
        self.store.add("policy.arm", t, pose_from_msg(msg)[0])

    def on_metrics(self, msg):
        t = time.time()
        self.health.tick(contract.POLICY_METRICS, t)
        try:
            m = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        flat = {}
        for k, v in m.items():
            if isinstance(v, dict):
                flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
            else:
                flat[k] = v
        for k, v in flat.items():
            if k != "t" and isinstance(v, (int, float)) and not isinstance(v, bool):
                self.store.add(f"metric.{k}", t, [v])
            elif k != "t" and v is None:
                self.store.add(f"metric.{k}", t, [np.nan])

    def on_policy_state(self, msg):
        self.health.tick(contract.POLICY_STATE)
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        old = self.policy_state
        with self._lock:
            self.policy_state = state
        if state.get("message") and state.get("message") != old.get("message"):
            msg = state["message"]
            # policy_node reports "<cmd>: ok" on success and "<cmd>: <reason>" when it refuses a command
            refused = ": " in msg and not msg.endswith(": ok") and not msg.startswith("home:")
            level = "error" if state.get("status") == "fault" else "warn" if refused else "info"
            self.events.add(level, "policy", state["message"])
        elif old and (state.get("status"), state.get("mode")) != (old.get("status"), old.get("mode")):
            self.events.add("info", "policy", f"{state.get('status')} ({state.get('mode')})")

    def image_to(self, channel, topic):
        def cb(msg):
            self.health.tick(topic)
            with self._lock:
                self._seq[channel] += 1
                self.camera[channel] = (msg, self._seq[channel])
        return cb

    def on_info(self, msg):
        self.camera_k = list(msg.k)

    def latest_image(self, channel):
        with self._lock:
            return self.camera[channel]

    # --- commands -------------------------------------------------------------------------
    def send_policy(self, cmd):
        if cmd not in POLICY_COMMANDS:
            raise ValueError(f"unknown policy command {cmd!r}")
        self.policy_cmd_pub.publish(String(data=cmd))
        if self.count_subscribers(contract.POLICY_COMMAND) == 0:
            return False, "no policy_node is running"
        return True, "sent"

    def send_manual(self, targets):
        """Forward dashboard manual targets; policy_node validates and clamps them."""
        if self.policy_state.get("status") != "manual":
            return False, "not in manual mode"
        self.manual_pub.publish(String(data=json.dumps(targets)))
        return True, "sent"

    def call_setbool(self, client, value, timeout_s=3.0):
        """Blocking SetBool call (run it off the event loop). Returns (success, message)."""
        if not client.wait_for_service(timeout_sec=0.5):
            return False, f"service {client.srv_name} not available"
        req = SetBool.Request()
        req.data = bool(value)
        done = concurrent.futures.Future()
        fut = client.call_async(req)
        fut.add_done_callback(lambda f: done.set_result(f.result()))
        try:
            res = done.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return False, f"{client.srv_name} timed out"
        return bool(res.success), res.message
