"""Runs a policy at policy.hz and is the only node that sends policy commands to the robot.

Commands (std_msgs/String on POLICY_COMMAND): start, stop, reset, home, live, preview.
- preview (default at startup): actions are published on /policy/action_* only, the robot does not move.
- live: actions also go to HAND_COMMAND / ARM_POSE_CMD. Mode can only change while idle.
Safety, applied to every action in both modes (so preview shows what live would send):
hand targets clamped to the ROMs and stepped by at most hand.max_step_deg per tick from the last
target; arm targets clipped to arm.workspace and stepped by arm.max_step_m / max_step_deg per tick.
Stale observations (older than *.stale_s) or a policy exception stop the run with status "fault".

Publishes POLICY_STATE (latched JSON) and POLICY_METRICS (JSON per tick):
latency_ms, loop_hz, obs_age_ms.{hand,arm,color}, hand_clip_deg, arm_clip_mm, policy.<info keys>.

    python -m deploy.nodes.policy_node
"""
import importlib
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

from deploy import config as cfg
from deploy.policies.base import Observation
from deploy.ros_util import (LATCHED_QOS, SENSOR_QOS, image_to_numpy, joint_state_msg, pose_from_msg,
                             pose_msg)
from imitate_common import contract, hand
from imitate_common.transforms import quat_normalize, step_pose


def load_policy(conf):
    module, cls = conf["policy"]["plugin"].split(":")
    return getattr(importlib.import_module(module), cls)(conf["policy"].get("checkpoint"), conf)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.conf = cfg.load()
        pc, ac = self.conf["policy"], self.conf["arm"]
        self.policy = load_policy(self.conf)
        self.uses_arm = bool(pc.get("uses_arm", True))
        self.hand_step = np.deg2rad(self.conf["hand"]["max_step_deg"])
        self.arm_step_m, self.arm_step_rad = ac["max_step_m"], np.deg2rad(ac["max_step_deg"])
        ws = ac["workspace"]
        self.ws_lo = np.array([ws["x"][0], ws["y"][0], ws["z"][0]])
        self.ws_hi = np.array([ws["x"][1], ws["y"][1], ws["z"][1]])
        self.home = None if ac.get("home") is None else np.asarray(ac["home"], dtype=np.float64)
        self.hand_stale, self.arm_stale = pc["stale_s"], ac["stale_s"]

        self.status, self.mode, self.message, self.steps = "idle", "preview", "", 0
        self.hand_q = self.arm = self.color = self.depth = self.K = None
        self.t_hand = self.t_arm = self.t_color = None
        self.last_hand = self.last_arm = None
        self.last_tick = None

        self.create_subscription(JointState, contract.HAND_STATE, self.on_hand, 10)
        self.create_subscription(PoseStamped, contract.ARM_POSE, self.on_arm, 10)
        if self.policy.needs_images:
            self.create_subscription(Image, contract.CAMERA_COLOR, self.on_color, SENSOR_QOS)
            self.create_subscription(Image, contract.CAMERA_DEPTH, self.on_depth, SENSOR_QOS)
            self.create_subscription(CameraInfo, contract.CAMERA_INFO, self.on_info, LATCHED_QOS)
        self.create_subscription(String, contract.POLICY_COMMAND, self.on_command, 10)
        self.action_hand_pub = self.create_publisher(JointState, contract.POLICY_ACTION_HAND, 10)
        self.action_arm_pub = self.create_publisher(PoseStamped, contract.POLICY_ACTION_ARM, 10)
        self.hand_cmd_pub = self.create_publisher(JointState, contract.HAND_COMMAND, 10)
        self.arm_cmd_pub = self.create_publisher(PoseStamped, contract.ARM_POSE_CMD, 10)
        self.metrics_pub = self.create_publisher(String, contract.POLICY_METRICS, 10)
        self.state_pub = self.create_publisher(String, contract.POLICY_STATE, LATCHED_QOS)
        self.create_timer(1.0 / pc["hz"], self.tick)
        self.create_timer(1.0, self.publish_state)
        self.publish_state()
        self.get_logger().info(f"policy {pc['plugin']} at {pc['hz']} Hz, mode preview")

    # --- inputs ---------------------------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_hand(self, msg):
        self.hand_q, self.t_hand = hand.reorder(msg.name, msg.position), self.now()

    def on_arm(self, msg):
        self.arm, self.t_arm = pose_from_msg(msg), self.now()

    def on_color(self, msg):
        self.color, self.t_color = image_to_numpy(msg), self.now()

    def on_depth(self, msg):
        self.depth = image_to_numpy(msg)

    def on_info(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def age(self, t):
        return None if t is None else self.now() - t

    def stale_reason(self):
        a = self.age(self.t_hand)
        if a is None or a > self.hand_stale:
            return "hand state missing or stale"
        if np.isnan(self.hand_q).any():
            return "hand state has unknown joints"
        if self.uses_arm:
            a = self.age(self.t_arm)
            if a is None or a > self.arm_stale:
                return "arm pose missing or stale"
        return None

    def observation(self):
        return Observation(t=self.now(), hand_q=self.hand_q.copy(),
                           arm_pos=self.arm[0].copy() if self.uses_arm else None,
                           arm_quat=self.arm[1].copy() if self.uses_arm else None,
                           color=self.color, depth=self.depth, camera_k=self.K)

    # --- commands -------------------------------------------------------------------------
    def on_command(self, msg):
        cmd = msg.data.strip().lower()
        ok, why = self.run_command(cmd)
        self.message = f"{cmd}: {'ok' if ok else why}"
        if ok:  # rclpy forbids one call site logging at two severities
            self.get_logger().info(self.message)
        else:
            self.get_logger().warning(self.message)
        self.publish_state()

    def run_command(self, cmd):
        if cmd == "stop":
            self.status = "idle"
            return True, ""
        if cmd == "reset":
            self.status, self.steps = "idle", 0
            return True, ""
        if cmd in ("live", "preview"):
            if self.status != "idle":
                return False, "stop first: mode only changes while idle"
            self.mode = cmd
            return True, ""
        if cmd in ("start", "home"):
            if self.status not in ("idle", "fault"):
                return False, f"already {self.status}"
            reason = self.stale_reason()
            if reason:
                return False, reason
            self.last_hand = self.hand_q.copy()
            self.last_arm = (self.arm[0].copy(), self.arm[1].copy()) if self.uses_arm else None
            if cmd == "start":
                try:
                    self.policy.reset(self.observation())
                except Exception as e:
                    return False, f"policy reset failed: {e}"
                self.steps = 0
            self.status = "running" if cmd == "start" else "homing"
            return True, ""
        return False, "unknown command"

    def fault(self, why):
        self.status, self.message = "fault", why
        self.get_logger().error(why)
        self.publish_state()

    # --- loop -----------------------------------------------------------------------------
    def tick(self):
        now = time.perf_counter()
        loop_hz = None if self.last_tick is None else 1.0 / max(now - self.last_tick, 1e-6)
        self.last_tick = now
        if self.status not in ("running", "homing"):
            return
        reason = self.stale_reason()
        if reason:
            return self.fault(reason)

        info, latency_ms = {}, 0.0
        if self.status == "running":
            t0 = time.perf_counter()
            try:
                action = self.policy.act(self.observation())
            except Exception as e:
                return self.fault(f"policy error: {e!r}")
            latency_ms = (time.perf_counter() - t0) * 1e3
            hand_t, arm_t = action.hand_q, (None if action.arm_pos is None else (action.arm_pos, action.arm_quat))
            info = {f"policy.{k}": float(v) for k, v in action.info.items()}
            self.steps += 1
        else:  # homing
            hand_t = hand.NEUTRAL_RAD
            arm_t = None if self.home is None or not self.uses_arm else (self.home[:3], self.home[3:])

        hand_clip = arm_clip = 0.0
        if hand_t is not None:
            raw = np.asarray(hand_t, dtype=np.float64)
            self.last_hand = hand.step_limit(self.last_hand, hand.clamp_rad(raw), self.hand_step)
            hand_clip = float(np.rad2deg(np.nanmax(np.abs(raw - self.last_hand))))
        if arm_t is not None and self.last_arm is not None:
            raw_pos = np.asarray(arm_t[0], dtype=np.float64)
            target = np.clip(raw_pos, self.ws_lo, self.ws_hi)
            pos, quat = step_pose(*self.last_arm, target, quat_normalize(arm_t[1]), self.arm_step_m, self.arm_step_rad)
            self.last_arm = (pos, quat)
            arm_clip = float(np.linalg.norm(raw_pos - pos) * 1e3)

        stamp = self.get_clock().now().to_msg()
        hand_msg = joint_state_msg(stamp, hand.JOINT_NAMES, self.last_hand)
        self.action_hand_pub.publish(hand_msg)
        if self.mode == "live":
            self.hand_cmd_pub.publish(hand_msg)
        if self.last_arm is not None:
            arm_msg = pose_msg(stamp, contract.ARM_FRAME, *self.last_arm)
            self.action_arm_pub.publish(arm_msg)
            if self.mode == "live":
                self.arm_cmd_pub.publish(arm_msg)

        if self.status == "homing":
            done_hand = np.max(np.abs(self.hand_q - hand.NEUTRAL_RAD)) < np.deg2rad(2)
            done_arm = arm_t is None or np.linalg.norm(self.arm[0] - arm_t[0]) < 0.005
            if self.mode == "preview" or (done_hand and done_arm):
                self.status, self.message = "idle", "home: reached" if self.mode == "live" else "home: preview only"
                self.publish_state()

        ages = {"hand": self.age(self.t_hand), "arm": self.age(self.t_arm), "color": self.age(self.t_color)}
        metrics = {"t": self.now(), "status": self.status, "latency_ms": latency_ms, "loop_hz": loop_hz,
                   "obs_age_ms": {k: None if v is None else v * 1e3 for k, v in ages.items()},
                   "hand_clip_deg": hand_clip, "arm_clip_mm": arm_clip, **info}
        self.metrics_pub.publish(String(data=json.dumps(metrics)))

    def publish_state(self):
        state = {"status": self.status, "mode": self.mode, "message": self.message, "steps": self.steps,
                 "plugin": self.conf["policy"]["plugin"], "uses_arm": self.uses_arm,
                 "home_configured": self.home is not None}
        self.state_pub.publish(String(data=json.dumps(state)))


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
