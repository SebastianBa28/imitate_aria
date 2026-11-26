"""ORCA v2 hand driver node (or a simulated hand with -p mock:=true).

Subscribes HAND_COMMAND (JointState, rad, any subset of joints by name), publishes HAND_STATE
(JointState, rad, all 17 joints) at hand.state_hz. Services: HAND_ESTOP (true = release torque and
disconnect; restart the node afterwards) and HAND_TORQUE.

Every command is clamped to the ranges of motion and, if hand.max_step_deg is set, to that far from the previous
command. Unlike rwr_system, a hand without calibration.yaml does not calibrate on its own:
start with -p calibrate:=true to do that deliberately (the fingers move through their full range).

    python -m deploy.nodes.hand_node --ros-args -p mock:=true
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from deploy import config as cfg
from deploy.ros_util import joint_state_msg
from imitate_common import contract, hand


class MockHandBackend:
    """First-order lag toward the last command, starting at neutral. Degrees, JOINT_NAMES order."""

    def __init__(self, tau_s):
        self.tau = tau_s
        self.pos = np.rad2deg(hand.NEUTRAL_RAD)
        self.vel = np.zeros(hand.NUM_JOINTS)
        self.target = self.pos.copy()
        self.torque = True
        self._t = time.monotonic()

    def read(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        if self.torque:
            new = self.pos + (self.target - self.pos) * (1.0 - np.exp(-dt / self.tau))
            self.vel = (new - self.pos) / max(dt, 1e-6)
            self.pos = new
        else:
            self.vel[:] = 0.0
        return self.pos.copy(), self.vel.copy()

    def command(self, pos_deg):
        self.target = np.asarray(pos_deg, dtype=np.float64)

    def set_torque(self, on):
        self.torque = on

    def disconnect(self):
        self.torque = False


class OrcaBackend:
    def __init__(self, config_path, calibrate):
        from deploy.hw.orca import OrcaHandV2

        self._hc = OrcaHandV2(config_path)
        if tuple(self._hc.joint_ids) != hand.JOINT_NAMES:
            raise RuntimeError(f"orca config joint_ids {self._hc.joint_ids} differ from imitate_common.hand.JOINT_NAMES")
        if not self._hc.calibrated and not calibrate:
            raise RuntimeError(f"no calibration.yaml next to {config_path}: restore your backup, "
                               "or start with -p calibrate:=true to calibrate (the hand moves)")
        self._hc.connect()
        self._hc.init_joints(calibrate=calibrate, move_to_neutral=True)

    def read(self):
        return self._hc.read_joint_state()

    def command(self, pos_deg):
        self._hc.set_joint_pos_array(pos_deg)

    def set_torque(self, on):
        self._hc.enable_torque() if on else self._hc.disable_torque()

    def disconnect(self):
        self._hc.disconnect()


class HandNode(Node):
    def __init__(self):
        super().__init__("hand_node")
        conf = cfg.load()
        self.hc = conf["hand"]
        mock = self.declare_parameter("mock", False).value
        calibrate = self.declare_parameter("calibrate", False).value

        if mock:
            self.backend = MockHandBackend(conf["mock"]["hand_tau_s"])
        else:
            self.backend = OrcaBackend(cfg.resolve(self.hc["config_dir"]) / "config.yaml", calibrate)
        self.get_logger().info(f"hand backend: {type(self.backend).__name__}")

        step = self.hc.get("max_step_deg")
        self.max_step = None if step is None else np.deg2rad(step)  # None: commands go straight through
        q0, _ = self.backend.read()
        q0 = np.deg2rad(q0)
        self.last_cmd = np.where(np.isnan(q0), hand.NEUTRAL_RAD, q0)

        self.state_pub = self.create_publisher(JointState, contract.HAND_STATE, 10)
        self.cmd_sub = self.create_subscription(JointState, contract.HAND_COMMAND, self.on_command, 10)
        self.timer = self.create_timer(1.0 / self.hc["state_hz"], self.publish_state)
        self.create_service(SetBool, contract.HAND_ESTOP, self.on_estop)
        self.create_service(SetBool, contract.HAND_TORQUE, self.on_torque)
        self.stopped = False

    def on_command(self, msg):
        if self.stopped:
            return
        q = hand.reorder(msg.name, msg.position)
        q = np.where(np.isnan(q), self.last_cmd, q)  # joints not in the message keep their last command
        q = hand.step_limit(self.last_cmd, hand.clamp_rad(q), self.max_step)
        self.last_cmd = q
        self.backend.command(np.rad2deg(q))

    def publish_state(self):
        pos, vel = self.backend.read()
        self.state_pub.publish(joint_state_msg(self.get_clock().now().to_msg(), hand.JOINT_NAMES,
                                               np.deg2rad(pos), np.deg2rad(vel)))

    def on_estop(self, request, response):
        if not request.data:
            response.success, response.message = False, "send true to stop"
            return response
        self.get_logger().warn("E-stop: releasing torque and disconnecting")
        self.stopped = True
        self.timer.cancel()
        self.destroy_subscription(self.cmd_sub)
        try:
            self.backend.set_torque(False)
        finally:
            self.backend.disconnect()
        response.success, response.message = True, "hand stopped; restart hand_node to use it again"
        return response

    def on_torque(self, request, response):
        if self.stopped:
            response.success, response.message = False, "hand is e-stopped"
            return response
        try:
            self.backend.set_torque(request.data)
            response.success, response.message = True, f"torque {'on' if request.data else 'off'}"
        except Exception as e:  # report the hardware error to the caller instead of crashing
            response.success, response.message = False, str(e)
        return response


def main():
    rclpy.init()
    node = HandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.stopped:
            node.backend.disconnect()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
