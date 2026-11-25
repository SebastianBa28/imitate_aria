"""Stands in for the external Franka controller: same two topics, pose follows the command with a lag.

    python -m deploy.nodes.mock_arm
"""
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from deploy import config as cfg
from deploy.ros_util import pose_from_msg, pose_msg
from imitate_common import contract
from imitate_common.transforms import quat_normalize, slerp


class MockArm(Node):
    def __init__(self):
        super().__init__("mock_arm")
        conf = cfg.load()["mock"]
        start = np.asarray(conf["arm_start"], dtype=np.float64)
        self.tau = conf["arm_tau_s"]
        self.pos, self.quat = start[:3], quat_normalize(start[3:])
        self.target_pos, self.target_quat = self.pos.copy(), self.quat.copy()
        self.pub = self.create_publisher(PoseStamped, contract.ARM_POSE, 10)
        self.create_subscription(PoseStamped, contract.ARM_POSE_CMD, self.on_cmd, 10)
        self._t = time.monotonic()
        self.create_timer(0.01, self.tick)

    def on_cmd(self, msg):
        if msg.header.frame_id and msg.header.frame_id != contract.ARM_FRAME:
            self.get_logger().warn(f"ignoring command in frame {msg.header.frame_id!r}", throttle_duration_sec=2.0)
            return
        self.target_pos, self.target_quat = pose_from_msg(msg)

    def tick(self):
        now = time.monotonic()
        a = 1.0 - np.exp(-(now - self._t) / self.tau)
        self._t = now
        self.pos = self.pos + (self.target_pos - self.pos) * a
        self.quat = slerp(self.quat, self.target_quat, a)
        self.pub.publish(pose_msg(self.get_clock().now().to_msg(), contract.ARM_FRAME, self.pos, self.quat))


def main():
    rclpy.init()
    node = MockArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
