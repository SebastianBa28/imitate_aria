"""Stands in for the external Franka controller: same two pose topics, pose follows the command with a lag.
Also publishes joint angles (contract.ARM_JOINT_STATE, ~30 Hz like the real controller) from IK on that pose.

    python -m deploy.nodes.mock_arm
"""
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from deploy import config as cfg
from deploy.ros_util import joint_state_msg, pose_from_msg, pose_msg
from imitate_common import contract
from imitate_common.arm_kinematics import Panda
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
        self.panda = Panda()
        self.q, _ = self.panda.ik(self.pos, self.quat)
        self.joint_pub = self.create_publisher(JointState, contract.ARM_JOINT_STATE, 10)
        self._t = time.monotonic()
        self.create_timer(0.01, self.tick)
        self.create_timer(1 / 30, self.publish_joints)

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

    def publish_joints(self):
        self.q, _ = self.panda.ik(self.pos, self.quat, self.q, iters=10)   # warm start: ~2 ms
        self.joint_pub.publish(joint_state_msg(self.get_clock().now().to_msg(), contract.ARM_JOINT_NAMES, self.q))


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
