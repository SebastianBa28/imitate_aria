"""Intel RealSense D435 through pyrealsense2 (not the ROS driver, which may not exist for macOS).

Publishes colour (rgb8), depth aligned to colour (16UC1, millimetres) and the colour CameraInfo.
On macOS, librealsense may need sudo for USB access: see spike/README.md.

    python -m deploy.nodes.realsense_node
"""
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from deploy import config as cfg
from deploy.ros_util import LATCHED_QOS, SENSOR_QOS, camera_info_msg, image_msg
from imitate_common import contract

FRAME = "camera_color_optical_frame"


class RealSenseNode(Node):
    def __init__(self):
        super().__init__("realsense_node")
        import pyrealsense2 as rs

        cam = cfg.load()["camera"]
        self.rs = rs
        self.pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, cam["width"], cam["height"], rs.format.rgb8, cam["fps"])
        rs_cfg.enable_stream(rs.stream.depth, cam["width"], cam["height"], rs.format.z16, cam["fps"])
        profile = self.pipeline.start(rs_cfg)
        self.align = rs.align(rs.stream.color)
        # D435 depth units are usually 1 mm; convert anyway so the topic is always mm
        self.depth_to_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.get_logger().info(f"RealSense started: {intr.width}x{intr.height}, depth scale {self.depth_to_mm:.3f} mm/unit")

        self.color_pub = self.create_publisher(Image, contract.CAMERA_COLOR, SENSOR_QOS)
        self.depth_pub = self.create_publisher(Image, contract.CAMERA_DEPTH, SENSOR_QOS)
        info_pub = self.create_publisher(CameraInfo, contract.CAMERA_INFO, LATCHED_QOS)
        info_pub.publish(camera_info_msg(self.get_clock().now().to_msg(), FRAME, intr.width, intr.height,
                                         intr.fx, intr.fy, intr.ppx, intr.ppy, list(intr.coeffs)))
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running and rclpy.ok():
            try:
                frames = self.align.process(self.pipeline.wait_for_frames(1000))
            except RuntimeError as e:  # timeout: camera unplugged or USB 2 bandwidth
                self.get_logger().warn(f"no frames: {e}", throttle_duration_sec=2.0)
                continue
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            stamp = self.get_clock().now().to_msg()
            self.color_pub.publish(image_msg(stamp, FRAME, np.asanyarray(color.get_data()), "rgb8"))
            d = np.asanyarray(depth.get_data())
            if abs(self.depth_to_mm - 1.0) > 1e-6:
                d = np.clip(d * self.depth_to_mm, 0, 65535)
            self.depth_pub.publish(image_msg(stamp, FRAME, d.astype(np.uint16), "16UC1"))

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.pipeline.stop()


def main():
    rclpy.init()
    node = RealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
