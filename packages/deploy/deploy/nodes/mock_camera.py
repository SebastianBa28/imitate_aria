"""Stands in for a camera: loops a video (mock.video) or draws a test pattern, plus a synthetic depth.

    python -m deploy.nodes.mock_camera                             # the RealSense (CAMERA_* topics)
    python -m deploy.nodes.mock_camera --ros-args -p camera:=oak   # the OAK (OAK_* topics, oak: size)
"""
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from deploy import config as cfg
from deploy.ros_util import LATCHED_QOS, SENSOR_QOS, camera_info_msg, image_msg
from imitate_common import contract


# which camera -> (config section, colour, depth, info topic, frame, HFOV deg, label, circle colour RGB)
CAMERAS = {
    "camera": ("camera", contract.CAMERA_COLOR, contract.CAMERA_DEPTH, contract.CAMERA_INFO,
               "camera_color_optical_frame", 69.0, "mock camera", (230, 160, 30)),
    "oak": ("oak", contract.OAK_COLOR, contract.OAK_DEPTH, contract.OAK_INFO,
            "oak_color_optical_frame", 69.0, "mock OAK", (60, 170, 230)),
}


class MockCamera(Node):
    def __init__(self):
        super().__init__("mock_camera")
        which = self.declare_parameter("camera", "camera").value
        if which not in CAMERAS:
            raise ValueError(f"camera must be one of {', '.join(CAMERAS)}, got {which!r}")
        section, color_topic, depth_topic, info_topic, self.frame_id, hfov, self.label, self.dot = CAMERAS[which]
        conf = cfg.load()
        cam = conf[section]
        self.w, self.h = cam["width"], cam["height"]
        video = conf["mock"].get("video")
        self.cap = None
        if video:
            path = cfg.resolve(video)
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                self.get_logger().warn(f"cannot open {path}, using the test pattern")
                self.cap = None
        self.color_pub = self.create_publisher(Image, color_topic, SENSOR_QOS)
        self.depth_pub = self.create_publisher(Image, depth_topic, SENSOR_QOS)
        info_pub = self.create_publisher(CameraInfo, info_topic, LATCHED_QOS)
        f = 0.5 * self.w / np.tan(np.deg2rad(hfov) / 2)  # D435 colour HFOV ~69 deg (OAK similar)
        info_pub.publish(camera_info_msg(self.get_clock().now().to_msg(), self.frame_id,
                                         self.w, self.h, f, f, self.w / 2, self.h / 2))
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        self.depth_base = (500 + 300 * yy / self.h + 50 * np.sin(xx / 40.0)).astype(np.float64)  # mm
        self.t0 = time.monotonic()
        self.create_timer(1.0 / cam["fps"], self.tick)

    def frame(self, t):
        if self.cap is not None:
            ok, bgr = self.cap.read()
            if not ok:  # loop
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self.cap.read()
            if ok:
                return cv2.cvtColor(cv2.resize(bgr, (self.w, self.h)), cv2.COLOR_BGR2RGB)
        img = np.full((self.h, self.w, 3), 40, np.uint8)
        cx = int(self.w / 2 + self.w / 3 * np.cos(t))
        cy = int(self.h / 2 + self.h / 4 * np.sin(2 * t))
        cv2.circle(img, (cx, cy), 40, self.dot, -1)
        cv2.putText(img, f"{self.label} {t:6.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        return img

    def tick(self):
        t = time.monotonic() - self.t0
        stamp = self.get_clock().now().to_msg()
        self.color_pub.publish(image_msg(stamp, self.frame_id, self.frame(t), "rgb8"))
        depth = self.depth_base + 100 * np.sin(t)
        self.depth_pub.publish(image_msg(stamp, self.frame_id, depth.astype(np.uint16), "16UC1"))


def main():
    rclpy.init()
    node = MockCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
