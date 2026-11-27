"""Luxonis OAK (second camera) through depthai 2.x, the same pipeline as the recorder used for in-scene demos.

Publishes colour (rgb8), stereo depth aligned to colour (16UC1, millimetres) and the colour CameraInfo on
contract.OAK_* topics, at oak.width x oak.height (station.yaml). The colour is the 1080p sensor scaled on the
device (full field of view, autofocus locked at the calibrated lens position so the intrinsics hold); depth
comes from the 640x400 mono pair and is aligned to the colour on the device. Stamps are the capture time.

    python -m deploy.nodes.oak_node
"""
import signal
import threading
import time
from datetime import timedelta
from fractions import Fraction

import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from deploy import config as cfg
from deploy.ros_util import LATCHED_QOS, SENSOR_QOS, camera_info_msg, image_msg
from imitate_common import contract

FRAME = "oak_color_optical_frame"
SENSOR_W, SENSOR_H = 1920, 1080


class OakNode(Node):
    def __init__(self):
        super().__init__("oak_node")
        import depthai as dai

        conf = cfg.load()["oak"]
        w, h, fps = conf["width"], conf["height"], conf["fps"]
        scale = Fraction(w, SENSOR_W)
        if Fraction(h, SENSOR_H) != scale or scale > 1:
            raise ValueError(f"oak.width/height {w}x{h} must be 1920x1080 scaled down by one factor (e.g. 960x540)")

        found, info = dai.Device.getFirstAvailableDevice()
        if not found:
            raise RuntimeError("no OAK camera found: is it plugged in, and not open in another program "
                               "(the recorder, another oak_node)?")
        self.device = dai.Device(info, maxUsbSpeed=dai.UsbSpeed.SUPER)
        usb = str(self.device.getUsbSpeed()).rsplit(".", 1)[-1]
        if usb not in ("SUPER", "SUPER_PLUS"):
            self.get_logger().warn(f"OAK on USB {usb}: use a USB 3 port/cable, or frames will drop")
        calib = self.device.readCalibration()
        a = dai.CameraBoardSocket.CAM_A
        lens = calib.getLensPosition(a)

        p = dai.Pipeline()
        rgb = p.create(dai.node.ColorCamera)
        rgb.setBoardSocket(a)
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setIspScale(scale.numerator, scale.denominator)
        rgb.setFps(fps)
        if lens:   # autofocus would change the intrinsics: hold the lens where it was calibrated
            rgb.initialControl.setManualFocus(lens)
        monos = []
        for sock in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C):
            m = p.create(dai.node.MonoCamera)
            m.setBoardSocket(sock)
            m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            m.setFps(fps)
            monos.append(m)
        st = p.create(dai.node.StereoDepth)
        st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        st.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        st.setLeftRightCheck(True)       # fewer wrong depths at occlusion edges (fingers)
        st.setExtendedDisparity(False)   # True halves the min depth (~35 -> ~20 cm) but costs fps and the median filter
        st.setSubpixel(True)
        st.setDepthAlign(a)
        st.setOutputSize(w, h)           # aligning at full 1080p on the device caps it at ~26 fps
        monos[0].out.link(st.left)
        monos[1].out.link(st.right)
        sync = p.create(dai.node.Sync)   # pairs colour and depth by capture time on the device
        sync.setSyncThreshold(timedelta(milliseconds=17))
        rgb.isp.link(sync.inputs["rgb"])
        st.depth.link(sync.inputs["depth"])
        xout = p.create(dai.node.XLinkOut)
        xout.setStreamName("sync")
        sync.out.link(xout.input)
        self.device.startPipeline(p)
        if conf.get("ir_projector_ma"):
            self.device.setIrLaserDotProjectorBrightness(float(conf["ir_projector_ma"]))
        self.queue = self.device.getOutputQueue("sync", maxSize=4, blocking=False)
        self.wall_minus_mono = time.time() - time.monotonic()   # depthai timestamps are host steady_clock

        K = calib.getCameraIntrinsics(a, w, h)
        coeffs = list(calib.getDistortionCoefficients(a))[:8]   # k1 k2 p1 p2 k3 k4 k5 k6
        self.color_pub = self.create_publisher(Image, contract.OAK_COLOR, SENSOR_QOS)
        self.depth_pub = self.create_publisher(Image, contract.OAK_DEPTH, SENSOR_QOS)
        info_pub = self.create_publisher(CameraInfo, contract.OAK_INFO, LATCHED_QOS)
        info_pub.publish(camera_info_msg(self.get_clock().now().to_msg(), FRAME, w, h, K[0][0], K[1][1],
                                         K[0][2], K[1][2], coeffs, model="rational_polynomial"))
        name = calib.getEepromData().productName or "OAK"
        self.get_logger().info(f"{name} ({self.device.getMxId()}) started: colour {w}x{h} @ {fps} fps, "
                               f"depth aligned to colour, USB {usb}, lens {lens}")
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        last = time.monotonic()
        while self.running and rclpy.ok():
            try:
                group = self.queue.tryGet()
            except RuntimeError as e:   # device gone (unplugged, or closing)
                if self.running:
                    self.get_logger().error(f"OAK stopped delivering: {e}")
                return
            if group is None:
                if time.monotonic() - last > 2.0:
                    self.get_logger().warn("no frames from the OAK for 2 s", throttle_duration_sec=2.0)
                time.sleep(0.002)
                continue
            last = time.monotonic()
            c, d = group["rgb"], group["depth"]
            t = c.getTimestamp().total_seconds() + self.wall_minus_mono
            stamp = Time(nanoseconds=int(t * 1e9)).to_msg()
            rgb = cv2.cvtColor(c.getCvFrame(), cv2.COLOR_BGR2RGB)
            self.color_pub.publish(image_msg(stamp, FRAME, rgb, "rgb8"))
            self.depth_pub.publish(image_msg(stamp, FRAME, d.getFrame(), "16UC1"))

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.device.close()


def main():
    rclpy.init()
    node = OakNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # closing the device takes ~1 s: a second Ctrl+C (ros2 launch forwards one) must not cut it short
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
