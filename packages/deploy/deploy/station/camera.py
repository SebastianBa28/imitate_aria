"""Turns the latest camera messages into JPEGs, once per new frame, shared by all dashboard clients."""
import threading

import cv2
import numpy as np

from deploy.ros_util import image_to_numpy

DEPTH_RANGE_MM = (250, 1500)  # mapped red (near) .. blue (far); 0 = no depth = black


def colorize_depth(depth_mm):
    d = depth_mm.astype(np.float32)
    lo, hi = DEPTH_RANGE_MM
    x = np.clip((hi - d) / (hi - lo), 0, 1)  # near = 1 = red in TURBO
    img = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth_mm == 0] = 0
    return img  # BGR


class CameraEncoder:
    def __init__(self, bridge, quality=70, max_width=640):
        self.bridge, self.quality, self.max_width = bridge, quality, max_width
        self._cache = {}  # channel -> (seq, jpeg bytes)
        self._lock = threading.Lock()

    def jpeg(self, channel):
        """(seq, bytes) of the newest frame, or (None, None) if no frame yet. Blocking: call off the event loop."""
        latest = self.bridge.latest_image(channel)
        if latest is None:
            return None, None
        msg, seq = latest
        with self._lock:
            cached = self._cache.get(channel)
            if cached and cached[0] == seq:
                return cached
        img = image_to_numpy(msg)
        if msg.encoding == "16UC1":
            bgr = colorize_depth(img)
        elif msg.encoding == "rgb8":
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = img
        if bgr.shape[1] > self.max_width:
            h = int(bgr.shape[0] * self.max_width / bgr.shape[1])
            bgr = cv2.resize(bgr, (self.max_width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return None, None
        with self._lock:
            self._cache[channel] = (seq, buf.tobytes())
            return self._cache[channel]
