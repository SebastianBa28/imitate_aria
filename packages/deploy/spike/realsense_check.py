"""Milestone 0, step 3: does the D435 stream on this machine? No ROS involved.

    python spike/realsense_check.py            # on macOS you may need: sudo -E env PATH=$PATH python ...
Reads 150 frames of aligned colour + depth, prints rate and the depth at the image centre, and saves
spike_realsense.png (colour | depth) in the current folder.
"""
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise SystemExit(f"FAIL: cannot import pyrealsense2 ({e})")

ctx = rs.context()
devices = list(ctx.query_devices())
if not devices:
    raise SystemExit("FAIL: no RealSense found (USB 3 cable/port? on macOS try running with sudo)")
for d in devices:
    print("found:", d.get_info(rs.camera_info.name), "serial", d.get_info(rs.camera_info.serial_number),
          "USB", d.get_info(rs.camera_info.usb_type_descriptor))

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
scale = profile.get_device().first_depth_sensor().get_depth_scale()
try:
    for _ in range(30):  # let auto-exposure settle
        pipe.wait_for_frames()
    t0 = time.time()
    for _ in range(150):
        frames = align.process(pipe.wait_for_frames())
    dt = time.time() - t0
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth = np.asanyarray(frames.get_depth_frame().get_data())
finally:
    pipe.stop()

c = depth[220:260, 300:340].astype(float) * scale
valid = c[c > 0]
print(f"OK: {150 / dt:.1f} fps aligned, depth scale {scale * 1000:.3f} mm/unit, "
      f"centre depth {np.median(valid) if valid.size else float('nan'):.3f} m, "
      f"valid depth {np.mean(depth > 0) * 100:.0f}% of pixels")
try:
    import cv2
    vis = cv2.applyColorMap(np.clip(depth * scale * 255 / 1.5, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite("spike_realsense.png", np.hstack([cv2.cvtColor(color, cv2.COLOR_RGB2BGR), vis]))
    print("saved spike_realsense.png")
except ImportError:
    pass
