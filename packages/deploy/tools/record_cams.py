#!/usr/bin/env python3
"""Record an in-scene human demonstration with every connected RGB-D camera at once (RealSense and
Luxonis OAK): colour + depth aligned to colour, per camera.

Camera only, no robot state: in AINA (arXiv 2511.16661) the in-scene demo is done by a person without
the robot and anchors the Aria demos in the robot frame. Keep the cameras fixed between recording and
deployment; putting the points in the robot frame needs a camera-to-robot calibration (not done yet).

    cd packages/deploy
    pixi run -e hw record-cams --name cup_insc                     # all cameras; ENTER stops
    pixi run -e hw record-cams --name cup_insc --cams d435 --preview --note "d435 left, oak right"

The cameras must be free: stop realsense_node first (dashboard Processes card, or the launch).
Modes: RealSense at its highest 30 fps modes (D435: colour 1920x1080, depth 1280x720, >= ~30 cm);
OAK colour 1920x1080 (1080p ISP of the IMX378; autofocus locked at the calibrated lens position so the
intrinsics hold), stereo depth from the 400p mono pair aligned to colour on the device at 960x540 and
upscaled (nearest) to 1920x1080 on the host.

Output: <repo>/raw_data/cams/<name>_<mon><day>[_N]/
    session.json                 cameras, note, start time
    <cam>/color/000000.jpg ...   RGB, JPEG quality --jpeg-quality    (<cam> = d435, oak, d435_1, ...)
    <cam>/depth/000000.png ...   uint16 millimetres aligned to colour (same pixel = same point), 0 = none
    <cam>/frames.csv             index, frame numbers, t_capture_s, t_host_s
    <cam>/meta.json              device, modes, colour intrinsics + distortion, results
Within a camera, frame i of color/ and depth/ is row i of frames.csv. Across cameras, pair frames by
t_capture_s (host wall-clock seconds at capture; both drivers sync their clocks to the host). Cameras are
not hardware-synced: pairs can be up to ~half a frame (17 ms) apart.
Distortion: RealSense colour is ~undistorted (zero coefficients); OAK colour is NOT, so undistort pixels
with meta.intrinsics.coeffs (OpenCV rational model, 14 values) before unprojecting.
"""
import argparse
import csv
import datetime
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
WARMUP_FRAMES = 30          # let auto-exposure settle before saving
MAX_IN_FLIGHT = 60          # frames per camera held for the writer threads (~2 s)
PREVIEW_SIZE = (640, 360)


def out_dir(data_dir, name):
    today = datetime.date.today()
    base = f"{name}_{today.strftime('%b').lower()}{today.day}"
    target, i = data_dir / base, 0
    while target.exists():
        i += 1
        target = data_dir / f"{base}_{i}"
    return target


class Camera(threading.Thread):
    """One camera: opens in __init__, warms up in run(), records between `go` and `stop`."""
    kind = "camera"

    def __init__(self, cam_id, root, args, pool, go, stop):
        super().__init__(daemon=True, name=cam_id)
        self.cam_id, self.dir, self.args, self.pool, self.go, self.stop = cam_id, root / cam_id, args, pool, go, stop
        self.ready = threading.Event()
        self.slots = threading.BoundedSemaphore(MAX_IN_FLIGHT)
        self.rows, self.errors, self.fatal = [], [], None
        self.meta = {"camera": cam_id, "kind": self.kind}
        self.preview = None
        self.jpeg = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        self.png = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # fast; PNG is lossless at any level

    def prepare_dirs(self):
        (self.dir / "color").mkdir(parents=True)
        (self.dir / "depth").mkdir()

    def save(self, i, bgr, depth_mm):
        cv2.imwrite(str(self.dir / "color" / f"{i:06d}.jpg"), bgr, self.jpeg)
        cv2.imwrite(str(self.dir / "depth" / f"{i:06d}.png"), depth_mm, self.png)

    def submit(self, fn, *a):
        self.slots.acquire()

        def job():
            try:
                fn(*a)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"frame {a[0]}: {e}")
            finally:
                self.slots.release()
        self.pool.submit(job)

    def run(self):
        try:
            self.loop()
        except Exception as e:  # noqa: BLE001  (unplugged, driver error): reported at the end
            self.fatal = f"{type(e).__name__}: {e}"
            self.ready.set()
        # closed by main() after the writer pool drained: RealSense alignment needs the pipeline alive

    def finish(self, frame_numbers):
        with open(self.dir / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "color_frame_number", "depth_frame_number", "t_capture_s", "t_host_s"])
            w.writerows(self.rows)
        nums = frame_numbers
        expected = (nums[-1] - nums[0]) // self.args.stride + 1 if nums else 0
        dropped = max(0, expected - len(nums))
        duration = self.rows[-1][3] - self.rows[0][3] if len(self.rows) > 1 else 0.0
        self.meta.update({"n_frames": len(self.rows), "dropped_frames": dropped, "duration_s": round(duration, 2),
                          "write_errors": self.errors[:20], "fatal_error": self.fatal})
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        rate = (len(self.rows) - 1) / duration if duration > 0 else 0.0
        return f"{self.cam_id}: {len(self.rows)} frames, {duration:.1f} s, {rate:.1f} fps, {dropped} dropped" + \
            (f", {len(self.errors)} write errors" if self.errors else "") + (f", FAILED: {self.fatal}" if self.fatal else "")

    def set_preview(self, rgb_or_bgr, is_rgb):
        small = cv2.resize(rgb_or_bgr, PREVIEW_SIZE)
        self.preview = cv2.cvtColor(small, cv2.COLOR_RGB2BGR) if is_rgb else small


class RealSense(Camera):
    kind = "realsense"

    def __init__(self, cam_id, root, args, pool, go, stop, device):
        super().__init__(cam_id, root, args, pool, go, stop)
        import pyrealsense2 as rs
        self.rs = rs
        info = lambda k: device.get_info(k) if device.supports(k) else None  # noqa: E731
        self.usb = info(rs.camera_info.usb_type_descriptor) or "?"
        if not self.usb.startswith("3"):
            print(f"WARNING: {cam_id} on USB {self.usb}: a RealSense needs USB 3 for these modes", flush=True)
        color_wh = self.best(device, rs.stream.color, rs.format.rgb8)
        depth_wh = self.best(device, rs.stream.depth, rs.format.z16)
        cfg = rs.config()
        cfg.enable_device(info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.color, *color_wh, rs.format.rgb8, args.fps)
        cfg.enable_stream(rs.stream.depth, *depth_wh, rs.format.z16, args.fps)
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            sys.exit(f"{cam_id}: could not start ({e}). Is realsense_node (or another app) using it?")
        self.to_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.local = threading.local()
        self.meta.update({
            "device": {"name": info(rs.camera_info.name), "serial": info(rs.camera_info.serial_number),
                       "firmware": info(rs.camera_info.firmware_version), "usb": self.usb},
            "fps": args.fps, "stride": args.stride, "color_size": list(color_wh), "depth_sensor_size": list(depth_wh),
            "depth": {"unit": "mm", "dtype": "uint16", "aligned_to": "color", "invalid": 0},
            "intrinsics": {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                           "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs)},
        })
        self.summary = f"{device.get_info(rs.camera_info.name)} colour {color_wh[0]}x{color_wh[1]}, " \
                       f"depth {depth_wh[0]}x{depth_wh[1]} @ {args.fps} fps, USB {self.usb}"

    def best(self, device, stream, fmt):
        sizes = {(p.as_video_stream_profile().width(), p.as_video_stream_profile().height())
                 for s in device.query_sensors() for p in s.get_stream_profiles()
                 if p.stream_type() == stream and p.format() == fmt and p.fps() == self.args.fps}
        if not sizes:
            sys.exit(f"{self.cam_id}: no {stream} {fmt} mode at {self.args.fps} fps")
        return max(sizes, key=lambda wh: wh[0] * wh[1])

    def write(self, i, frames):
        if not hasattr(self.local, "align"):
            self.local.align = self.rs.align(self.rs.stream.color)   # rs.align is not thread-safe
        aligned = self.local.align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data())
        depth = np.asanyarray(aligned.get_depth_frame().get_data())
        if abs(self.to_mm - 1.0) > 1e-9:
            depth = np.round(depth * self.to_mm).astype(np.uint16)
        self.save(i, cv2.cvtColor(color, cv2.COLOR_RGB2BGR), depth)

    def loop(self):
        rs = self.rs
        for _ in range(WARMUP_FRAMES):
            self.pipeline.wait_for_frames(5000)
        self.ready.set()
        self.go.wait()
        seen, nums = 0, []
        while not self.stop.is_set():
            frames = self.pipeline.wait_for_frames(5000)
            seen += 1
            if (seen - 1) % self.args.stride:
                continue
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                continue
            host = time.time()
            synced = c.get_frame_timestamp_domain() in (rs.timestamp_domain.global_time, rs.timestamp_domain.system_time)
            t_cap = c.get_timestamp() / 1000.0 if synced else host
            i = len(self.rows)
            self.rows.append((i, c.get_frame_number(), d.get_frame_number(), round(t_cap, 6), round(host, 6)))
            nums.append(c.get_frame_number())
            frames.keep()   # hold the frames past this iteration for the writer thread
            self.submit(self.write, i, frames)
            if self.args.preview and i % 3 == 0:
                self.set_preview(np.asanyarray(c.get_data()), True)
        self.nums = nums

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


class Oak(Camera):
    kind = "oak"

    def __init__(self, cam_id, root, args, pool, go, stop, info):
        super().__init__(cam_id, root, args, pool, go, stop)
        import depthai as dai
        self.dai = dai
        self.device = dai.Device(info, maxUsbSpeed=dai.UsbSpeed.SUPER)
        usb = str(self.device.getUsbSpeed()).rsplit(".", 1)[-1]
        if usb not in ("SUPER", "SUPER_PLUS"):
            print(f"WARNING: {cam_id} on USB {usb}: use a USB 3 port/cable", flush=True)
        calib = self.device.readCalibration()
        a = dai.CameraBoardSocket.CAM_A
        w, h = 1920, 1080
        lens = calib.getLensPosition(a)

        p = dai.Pipeline()
        rgb = p.create(dai.node.ColorCamera)
        rgb.setBoardSocket(a)
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setFps(args.fps)
        if lens:   # autofocus would change the intrinsics: hold the lens where it was calibrated
            rgb.initialControl.setManualFocus(lens)
        mono = []
        for sock in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C):
            m = p.create(dai.node.MonoCamera)
            m.setBoardSocket(sock)
            m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            m.setFps(args.fps)
            mono.append(m)
        st = p.create(dai.node.StereoDepth)
        st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        st.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        st.setLeftRightCheck(True)       # fewer wrong depths at occlusion edges (fingers)
        st.setExtendedDisparity(False)   # True halves the min depth (~35 -> ~20 cm) but costs fps and the median filter
        st.setSubpixel(True)
        st.setDepthAlign(a)
        # aligning at full 1080p on the device caps it at ~26 fps (+150 ms latency); the depth comes from
        # 640x400 monos anyway, so align at 960x540 and upscale (nearest) on the host: no information lost
        st.setOutputSize(w // 2, h // 2)
        mono[0].out.link(st.left)
        mono[1].out.link(st.right)
        sync = p.create(dai.node.Sync)   # pairs colour and depth by timestamp on the device
        sync.setSyncThreshold(datetime.timedelta(milliseconds=17))
        rgb.isp.link(sync.inputs["rgb"])
        st.depth.link(sync.inputs["depth"])
        xout = p.create(dai.node.XLinkOut)
        xout.setStreamName("sync")
        sync.out.link(xout.input)
        self.device.startPipeline(p)
        if args.oak_ir_ma:
            self.device.setIrLaserDotProjectorBrightness(args.oak_ir_ma)
        self.queue = self.device.getOutputQueue("sync", maxSize=8, blocking=False)
        self.clock_offset = time.time() - time.monotonic()   # depthai timestamps are host steady_clock

        K = np.array(calib.getCameraIntrinsics(a, w, h))
        name = self.device.getDeviceName() if hasattr(self.device, "getDeviceName") else calib.getEepromData().productName
        self.meta.update({
            "device": {"name": name, "serial": info.getMxId(), "usb": usb, "lens_position": lens,
                       "sensors": {str(k).rsplit(".", 1)[-1]: v for k, v in self.device.getCameraSensorNames().items()}},
            "fps": args.fps, "stride": args.stride, "color_size": [w, h], "depth_sensor_size": [640, 400],
            "depth": {"unit": "mm", "dtype": "uint16", "aligned_to": "color", "invalid": 0,
                      "aligned_on_device_at": [w // 2, h // 2], "upscaled": "nearest, to the colour size"},
            "intrinsics": {"width": w, "height": h, "fx": K[0, 0], "fy": K[1, 1], "ppx": K[0, 2], "ppy": K[1, 2],
                           "model": "opencv_rational (k1,k2,p1,p2,k3,k4,k5,k6,s1..s4,tx,ty)",
                           "coeffs": list(calib.getDistortionCoefficients(a))},
            "ir_dot_projector_ma": args.oak_ir_ma,
        })
        self.summary = f"{name} colour {w}x{h}, depth 640x400 stereo aligned to colour (960x540 -> {w}x{h}) " \
                       f"@ {args.fps} fps, USB {usb}"

    def write(self, i, bgr, depth):
        h, w = bgr.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        self.save(i, bgr, depth)

    def loop(self):
        got = 0
        while got < WARMUP_FRAMES:
            if self.queue.get() is not None:
                got += 1
        self.ready.set()
        self.go.wait()
        self.queue.tryGetAll()   # drop what queued while the other cameras warmed up
        seen, nums = 0, []
        while not self.stop.is_set():
            group = self.queue.get()
            seen += 1
            if (seen - 1) % self.args.stride:
                continue
            c, d = group["rgb"], group["depth"]
            host = time.time()
            t_cap = c.getTimestamp().total_seconds() + self.clock_offset
            i = len(self.rows)
            self.rows.append((i, c.getSequenceNum(), d.getSequenceNum(), round(t_cap, 6), round(host, 6)))
            nums.append(c.getSequenceNum())
            bgr = c.getCvFrame()
            self.submit(self.write, i, bgr, d.getFrame())
            if self.args.preview and i % 3 == 0:
                self.set_preview(bgr, False)
        self.nums = nums

    def close(self):
        try:
            self.device.close()
        except Exception:  # noqa: BLE001
            pass


def discover(want):
    """[(kind, handle)] for connected cameras of the wanted kinds."""
    found = []
    if "d435" in want or "realsense" in want:
        try:
            import pyrealsense2 as rs
            found += [("d435", d) for d in rs.context().query_devices()]
        except ImportError:
            print("pyrealsense2 missing: RealSense cameras skipped (use the hw env)")
    if "oak" in want:
        try:
            import depthai as dai
            found += [("oak", d) for d in dai.Device.getAllAvailableDevices()]
        except ImportError:
            print("depthai missing: OAK cameras skipped (use the hw env)")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="demo name, e.g. cup_insc")
    ap.add_argument("--cams", default="d435,oak", help="kinds to record, comma-separated: d435, oak (default both)")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "raw_data" / "cams"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="save every Nth frame (3 = 10 Hz, AINA's rate)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--oak-ir-ma", type=int, default=0, help="OAK-D Pro IR dot projector current (mA, 0 = off)")
    ap.add_argument("--note", default="", help="free text saved in session.json, e.g. camera placement")
    ap.add_argument("--preview", action="store_true", help="show small live colour windows")
    args = ap.parse_args()

    want = {k.strip().lower() for k in args.cams.split(",") if k.strip()}
    found = discover(want)
    if not found:
        sys.exit(f"no camera found for --cams {args.cams} (USB 3 port and cable? used by another program?)")

    root = out_dir(Path(args.data_dir), args.name)   # created once every camera opened
    pool = ThreadPoolExecutor(max_workers=8 * len(found))
    go, stop = threading.Event(), threading.Event()
    cams, counts = [], {}
    try:
        for kind, handle in found:
            n = counts.get(kind, 0)
            counts[kind] = n + 1
            cam_id = kind if n == 0 else f"{kind}_{n}"
            cls = RealSense if kind == "d435" else Oak
            cams.append(cls(cam_id, root, args, pool, go, stop, handle))
    except BaseException:
        for c in cams:
            c.close()
        raise
    root.mkdir(parents=True)
    for c in cams:
        c.prepare_dirs()
    (root / "session.json").write_text(json.dumps({
        "name": args.name, "note": args.note, "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "cameras": [c.cam_id for c in cams], "pairing": "nearest t_capture_s (frames.csv); not hardware-synced",
    }, indent=2))
    print(f"Recording to {root}")
    for c in cams:
        print(f"  {c.cam_id}: {c.summary}")
    for c in cams:
        c.start()
    for c in cams:
        c.ready.wait(30)
    failed = [c for c in cams if c.fatal or not c.ready.is_set()]
    if failed:
        stop.set()
        go.set()
        for c in cams:
            c.join(10)
            c.close()
        sys.exit("camera failed while warming up: " + "; ".join(f"{c.cam_id}: {c.fatal or 'timeout'}" for c in failed))

    def wait_enter():
        try:
            input()
        except EOFError:
            pass
        stop.set()

    threading.Thread(target=wait_enter, daemon=True).start()
    go.set()
    t0 = time.time()
    print("RECORDING. Press ENTER to stop.", flush=True)
    try:
        while not stop.is_set():
            if any(c.fatal for c in cams):
                print("\nA camera failed: stopping.", flush=True)
                break
            if args.preview:
                for c in cams:
                    if c.preview is not None:
                        cv2.imshow(f"{c.cam_id} (ENTER in the terminal stops)", c.preview)
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
            counts_txt = ", ".join(f"{c.cam_id} {len(c.rows)}" for c in cams)
            print(f"\r  {time.time() - t0:5.1f} s  {counts_txt} frames", end="", flush=True)
    except KeyboardInterrupt:
        pass
    stop.set()
    print("\nStopping, writing the last frames...", flush=True)
    for c in cams:
        c.join(10)
    pool.shutdown(wait=True)
    for c in cams:
        c.close()
    if args.preview:
        cv2.destroyAllWindows()
    bad = False
    for c in cams:
        line = c.finish(getattr(c, "nums", []))
        print(line)
        m = c.meta
        exp = m["n_frames"] + m["dropped_frames"]
        if c.fatal or c.errors or m["dropped_frames"] > 0.02 * max(exp, 1):
            bad = True
    if len(cams) > 1:
        starts = [c.rows[0][3] for c in cams if c.rows]
        if len(starts) > 1:
            print(f"Cameras started within {1000 * (max(starts) - min(starts)):.0f} ms of each other")
    print(f"Saved to {root}")
    if bad:
        print("WARNING: frames were dropped or failed: try --stride 3 (10 Hz) or fewer cameras")


if __name__ == "__main__":
    main()
