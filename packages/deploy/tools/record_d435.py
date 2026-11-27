#!/usr/bin/env python3
"""Record an in-scene human demonstration with the RealSense D435 (colour + depth aligned to colour).

Camera only, no robot state: in AINA (arXiv 2511.16661) the in-scene demo is done by a person without
the robot and anchors the Aria demos in the robot frame. Keep the camera fixed between recording and
deployment; turning these points into the robot frame needs the camera-to-robot calibration (not done yet).

    cd packages/deploy
    pixi run -e hw record-d435 --name cup_insc            # ENTER stops
    pixi run -e hw record-d435 --name cup_insc --preview --note "camera on left mount"

The camera must be free: stop realsense_node first (dashboard Processes card, or the launch).
Default: the highest modes at 30 fps (colour 1920x1080, depth 1280x720; depth needs >= ~30 cm at 720p).

Output: <repo>/raw_data/d435/<name>_<mon><day>[_N]/
    color/000000.jpg ...   RGB, JPEG quality --jpeg-quality
    depth/000000.png ...   uint16 millimetres, aligned to colour (same size, same pixel = same point),
                           0 = no depth
    frames.csv             index, color/depth frame numbers, device timestamp (ms), host time (s)
    meta.json              device, modes, colour intrinsics (= depth's, after alignment), depth scale, note
Frame i of color/ and depth/ is row i of frames.csv. Gaps in color_frame_number are dropped frames.
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
import pyrealsense2 as rs

REPO_ROOT = Path(__file__).resolve().parents[3]
WARMUP_FRAMES = 30          # let auto-exposure settle before saving
MAX_IN_FLIGHT = 60          # frames held for the writer threads (~2 s); beyond that the camera drops frames


def best_mode(device, stream, fmt, fps):
    """Largest (width, height) the device offers for this stream/format at this fps."""
    sizes = set()
    for s in device.query_sensors():
        for p in s.get_stream_profiles():
            if p.stream_type() == stream and p.format() == fmt and p.fps() == fps:
                v = p.as_video_stream_profile()
                sizes.add((v.width(), v.height()))
    if not sizes:
        sys.exit(f"the camera offers no {stream} {fmt} mode at {fps} fps")
    return max(sizes, key=lambda wh: wh[0] * wh[1])


def parse_size(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def out_dir(data_dir, name):
    today = datetime.date.today()
    base = f"{name}_{today.strftime('%b').lower()}{today.day}"
    target, i = data_dir / base, 0
    while target.exists():
        i += 1
        target = data_dir / f"{base}_{i}"
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="demo name, e.g. cup_insc")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "raw_data" / "d435"))
    ap.add_argument("--color", type=parse_size, help="colour WxH (default: the largest at --fps)")
    ap.add_argument("--depth", type=parse_size, help="depth WxH (default: the largest at --fps)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="save every Nth frame (default: all)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--note", default="", help="free text saved in meta.json, e.g. where the camera is")
    ap.add_argument("--preview", action="store_true", help="show a small live colour window")
    args = ap.parse_args()

    devices = rs.context().query_devices()
    if len(devices) == 0:
        sys.exit("no RealSense camera found (USB 3 port and cable?)")
    dev = devices[0]
    usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    if not usb.startswith("3"):
        print(f"WARNING: camera on USB {usb}; the D435 needs USB 3 to deliver frames", flush=True)
    color_wh = args.color or best_mode(dev, rs.stream.color, rs.format.rgb8, args.fps)
    depth_wh = args.depth or best_mode(dev, rs.stream.depth, rs.format.z16, args.fps)

    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.color, *color_wh, rs.format.rgb8, args.fps)
    cfg.enable_stream(rs.stream.depth, *depth_wh, rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(cfg)
    except RuntimeError as e:
        sys.exit(f"could not start the camera ({e}). Is realsense_node (or another app) using it?")
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()   # metres per unit
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    out = out_dir(Path(args.data_dir), args.name)
    (out / "color").mkdir(parents=True)
    (out / "depth").mkdir()
    meta = {
        "name": args.name, "note": args.note, "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "device": {"name": dev.get_info(rs.camera_info.name), "serial": dev.get_info(rs.camera_info.serial_number),
                   "firmware": dev.get_info(rs.camera_info.firmware_version), "usb": usb},
        "fps": args.fps, "stride": args.stride, "color_size": list(color_wh), "depth_sensor_size": list(depth_wh),
        "depth": {"unit": "mm", "dtype": "uint16", "aligned_to": "color", "invalid": 0,
                  "sensor_scale_m": depth_scale},
        "intrinsics": {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                       "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs)},
        "color_format": "RGB (saved as JPEG, read back with cv2 as BGR)",
    }
    print(f"Recording to {out}\n  colour {color_wh[0]}x{color_wh[1]}, depth {depth_wh[0]}x{depth_wh[1]} "
          f"(aligned to colour) @ {args.fps} fps, USB {usb}", flush=True)

    to_mm = depth_scale * 1000.0
    local = threading.local()
    slots = threading.BoundedSemaphore(MAX_IN_FLIGHT)
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    png = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # fast; depth PNGs are lossless at any level
    errors = []

    def write(i, frames):
        try:
            if not hasattr(local, "align"):
                local.align = rs.align(rs.stream.color)   # one per thread: rs.align is not thread-safe
            aligned = local.align.process(frames)
            color = np.asanyarray(aligned.get_color_frame().get_data())
            depth = np.asanyarray(aligned.get_depth_frame().get_data())
            if abs(to_mm - 1.0) > 1e-9:
                depth = np.round(depth * to_mm).astype(np.uint16)
            cv2.imwrite(str(out / "color" / f"{i:06d}.jpg"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR), jpeg)
            cv2.imwrite(str(out / "depth" / f"{i:06d}.png"), depth, png)
        except Exception as e:  # noqa: BLE001
            errors.append(f"frame {i}: {e}")
        finally:
            slots.release()

    stop = threading.Event()

    def wait_enter():
        try:
            input()
        except EOFError:
            pass
        stop.set()

    rows = []
    pool = ThreadPoolExecutor(max_workers=8)
    seen = 0
    try:
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames(5000)
        threading.Thread(target=wait_enter, daemon=True).start()
        print("RECORDING. Press ENTER to stop.", flush=True)
        t0 = time.time()
        while not stop.is_set():
            frames = pipeline.wait_for_frames(5000)
            seen += 1
            if (seen - 1) % args.stride:
                continue
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                continue
            i = len(rows)
            rows.append((i, c.get_frame_number(), d.get_frame_number(), f"{c.get_timestamp():.3f}", f"{time.time():.6f}"))
            frames.keep()   # hold the frames past this loop iteration for the writer thread
            slots.acquire()
            pool.submit(write, i, frames)
            if args.preview and i % 3 == 0:
                small = cv2.resize(np.asanyarray(c.get_data()), (640, 360))
                cv2.imshow("D435 (ENTER in the terminal stops)", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            if i % (args.fps // args.stride or 1) == 0:
                print(f"\r  {i} frames, {time.time() - t0:5.1f} s", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping, writing the last frames...", flush=True)
        pool.shutdown(wait=True)
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()

    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "color_frame_number", "depth_frame_number", "t_device_ms", "t_host_s"])
        w.writerows(rows)
    nums = [r[1] for r in rows]
    expected = (nums[-1] - nums[0]) // args.stride + 1 if nums else 0
    dropped = max(0, expected - len(nums))
    duration = float(rows[-1][4]) - float(rows[0][4]) if len(rows) > 1 else 0.0
    meta.update({"n_frames": len(rows), "dropped_frames": dropped, "duration_s": round(duration, 2),
                 "write_errors": errors[:20]})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    rate = (len(rows) - 1) / duration if duration > 0 else 0.0
    print(f"Saved {len(rows)} frames ({duration:.1f} s, {rate:.1f} fps saved, {dropped} dropped) to {out}")
    if errors:
        print(f"WARNING: {len(errors)} frames failed to write, first: {errors[0]}")
    if dropped > 0.02 * max(expected, 1):
        print("WARNING: more than 2% of frames dropped: the disk or CPU can't keep up; try --stride 3 "
              "(10 Hz, the rate AINA uses) or a smaller --color")


if __name__ == "__main__":
    main()
