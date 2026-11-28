#!/usr/bin/env python3
"""
Pack in-scene recordings (record_cams.py: one JPEG + one 16-bit PNG per frame and camera) into two video
files per camera, ~5-10x smaller and 2 files instead of thousands:

    <cam>/color.mp4   H.264 (libx264, --crf, default 18 = visually lossless), yuv420p
    <cam>/depth.mkv   FFV1 gray16le: LOSSLESS, every depth value (mm) is kept bit-exact

frames.csv / meta.json / session.json are kept; meta.json gets a "packed" entry. Every pack is verified before
anything is deleted: same frame count, depth bit-exact, colour PSNR above --min-psnr. With --delete-frames the
color/ and depth/ image folders are then removed. Recordings already packed are skipped, so re-running is safe.
All scene_*.py scripts read raw and packed recordings alike.

    uv run python packages/data/scene_pack.py raw_data/scene/cup_insc_nov27            # one recording
    uv run python packages/data/scene_pack.py raw_data/scene/* --delete-frames          # all, free the space
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from aina.scene_io import SceneCamera, is_recording


def encode(path, frames, fps, codec, pix_fmt, in_fmt, options):
    import av

    n = 0
    with av.open(path, "w") as out:
        s = out.add_stream(codec, rate=int(round(fps)))
        s.pix_fmt = pix_fmt
        s.options = options
        for img in frames:
            if n == 0:
                s.height, s.width = img.shape[:2]
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format=in_fmt)):
                out.mux(p)
            n += 1
        for p in s.encode():
            out.mux(p)
    return n


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def dir_mb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6


def pack_camera(cam_dir, args):
    raw = SceneCamera(cam_dir)
    if raw.packed:
        print(f"  {raw.id}: already packed")
        return True
    n = len(raw)
    idxs = range(n)
    fps = float(raw.meta.get("fps", 30)) / max(int(raw.meta.get("stride", 1)), 1)
    cpath, dpath = os.path.join(cam_dir, "color.mp4"), os.path.join(cam_dir, "depth.mkv")
    t0 = time.time()
    nc = encode(cpath + ".tmp.mp4", raw.colors(idxs), fps, "libx264", "yuv420p", "rgb24",
                {"crf": str(args.crf), "preset": args.preset})
    nd = encode(dpath + ".tmp.mkv", raw.depths(idxs), fps, "ffv1", "gray16le", "gray16le",
                {"level": "3", "g": "1", "slices": "16", "slicecrc": "1"})
    t_enc = time.time() - t0

    # verify against the original images before trusting the videos
    from aina.scene_io import _decode
    ok = nc == nd == n
    worst, psnrs, bad_depth = 99.0, [], 0
    if ok:
        dec_c = _decode(cpath + ".tmp.mp4", idxs, "rgb24")
        dec_d = _decode(dpath + ".tmp.mkv", idxs, "gray16le")
        for a, b, c, d in zip(raw.colors(idxs), dec_c, raw.depths(idxs), dec_d):
            p = psnr(a, b)
            psnrs.append(p)
            worst = min(worst, p)
            bad_depth += int(not np.array_equal(c, d))
    if not ok or bad_depth or worst < args.min_psnr:
        for p in (cpath + ".tmp.mp4", dpath + ".tmp.mkv"):
            if os.path.exists(p):
                os.remove(p)
        print(f"  {raw.id}: FAILED verification (frames {nc}/{nd}/{n}, depth mismatches {bad_depth}, "
              f"worst PSNR {worst:.1f} dB): nothing changed")
        return False
    before = dir_mb(os.path.join(cam_dir, "color")) + dir_mb(os.path.join(cam_dir, "depth"))
    after = (os.path.getsize(cpath + ".tmp.mp4") + os.path.getsize(dpath + ".tmp.mkv")) / 1e6
    raw.meta["packed"] = {"color": "color.mp4", "color_codec": f"h264 crf {args.crf} yuv420p",
                          "depth": "depth.mkv", "depth_codec": "ffv1 gray16le (lossless)", "frames": n,
                          "color_psnr_db_mean": round(float(np.mean(psnrs)), 2), "color_psnr_db_min": round(float(worst), 2),
                          "mb_before": round(before, 1), "mb_after": round(after, 1)}
    meta_path = os.path.join(cam_dir, "meta.json")   # written atomically, then the videos go in place
    with open(meta_path + ".tmp", "w") as f:
        json.dump(raw.meta, f, indent=2)
    os.replace(meta_path + ".tmp", meta_path)
    os.replace(cpath + ".tmp.mp4", cpath)
    os.replace(dpath + ".tmp.mkv", dpath)
    print(f"  {raw.id}: {n} frames, {before:.0f} MB -> {after:.0f} MB ({before / max(after, 1e-6):.1f}x), "
          f"colour PSNR mean {np.mean(psnrs):.1f} / min {worst:.1f} dB, depth bit-exact, {t_enc:.0f} s")
    if args.delete_frames:
        shutil.rmtree(os.path.join(cam_dir, "color"))
        shutil.rmtree(os.path.join(cam_dir, "depth"))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", help="recording folders (with session.json)")
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality, lower = better/larger (default 18)")
    ap.add_argument("--preset", default="medium", help="libx264 preset (default medium)")
    ap.add_argument("--min-psnr", type=float, default=35.0, help="refuse a colour pack below this PSNR (dB)")
    ap.add_argument("--delete-frames", action="store_true",
                    help="after a verified pack, delete the color/ and depth/ image folders")
    args = ap.parse_args()

    failed = []
    for rec in args.recordings:
        if not is_recording(rec):
            print(f"{rec}: no session.json, skipped")
            continue
        print(rec)
        with open(os.path.join(rec, "session.json")) as f:
            cams = json.load(f)["cameras"]
        for c in cams:
            cam_dir = os.path.join(rec, c)
            if os.path.isdir(cam_dir) and not pack_camera(cam_dir, args):
                failed.append(cam_dir)
            elif args.delete_frames and os.path.isdir(os.path.join(cam_dir, "color")) and SceneCamera(cam_dir).packed:
                # packed earlier without --delete-frames
                shutil.rmtree(os.path.join(cam_dir, "color"))
                shutil.rmtree(os.path.join(cam_dir, "depth"), ignore_errors=True)
    if failed:
        sys.exit(f"{len(failed)} camera(s) not packed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
