#!/usr/bin/env python3
"""
Process an in-scene demo (fixed RGB-D cameras, record_cams.py) with the AINA recipe, into the same
<demo>_aina.npz format as post_process.py, in the calibrated TABLE frame (scene_calib.py).

Per camera:
  1. Grounding DINO box + SAM 2 mask per object on the first frame (same PROMPTS as the Aria pipeline)
  2. N_POINTS[k] // n_cameras points per object, tracked with CoTracker3 through the clip
  3. lifted to 3D with that camera's own aligned sensor depth (median in a LIFT_RADIUS window), then into the
     world with T_world_cam. Both cameras' points are concatenated per object (same slot count as Aria).
Hands (WiLoR on every frame of every camera, see aina/hands_wilor.py):
  both cameras see a hand -> its 21 keypoints are triangulated (DLT on undistorted pixels), kept if they
  reproject within --max-reproj px; otherwise -> the best single view, placed metrically from the hand's size
  and corrected with the sensor depth. Stored in Aria landmark order, so dataset_io.py works unchanged.

    uv run python packages/data/scene_process.py raw_data/scene/cup_insc_nov30            # calib found automatically
    uv run python packages/data/scene_process.py raw_data/scene/cup_insc_nov30 --calib raw_data/scene/calib_nov30/calib.json
    uv run python packages/data/scene_process.py raw_data/scene/cup_insc_nov27 --cams oak --calib none   # no calibration:
                                                       # one camera only, its camera frame is the world
    uv run python packages/data/scene_process.py ... --append-to   # also add to datasets/imitate-aria-scene

Output, in --out-dir/<demo>/ (default output/scene/<demo>/), NaN = missing:
  <demo>_aina.npz
    per frame (T):   rgb_ts (ns, host clock of the reference camera), time_s, frame_idx (C,T) per camera (-1 = no
                     frame within --max-dt)
    per point (N):   points_world (T,N,3), depth (T,N) (along its camera's axis), tracks (T,N,2) (its camera's
                     pixels), vis (T,N), obj_id (N), point_cam (N) index into cameras
    hands:           hand_landmarks_world (T,2,21,3) Aria order, axis 1 = (left, right); hand_keypoints_world,
                     T_world_wrist as in post_process.py; hand_uv (C,T,2,21,2) the fused landmarks reprojected into
                     each camera; hand_source (T,2): 0 none, 1 single view, 2 triangulated; hand_reproj_px (T,2);
                     hand_kp2d (C,T,2,21,2) raw WiLoR keypoints (OpenPose order), hand_conf (C,T,2)
    per camera (C):  cameras, T_world_cam (C,4,4), K (C,3,3), boxes0_<cam>, masks0_<cam>
    per object (K):  object_names, n_points (dataset point slots per object)
    metadata:        source="scene", recording, calib_file, fps, start, stride, frame_hw (per camera, (C,2)),
                     videos (JSON {dataset feature: file}), seg_files
  <demo>_<cam>_rgb.mp4 / _overlay.mp4 / _depth.mp4   per camera, frame t = row t (overlay: tracked points of that
                     camera + the fused hands reprojected; depth: sensor depth, turbo, red = near)
  <demo>_<cam>_seg.jpg   first frame with the boxes and masks
"""
import argparse
import json
import os
import shutil
import sys
import warnings

import cv2
import numpy as np
import torch

from aina.config import LIFT_RADIUS, N_POINTS, PROMPTS
from aina.geometry import inv, transform, triangulate
from aina.hands import WRIST_KEYPOINTS, openpose_to_aria, wrist_poses
from aina.lift import lift_points
from aina.overlay import draw_overlay
from aina.scene_io import SceneRecording, find_calib, load_calib
from aina.segment import boxes_grounding_dino, draw_segmentation, masks_sam2
from aina.track import track_points
from dataset_io import EpisodeError, EpisodeWriter

DEFAULT_ROOT = os.path.join("datasets", "imitate-aria-scene")
DEFAULT_REPO_ID = "SebastianBa28/imitate-aria-scene"
DEPTH_VIS_RANGE = (0.3, 1.5)   # m, red..blue in the depth videos
TRACK_SCALE = 0.5              # frames handed to CoTracker at half resolution (it works at <= 512 px anyway)


# ----------------------------------------------------------------------------- objects
def objects_for_camera(cam, idxs, prompts, n_points, args):
    """Segment the first frame, track, lift. Returns dict with per-point arrays in the camera frame."""
    rgb0 = cam.color(int(idxs[0]))
    print(f"  [{cam.id}] boxes + masks on frame {int(idxs[0])}")
    boxes = boxes_grounding_dino(rgb0, prompts, args.device, args.box_thr)
    masks = masks_sam2(rgb0, boxes) if boxes else []
    seg = draw_segmentation(rgb0, boxes, masks)
    if not boxes:
        return None, seg
    names = [p for p, _ in boxes]
    n_k = [n_points[prompts.index(p)] for p in names]

    print(f"  [{cam.id}] CoTracker3 on {len(idxs)} frames")
    small = [cv2.resize(f, None, fx=TRACK_SCALE, fy=TRACK_SCALE, interpolation=cv2.INTER_AREA)
             for f in cam.colors(idxs)]
    masks_small = [cv2.resize(m.astype(np.uint8), small[0].shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                   for m in masks]
    tracks, vis, obj_local = track_points(small, masks_small, n_k, args.device, tracker=args.tracker)
    tracks = tracks / TRACK_SCALE
    del small

    print(f"  [{cam.id}] lifting with sensor depth")
    T, N = vis.shape
    P_cam = np.full((T, N, 3), np.nan)
    for t, dimg in enumerate(cam.depths(idxs)):
        if vis[t].any():
            P_cam[t] = lift_points(cam.depth_to_points(dimg), tracks[t], vis[t], LIFT_RADIUS)
    obj_id = np.array([prompts.index(names[k]) for k in obj_local], np.int32)   # global prompt index
    return dict(tracks=tracks, vis=vis, P_cam=P_cam, obj_id=obj_id, boxes=boxes, masks=masks), seg


# ----------------------------------------------------------------------------- hands
def hands_for_camera(wilor, cam, idxs, max_dist):
    """Per frame and side: WiLoR 2D keypoints (21,2), camera-frame single-view 3D (21,3), confidence."""
    T = len(idxs)
    kp2d = np.full((T, 2, 21, 2), np.nan)
    single = np.full((T, 2, 21, 3), np.nan)
    conf = np.zeros((T, 2))
    from aina.hands_wilor import place_metric
    for t, (rgb, dimg) in enumerate(zip(cam.colors(idxs), cam.depths(idxs))):
        hands = wilor(rgb)
        if not hands:
            continue
        buf = cam.depth_to_points(dimg)
        for s, h in hands.items():
            P = place_metric(h["kp3d"], cam.undistort_px(h["kp2d"]), buf, h["kp2d"])
            if P is None or np.linalg.norm(P[0]) > max_dist:   # e.g. people in the background
                continue
            kp2d[t, s], single[t, s], conf[t, s] = h["kp2d"], P, h["conf"]
        if (t + 1) % 50 == 0:
            print(f"    {t + 1}/{T} frames")
    det = np.isfinite(kp2d[..., 0, 0]).mean(0)
    print(f"  [{cam.id}] hands detected: left {det[0]:.0%}, right {det[1]:.0%} of frames")
    return kp2d, single, conf


def fuse_hands(cams, cam_ids, T_world_cam, kp2d, single, conf, max_reproj):
    """kp2d (C,T,2,21,2), single (C,T,2,21,3) camera frame, conf (C,T,2) -> world landmarks (T,2,21,3) in OpenPose
    order, source (T,2), reprojection error (T,2)."""
    C, T = kp2d.shape[:2]
    out = np.full((T, 2, 21, 3), np.nan)
    source = np.zeros((T, 2), np.int8)
    err = np.full((T, 2), np.nan)
    single_w = np.stack([transform(T_world_cam[c], single[i]) for i, c in enumerate(cam_ids)])  # (C,T,2,21,3)
    P = [inv(T_world_cam[c])[:3] for c in cam_ids]
    for t in range(T):
        for s in range(2):
            seen = [i for i in range(C) if np.isfinite(kp2d[i, t, s]).all()]
            if len(seen) >= 2:
                nrm = [cams[cam_ids[i]].undistort_px(kp2d[i, t, s]) for i in seen]
                X = np.stack([triangulate([P[i] for i in seen], [n[j] for n in nrm]) for j in range(21)])
                e = []
                for i in seen:
                    uv = cams[cam_ids[i]].project(transform(inv(T_world_cam[cam_ids[i]]), X))
                    e.append(np.linalg.norm(uv - kp2d[i, t, s], axis=1))
                e = np.max(np.stack(e), 0)                       # (21,) worst view per joint
                if np.median(e) <= max_reproj:
                    bad = e > 3 * max_reproj                     # single joints that disagree: single-view mean
                    if bad.any():
                        X[bad] = np.nanmean(single_w[seen][:, t, s][:, bad], 0)
                    out[t, s], source[t, s], err[t, s] = X, 2, float(np.median(e))
                    continue
            if seen:
                i = max(seen, key=lambda i: conf[i, t, s])
                out[t, s], source[t, s] = single_w[i, t, s], 1
    return out, source, err


# ----------------------------------------------------------------------------- videos
def depth_color(depth_mm, size):
    lo, hi = DEPTH_VIS_RANGE
    z = cv2.resize(depth_mm, size, interpolation=cv2.INTER_NEAREST).astype(np.float32) / 1000.0
    level = (255 * (hi - np.clip(z, lo, hi)) / (hi - lo)).astype(np.uint8)
    img = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
    img[z <= 0] = 0
    return img


def write_videos(out_dir, stem, cam, idxs, fps, width, tracks, vis, has3d, obj_id_local, names, t_sec, hand_uv):
    W, H = cam.size
    vw, vh = int(width), int(round(H * width / W))
    s = vw / W
    files = {k: f"{stem}_{cam.id}_{k}.mp4" for k in ("rgb", "overlay", "depth")}
    writers = {k: cv2.VideoWriter(os.path.join(out_dir, f), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
               for k, f in files.items()}
    for t, (rgb, dimg) in enumerate(zip(cam.colors(idxs), cam.depths(idxs))):
        small = cv2.cvtColor(cv2.resize(rgb, (vw, vh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        writers["rgb"].write(small)
        if tracks is not None:
            ov = draw_overlay(small, tracks[t] * s, vis[t], has3d[t], obj_id_local, names, t_sec[t], hand_uv[t] * s)
        else:
            ov = draw_overlay(small, np.zeros((0, 2)), np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, int),
                              names, t_sec[t], hand_uv[t] * s)
        writers["overlay"].write(ov)
        writers["depth"].write(depth_color(dimg, (vw, vh)))
    for w in writers.values():
        w.release()
    return files, (vh, vw)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="demo folder (record_cams.py, raw or packed)")
    ap.add_argument("--calib", default=None,
                    help="calib.json (default: the same day's calibration clip next to the demo); 'none' = no "
                         "calibration, one camera only, in its own camera frame")
    ap.add_argument("--cams", default=None, help="comma-separated cameras to use (default: all in the calibration)")
    ap.add_argument("--ref", default=None, help="camera whose timeline defines the frames (default: oak if used)")
    ap.add_argument("--stride", type=int, default=3, help="every Nth reference frame (default 3: 30 -> 10 Hz)")
    ap.add_argument("--start", type=int, default=0, help="first reference frame (objects must be visible there)")
    ap.add_argument("--stop", type=int, default=0, help="last reference frame, exclusive (0 = end)")
    ap.add_argument("--max-dt", type=float, default=0.020, help="max capture time gap of paired frames (s)")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--tracker", choices=["online", "offline"], default="online")
    ap.add_argument("--no-hands", dest="hands", action="store_false", help="skip WiLoR")
    ap.add_argument("--max-reproj", type=float, default=12.0,
                    help="triangulated hands must reproject within this many px (median over joints)")
    ap.add_argument("--max-hand-dist", type=float, default=1.5, help="ignore hands farther from a camera (m)")
    ap.add_argument("--out-dir", default=os.path.join("output", "scene"))
    ap.add_argument("--video-width", type=int, default=640)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--append-to", nargs="?", const=DEFAULT_ROOT, default=None, metavar="DIR",
                    help=f"append the result to the LeRobot dataset in DIR (default {DEFAULT_ROOT})")
    ap.add_argument("--keep-output", action="store_true", help="with --append-to: keep the output folder")
    args = ap.parse_args()

    rec = SceneRecording(args.recording)
    stem = rec.name
    prompts = PROMPTS
    if args.calib is None:
        args.calib = find_calib(rec.dir)
        if args.calib is None:
            sys.exit(f"no calibration found for {stem} (a solved calib* clip of the same day next to it): run "
                     "scene_calib.py on the calibration clip, pass --calib PATH, or --calib none --cams <one camera>")
        print(f"calibration: {args.calib}")
    calib = None if args.calib == "none" else load_calib(args.calib)

    want = args.cams.split(",") if args.cams else (list(calib["cameras"]) if calib else list(rec.cams))
    cam_ids = [c for c in want if c in rec.cams]
    if calib:
        missing = [c for c in cam_ids if c not in calib["cameras"]]
        if missing:
            sys.exit(f"cameras {missing} are not in the calibration {args.calib}")
        T_world_cam = {c: calib["cameras"][c]["T_world_cam"] for c in cam_ids}
        for c in cam_ids:
            if tuple(calib["cameras"][c]["size"]) != rec.cams[c].size or \
                    not np.allclose(calib["cameras"][c]["K"], rec.cams[c].K, atol=0.5):
                sys.exit(f"{c}: intrinsics/size differ between the calibration clip and this demo (another device "
                         "or mode?)")
    else:
        if len(cam_ids) != 1:
            sys.exit(f"--calib none needs exactly one camera (--cams one of {list(rec.cams)})")
        T_world_cam = {cam_ids[0]: np.eye(4)}
        print(f"no calibration: world = {cam_ids[0]} camera frame (x right, y down, z forward)")
    if not cam_ids:
        sys.exit(f"no usable camera (recording has {list(rec.cams)})")
    cams = {c: rec.cams[c] for c in cam_ids}
    ref = args.ref or ("oak" if "oak" in cam_ids else cam_ids[0])

    writer = None
    if args.append_to:
        try:
            writer = EpisodeWriter(args.append_to, DEFAULT_REPO_ID)
        except EpisodeError as e:
            sys.exit(str(e))
        if stem in writer:
            print(f"{stem} is already in {args.append_to}; nothing to do")
            return

    # frames: reference timeline, paired frames of the other cameras
    t_ref, pairing = rec.pair(ref, stride=args.stride, max_dt=args.max_dt)
    sel = np.arange(len(t_ref))
    sel = sel[(pairing[ref] >= args.start) & ((pairing[ref] < args.stop) if args.stop > 0 else True)]
    keep = np.all([pairing[c][sel] >= 0 for c in cam_ids], axis=0)   # every camera has a frame
    if (~keep).any():
        print(f"  dropping {(~keep).sum()} frames without a partner within {args.max_dt * 1000:.0f} ms")
    sel = sel[keep]
    if len(sel) < 2:
        sys.exit("fewer than 2 frames to process")
    idx = {c: pairing[c][sel].astype(int) for c in cam_ids}
    t_s = t_ref[sel] - t_ref[sel][0]
    fps = float(cams[ref].meta.get("fps", 30)) / args.stride
    T = len(sel)
    print(f"{stem}: cameras {cam_ids} (ref {ref}), {T} frames at {fps:.1f} Hz ({t_s[-1]:.1f} s)")

    # objects, per camera, half the points each
    n_cam = len(cam_ids)
    n_per_cam = [max(1, n // n_cam) for n in N_POINTS]
    obj, segs = {}, {}
    for c in cam_ids:
        obj[c], segs[c] = objects_for_camera(cams[c], idx[c], prompts, n_per_cam, args)
        torch.cuda.empty_cache()

    # hands
    C = len(cam_ids)
    kp2d = np.full((C, T, 2, 21, 2), np.nan)
    single = np.full((C, T, 2, 21, 3), np.nan)
    conf = np.zeros((C, T, 2))
    if args.hands:
        from aina.hands_wilor import WiLoR
        print("hands: WiLoR")
        wilor = WiLoR(args.device)
        for i, c in enumerate(cam_ids):
            kp2d[i], single[i], conf[i] = hands_for_camera(wilor, cams[c], idx[c], args.max_hand_dist)
        del wilor
        torch.cuda.empty_cache()
    hands_op, hand_source, hand_err = fuse_hands(cams, cam_ids, T_world_cam, kp2d, single, conf, args.max_reproj)
    hand_world = openpose_to_aria(hands_op)                                   # (T,2,21,3)
    hand_uv = np.stack([cams[c].project(transform(inv(T_world_cam[c]), hand_world.reshape(-1, 3)))
                        .reshape(T, 2, 21, 2) for c in cam_ids])              # (C,T,2,21,2)

    # assemble object points: camera after camera, object order kept within each
    names = [p for p in prompts if any(obj[c] is not None and prompts.index(p) in obj[c]["obj_id"] for c in cam_ids)]
    if not names:
        sys.exit("no object found in any camera")
    pts_w, depth, tracks, vis, obj_id, point_cam = [], [], [], [], [], []
    for i, c in enumerate(cam_ids):
        o = obj[c]
        if o is None:
            continue
        pts_w.append(transform(T_world_cam[c], o["P_cam"]))
        depth.append(o["P_cam"][..., 2])
        tracks.append(o["tracks"])
        vis.append(o["vis"])
        obj_id.append(np.array([names.index(prompts[k]) for k in o["obj_id"]], np.int32))
        point_cam.append(np.full(len(o["obj_id"]), i, np.int32))
    points_world, depth = np.concatenate(pts_w, 1), np.concatenate(depth, 1)
    tracks, vis = np.concatenate(tracks, 1), np.concatenate(vis, 1)
    obj_id, point_cam = np.concatenate(obj_id), np.concatenate(point_cam)

    # save
    out_dir = os.path.join(args.out_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    videos, seg_files, frame_hw = {}, [], []
    for i, c in enumerate(cam_ids):
        sel_p = point_cam == i
        files, vhw = write_videos(out_dir, stem, cams[c], idx[c], fps, args.video_width,
                                  tracks[:, sel_p] if sel_p.any() else None, vis[:, sel_p],
                                  np.isfinite(depth[:, sel_p]), obj_id[sel_p], names, t_s, hand_uv[i])
        videos[f"observation.images.{c}"] = files["rgb"]
        videos[f"debug.overlay_{c}"] = files["overlay"]
        videos[f"debug.depth_{c}"] = files["depth"]
        seg = f"{stem}_{c}_seg.jpg"
        cv2.imwrite(os.path.join(out_dir, seg), segs[c])
        seg_files.append(seg)
        frame_hw.append(cams[c].size[::-1])
        print(f"  wrote {c} videos ({vhw[1]}x{vhw[0]}) and {seg}")

    extra = {}
    for c in cam_ids:
        o = obj[c]
        extra[f"boxes0_{c}"] = np.array([b for _, b in o["boxes"]], np.float32).reshape(-1, 4) if o else np.zeros((0, 4))
        extra[f"masks0_{c}"] = np.stack(o["masks"]).astype(bool) if o else np.zeros((0, 1, 1), bool)
        extra[f"box_names_{c}"] = np.array([p for p, _ in o["boxes"]]) if o else np.array([], str)
    rgb_ts = np.round(t_ref[sel] * 1e9).astype(np.int64)
    npz_path = os.path.join(out_dir, f"{stem}_aina.npz")
    np.savez_compressed(
        npz_path,
        rgb_ts=rgb_ts, time_s=t_s, frame_idx=np.stack([idx[c] for c in cam_ids]).astype(np.int32),
        points_world=points_world.astype(np.float32), depth=depth.astype(np.float32),
        tracks=tracks.astype(np.float32), vis=vis.astype(bool), obj_id=obj_id, point_cam=point_cam,
        hand_landmarks_world=hand_world.astype(np.float32),
        hand_keypoints_world=hand_world[:, :, list(WRIST_KEYPOINTS)].astype(np.float32),
        T_world_wrist=wrist_poses(hand_world), hand_uv=hand_uv.astype(np.float32),
        hand_source=hand_source, hand_reproj_px=hand_err.astype(np.float32),
        hand_kp2d=kp2d.astype(np.float32), hand_conf=conf.astype(np.float32),
        cameras=np.array(cam_ids), T_world_cam=np.stack([T_world_cam[c] for c in cam_ids]),
        K=np.stack([cams[c].K for c in cam_ids]),
        object_names=np.array(names), n_points=np.array([N_POINTS[prompts.index(p)] for p in names], np.int32),
        source=np.array("scene"), recording=np.array(rec.dir), calib_file=np.array(str(args.calib)),
        fps=np.float64(fps), start=np.int32(args.start), stride=np.int32(args.stride),
        frame_hw=np.array(frame_hw, np.int32), videos=np.array(json.dumps(videos)), seg_files=np.array(seg_files),
        **extra,
    )
    print(f"  wrote {npz_path} ({os.path.getsize(npz_path) / 1e6:.1f} MB)")

    # summary
    stats = {"recording": stem, "frames": T, "cameras": cam_ids, "objects": {}, "hands": {}}
    for k, name in enumerate(names):
        o = {}
        for i, c in enumerate(cam_ids):
            s = (obj_id == k) & (point_cam == i)
            if s.any():
                o[c] = {"points": int(s.sum()), "vis_pct": round(100 * float(vis[:, s].mean()), 1),
                        "3d_given_vis_pct": round(100 * float(np.isfinite(depth[:, s]).sum() / max(vis[:, s].sum(), 1)), 1)}
        cents = []
        for i, c in enumerate(cam_ids):   # first-frame centroid per camera: a direct calibration check
            s = (obj_id == k) & (point_cam == i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cents.append(np.nanmedian(points_world[0, s], 0) if s.any() else np.full(3, np.nan))
        if len(cents) >= 2 and np.isfinite(cents[0]).all() and np.isfinite(cents[1]).all():
            o["frame0_centroid_gap_cm"] = round(float(np.linalg.norm(cents[0] - cents[1])) * 100, 2)
        stats["objects"][name] = o
        print(f"  '{name}': " + ", ".join(f"{c}: {v['points']} pts, visible {v['vis_pct']}%, 3D {v['3d_given_vis_pct']}% "
                                          f"of visible" for c, v in o.items() if isinstance(v, dict))
              + (f"; frame-0 centroid gap between cameras {o['frame0_centroid_gap_cm']} cm (calibration check, "
                 "points sample different sides of the object)" if "frame0_centroid_gap_cm" in o else ""))
    for s, side in enumerate(("left", "right")):
        h = {"tracked_pct": round(100 * float((hand_source[:, s] > 0).mean()), 1),
             "triangulated_pct": round(100 * float((hand_source[:, s] == 2).mean()), 1),
             "single_view_pct": round(100 * float((hand_source[:, s] == 1).mean()), 1)}
        if np.isfinite(hand_err[:, s]).any():
            h["median_reproj_px"] = round(float(np.nanmedian(hand_err[:, s])), 2)
        for i, c in enumerate(cam_ids):
            h[f"detected_{c}_pct"] = round(100 * float(np.isfinite(kp2d[i, :, s, 0, 0]).mean()), 1)
        stats["hands"][side] = h
        print(f"  {side} hand: {h}")
    with open(os.path.join(out_dir, f"{stem}_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    if writer is not None:
        print(f"appending to {args.append_to}")
        try:
            with writer:
                writer.add(out_dir)
        except EpisodeError as e:
            sys.exit(f"not appended, output kept in {out_dir}: {e}")
        if not args.keep_output:
            shutil.rmtree(out_dir)
            print(f"  removed {out_dir} (pass --keep-output to keep it)")


if __name__ == "__main__":
    main()
