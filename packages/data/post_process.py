#!/usr/bin/env python3
"""
Post-process an Aria Gen2 .vrs recording.

Official AINA recipe (arXiv 2511.16661):
  1. Segment ONCE, on the first sampled frame only: Grounding DINO box -> SAM 2 mask per object
  2. Sample a fixed number of points inside each mask (N_POINTS, one count per prompt)
  3. CoTracker3 (--tracker online by default, or offline) follows all points through the whole clip
     (forward from the first frame), with per-point visibility
     (so you can see when an object is hidden in the hand). No re-detection, no re-segmentation.
  4. Stereo depth + lift to 3D: the two front SLAM cameras (slam-front-left / -right) are
     rectified to a virtual pinhole pair along their baseline, FoundationStereo gives the
     disparity d, depth is Z = f * B / d (B = norm of the translation between the two cameras).
     The 3D points are projected into the RGB camera, every tracked point takes the depth at
     its pixel, and the VIO glasses pose puts it in the world frame.

Output, in --out-dir/<vrs stem>/ (default output/<vrs stem>/), raw and complete (NaN = missing: hidden, no depth, no pose):
  <vrs stem>_aina.npz
    per frame (T):   rgb_ts (ns, device time), time_s, T_world_device (4,4), session_idx
    per point (N):   points_world (T,N,3), depth (T,N), tracks (T,N,2) in frame pixels, vis (T,N), obj_id (N)
    hands:           hand_landmarks_device / hand_landmarks_world (T,2,21,3), hand_uv (T,2,21,2) in frame
                     pixels; axis 1 = (left, right); on-device hand tracking, NaN = hand not tracked
                     hand_keypoints_world (T,2,3,3): wrist, index knuckle (MCP), pinky knuckle (MCP) = landmarks 5, 8, 17
                     T_world_wrist (T,2,4,4): wrist pose from those 3 points, origin at the wrist, +z to the
                     knuckles' midpoint, +x index -> pinky knuckle, +y = z cross x (out of the palm for the right
                     hand, the ORCA palm-frame axes; out of the back for the left); NaN = hand not tracked
    per object (K):  object_names, boxes0 (K,4), masks0 (K,H,W)   (on frame 0, frame pixels)
    metadata:        vrs_path, fps, start, stride, scale, tracker, frame_hw, n_points (per object),
                     video_file, overlay_file, depth_video_file, depth_video_range_m, seg_file, video_hw
  <vrs stem>_rgb.mp4   the same T frames, downscaled to --video-width, at fps: video frame t = row t
  <vrs stem>_overlay.mp4   same frames with the tracked points drawn on, for checking (--no-overlay to skip):
                       colour per object; filled = visible with 3D, hollow = visible but no depth,
                       small red = hidden; hand skeletons: left white, right yellow
  <vrs stem>_depth.mp4   FoundationStereo depth in the rectified left SLAM view (512x512), frame t = row t
                       (--no-depth-video to skip): turbo colour map shaded by surface normals, red = near (DEPTH_VIS_RANGE[0]), blue = far
                       (DEPTH_VIS_RANGE[1]), black = no depth / no SLAM pair
  <vrs stem>_seg.jpg   frame 0 (frame resolution) with the Grounding DINO boxes + SAM 2 masks drawn on

    python post_process.py --vrs ../../raw_data/rubiks1_20261123_231401.vrs
    python post_process.py --vrs ... --stop 60        # quick test on the first frames

Setup: run `bash scripts/setup.sh` from the repo root once. It installs the Python environment
with uv and puts FoundationStereo + its checkpoint in third_party/FoundationStereo, where this
script finds it by default (override with --fs-path or $FOUNDATION_STEREO_PATH). Needs an NVIDIA GPU.
Run with `uv run python packages/data/post_process.py ...` (or activate .venv).

Frames: "world" = on-device VIO odometry frame (gravity-aligned, Z up), only consistent within
one session_idx. "device" = glasses frame. T_a_b maps b-frame points into frame a. Missing data
is NaN, never zero.
"""
import argparse
import contextlib
import functools
import os
import sys
import warnings

import cv2
import numpy as np
import torch
from scipy.ndimage import map_coordinates

FS_CKPT = os.path.join("pretrained_models", "23-51-11", "model_best_bp2.pth")
FS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "FoundationStereo")
PROMPTS = ["rubiks cube", "laptop"]  # objects to track (Grounding DINO text prompts), in obj_id order
N_POINTS = [250, 450]                # points sampled per object, same order as PROMPTS
MAX_POSE_DT_NS = 50_000_000  # frames whose nearest valid VIO sample is further away get no pose
STEREO_SIZE = 768             # rectified SLAM image side (px), at native focal length: ~115 deg FOV
DEPTH_VIS_RANGE = (0.3, 0.8)  # m, mapped red..blue (turbo) in <stem>_depth.mp4 (farther clips to blue)
SETTLE_WINDOW_S = 0.5          # auto start: brightness must stay steady this long...
SETTLE_TOL = 0.02              # ...within this fraction of the frame's mean brightness
LIFT_RADIUS = 7              # px window (2r+1)^2 around a tracked point to read its depth


@contextlib.contextmanager
def quiet_stderr():
    """Silence C/C++ stderr (fd 2) inside the block: projectaria_tools/VRS log every stream activation and
    H.265 decoder fallback there, with no log-level setting. Python exceptions still propagate normally."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def as_matrix(T):
    """Sophus SE3 (or ndarray) -> (4,4) float64."""
    return np.asarray(T.to_matrix() if hasattr(T, "to_matrix") else T, dtype=np.float64)


def _unit(v):
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------- frames
def read_vrs(dp, start, stop, stride, scale):
    """Sampled RGB frames (downscaled by `scale`) and their device-time capture timestamps."""
    sid = dp.get_stream_id_from_label("camera-rgb")
    n = dp.get_num_data(sid)
    stop = n if stop <= 0 else min(stop, n)
    frames, ts = [], []
    for i in range(start, stop, stride):
        d, rec = dp.get_image_data_by_index(sid, i)
        if not d.is_valid():
            continue
        img = d.to_numpy_array()
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(img)
        ts.append(rec.capture_timestamp_ns)
    print(f"read {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]}) from {n} total")
    return frames, np.array(ts, dtype=np.int64)


def auto_start(frames, rgb_ts, T_wd, window_s=SETTLE_WINDOW_S, tol=SETTLE_TOL):
    """First sampled frame where the RGB auto-exposure has settled (mean brightness stays within tol of
    that frame's for the next window_s seconds) and the frame has a VIO pose. The glasses take ~1-3 s to
    adjust exposure at the start of a recording, and frames without a pose get no 3D anyway.
    Returns (index, settle index, pose index); settle/pose index is None if never reached."""
    b = np.array([f[::16, ::16].mean() for f in frames])
    fps = 1e9 / np.median(np.diff(rgb_ts)) if len(rgb_ts) > 1 else 10.0
    w = max(2, int(round(window_s * fps)))
    settle = next((i for i in range(len(b) - w + 1)
                   if np.abs(b[i:i + w] - b[i]).max() <= tol * max(b[i], 1.0)), None)
    pose = next((i for i in range(len(T_wd)) if np.isfinite(T_wd[i]).all()), None)
    return max(settle or 0, pose or 0), settle, pose


def read_synced(dp, rgb_ts):
    """For every RGB timestamp (master clock, as in AINA): the closest front SLAM pair and the
    glasses pose from the nearest valid VIO sample.

    Returns slam_l, slam_r (lists of uint8 images), T_world_device (T,4,4, NaN if no pose),
    session_idx (T,) int (-1 if no pose).
    """
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions, TrackingQuality, VioStatus

    sl_id = dp.get_stream_id_from_label("slam-front-left")
    sr_id = dp.get_stream_id_from_label("slam-front-right")
    vio_id = dp.get_stream_id_from_label("vio")
    if sl_id is None or sr_id is None or vio_id is None:
        sys.exit("VRS needs the streams slam-front-left, slam-front-right and vio.")

    slam_l, slam_r = [], []
    for t in rgb_ts:
        pair = []
        for sid in (sl_id, sr_id):
            d, _ = dp.get_image_data_by_time_ns(sid, int(t), TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
            pair.append(d.to_numpy_array() if d.is_valid() else None)
        slam_l.append(pair[0])
        slam_r.append(pair[1])

    # all valid VIO samples (same filter + pose chain as the old extract_hand_trajectories.py)
    v_ts, v_T, v_uid = [], [], []
    for i in range(dp.get_num_data(vio_id)):
        vio = dp.get_vio_data_by_index(vio_id, i)
        if vio.status != VioStatus.VALID or vio.pose_quality != TrackingQuality.GOOD:
            continue
        v_ts.append(vio.capture_timestamp_ns)
        v_T.append(as_matrix(vio.transform_odometry_bodyimu @ vio.transform_bodyimu_device))
        v_uid.append(str(getattr(vio, "frontend_session_uid", 0)))
    if not v_ts:
        sys.exit("No valid VIO samples (status VALID + quality GOOD) in this recording.")
    v_ts = np.array(v_ts, dtype=np.int64)
    v_T = np.stack(v_T)
    order = {u: k for k, u in enumerate(dict.fromkeys(v_uid))}
    v_sess = np.array([order[u] for u in v_uid], dtype=np.int32)

    j = np.clip(np.searchsorted(v_ts, rgb_ts), 1, len(v_ts) - 1)
    j = np.where(np.abs(v_ts[j - 1] - rgb_ts) <= np.abs(v_ts[j] - rgb_ts), j - 1, j)
    ok = np.abs(v_ts[j] - rgb_ts) <= MAX_POSE_DT_NS
    T_wd = np.where(ok[:, None, None], v_T[j], np.nan)
    sess = np.where(ok, v_sess[j], -1)
    n_slam = sum(a is not None and b is not None for a, b in zip(slam_l, slam_r))
    print(f"  SLAM pairs for {n_slam}/{len(rgb_ts)} frames, VIO pose for {ok.sum()}/{len(rgb_ts)}, "
          f"{v_sess.max() + 1} VIO session(s)")
    return slam_l, slam_r, T_wd, sess


HANDS = ("left", "right")
# Aria hand landmarks: 0-4 fingertips (thumb..pinky), 5 wrist, 6-7 thumb, 8-10 index, 11-13 middle,
# 14-16 ring, 17-19 pinky (proximal -> distal), 20 palm centre. Chains as in projectaria_tools' rerun helpers.
WRIST_KEYPOINTS = (5, 8, 17)  # wrist, index knuckle (MCP), pinky knuckle (MCP)
HAND_CHAINS = [(5, 6, 7, 0), (5, 8, 9, 10, 1), (5, 11, 12, 13, 2), (5, 14, 15, 16, 3), (5, 17, 18, 19, 4),
               (6, 8, 11, 14, 17)]


def read_hands(dp, rgb_ts):
    """On-device hand tracking interpolated to every RGB timestamp: (T,2,21,3) landmarks in the device frame,
    hand axis 0 = left, 1 = right, NaN where a hand is not tracked."""
    from projectaria_tools.core.sensor_data import TimeDomain

    out = np.full((len(rgb_ts), 2, 21, 3), np.nan)
    hid = dp.get_stream_id_from_label("handtracking")
    if hid is None:
        print("  no handtracking stream in this VRS: hand landmarks will be NaN")
        return out
    for t, ts in enumerate(rgb_ts):
        hp = dp.get_interpolated_hand_pose_data(hid, int(ts), TimeDomain.DEVICE_TIME)
        if hp is None:
            continue
        for h, side in enumerate(HANDS):
            hand = getattr(hp, f"{side}_hand", None)
            if hand is not None:
                out[t, h] = np.asarray(hand.landmark_positions_device, dtype=np.float64).reshape(21, 3)
    found = np.isfinite(out).all((2, 3)).mean(0)
    print(f"  hands tracked in {found[0]:.0%} (left) / {found[1]:.0%} (right) of frames")
    return out


def wrist_poses(lm):
    """(..., 21, 3) landmarks -> (..., 4, 4) wrist poses in the same frame (see T_world_wrist in the docstring;
    same axes as imitate_common.transforms.wrist_frame). NaN where a keypoint is missing or degenerate."""
    w, i, p = (lm[..., k, :] for k in WRIST_KEYPOINTS)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (i + p) / 2 - w
        z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        x = (p - i) - np.sum((p - i) * z, axis=-1, keepdims=True) * z
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    T = np.full(lm.shape[:-2] + (4, 4), np.nan)
    T[..., :3, 0], T[..., :3, 1], T[..., :3, 2], T[..., :3, 3] = x, np.cross(z, x), z, w
    T[..., 3, :] = [0, 0, 0, 1]
    bad = ~np.isfinite(T).all((-2, -1))
    T[bad] = np.nan
    return T


# ----------------------------------------------------------------------------- step 1: segmentation-frame boxes
def boxes_grounding_dino(rgb, prompts, device, box_thr):
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    img = Image.fromarray(rgb)
    boxes = []
    for p in prompts:
        text = p.lower().strip().rstrip(".") + "."
        inp = proc(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        try:
            res = proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=box_thr, target_sizes=[img.size[::-1]])[0]
        except TypeError:  # older transformers
            res = proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=box_thr, target_sizes=[img.size[::-1]])[0]
        if len(res["boxes"]) == 0:
            print(f"  Grounding DINO: nothing for '{p}' (lower --box-thr or pick another start frame with --start)")
            continue
        k = int(res["scores"].argmax())
        b = res["boxes"][k].cpu().numpy().tolist()
        print(f"  '{p}': box {[round(v) for v in b]} score {float(res['scores'][k]):.2f}")
        boxes.append((p, b))
    del model
    torch.cuda.empty_cache()
    return boxes


# ----------------------------------------------------------------------------- step 2: segmentation-frame masks
def masks_sam2(rgb, boxes):
    """SAM 2 in image mode on one frame only: one (H,W) bool mask per box, same order."""
    from ultralytics import SAM

    h, w = rgb.shape[:2]
    res = SAM("sam2.1_l.pt")(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), bboxes=[b for _, b in boxes], verbose=False)[0]
    if res.masks is None:
        return [np.zeros((h, w), bool) for _ in boxes]
    masks = []
    for m in res.masks.data.cpu().numpy():
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(m.astype(bool))
    for (p, _), m in zip(boxes, masks):
        print(f"  '{p}': mask {m.sum()} px ({m.mean():.3%} of frame)")
    return masks


# ----------------------------------------------------------------------------- step 3: CoTracker3
def _track_online(model, small, q_xy, device):
    """Online CoTracker3 over small[0..], all points queried on frame 0: feeds overlapping windows of
    2*step frames every step frames, as in CoTracker's online_demo.py. Returns tracks (T,N,2), vis (T,N)."""
    T, step = len(small), model.step
    frames = list(small) + [small[-1]] * max(0, 2 * step + 1 - len(small))  # pad very short segments
    q = torch.cat([torch.zeros_like(q_xy[:, :1]), q_xy], 1)[None].to(device)  # (1,N,3), query frame 0

    def run(chunk, first):
        video = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2)[None].float().to(device)
        return model(video, is_first_step=first, queries=q, add_support_grid=True)

    first = True
    for i in range(len(frames)):
        if i % step == 0 and i != 0:
            run(frames[max(0, i - 2 * step):i], first)
            first = False
    tracks, vis = run(frames[-(i % step) - step - 1:], first)
    return tracks[0, :T].cpu().numpy(), vis[0, :T].cpu().numpy()


def track_points(frames, masks0, n_points, device, max_side=512, tracker='online'):
    """Sample points inside each mask (masks0 are on frames[0]) and track them all with CoTracker3 from frame 0.
    Object k gets n_points[k] points, sampled from its mask downscaled to the tracking resolution.

    tracker="online": CoTracker3 online, sliding 16-frame windows, so GPU memory does not grow with clip length.
    tracker="offline": CoTracker3 offline, the whole clip in one call (more robust to occlusions, but memory
    grows with clip length); its feature encoder runs 16 frames at a time (identical result, less memory).

    Returns tracks (T,N,2) in frame pixel coords, vis (T,N) bool, obj_id (N,) index into masks0.
    """
    h, w = frames[0].shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for f in frames]
    sh, sw = small[0].shape[:2]
    rng = np.random.default_rng(0)
    pts, obj_id = [], []
    for k, (m, n_k) in enumerate(zip(masks0, n_points)):
        ms = cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
        ys, xs = np.nonzero(ms)
        if len(xs) == 0:
            print(f"  object {k}: empty mask, no points")
            continue
        sel = rng.choice(len(xs), size=min(int(n_k), len(xs)), replace=False)
        print(f"  object {k}: {m.sum()} mask px -> {len(sel)} points")
        pts += [[0, xs[i], ys[i]] for i in sel]  # (query frame, x, y)
        obj_id += [k] * len(sel)
    if not pts:
        raise SystemExit("all segmentation masks are empty; nothing to track")
    q = torch.tensor(pts, dtype=torch.float32)
    if tracker == "online":
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
        with torch.no_grad():
            tracks, vis = _track_online(model, small, q[:, 1:], device)
    else:
        video = torch.from_numpy(np.stack(small)).permute(0, 3, 1, 2)[None].float().to(device)
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        # its feature encoder runs on all frames at once below 200 frames, which dominates GPU memory on long
        # clips; 16 frames at a time gives identical features
        model.model.forward = functools.partial(model.model.forward, fmaps_chunk_size=16)
        with torch.no_grad():
            tracks, vis = model(video, queries=q[None].to(device))
        tracks, vis = tracks[0].cpu().numpy(), vis[0].cpu().numpy()
        del video
    tracks = tracks / s  # T,N,2 in frame coords
    vis = vis.astype(bool)  # T,N
    del model
    torch.cuda.empty_cache()
    print(f"  tracked {len(obj_id)} points over {len(frames)} frames ({tracker}), mean visibility {vis.mean():.0%}")
    return tracks, vis, np.array(obj_id, dtype=np.int32)


# ----------------------------------------------------------------------------- step 4: stereo depth
class RectifiedStereo:
    """
    Virtual rectified pinhole pair built from two calibrated (fisheye) cameras.

    Both virtual cameras keep the real camera centers and only change orientation:
    x axis = baseline direction, z axis = mean optical axis (made orthogonal to x).
    Disparity is then d = u_left - u_right = f * B / Z, with Z along the rectified z axis.
    """

    def __init__(self, calib_left, calib_right, size=STEREO_SIZE, f=None):
        """size: rectified image side (px). f: rectified focal length (px), default the left camera's native
        focal length, so the image centre keeps the SLAM camera's full resolution (a smaller f downsamples)."""
        T_l = as_matrix(calib_left.get_transform_device_camera())
        T_r = as_matrix(calib_right.get_transform_device_camera())
        c_l, c_r = T_l[:3, 3], T_r[:3, 3]
        b = c_r - c_l
        self.baseline = float(np.linalg.norm(b))
        x = b / self.baseline
        z = T_l[:3, 2] + T_r[:3, 2]
        z = _unit(z - (z @ x) * x)
        y = np.cross(z, x)
        self.R_dev_rect = np.stack([x, y, z], axis=1)   # columns: rect axes in device coords
        self.c_left = c_l

        w = h = int(size)
        self.size = (w, h)
        self.f = float(f) if f else float(calib_left.get_focal_lengths()[0])
        self.cx, self.cy = w / 2.0, h / 2.0

        self.map_l = self._build_map(calib_left, T_l)
        self.map_r = self._build_map(calib_right, T_r)
        ones = np.full((self.size[1], self.size[0]), 255, np.uint8)
        valid = cv2.remap(ones, *self.map_l, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.valid_left = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        valid_r = cv2.remap(ones, *self.map_r, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.valid_right = cv2.erode(valid_r.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    def _build_map(self, calib, T_c):
        w, h = self.size
        R_cr = T_c[:3, :3].T @ self.R_dev_rect       # rect-frame vector -> camera-frame vector
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        rays = np.stack([(us - self.cx) / self.f, (vs - self.cy) / self.f, np.ones((h, w))], -1)
        rays = rays.reshape(-1, 3) @ R_cr.T
        mx = np.full(len(rays), -1.0, np.float32)
        my = np.full(len(rays), -1.0, np.float32)
        for i, r in enumerate(rays):
            p = calib.project(r)
            if p is not None:
                mx[i], my[i] = p[0], p[1]
        return mx.reshape(h, w), my.reshape(h, w)

    def rectify(self, img_l, img_r):
        kw = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.remap(img_l, *self.map_l, **kw), cv2.remap(img_r, *self.map_r, **kw)

    def disparity_to_points(self, disp, z_min=0.15, z_max=2.0):
        """disparity (H,W) -> points (H,W,3) in the DEVICE frame and a validity mask."""
        with np.errstate(invalid="ignore", divide="ignore"):
            valid = np.isfinite(disp) & (disp > 0.5)
            Z = np.where(valid, self.f * self.baseline / np.where(valid, disp, 1.0), np.nan)
            valid &= (Z >= z_min) & (Z <= z_max)
        h, w = disp.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        X = (us - self.cx) / self.f * Z
        Y = (vs - self.cy) / self.f * Z
        P_rect = np.stack([X, Y, Z], -1)
        P_dev = P_rect @ self.R_dev_rect.T + self.c_left
        return P_dev, valid

    def compute_points(self, img_l, img_r, disparity_fn, z_range=(0.15, 3.0)):
        l, r = self.rectify(img_l, img_r)
        disp = disparity_fn(l, r)
        P, valid = self.disparity_to_points(disp, *z_range)
        valid &= self.valid_left
        # the matched right pixel (u - d, v) must be inside the right camera's view: the SLAM cameras are toed
        # out, so much of the left view has no right counterpart and FoundationStereo returns guesses there
        h, w = disp.shape
        ur = np.round(np.arange(w)[None, :] - np.nan_to_num(disp, nan=-1.0)).astype(np.int64)
        inside = (ur >= 0) & (ur < w)
        vs = np.broadcast_to(np.arange(h)[:, None], (h, w))
        inside[inside] = self.valid_right[vs[inside], ur[inside]]
        valid &= inside
        return P, valid, disp, (l, r)


class FoundationStereoDisparity:
    """FoundationStereo (NVlabs) as a disparity function: (left_u8, right_u8) -> float32 disparity px,
    NaN where invalid. Mirrors AINA's foundation_stereo_wrapper.py and FoundationStereo's run_demo.py."""

    def __init__(self, fs_path, iters=32):
        if not fs_path or not os.path.isdir(fs_path):
            sys.exit(f"FoundationStereo not found at {fs_path}: run `bash scripts/setup.sh` from the repo root, "
                     "or pass --fs-path / set FOUNDATION_STEREO_PATH.")
        ckpt = os.path.join(fs_path, FS_CKPT)
        cfg_path = os.path.join(os.path.dirname(ckpt), "cfg.yaml")
        if not (os.path.isfile(ckpt) and os.path.isfile(cfg_path)):
            sys.exit(f"FoundationStereo checkpoint missing: need {ckpt} and {cfg_path}.")
        if not torch.cuda.is_available():
            sys.exit("FoundationStereo needs a CUDA GPU.")
        sys.path.insert(0, os.path.abspath(fs_path))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = iters
        with warnings.catch_warnings():  # its DINOv2 (torch.hub) always reports its xFormers check; we use PyTorch SDPA
            warnings.filterwarnings("ignore", message="xFormers is", category=UserWarning)
            self.model = FoundationStereo(cfg)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)  # no FS source patch needed
        self.model.load_state_dict(state["model"])
        self.model.cuda().eval()
        self.InputPadder = InputPadder
        self.iters = iters

    @staticmethod
    def _to_tensor(img):
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, -1)  # grayscale SLAM -> 3 channels
        return torch.as_tensor(img).cuda().float()[None].permute(0, 3, 1, 2)

    def __call__(self, left, right):
        img0, img1 = self._to_tensor(left), self._to_tensor(right)
        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            disp = self.model.forward(img0, img1, iters=self.iters, test_mode=True)
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(left.shape[:2]).astype(np.float32)
        xx = np.arange(disp.shape[1])[None, :]
        disp[(xx - disp) < 0] = np.nan  # would match outside the right image
        return disp


class FastProjector:
    """
    Vectorised projection of device-frame points into a camera, using a lookup table
    sampled once from calib.project() on the tangent plane (x/z, y/z) and bilinearly
    interpolated. Accurate to a fraction of a pixel; far faster than a Python loop.
    """

    def __init__(self, calib, T_dev_cam, max_tan=2.5, n=257):
        T = as_matrix(T_dev_cam)
        self.R, self.t = T[:3, :3], T[:3, 3]
        self.max_tan, self.n = max_tan, n
        ts = np.linspace(-max_tan, max_tan, n)
        lut = np.full((n, n, 2), np.nan, np.float32)
        for i in range(n):          # row = y/z
            for j in range(n):      # col = x/z
                p = calib.project(np.array([ts[j], ts[i], 1.0]))
                if p is not None:
                    lut[i, j] = p
        self.lut_u, self.lut_v = lut[..., 0], lut[..., 1]
        self.step = ts[1] - ts[0]

    def to_camera(self, P_dev):
        return (P_dev - self.t) @ self.R

    def project(self, P_dev):
        """(N,3) device-frame points -> (N,2) pixels, NaN if not visible."""
        Pc = self.to_camera(P_dev)
        z = Pc[:, 2]
        ok = z > 1e-6
        zz = np.where(ok, z, 1.0)
        ci = (Pc[:, 0] / zz + self.max_tan) / self.step
        ri = (Pc[:, 1] / zz + self.max_tan) / self.step
        ok &= (ci >= 0) & (ci <= self.n - 1) & (ri >= 0) & (ri <= self.n - 1)
        out = np.full((len(P_dev), 2), np.nan)
        if ok.any():
            coords = np.vstack([ri[ok], ci[ok]])
            out[ok, 0] = map_coordinates(self.lut_u, coords, order=1, mode="nearest")
            out[ok, 1] = map_coordinates(self.lut_v, coords, order=1, mode="nearest")
        return out


def zbuffer_points(P_dev, proj, pix_scale, out_hw):
    """Splat device-frame points into an image of size out_hw (the downscaled RGB frame),
    keeping the point nearest to the RGB camera per pixel. Returns (H,W,3), NaN where empty."""
    H, W = out_hw
    uv = proj.project(P_dev) * pix_scale
    z = proj.to_camera(P_dev)[:, 2]
    ok = np.isfinite(uv).all(1)
    u = np.round(uv[ok, 0]).astype(np.int64)
    v = np.round(uv[ok, 1]).astype(np.int64)
    P, z = P_dev[ok], z[ok]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, P, z = u[inside], v[inside], P[inside], z[inside]
    buf = np.full((H, W, 3), np.nan)
    if len(u):
        lin = v * W + u
        order = np.lexsort((z, lin))        # by pixel, then nearest first
        lin_s = lin[order]
        first = np.r_[True, lin_s[1:] != lin_s[:-1]]
        keep = order[first]
        buf[v[keep], u[keep]] = P[keep]
    return buf


def lift_tracks(tracks, vis, frame_hw, slam_l, slam_r, T_wd, stereo, fs, proj_rgb, pix_scale, depth_video=False):
    """Give every visible tracked point a 3D position: device frame from stereo depth, then world.

    Returns points_world (T,N,3) and depth (T,N) (distance along the RGB optical axis), NaN where
    the point is hidden, has no stereo depth nearby, or the frame has no SLAM pair / VIO pose.
    With depth_video, also returns one uint8 depth image per frame (rectified left view, see depth_to_color),
    and runs FoundationStereo on every frame with a SLAM pair, not only the ones that need 3D points.
    """
    T, N = vis.shape
    H, W = frame_hw
    r = LIFT_RADIUS
    pts_world = np.full((T, N, 3), np.nan)
    depth = np.full((T, N), np.nan)
    color = [np.zeros((*stereo.size[::-1], 3), np.uint8) for _ in range(T)] if depth_video else None
    for t in range(T):
        if slam_l[t] is None or slam_r[t] is None:
            continue
        need_3d = np.isfinite(T_wd[t]).all() and vis[t].any()
        if not (need_3d or depth_video):
            continue
        P, valid, disp, _ = stereo.compute_points(slam_l[t], slam_r[t], disparity_fn=fs)
        if depth_video:
            color[t] = depth_to_color(stereo, disp, valid)
        if not need_3d:
            continue
        buf = zbuffer_points(P[valid], proj_rgb, pix_scale, (H, W))
        for n in np.nonzero(vis[t])[0]:
            u, v = int(round(tracks[t, n, 0])), int(round(tracks[t, n, 1]))
            if u < -r or v < -r or u >= W + r or v >= H + r:
                continue
            win = buf[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1].reshape(-1, 3)
            win = win[np.isfinite(win).all(1)]
            if len(win) == 0:
                continue
            p_dev = np.median(win, axis=0)
            depth[t, n] = proj_rgb.to_camera(p_dev[None])[0, 2]
            pts_world[t, n] = T_wd[t, :3, :3] @ p_dev + T_wd[t, :3, 3]
        if (t + 1) % 25 == 0 or t == T - 1:
            print(f"  depth {t + 1}/{T} frames")
    return pts_world, depth, color


def depth_to_color(stereo, disp, valid):
    """Disparity in the rectified left view -> BGR uint8 depth image: turbo colour map over DEPTH_VIS_RANGE
    (red = near, blue = far, farther clips to blue), darkened by Lambert shading of the surface normals so
    thin objects and edges show (a 1-2 cm laptop is invisible in colour alone). Black = no depth."""
    lo, hi = DEPTH_VIS_RANGE
    with np.errstate(invalid="ignore", divide="ignore"):
        z = stereo.f * stereo.baseline / disp
    ok = valid & np.isfinite(z)
    level = np.nan_to_num(255 * (hi - np.clip(z, lo, hi)) / (hi - lo)).astype(np.uint8)
    img = cv2.applyColorMap(level, cv2.COLORMAP_TURBO).astype(np.float32)
    # normals of the rectified-frame surface (x right, y down, z forward), lit from the upper left of the camera
    h, w = z.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    zf = np.where(ok, z, np.nan)
    P = np.stack([(us - stereo.cx) / stereo.f * zf, (vs - stereo.cy) / stereo.f * zf, zf], -1)
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    with np.errstate(invalid="ignore"):
        n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.4, -0.5, -1.0]) / np.linalg.norm([-0.4, -0.5, -1.0])
    shade = np.nan_to_num(np.abs(n @ light), nan=1.0)
    img *= (0.3 + 0.7 * shade)[..., None]
    img[~ok] = 0
    return img.astype(np.uint8)


# ----------------------------------------------------------------------------- output
OVERLAY_COLORS = [(0, 220, 0), (255, 140, 0), (255, 0, 255), (0, 200, 255), (255, 255, 0)]  # BGR, per object
HIDDEN_COLOR = (0, 0, 255)


HAND_COLORS = [(255, 255, 255), (0, 255, 255)]  # BGR: left white, right yellow


def draw_overlay(img_bgr, uv, vis_t, has3d_t, obj_id, names, t_sec, hand_uv_t=None):
    """Tracked points on one (already downscaled, BGR) frame. Per object colour; filled dot = visible with
    3D position, hollow = visible but no depth, small red dot = hidden (CoTracker's guess).
    hand_uv_t (2,21,2): hand keypoints (left white, right yellow) drawn as a skeleton."""
    out = img_bgr.copy()
    if hand_uv_t is not None:
        for h in range(2):
            pts = hand_uv_t[h]
            if not np.isfinite(pts).any():
                continue
            for chain in HAND_CHAINS:
                for a, b in zip(chain[:-1], chain[1:]):
                    if np.isfinite(pts[[a, b]]).all():
                        cv2.line(out, tuple(int(x) for x in pts[a]), tuple(int(x) for x in pts[b]),
                                 HAND_COLORS[h], 1, cv2.LINE_AA)
            for q in pts:
                if np.isfinite(q).all():
                    cv2.circle(out, (int(q[0]), int(q[1])), 2, HAND_COLORS[h], -1)
    for (u, v), vv, h3, o in zip(uv, vis_t, has3d_t, obj_id):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        c = (int(u), int(v))
        if not vv:
            cv2.circle(out, c, 1, HIDDEN_COLOR, -1)
        elif h3:
            cv2.circle(out, c, 3, OVERLAY_COLORS[o % len(OVERLAY_COLORS)], -1)
        else:
            cv2.circle(out, c, 3, OVERLAY_COLORS[o % len(OVERLAY_COLORS)], 1)
    parts = [f"t={t_sec:.1f}s"]
    for k, name in enumerate(names):
        sel = obj_id == k
        if sel.any():
            parts.append(f"{name} {has3d_t[sel].mean():.0%} 3D")
    text = " | ".join(parts)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for k, name in enumerate(names):  # legend
        y = 40 + 18 * k
        cv2.circle(out, (14, y - 4), 4, OVERLAY_COLORS[k % len(OVERLAY_COLORS)], -1)
        cv2.putText(out, name, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_segmentation(rgb, boxes, masks0):
    """Segmentation frame with the Grounding DINO boxes and SAM 2 masks drawn on (BGR, frame resolution)."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for k, ((name, b), m) in enumerate(zip(boxes, masks0)):
        c = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
        out[m] = (0.45 * out[m] + 0.55 * np.array(c)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, c, 2)
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"{name} ({m.sum()} px)"
        cv2.putText(out, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
    return out


def save_outputs(out_dir, stem, args, frames, rgb_ts, T_world_device, session_idx, boxes, masks0,
                 tracks, vis, obj_id, points_world, depth, hand_lm_device, hand_uv, depth_color=None):
    """Write <stem>_aina.npz (all results, raw, NaN for missing), <stem>_rgb.mp4 (the same T frames,
    downscaled to --video-width) and, unless --no-overlay, <stem>_overlay.mp4 (same frames with the
    tracked points drawn on). Frame t of the videos is row t of every per-frame array."""
    os.makedirs(out_dir, exist_ok=True)
    H, W = frames[0].shape[:2]
    fps = float(1e9 / np.median(np.diff(rgb_ts))) if len(rgb_ts) > 1 else 0.0

    video_file, overlay_file, video_hw = "", "", (0, 0)
    if args.video_width > 0:
        vw_, vh_ = int(args.video_width), int(round(H * args.video_width / W))

        def open_writer(name):
            path = os.path.join(out_dir, name)
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw_, vh_))
            if not w.isOpened():
                sys.exit(f"could not open a video writer for {path}")
            return w, path

        video_file = f"{stem}_rgb.mp4"
        writer, vpath = open_writer(video_file)
        ov_writer = ov_path = None
        if args.overlay:
            overlay_file = f"{stem}_overlay.mp4"
            ov_writer, ov_path = open_writer(overlay_file)
        names = [p for p, _ in boxes]
        has3d = np.isfinite(depth)
        t_sec = (rgb_ts - rgb_ts[0]) / 1e9
        s_v = vw_ / W  # frame pixels -> video pixels
        for t, f in enumerate(frames):
            small = cv2.cvtColor(cv2.resize(f, (vw_, vh_), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            writer.write(small)
            if ov_writer is not None:
                ov_writer.write(draw_overlay(small, tracks[t] * s_v, vis[t], has3d[t], obj_id, names, t_sec[t],
                                             hand_uv[t] * s_v))
        writer.release()
        if ov_writer is not None:
            ov_writer.release()
        video_hw = (vh_, vw_)
        for path in (vpath, ov_path):
            if path:
                print(f"  wrote {path}  ({len(frames)} frames, {vw_}x{vh_}, {fps:.2f} fps, "
                      f"{os.path.getsize(path) / 1e6:.1f} MB)")

    depth_video_file = ""
    if depth_color is not None:
        depth_video_file = f"{stem}_depth.mp4"
        dpath = os.path.join(out_dir, depth_video_file)
        dh, dw = depth_color[0].shape[:2]
        dw_ = cv2.VideoWriter(dpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dw, dh))
        if not dw_.isOpened():
            sys.exit(f"could not open a video writer for {dpath}")
        for img in depth_color:
            dw_.write(img)
        dw_.release()
        print(f"  wrote {dpath}  ({len(depth_color)} frames, {dw}x{dh}, {os.path.getsize(dpath) / 1e6:.1f} MB)")

    seg_file = f"{stem}_seg.jpg"
    seg_path = os.path.join(out_dir, seg_file)
    cv2.imwrite(seg_path, draw_segmentation(frames[0], boxes, masks0))
    print(f"  wrote {seg_path}  (frame 0 boxes + SAM 2 masks)")

    npz_path = os.path.join(out_dir, f"{stem}_aina.npz")
    hand_lm_world = (np.einsum("tij,thkj->thki", T_world_device[:, :3, :3], hand_lm_device)
                     + T_world_device[:, None, None, :3, 3])
    np.savez_compressed(
        npz_path,
        # per frame (T)
        rgb_ts=rgb_ts.astype(np.int64),
        time_s=(rgb_ts - rgb_ts[0]) / 1e9,
        T_world_device=T_world_device,
        session_idx=session_idx.astype(np.int32),
        # per point (N)
        points_world=points_world.astype(np.float32),
        depth=depth.astype(np.float32),
        tracks=tracks.astype(np.float32),
        vis=vis.astype(bool),
        obj_id=obj_id.astype(np.int32),
        # hands (on-device hand tracking), axis 1 = (left, right), 21 Aria landmarks
        hand_landmarks_device=hand_lm_device.astype(np.float32),
        hand_landmarks_world=hand_lm_world.astype(np.float32),
        hand_uv=hand_uv.astype(np.float32),
        hand_keypoints_world=hand_lm_world[:, :, list(WRIST_KEYPOINTS)].astype(np.float32),
        T_world_wrist=wrist_poses(hand_lm_world),
        # per object (K)
        object_names=np.array([p for p, _ in boxes]),
        boxes0=np.array([b for _, b in boxes], dtype=np.float32).reshape(-1, 4),
        masks0=np.stack(masks0).astype(bool),
        # metadata
        vrs_path=np.array(args.vrs),
        fps=np.float64(fps),
        start=np.int32(args.start),
        stride=np.int32(args.stride),
        scale=np.float64(args.scale),
        tracker=args.tracker,
        frame_hw=np.array([H, W], dtype=np.int32),
        n_points=np.array(N_POINTS, np.int32),
        video_file=np.array(video_file),
        overlay_file=np.array(overlay_file),
        depth_video_file=np.array(depth_video_file),
        depth_video_range_m=np.array(DEPTH_VIS_RANGE, np.float32),
        seg_file=np.array(seg_file),
        video_hw=np.array(video_hw, dtype=np.int32),
    )
    print(f"  wrote {npz_path}  ({os.path.getsize(npz_path) / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--start", type=int, default=None,
                    help="first RGB frame; default: auto, the first frame with settled exposure and a VIO pose")
    ap.add_argument("--stop", type=int, default=0, help="0 = until the end")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="downscale frames (memory/speed)")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--tracker", choices=["online", "offline"], default="online",
                    help="CoTracker3 model: online (sliding windows, memory independent of clip length) or offline "
                         "(whole clip at once, better through occlusions, needs more memory) (default online)")
    ap.add_argument("--fs-path", default=os.environ.get("FOUNDATION_STEREO_PATH") or os.path.normpath(FS_DEFAULT),
                    help="FoundationStereo repo root (default: $FOUNDATION_STEREO_PATH, else third_party/FoundationStereo)")
    ap.add_argument("--out-dir", default="output", help="results go to <out-dir>/<vrs stem>/ (default: output/)")
    ap.add_argument("--video-width", type=int, default=640,
                    help="width of the saved RGB video, height keeps the aspect ratio (0 = no video)")
    ap.add_argument("--no-overlay", dest="overlay", action="store_false",
                    help="don't write <stem>_overlay.mp4 (the video with the tracked points drawn on)")
    ap.add_argument("--no-depth-video", dest="depth_video", action="store_false",
                    help="don't write <stem>_depth.mp4 (FoundationStereo depth, turbo colour map, red = near)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    prompts = PROMPTS
    assert len(N_POINTS) == len(PROMPTS), "N_POINTS needs one count per entry in PROMPTS"
    args.vrs = os.path.abspath(args.vrs)
    print("device:", args.device)

    from projectaria_tools.core import data_provider

    with quiet_stderr():
        dp = data_provider.create_vrs_data_provider(args.vrs)
    if dp is None:
        sys.exit(f"could not open {args.vrs}")
    calib = dp.get_device_calibration()
    cal_rgb = calib.get_camera_calib("camera-rgb")
    cal_l = calib.get_camera_calib("slam-front-left")
    cal_r = calib.get_camera_calib("slam-front-right")
    if cal_rgb is None or cal_l is None or cal_r is None:
        sys.exit("Calibration for camera-rgb / slam-front-left / slam-front-right is missing.")

    with quiet_stderr():  # our own progress prints go to stdout and still show
        frames, rgb_ts = read_vrs(dp, args.start or 0, args.stop, args.stride, args.scale)
        slam_l, slam_r, T_world_device, session_idx = read_synced(dp, rgb_ts)
        hand_lm_device = read_hands(dp, rgb_ts)

    if args.start is None:  # auto: skip the auto-exposure ramp and the frames before the first VIO pose
        k, settle, pose = auto_start(frames, rgb_ts, T_world_device)
        if settle is None:
            print("  WARNING: brightness never settled; check the segmentation image")
        print(f"  auto start at sampled frame {k} = RGB frame {k * args.stride} ({k / max(len(frames), 1):.0%} "
              f"of the clip; exposure settled at {settle}, first pose at {pose})")
        if k >= len(frames) - 1:
            sys.exit("auto start left no frames; pass --start N")
        frames, rgb_ts, T_world_device, session_idx = frames[k:], rgb_ts[k:], T_world_device[k:], session_idx[k:]
        slam_l, slam_r, hand_lm_device = slam_l[k:], slam_r[k:], hand_lm_device[k:]
        args.start = k * args.stride  # recorded in the npz as the RGB frame the clip starts at

    seg_rgb = frames[0]
    print("step 1: boxes on frame 0")
    boxes = boxes_grounding_dino(seg_rgb, prompts, args.device, args.box_thr)
    if not boxes:
        raise SystemExit("no boxes on frame 0; lower --box-thr or pick another start frame with --start")

    print("step 2: SAM2 masks on frame 0")
    masks0 = masks_sam2(seg_rgb, boxes)

    print("step 3: CoTracker3 point tracking")
    tracks, vis, obj_id = track_points(frames, masks0, N_POINTS,
                                      args.device, tracker=args.tracker)

    print("step 4: stereo depth (FoundationStereo) + lift to 3D")
    fs = FoundationStereoDisparity(args.fs_path)
    print("  building rectification + RGB projection tables (one-off)...")
    stereo = RectifiedStereo(cal_l, cal_r)
    print(f"  baseline {stereo.baseline * 100:.2f} cm, rectified f {stereo.f:.1f} px, size {stereo.size}")
    proj_rgb = FastProjector(cal_rgb, cal_rgb.get_transform_device_camera())
    H, W = frames[0].shape[:2]
    pix_scale = W / float(cal_rgb.get_image_size()[0])  # calibration pixels -> downscaled frame pixels
    hand_uv = (proj_rgb.project(hand_lm_device.reshape(-1, 3)) * pix_scale).reshape(*hand_lm_device.shape[:3], 2)
    points_world, depth, depth_color = lift_tracks(tracks, vis, (H, W), slam_l, slam_r, T_world_device,
                                                  stereo, fs, proj_rgb, pix_scale, args.depth_video)
    del fs
    torch.cuda.empty_cache()

    print("saving results")
    stem = os.path.splitext(os.path.basename(args.vrs))[0]
    save_outputs(os.path.join(args.out_dir, stem), stem, args, frames, rgb_ts,
                 T_world_device, session_idx, boxes, masks0, tracks, vis, obj_id, points_world, depth,
                 hand_lm_device, hand_uv, depth_color)

    # Sanity summary. "jitter" = how much each point moves in world during the first second that
    # has 3D data (VIO often has no valid pose for the first ~0.5 s); objects should be still then.
    t_s = (rgb_ts - rgb_ts[0]) / 1e9
    for k, (p, _) in enumerate(boxes):
        sel = obj_id == k
        if not sel.any():
            continue
        pw = points_world[:, sel]                       # (T,n,3)
        has = np.isfinite(depth[:, sel])
        frac = has.sum() / max(vis[:, sel].sum(), 1)
        med = np.nanmedian(depth[:, sel]) if has.any() else np.nan
        jitter = np.nan
        f3d = np.nonzero(has.any(1))[0]
        if len(f3d):
            win = (t_s >= t_s[f3d[0]]) & (t_s <= t_s[f3d[0]] + 1.0)
            pts = pw[win]                               # (t,n,3)
            ok = np.isfinite(pts).all(2).sum(0) >= 2    # points seen with depth in >= 2 window frames
            if ok.any():
                with warnings.catch_warnings():         # all-NaN slices are expected (hidden / no depth)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    jitter = np.median(np.linalg.norm(np.nanstd(pts[:, ok], axis=0), axis=1))
        print(f"  '{p}': depth for {frac:.0%} of visible points, median depth {med:.3f} m, "
              f"median point jitter over first 1 s with 3D {'n/a' if np.isnan(jitter) else f'{jitter * 100:.1f} cm'}")


if __name__ == "__main__":
    main()
