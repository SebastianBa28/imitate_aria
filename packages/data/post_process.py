#!/usr/bin/env python3
"""
Post-process an Aria Gen2 .vrs recording.

Official AINA recipe (arXiv 2511.16661):
  1. Segment ONCE, on one frame only (the first, or --seg-frame): Grounding DINO box (or --roi) -> SAM 2 mask per object
  2. Sample --n-points points inside each of those masks
  3. CoTracker3 follows all points through the whole clip (forward and backward from the segmentation
     frame), with per-point visibility
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
    per object (K):  object_names, boxes0 (K,4), masks0 (K,H,W)   (on frame seg_frame, frame pixels)
    metadata:        vrs_path, fps, start, stride, scale, seg_frame, frame_hw, n_points, video_file, overlay_file, seg_file, video_hw
  <vrs stem>_rgb.mp4   the same T frames, downscaled to --video-width, at fps: video frame t = row t
  <vrs stem>_overlay.mp4   same frames with the tracked points drawn on, for checking (--no-overlay to skip):
                       colour per object; filled = visible with 3D, hollow = visible but no depth,
                       small red = hidden; hand skeletons: left white, right yellow
  <vrs stem>_seg.jpg   segmentation frame (frame resolution) with the Grounding DINO boxes + SAM 2 masks drawn on

    python post_process.py --vrs ../../sample_data/demo20_20261116_175133.vrs --prompt "red cube" --prompt "cardboard box"
    python post_process.py --vrs ... --roi            # draw the boxes yourself on the segmentation frame instead of Grounding DINO
    python post_process.py --vrs ... --stop 60        # quick test on the first frames
    python post_process.py --vrs ... --seg-frame 5    # segment on output frame 5 (e.g. frame 0 badly exposed)

Setup: run `bash scripts/setup.sh` from the repo root once. It installs the Python environment
with uv and puts FoundationStereo + its checkpoint in third_party/FoundationStereo, where this
script finds it by default (override with --fs-path or $FOUNDATION_STEREO_PATH). Needs an NVIDIA GPU.
Run with `uv run python packages/data/post_process.py ...` (or activate .venv).

Frames: "world" = on-device VIO odometry frame (gravity-aligned, Z up), only consistent within
one session_idx. "device" = glasses frame. T_a_b maps b-frame points into frame a. Missing data
is NaN, never zero.
"""
import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch
from scipy.ndimage import map_coordinates

FS_CKPT = os.path.join("pretrained_models", "23-51-11", "model_best_bp2.pth")
FS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "FoundationStereo")
MAX_POSE_DT_NS = 50_000_000  # frames whose nearest valid VIO sample is further away get no pose
LIFT_RADIUS = 7              # px window (2r+1)^2 around a tracked point to read its depth


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


# ----------------------------------------------------------------------------- step 1: segmentation-frame boxes
def boxes_grounding_dino(rgb, prompts, device, model_id, box_thr, text_thr):
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

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
                out, inp.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[img.size[::-1]])[0]
        except TypeError:  # older transformers
            res = proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=box_thr, text_threshold=text_thr, target_sizes=[img.size[::-1]])[0]
        if len(res["boxes"]) == 0:
            print(f"  Grounding DINO: nothing for '{p}' (lower --box-thr or use --roi)")
            continue
        k = int(res["scores"].argmax())
        b = res["boxes"][k].cpu().numpy().tolist()
        print(f"  '{p}': box {[round(v) for v in b]} score {float(res['scores'][k]):.2f}")
        boxes.append((p, b))
    del model
    torch.cuda.empty_cache()
    return boxes


def boxes_manual(rgb, prompts):
    boxes = []
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for p in prompts:
        x, y, w, h = cv2.selectROI(f"draw a box around: {p}  (ENTER to confirm)", bgr, showCrosshair=False)
        cv2.destroyAllWindows()
        if w > 0 and h > 0:
            boxes.append((p, [x, y, x + w, y + h]))
    return boxes


# ----------------------------------------------------------------------------- step 2: segmentation-frame masks
def masks_sam2(rgb, boxes, sam_model):
    """SAM 2 in image mode on one frame only: one (H,W) bool mask per box, same order."""
    from ultralytics import SAM

    h, w = rgb.shape[:2]
    res = SAM(sam_model)(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), bboxes=[b for _, b in boxes], verbose=False)[0]
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
def track_points(frames, masks0, n_points, device, seg_frame=0, max_side=512):
    """Sample up to n_points inside each mask (masks0 are on frames[seg_frame]) and track them all with
    CoTracker3. The offline model tracks forward and backward from the query frame, so every frame gets a track.

    Returns tracks (T,N,2) in frame pixel coords, vis (T,N) bool, obj_id (N,) index into masks0.
    """
    h, w = frames[0].shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for f in frames]
    sh, sw = small[0].shape[:2]
    rng = np.random.default_rng(0)
    pts, obj_id = [], []
    for k, m in enumerate(masks0):
        ms = cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
        ys, xs = np.nonzero(ms)
        if len(xs) == 0:
            print(f"  object {k}: empty mask, no points")
            continue
        sel = rng.choice(len(xs), size=min(n_points, len(xs)), replace=False)
        pts += [[seg_frame, xs[i], ys[i]] for i in sel]  # (query frame, x, y)
        obj_id += [k] * len(sel)
    if not pts:
        raise SystemExit("all segmentation masks are empty; nothing to track")
    q = torch.tensor(pts, dtype=torch.float32, device=device)[None]
    video = torch.from_numpy(np.stack(small)).permute(0, 3, 1, 2)[None].float().to(device)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    with torch.no_grad():
        tracks, vis = model(video, queries=q)
    tracks = tracks[0].cpu().numpy() / s  # T,N,2 in frame coords
    vis = vis[0].cpu().numpy().astype(bool)  # T,N
    del model, video
    torch.cuda.empty_cache()
    print(f"  tracked {len(obj_id)} points over {len(frames)} frames, mean visibility {vis.mean():.0%}")
    return tracks, vis, np.array(obj_id, dtype=np.int32)


# ----------------------------------------------------------------------------- step 4: stereo depth
class RectifiedStereo:
    """
    Virtual rectified pinhole pair built from two calibrated (fisheye) cameras.

    Both virtual cameras keep the real camera centers and only change orientation:
    x axis = baseline direction, z axis = mean optical axis (made orthogonal to x).
    Disparity is then d = u_left - u_right = f * B / Z, with Z along the rectified z axis.
    """

    def __init__(self, calib_left, calib_right, hfov_deg=130.0, size=None):
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

        w, h = size if size else (int(v) for v in calib_left.get_image_size())
        self.size = (int(w), int(h))
        self.f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.cx, self.cy = w / 2.0, h / 2.0

        self.map_l = self._build_map(calib_left, T_l)
        self.map_r = self._build_map(calib_right, T_r)
        ones = np.full((self.size[1], self.size[0]), 255, np.uint8)
        valid = cv2.remap(ones, *self.map_l, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.valid_left = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

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


def lift_tracks(tracks, vis, frame_hw, slam_l, slam_r, T_wd, stereo, fs, proj_rgb, pix_scale):
    """Give every visible tracked point a 3D position: device frame from stereo depth, then world.

    Returns points_world (T,N,3) and depth (T,N) (distance along the RGB optical axis), NaN where
    the point is hidden, has no stereo depth nearby, or the frame has no SLAM pair / VIO pose.
    """
    T, N = vis.shape
    H, W = frame_hw
    r = LIFT_RADIUS
    pts_world = np.full((T, N, 3), np.nan)
    depth = np.full((T, N), np.nan)
    for t in range(T):
        if slam_l[t] is None or slam_r[t] is None or not np.isfinite(T_wd[t]).all() or not vis[t].any():
            continue
        P, valid, _, _ = stereo.compute_points(slam_l[t], slam_r[t], disparity_fn=fs)
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
    return pts_world, depth


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
                 tracks, vis, obj_id, points_world, depth, hand_lm_device, hand_uv):
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

    seg_file = f"{stem}_seg.jpg"
    seg_path = os.path.join(out_dir, seg_file)
    cv2.imwrite(seg_path, draw_segmentation(frames[args.seg_frame], boxes, masks0))
    print(f"  wrote {seg_path}  (frame {args.seg_frame} boxes + SAM 2 masks)")

    npz_path = os.path.join(out_dir, f"{stem}_aina.npz")
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
        hand_landmarks_world=(np.einsum("tij,thkj->thki", T_world_device[:, :3, :3], hand_lm_device)
                              + T_world_device[:, None, None, :3, 3]).astype(np.float32),
        hand_uv=hand_uv.astype(np.float32),
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
        seg_frame=np.int32(args.seg_frame),
        frame_hw=np.array([H, W], dtype=np.int32),
        n_points=np.int32(args.n_points),
        video_file=np.array(video_file),
        overlay_file=np.array(overlay_file),
        seg_file=np.array(seg_file),
        video_hw=np.array(video_hw, dtype=np.int32),
    )
    print(f"  wrote {npz_path}  ({os.path.getsize(npz_path) / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--prompt", action="append", default=None,
                    help="object to track, repeat for several (default: 'red cube' and 'cardboard box')")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=0, help="0 = until the end")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0, help="downscale frames (memory/speed)")
    ap.add_argument("--seg-frame", type=int, default=15,
                    help="output frame (after --start/--stride, = row t of the npz/video) to segment on (default 0)")
    ap.add_argument("--roi", action="store_true", help="draw the segmentation-frame boxes by hand instead of Grounding DINO")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.2)
    ap.add_argument("--sam-model", default="sam2.1_l.pt",
                    help="SAM 2.1 checkpoint (ultralytics auto-downloads): sam2.1_t/s/b/l.pt (default l, largest)")
    ap.add_argument("--n-points", type=int, default=250, help="points sampled per object mask (default 200)")
    ap.add_argument("--fs-path", default=os.environ.get("FOUNDATION_STEREO_PATH") or os.path.normpath(FS_DEFAULT),
                    help="FoundationStereo repo root (default: $FOUNDATION_STEREO_PATH, else third_party/FoundationStereo)")
    ap.add_argument("--out-dir", default="output", help="results go to <out-dir>/<vrs stem>/ (default: output/)")
    ap.add_argument("--video-width", type=int, default=640,
                    help="width of the saved RGB video, height keeps the aspect ratio (0 = no video)")
    ap.add_argument("--no-overlay", dest="overlay", action="store_false",
                    help="don't write <stem>_overlay.mp4 (the video with the tracked points drawn on)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    prompts = args.prompt or ["red cube", "cardboard box"]
    args.vrs = os.path.abspath(args.vrs)
    print("device:", args.device)

    from projectaria_tools.core import data_provider

    dp = data_provider.create_vrs_data_provider(args.vrs)
    calib = dp.get_device_calibration()
    cal_rgb = calib.get_camera_calib("camera-rgb")
    cal_l = calib.get_camera_calib("slam-front-left")
    cal_r = calib.get_camera_calib("slam-front-right")
    if cal_rgb is None or cal_l is None or cal_r is None:
        sys.exit("Calibration for camera-rgb / slam-front-left / slam-front-right is missing.")

    frames, rgb_ts = read_vrs(dp, args.start, args.stop, args.stride, args.scale)
    slam_l, slam_r, T_world_device, session_idx = read_synced(dp, rgb_ts)
    hand_lm_device = read_hands(dp, rgb_ts)

    if not 0 <= args.seg_frame < len(frames):
        sys.exit(f"--seg-frame {args.seg_frame} is outside the {len(frames)} sampled frames")
    seg_rgb = frames[args.seg_frame]
    print(f"step 1: boxes on frame {args.seg_frame}")
    boxes = boxes_manual(seg_rgb, prompts) if args.roi else boxes_grounding_dino(
        seg_rgb, prompts, args.device, args.gdino, args.box_thr, args.text_thr)
    if not boxes:
        raise SystemExit("no boxes on the segmentation frame; use --roi, lower --box-thr or try another --seg-frame")

    print(f"step 2: SAM2 masks on frame {args.seg_frame}")
    masks0 = masks_sam2(seg_rgb, boxes, args.sam_model)

    print("step 3: CoTracker3 point tracking")
    tracks, vis, obj_id = track_points(frames, masks0, args.n_points, args.device, args.seg_frame)

    print("step 4: stereo depth (FoundationStereo) + lift to 3D")
    fs = FoundationStereoDisparity(args.fs_path)
    print("  building rectification + RGB projection tables (one-off)...")
    stereo = RectifiedStereo(cal_l, cal_r)
    print(f"  baseline {stereo.baseline * 100:.2f} cm, rectified f {stereo.f:.1f} px, size {stereo.size}")
    proj_rgb = FastProjector(cal_rgb, cal_rgb.get_transform_device_camera())
    H, W = frames[0].shape[:2]
    pix_scale = W / float(cal_rgb.get_image_size()[0])  # calibration pixels -> downscaled frame pixels
    hand_uv = (proj_rgb.project(hand_lm_device.reshape(-1, 3)) * pix_scale).reshape(*hand_lm_device.shape[:3], 2)
    points_world, depth = lift_tracks(tracks, vis, (H, W), slam_l, slam_r, T_world_device,
                                      stereo, fs, proj_rgb, pix_scale)
    del fs
    torch.cuda.empty_cache()

    print("saving results")
    stem = os.path.splitext(os.path.basename(args.vrs))[0]
    save_outputs(os.path.join(args.out_dir, stem), stem, args, frames, rgb_ts,
                 T_world_device, session_idx, boxes, masks0, tracks, vis, obj_id, points_world, depth,
                 hand_lm_device, hand_uv)

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
