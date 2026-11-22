#!/usr/bin/env python3
"""
Post-process an Aria Gen2 .vrs recording.

Official AINA recipe (arXiv 2511.16661):
  1. Segment ONCE, on the first frame only: Grounding DINO box (or --roi) -> SAM 2 mask per object
  2. Sample --n-points points inside each first-frame mask
  3. CoTracker3 follows all points through the whole clip, with per-point visibility
     (so you can see when an object is hidden in the hand). No re-detection, no re-segmentation.
  4. Stereo depth + lift to 3D: the two front SLAM cameras (slam-front-left / -right) are
     rectified to a virtual pinhole pair along their baseline, FoundationStereo gives the
     disparity d, depth is Z = f * B / d (B = norm of the translation between the two cameras).
     The 3D points are projected into the RGB camera, every tracked point takes the depth at
     its pixel, and the VIO glasses pose puts it in the world frame.

Output: none yet (output logic to be rebuilt).

    python post_process.py --vrs ../../sample_data/demo20_20261116_175133.vrs --prompt "red cube" --prompt "cardboard box"
    python post_process.py --vrs ... --roi            # draw the boxes yourself on frame 0 instead of Grounding DINO
    python post_process.py --vrs ... --stop 60        # quick test on the first frames

Deps: projectaria_tools, opencv-python, numpy, scipy, torch, transformers (Grounding DINO),
ultralytics (SAM 2), and FoundationStereo (needs CUDA):
    git clone https://github.com/NVlabs/FoundationStereo
    # download the "23-51-11" checkpoint folder (model_best_bp2.pth + cfg.yaml) into
    # FoundationStereo/pretrained_models/23-51-11/   (link in the FoundationStereo README)
    pip install omegaconf timm einops      # into .venv. Do NOT install the xformers / flash-attn
                                           # from FoundationStereo's env file: they downgrade torch.
    export FOUNDATION_STEREO_PATH=/path/to/FoundationStereo     # or pass --fs-path

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
MAX_POSE_DT_NS = 50_000_000  # frames whose nearest valid VIO sample is further away get no pose
LIFT_RADIUS = 3              # px window (2r+1)^2 around a tracked point to read its depth


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


# ----------------------------------------------------------------------------- step 1: first-frame boxes
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


# ----------------------------------------------------------------------------- step 2: first-frame masks
def masks_sam2(rgb, boxes, sam_model):
    """SAM 2 in image mode on frame 0 only: one (H,W) bool mask per box, same order."""
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
def track_points(frames, masks0, n_points, device, max_side=512):
    """Sample up to n_points inside each first-frame mask and track them all with CoTracker3.

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
        pts += [[0, xs[i], ys[i]] for i in sel]  # (query frame, x, y)
        obj_id += [k] * len(sel)
    if not pts:
        raise SystemExit("all first-frame masks are empty; nothing to track")
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

    def __init__(self, calib_left, calib_right, hfov_deg=100.0, size=None):
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

    def disparity_to_points(self, disp, z_min=0.15, z_max=3.0):
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
            sys.exit("FoundationStereo not found: pass --fs-path or set FOUNDATION_STEREO_PATH "
                     "(see the setup notes at the top of this script).")
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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--prompt", action="append", default=None,
                    help="object to track, repeat for several (default: 'red cube' and 'cardboard box')")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=0, help="0 = until the end")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.5, help="downscale frames (memory/speed)")
    ap.add_argument("--roi", action="store_true", help="draw first-frame boxes by hand instead of Grounding DINO")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.2)
    ap.add_argument("--sam-model", default="sam2.1_l.pt",
                    help="SAM 2.1 checkpoint (ultralytics auto-downloads): sam2.1_t/s/b/l.pt (default l, largest)")
    ap.add_argument("--n-points", type=int, default=200, help="points sampled per object mask (default 200)")
    ap.add_argument("--fs-path", default=os.environ.get("FOUNDATION_STEREO_PATH"),
                    help="FoundationStereo repo root (default: $FOUNDATION_STEREO_PATH)")
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

    print("step 1: first-frame boxes")
    boxes = boxes_manual(frames[0], prompts) if args.roi else boxes_grounding_dino(
        frames[0], prompts, args.device, args.gdino, args.box_thr, args.text_thr)
    if not boxes:
        raise SystemExit("no first-frame boxes; use --roi or lower --box-thr")

    print("step 2: SAM2 masks on the first frame")
    masks0 = masks_sam2(frames[0], boxes, args.sam_model)

    print("step 3: CoTracker3 point tracking")
    tracks, vis, obj_id = track_points(frames, masks0, args.n_points, args.device)

    print("step 4: stereo depth (FoundationStereo) + lift to 3D")
    fs = FoundationStereoDisparity(args.fs_path)
    print("  building rectification + RGB projection tables (one-off)...")
    stereo = RectifiedStereo(cal_l, cal_r)
    print(f"  baseline {stereo.baseline * 100:.2f} cm, rectified f {stereo.f:.1f} px, size {stereo.size}")
    proj_rgb = FastProjector(cal_rgb, cal_rgb.get_transform_device_camera())
    H, W = frames[0].shape[:2]
    pix_scale = W / float(cal_rgb.get_image_size()[0])  # calibration pixels -> downscaled frame pixels
    points_world, depth = lift_tracks(tracks, vis, (H, W), slam_l, slam_r, T_world_device,
                                      stereo, fs, proj_rgb, pix_scale)
    del fs
    torch.cuda.empty_cache()

    t_s = (rgb_ts - rgb_ts[0]) / 1e9
    first_sec = t_s <= 1.0
    for k, (p, _) in enumerate(boxes):
        sel = obj_id == k
        if not sel.any():
            continue
        v = vis[:, sel]
        has = np.isfinite(depth[:, sel])
        frac = has.sum() / max(v.sum(), 1)
        med = np.nanmedian(depth[:, sel]) if has.any() else np.nan
        with warnings.catch_warnings():  # all-NaN slices are expected (hidden / no depth)
            warnings.simplefilter("ignore", RuntimeWarning)
            cen = np.nanmean(points_world[first_sec][:, sel], axis=1)  # (t,3) centroid per frame
            spread = np.linalg.norm(np.nanstd(cen, axis=0))
        print(f"  '{p}': depth for {frac:.0%} of visible points, median depth {med:.3f} m, "
              f"centroid spread over first 1 s {spread * 100:.1f} cm")

    # TODO: output (to be rebuilt). Results available here:
    #   frames          list[(H,W,3) uint8 RGB]        sampled RGB frames (T of them)
    #   rgb_ts          (T,) int64                     device-time capture timestamp of each frame (ns)
    #   T_world_device  (T,4,4)                        glasses pose (NaN if no VIO pose near the frame)
    #   session_idx     (T,) int                       VIO session (world frame only consistent within one; -1 = none)
    #   boxes           list[(prompt, [x1,y1,x2,y2])]  first-frame boxes, one per object
    #   masks0          list[(H,W) bool]               first-frame SAM2 masks, same order as boxes
    #   tracks          (T,N,2) float                  CoTracker point positions in frame pixels
    #   vis             (T,N) bool                     per-point visibility
    #   obj_id          (N,) int                       which object (index into boxes) each point belongs to
    #   points_world    (T,N,3) float                  3D point positions in world (NaN: hidden / no depth / no pose)
    #   depth           (T,N) float                    depth along the RGB optical axis (m), NaN as above


if __name__ == "__main__":
    main()
