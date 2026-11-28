"""Read in-scene recordings made by packages/deploy/tools/record_cams.py (branch in-scene), raw or packed.

Recording folder (one per demo or calibration clip):
    session.json                  cameras, note, start time
    <cam>/frames.csv              index, colour/depth frame numbers, t_capture_s (host clock), t_host_s
    <cam>/meta.json               device, sizes, colour intrinsics (fx, fy, ppx, ppy, coeffs)
    <cam>/color/000000.jpg ...    raw: RGB JPEG          | packed (scene_pack.py): <cam>/color.mp4 (H.264)
    <cam>/depth/000000.png ...    raw: uint16 mm, 0=none | packed: <cam>/depth.mkv (FFV1 gray16le, lossless)
Depth is aligned to colour (same pixel = same point), in millimetres along the optical axis.
Frame i of colour/depth = row i of frames.csv. Cameras are not hardware-synced: pair by t_capture_s.
OAK colour is distorted (OpenCV rational model, 14 coeffs); RealSense colour has zero coefficients.
"""
import csv
import glob
import json
import os

import cv2
import numpy as np


def is_recording(folder):
    return os.path.isfile(os.path.join(folder, "session.json"))


class SceneCamera:
    """One camera of a recording. Frames are read lazily and in order (fast for both JPEG folders and videos)."""

    def __init__(self, cam_dir):
        self.dir = cam_dir
        self.id = os.path.basename(os.path.normpath(cam_dir))
        with open(os.path.join(cam_dir, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(cam_dir, "frames.csv")) as f:
            rows = list(csv.DictReader(f))
        self.t = np.array([float(r["t_capture_s"]) for r in rows])
        intr = self.meta["intrinsics"]
        self.size = (int(intr["width"]), int(intr["height"]))      # (W, H)
        self.K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], np.float64)
        coeffs = np.asarray(intr.get("coeffs", []), np.float64)
        self.dist = coeffs if np.any(coeffs) else np.zeros(5)
        self.packed = os.path.isfile(os.path.join(cam_dir, "color.mp4")) and \
            os.path.isfile(os.path.join(cam_dir, "depth.mkv"))
        if not self.packed:
            n_c = len(glob.glob(os.path.join(cam_dir, "color", "*.jpg")))
            n_d = len(glob.glob(os.path.join(cam_dir, "depth", "*.png")))
            if n_c == 0:
                raise FileNotFoundError(f"{cam_dir}: no color/*.jpg and no color.mp4 + depth.mkv")
            if not (n_c == n_d == len(self.t)):
                print(f"  WARNING {self.id}: {n_c} colour / {n_d} depth images, {len(self.t)} rows in frames.csv")
        self._rays = None

    def __len__(self):
        return len(self.t)

    # ----------------------------------------------------------------- frames
    def colors(self, idxs):
        """RGB uint8 frames for the sorted indices idxs (generator)."""
        if not self.packed:
            for i in idxs:
                yield cv2.cvtColor(cv2.imread(os.path.join(self.dir, "color", f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
            return
        yield from _decode(os.path.join(self.dir, "color.mp4"), idxs, "rgb24")

    def depths(self, idxs):
        """uint16 depth images (mm, 0 = none) for the sorted indices idxs (generator)."""
        if not self.packed:
            for i in idxs:
                yield cv2.imread(os.path.join(self.dir, "depth", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            return
        yield from _decode(os.path.join(self.dir, "depth.mkv"), idxs, "gray16le")

    def color(self, i):
        return next(self.colors([i]))

    # ----------------------------------------------------------------- geometry
    def undistort_px(self, uv):
        """(N,2) distorted pixels -> (N,2) normalized image coords (x/z, y/z). NaN stays NaN."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        out = np.full_like(uv, np.nan)
        ok = np.isfinite(uv).all(1)
        if ok.any():
            out[ok] = cv2.undistortPoints(uv[ok].reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        return out

    def rays(self):
        """(H,W,2) normalized coords of every pixel (cached)."""
        if self._rays is None:
            W, H = self.size
            us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
            self._rays = self.undistort_px(np.stack([us, vs], -1).reshape(-1, 2)).reshape(H, W, 2)
        return self._rays

    def depth_to_points(self, depth_mm, z_range=(0.15, 3.0)):
        """uint16 depth image -> (H,W,3) points in the camera frame (m), NaN where no depth / out of range."""
        z = depth_mm.astype(np.float64) / 1000.0
        z[(z < z_range[0]) | (z > z_range[1])] = np.nan
        r = self.rays()
        return np.stack([r[..., 0] * z, r[..., 1] * z, z], -1)

    def project(self, P_cam):
        """(N,3) camera-frame points -> (N,2) distorted pixels, NaN behind the camera or for NaN input."""
        P = np.asarray(P_cam, np.float64).reshape(-1, 3)
        out = np.full((len(P), 2), np.nan)
        ok = np.isfinite(P).all(1) & (P[:, 2] > 1e-6)
        if ok.any():
            uv, _ = cv2.projectPoints(P[ok], np.zeros(3), np.zeros(3), self.K, self.dist)
            out[ok] = uv.reshape(-1, 2)
        return out


def _decode(path, idxs, fmt):
    import av

    want = list(idxs)
    if not want:
        return
    k = 0
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i < want[k]:
                continue
            arr = frame.to_ndarray(format=fmt)
            while k < len(want) and want[k] == i:
                yield arr
                k += 1
            if k == len(want):
                return
    if k < len(want):
        raise IndexError(f"{path}: frame {want[k]} requested, the video has fewer frames")


class SceneRecording:
    """A recording folder: its cameras plus session info."""

    def __init__(self, folder):
        if not is_recording(folder):
            raise FileNotFoundError(f"{folder}: no session.json (not a record_cams.py recording)")
        self.dir = os.path.abspath(folder)
        self.name = os.path.basename(os.path.normpath(folder))
        with open(os.path.join(folder, "session.json")) as f:
            self.session = json.load(f)
        self.cams = {c: SceneCamera(os.path.join(folder, c)) for c in self.session["cameras"]
                     if os.path.isdir(os.path.join(folder, c))}

    def pair(self, ref, stride=1, max_dt=0.020):
        """Frame pairing on camera `ref`'s timeline: every stride-th ref frame, and for each other camera the
        frame nearest in t_capture_s. Returns (t (T,), {cam: (T,) frame index, -1 if > max_dt away})."""
        r = self.cams[ref]
        ridx = np.arange(0, len(r), stride)
        t = r.t[ridx]
        out = {ref: ridx}
        for cid, cam in self.cams.items():
            if cid == ref:
                continue
            j = np.clip(np.searchsorted(cam.t, t), 1, len(cam.t) - 1)
            j = np.where(np.abs(cam.t[j - 1] - t) <= np.abs(cam.t[j] - t), j - 1, j)
            out[cid] = np.where(np.abs(cam.t[j] - t) <= max_dt, j, -1)
        return t, out


def load_calib(path):
    """calib.json (scene_calib.py) -> dict with numpy T_world_cam / K / dist per camera."""
    with open(path) as f:
        c = json.load(f)
    for cam in c["cameras"].values():
        for k in ("T_world_cam", "K", "dist"):
            cam[k] = np.asarray(cam[k], np.float64)
    if c.get("T_base_world") is not None:
        c["T_base_world"] = np.asarray(c["T_base_world"], np.float64)
    return c


def find_calib(rec_dir):
    """The calib.json of a calibration clip (folder name starting with 'calib', same parent folder, already
    solved by scene_calib.py) from the same day: the latest one recorded before this demo, else the first one
    after it. None if there is none."""
    parent = os.path.dirname(os.path.abspath(rec_dir))
    with open(os.path.join(rec_dir, "session.json")) as f:
        started = json.load(f)["started"]
    before, after = None, None
    for d in glob.glob(os.path.join(parent, "calib*")):
        cj, sj = os.path.join(d, "calib.json"), os.path.join(d, "session.json")
        if not (os.path.isfile(cj) and os.path.isfile(sj)):
            continue
        with open(sj) as f:
            s = json.load(f)["started"]
        if s[:10] != started[:10]:
            continue
        if s <= started and (before is None or s > before[0]):
            before = (s, cj)
        if s > started and (after is None or s < after[0]):
            after = (s, cj)
    best = before or after
    return best[1] if best else None
