"""3D hand pose from RGB with WiLoR (HaMeR-style ViT regressing MANO, plus its own YOLO hand detector),
via the wilor-mini package. Weights download on first use into ~/.cache/wilor_mini (Hugging Face
warmshao/WiLoR-mini; MANO_RIGHT.pkl comes from there too: MANO is under its own non-commercial licence).

WiLoR gives per detected hand: left/right, 21 keypoints in OpenPose order as 2D pixels and as a root-relative
3D shape in camera orientation (metric, MANO scale). Its own camera translation assumes a fake focal length,
so the metric placement is done here with the real intrinsics (place_metric) and, where available, the
sensor depth.
"""
import os
import pickle
import sys
import types

import numpy as np
import torch

from aina.lift import lift_points

CACHE = os.path.expanduser(os.environ.get("WILOR_CACHE", "~/.cache/wilor_mini"))


def _dechumpify_mano(path):
    """MANO_RIGHT.pkl stores shapedirs as a chumpy object (chumpy does not install on Python 3.12 / numpy 2).
    Rewrite the file once with plain numpy arrays, so smplx can load it without chumpy."""
    class Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {})

        def value(self):
            if hasattr(self, "x"):                                   # chumpy.Ch: the array itself
                return np.asarray(self.x)
            if hasattr(self, "a") and hasattr(self, "idxs"):         # chumpy.reordering.Select
                a = self.a.value() if isinstance(self.a, Stub) else np.asarray(self.a)
                return a.ravel()[self.idxs].reshape(self.preferred_shape)
            raise TypeError(f"unknown chumpy object {type(self).__name__}")

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith("chumpy"):
                return type(name, (Stub,), {})
            return super().find_class(mod, name)

    with open(path, "rb") as f:
        d = U(f, encoding="latin1").load()
    if not any(isinstance(v, Stub) for v in d.values()):
        return
    d = {k: (v.value() if isinstance(v, Stub) else v) for k, v in d.items()}
    with open(path + ".tmp", "wb") as f:
        pickle.dump(d, f)
    os.replace(path + ".tmp", path)
    print(f"  converted {path} to plain numpy (no chumpy needed)")


def _ensure_weights():
    from huggingface_hub import hf_hub_download

    for name in ("mano_mean_params.npz", "MANO_RIGHT.pkl", "wilor_final.ckpt", "detector.pt"):
        if not os.path.exists(os.path.join(CACHE, "pretrained_models", name)):
            print(f"  downloading WiLoR {name} into {CACHE}")
            hf_hub_download(repo_id="warmshao/WiLoR-mini", subfolder="pretrained_models", filename=name, local_dir=CACHE)
    _dechumpify_mano(os.path.join(CACHE, "pretrained_models", "MANO_RIGHT.pkl"))


class WiLoR:
    """hands = WiLoR(device)(rgb) -> {side (0 left, 1 right): dict(kp2d (21,2) px, kp3d (21,3) m root-relative in
    camera orientation, conf, box)}, best detection per side, OpenPose keypoint order."""

    def __init__(self, device="cuda", conf=0.3):
        _ensure_weights()
        if "chumpy" not in sys.modules:   # smplx imports nothing from chumpy once the pkl is plain numpy
            sys.modules["chumpy"] = types.ModuleType("chumpy")
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

        dev = torch.device(device)
        self.pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float16 if dev.type == "cuda" else
                                                      torch.float32, verbose=False, wilor_pretrained_dir=CACHE)
        self.conf = conf

    @torch.no_grad()
    def __call__(self, rgb):
        det = self.pipe.hand_detector(rgb, conf=self.conf, verbose=False)[0]
        best = {}
        for b in det.boxes:
            side, c = int(b.cls.item()), float(b.conf.item())
            if side not in best or c > best[side][1]:
                best[side] = (b.xyxy.cpu().numpy().reshape(4), c)
        if not best:
            return {}
        sides = sorted(best)
        boxes = np.stack([best[s][0] for s in sides])
        res = self.pipe.predict_with_bboxes(rgb, boxes, sides)
        out = {}
        for s, r in zip(sides, res):
            p = r["wilor_preds"]
            out[s] = dict(kp2d=p["pred_keypoints_2d"][0].astype(np.float64),
                          kp3d=p["pred_keypoints_3d"][0].astype(np.float64),
                          conf=best[s][1], box=best[s][0])
        return out


def place_metric(kp3d, nrm, buf_cam=None, uv=None, radius=4):
    """Put a root-relative metric hand shape kp3d (21,3) (camera orientation) into the camera frame.

    nrm (21,2): the keypoints' undistorted normalized coords (x/z, y/z). The translation t that makes kp3d + t
    project onto them is linear least squares (the hand's known metric size fixes the distance). With a
    camera-frame point buffer buf_cam (H,W,3) and the keypoints' pixels uv, the distance is then corrected by
    the median offset to the sensor depth at the keypoints (robust to the object or the other hand in front).
    Returns (21,3) camera-frame points, or None if degenerate."""
    X = kp3d
    A = np.zeros((42, 3))
    b = np.zeros(42)
    A[0::2, 0], A[0::2, 2], b[0::2] = 1, -nrm[:, 0], nrm[:, 0] * X[:, 2] - X[:, 0]
    A[1::2, 1], A[1::2, 2], b[1::2] = 1, -nrm[:, 1], nrm[:, 1] * X[:, 2] - X[:, 1]
    ok = np.isfinite(A).all(1) & np.isfinite(b)
    if ok.sum() < 8:
        return None
    t = np.linalg.lstsq(A[ok], b[ok], rcond=None)[0]
    if buf_cam is not None and uv is not None:
        P = lift_points(buf_cam, uv, np.ones(len(uv), bool), radius)
        dz = P[:, 2] - (X[:, 2] + t[2])
        dz = dz[np.isfinite(dz) & (np.abs(dz) < 0.25)]
        if len(dz) >= 5:
            t[2] += float(np.median(dz))
            # keep the projection: re-solve x, y for the corrected distance
            z = X[:, 2] + t[2]
            t[0] = np.nanmean(nrm[:, 0] * z - X[:, 0])
            t[1] = np.nanmean(nrm[:, 1] * z - X[:, 1])
    if not np.isfinite(t).all() or t[2] <= 0.05:
        return None
    return X + t
