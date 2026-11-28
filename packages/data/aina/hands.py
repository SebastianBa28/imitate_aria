"""Hand landmark conventions shared by both pipelines: everything is stored in Aria's 21-landmark order.

Aria (projectaria_tools HandLandmark): 0-4 fingertips (thumb..pinky), 5 wrist, 6 thumb intermediate (MCP),
7 thumb distal (IP), 8-10 index proximal/intermediate/distal (MCP/PIP/DIP), 11-13 middle, 14-16 ring,
17-19 pinky, 20 palm centre.
"""
import numpy as np

HANDS = ("left", "right")
# 14-16 ring, 17-19 pinky (proximal -> distal), 20 palm centre. Chains as in projectaria_tools' rerun helpers.
WRIST_KEYPOINTS = (5, 8, 17)  # wrist, index knuckle (MCP), pinky knuckle (MCP)
HAND_CHAINS = [(5, 6, 7, 0), (5, 8, 9, 10, 1), (5, 11, 12, 13, 2), (5, 14, 15, 16, 3), (5, 17, 18, 19, 4),
               (6, 8, 11, 14, 17)]

# OpenPose / MANO-21 order (WiLoR, HaMeR): 0 wrist, thumb 1 CMC 2 MCP 3 IP 4 tip, index 5 MCP 6 PIP 7 DIP 8 tip,
# middle 9-12, ring 13-16, pinky 17-20. ARIA_FROM_OPENPOSE[a] = OpenPose index of Aria landmark a; the thumb
# CMC (1) has no Aria counterpart; the palm centre (-1) is the midpoint of the wrist and the middle MCP.
ARIA_FROM_OPENPOSE = [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, -1]


def openpose_to_aria(kp):
    """(..., 21, D) OpenPose-ordered keypoints -> (..., 21, D) in Aria landmark order."""
    idx = np.array(ARIA_FROM_OPENPOSE)
    out = kp[..., np.where(idx < 0, 0, idx), :].copy()
    out[..., 20, :] = (kp[..., 0, :] + kp[..., 9, :]) / 2
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
