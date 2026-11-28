"""Debug drawing: tracked points and hand skeletons on a frame."""
import cv2
import numpy as np

from aina.hands import HAND_CHAINS

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
