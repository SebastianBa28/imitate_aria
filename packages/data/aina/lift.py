"""Give tracked 2D points a 3D position from a per-pixel 3D buffer (median over a small window)."""
import numpy as np

from aina.config import LIFT_RADIUS


def lift_points(buf, uv, vis, r=LIFT_RADIUS):
    """buf (H,W,3): 3D point per pixel (any frame), NaN where empty. uv (N,2) pixels, vis (N,) bool.
    Returns (N,3) in buf's frame: median of the valid buffer points in the (2r+1)^2 window around each visible
    point (robust to edges and holes), NaN where hidden, outside the image or no depth nearby."""
    H, W = buf.shape[:2]
    out = np.full((len(uv), 3), np.nan)
    for n in np.nonzero(vis)[0]:
        if not np.isfinite(uv[n]).all():
            continue
        u, v = int(round(uv[n, 0])), int(round(uv[n, 1]))
        if u < -r or v < -r or u >= W + r or v >= H + r:
            continue
        win = buf[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1].reshape(-1, 3)
        win = win[np.isfinite(win).all(1)]
        if len(win):
            out[n] = np.median(win, axis=0)
    return out
