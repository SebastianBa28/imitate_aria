"""Step 3 of AINA: sample points in the frame-0 masks and track them with CoTracker3."""
import functools

import cv2
import numpy as np
import torch


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
