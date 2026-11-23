#!/usr/bin/env python3
"""
Build a LeRobot dataset from post_process.py outputs and (optionally) push it to the Hugging Face Hub.

Every output/<recording>/ folder with a <recording>_aina.npz + <recording>_rgb.mp4 becomes one episode.
All 3D data stays in the raw VIO world frame of its recording (no re-framing).

Features per frame (float32 unless noted; missing values filled, see the *_valid flags):
    observation.images.rgb        video (H,W,3)   the downscaled RGB video saved by post_process
    observation.state             (69,)  right-hand 21 keypoints (63) + centroid of each object (2 x 3)
    action                        (63,)  right-hand 21 keypoints at the NEXT frame (last frame repeats)
    observation.hands             (2,21,3)  both hands' keypoints, axis 0 = (left, right)
    observation.hands_valid       (2,)   1 = hand tracked and frame has a pose
    observation.object_points     (N,3)  tracked object points (N = points of all objects)
    observation.object_points_valid (N,) 1 = point visible with a 3D position
    observation.object_centroids_valid (K,)  1 = object had >= 1 valid point this frame
    observation.point_object_id   (N,)   index into the object names (same every frame)
    observation.T_world_device    (4,4)  glasses pose
Missing values: hand / centroid entries are filled from the nearest frame where they were valid
(forward, then backward fill); object points are 0 where invalid. NaN never enters the dataset,
because LeRobot's statistics cannot handle it.

    uv run python packages/data/log.py                       # build locally into datasets/imitate-aria
    uv run python packages/data/log.py --push                # build and upload (private by default)
    uv run python packages/data/log.py --push --public

Uploading needs a Hugging Face login once:  uv run hf auth login   (paste a token with write access)
"""
import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

REPO_ID = "SebastianBa28/imitate-aria"
RIGHT = 1  # hand axis: 0 = left, 1 = right


def find_episodes(input_dir):
    eps = []
    for npz in sorted(glob.glob(os.path.join(input_dir, "*", "*_aina.npz"))):
        d = np.load(npz)
        vid = os.path.join(os.path.dirname(npz), str(d["video_file"]))
        if not str(d["video_file"]) or not os.path.isfile(vid):
            print(f"  skipping {npz}: no RGB video next to it (was post_process run with --video-width 0?)")
            continue
        eps.append((npz, vid))
    return eps


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def fill_time(x, valid):
    """Fill invalid rows of x (T,...) from the nearest valid row: forward fill, then backward fill.
    Rows stay 0 if nothing is ever valid."""
    x = np.where(valid.reshape(valid.shape + (1,) * (x.ndim - 1)), x, 0.0).astype(np.float32)
    if not valid.any():
        return x
    idx = np.where(valid, np.arange(len(valid)), -1)
    idx = np.maximum.accumulate(idx)                 # last valid index so far (-1 before the first)
    first = int(np.argmax(valid))
    idx[idx < 0] = first
    return x[idx]


def episode_arrays(d):
    """Per-frame arrays for one episode, NaN-free, in world coordinates."""
    T = len(d["rgb_ts"])
    pose_ok = np.isfinite(d["T_world_device"]).all((1, 2))
    hands = d["hand_landmarks_world"].astype(np.float64)              # (T,2,21,3)
    hands_valid = np.isfinite(hands).all((2, 3)) & pose_ok[:, None]   # (T,2)

    pts = d["points_world"].astype(np.float64)                        # (T,N,3)
    pts_valid = np.isfinite(pts).all(2)                               # (T,N)
    obj_id = d["obj_id"]
    K = len(d["object_names"])
    cents = np.zeros((T, K, 3))
    cents_valid = np.zeros((T, K), bool)
    for k in range(K):
        sel = obj_id == k
        n = pts_valid[:, sel].sum(1)
        cents_valid[:, k] = n > 0
        s = np.where(pts_valid[:, sel, None], pts[:, sel], 0.0).sum(1)
        cents[cents_valid[:, k], k] = s[cents_valid[:, k]] / n[cents_valid[:, k], None]

    right = fill_time(hands[:, RIGHT], hands_valid[:, RIGHT])          # (T,21,3)
    cents_f = np.stack([fill_time(cents[:, k], cents_valid[:, k]) for k in range(K)], 1)  # (T,K,3)
    state = np.concatenate([right.reshape(T, -1), cents_f.reshape(T, -1)], 1).astype(np.float32)
    action = np.concatenate([right[1:], right[-1:]], 0).reshape(T, -1).astype(np.float32)
    hands_f = np.stack([fill_time(hands[:, h], hands_valid[:, h]) for h in range(2)], 1).astype(np.float32)

    return dict(
        state=state,
        action=action,
        hands=hands_f,
        hands_valid=hands_valid.astype(np.float32),
        points=np.where(pts_valid[..., None], pts, 0.0).astype(np.float32),
        points_valid=pts_valid.astype(np.float32),
        cents_valid=cents_valid.astype(np.float32),
        obj_id=obj_id.astype(np.int32),
        T_wd=np.where(pose_ok[:, None, None], d["T_world_device"], np.eye(4)).astype(np.float32),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="output", help="folder with one sub-folder per recording (default: output/)")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--root", default=None, help="local dataset folder (default: datasets/<repo name>)")
    ap.add_argument("--task", default=None,
                    help="task description stored with every frame (default: 'put the <object 0> in the <object 1>')")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing local dataset folder first")
    ap.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub after building")
    ap.add_argument("--public", action="store_true", help="make the Hub dataset public (default: private)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root or os.path.join("datasets", args.repo_id.split("/")[-1])
    if os.path.exists(root):
        if not args.overwrite:
            sys.exit(f"{root} already exists; pass --overwrite to rebuild it from scratch.")
        shutil.rmtree(root)

    if args.push:  # fail early, before spending time building
        from huggingface_hub import HfApi
        try:
            user = HfApi().whoami()["name"]
        except Exception:
            sys.exit("Not logged in to Hugging Face. Run once:  uv run hf auth login   (token with write access)")
        print(f"logged in to the Hub as {user}")

    episodes = find_episodes(args.input)
    if not episodes:
        sys.exit(f"no <recording>/<recording>_aina.npz + _rgb.mp4 found under {args.input}/")
    print(f"found {len(episodes)} episode(s) under {args.input}/")

    # dataset-wide checks: one fps, one video size, one point count, same objects
    metas = [np.load(n) for n, _ in episodes]
    fps_all = sorted({int(round(float(m["fps"]))) for m in metas})
    if len(fps_all) != 1:
        sys.exit(f"episodes have different fps {fps_all}: re-run post_process with the same --stride for all")
    hw_all = {tuple(m["video_hw"]) for m in metas}
    if len(hw_all) != 1:
        sys.exit(f"episodes have different video sizes {hw_all}: re-run post_process with the same --video-width")
    names0 = list(metas[0]["object_names"])
    for (n, _), m in zip(episodes, metas):
        if list(m["object_names"]) != names0:
            sys.exit(f"{n} tracks {list(m['object_names'])}, but the first episode tracks {names0}")
    N = max(len(m["obj_id"]) for m in metas)
    K = len(names0)
    fps = fps_all[0]
    H, W = hw_all.pop()
    task = args.task or (f"put the {names0[0]} in the {names0[1]}" if K >= 2 else f"move the {names0[0]}")

    features = {
        "observation.images.rgb": {"dtype": "video", "shape": (int(H), int(W), 3),
                                   "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (63 + 3 * K,), "names": None},
        "action": {"dtype": "float32", "shape": (63,), "names": None},
        "observation.hands": {"dtype": "float32", "shape": (2, 21, 3), "names": None},
        "observation.hands_valid": {"dtype": "float32", "shape": (2,), "names": ["left", "right"]},
        "observation.object_points": {"dtype": "float32", "shape": (N, 3), "names": None},
        "observation.object_points_valid": {"dtype": "float32", "shape": (N,), "names": None},
        "observation.object_centroids_valid": {"dtype": "float32", "shape": (K,), "names": names0},
        "observation.point_object_id": {"dtype": "int32", "shape": (N,), "names": None},
        "observation.T_world_device": {"dtype": "float32", "shape": (4, 4), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features, root=root,
                               robot_type="aria_gen2_human", use_videos=True)

    for (npz, vid), d in zip(episodes, metas):
        frames = read_video(vid)
        T = len(d["rgb_ts"])
        if len(frames) != T:
            print(f"  skipping {npz}: video has {len(frames)} frames, npz has {T}")
            continue
        a = episode_arrays(d)
        n = a["points"].shape[1]
        if n < N:  # pad episodes with fewer points (small first-frame mask) as invalid
            a["points"] = np.concatenate([a["points"], np.zeros((T, N - n, 3), np.float32)], 1)
            a["points_valid"] = np.concatenate([a["points_valid"], np.zeros((T, N - n), np.float32)], 1)
            a["obj_id"] = np.concatenate([a["obj_id"], np.full(N - n, -1, np.int32)])
        for t in range(T):
            ds.add_frame({
                "task": task,
                "observation.images.rgb": frames[t],
                "observation.state": a["state"][t],
                "action": a["action"][t],
                "observation.hands": a["hands"][t],
                "observation.hands_valid": a["hands_valid"][t],
                "observation.object_points": a["points"][t],
                "observation.object_points_valid": a["points_valid"][t],
                "observation.object_centroids_valid": a["cents_valid"][t],
                "observation.point_object_id": a["obj_id"],
                "observation.T_world_device": a["T_wd"][t],
            })
        ds.save_episode()
        print(f"  episode {ds.num_episodes - 1}: {os.path.basename(os.path.dirname(npz))}  ({T} frames)")

    ds.finalize()
    print(f"dataset written to {root}: {ds.num_episodes} episodes, {ds.num_frames} frames at {fps} fps, task '{task}'")

    if args.push:
        ds.push_to_hub(private=not args.public)
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}"
              f"  ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
