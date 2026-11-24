#!/usr/bin/env python3
"""
Build a LeRobot dataset from post_process.py outputs and (optionally) push it to the Hugging Face Hub.

Every output/<recording>/ folder with a <recording>_aina.npz + <recording>_rgb.mp4 becomes one episode.
All 3D data stays in the raw VIO world frame of its recording (no re-framing).

Features per frame (float32, AINA-style; missing values are 0 with a *_valid flag, never NaN or filled,
because LeRobot's statistics cannot handle NaN):
    observation.images.rgb          video (H,W,3)  the downscaled RGB video saved by post_process
    debug.overlay                   video (H,W,3)  the same frames with tracked points + hands drawn on
    debug.depth                     video (S,S,3)  FoundationStereo depth video (not policy inputs: debug.*)
    observation.state               (15,)  right-hand 5 fingertips (thumb..pinky) x xyz
    observation.state_valid         (1,)   1 = right hand tracked with a pose this frame
    observation.object_points       (N,3)  tracked object points (N = points of all objects)
    observation.object_points_valid (N,)   1 = point visible with a 3D position
    action                          (15,)  right-hand fingertips at the NEXT frame (last frame repeats)
    action_valid                    (1,)   observation.state_valid of that next frame
Which object each point belongs to is the same for every frame, so it is written once to
meta/objects.json: {"object_names": [...], "point_object_id": [N ints, -1 = padding]}.

    uv run python packages/data/log.py                       # build locally into datasets/imitate-aria
    uv run python packages/data/log.py --push                # build and upload (private by default)
    uv run python packages/data/log.py --push --public

Uploading needs a Hugging Face login once:  uv run hf auth login   (paste a token with write access)
"""
import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

REPO_ID = "SebastianBa28/imitate-aria"
RIGHT = 1  # hand axis: 0 = left, 1 = right
FINGERTIPS = [0, 1, 2, 3, 4]  # Aria hand landmarks: thumb, index, middle, ring, pinky tips


VIDEOS = {  # dataset feature -> npz field naming the post_process video file
    "observation.images.rgb": "video_file",
    "debug.overlay": "overlay_file",   # tracked points + hands drawn on; debug.* so policies don't use it as input
    "debug.depth": "depth_video_file",  # FoundationStereo depth (rectified left SLAM view), turbo + shading
}


def find_episodes(input_dir):
    """(npz path, {feature: video path}) for every recording that has all the VIDEOS files."""
    eps = []
    for npz in sorted(glob.glob(os.path.join(input_dir, "*", "*_aina.npz"))):
        d = np.load(npz)
        vids = {}
        for feat, field in VIDEOS.items():
            name = str(d[field]) if field in d else ""
            path = os.path.join(os.path.dirname(npz), name)
            if name and os.path.isfile(path):
                vids[feat] = path
        missing = [VIDEOS[f] for f in VIDEOS if f not in vids]
        if missing:
            print(f"  skipping {npz}: missing {', '.join(missing)} (re-run post_process without "
                  "--video-width 0 / --no-overlay / --no-depth-video)")
            continue
        eps.append((npz, vids))
    return eps


def video_hw(path):
    cap = cv2.VideoCapture(path)
    hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    return hw


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


def episode_arrays(d):
    """Per-frame arrays for one episode, NaN-free, in world coordinates."""
    tips = d["hand_landmarks_world"][:, RIGHT, FINGERTIPS].astype(np.float32)   # (T,5,3), NaN without a pose
    tips_valid = np.isfinite(tips).all((1, 2))                                  # (T,)
    tips = np.where(tips_valid[:, None, None], tips, 0.0).reshape(len(tips), -1).astype(np.float32)
    pts = d["points_world"].astype(np.float32)                                  # (T,N,3)
    pts_valid = np.isfinite(pts).all(2)
    nxt = np.r_[np.arange(1, len(tips)), len(tips) - 1]                         # t+1, last frame repeats
    return dict(
        state=tips,
        state_valid=tips_valid[:, None].astype(np.float32),
        points=np.where(pts_valid[..., None], pts, 0.0).astype(np.float32),
        points_valid=pts_valid.astype(np.float32),
        action=tips[nxt],
        action_valid=tips_valid[nxt, None].astype(np.float32),
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

    # dataset-wide checks: one fps, one video size, same objects
    metas = [np.load(n) for n, _ in episodes]
    fps_all = sorted({int(round(float(m["fps"]))) for m in metas})
    if len(fps_all) != 1:
        sys.exit(f"episodes have different fps {fps_all}: re-run post_process with the same --stride for all")
    sizes = {}
    for feat in VIDEOS:
        hw = {video_hw(v[feat]) for _, v in episodes}
        if len(hw) != 1:
            sys.exit(f"episodes have different {feat} video sizes {hw}: re-run post_process with the same "
                     "--video-width / STEREO_SIZE for all")
        sizes[feat] = hw.pop()
    names0 = [str(x) for x in metas[0]["object_names"]]
    for (n, _), m in zip(episodes, metas):
        if list(m["object_names"]) != names0:
            sys.exit(f"{n} tracks {list(m['object_names'])}, but the first episode tracks {names0}")
    N = max(len(m["obj_id"]) for m in metas)
    K = len(names0)
    fps = fps_all[0]
    task = args.task or (f"put the {names0[0]} in the {names0[1]}" if K >= 2 else f"move the {names0[0]}")

    features = {
        **{feat: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
           for feat, (h, w) in sizes.items()},
        "observation.state": {"dtype": "float32", "shape": (15,), "names": None},
        "observation.state_valid": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.object_points": {"dtype": "float32", "shape": (N, 3), "names": None},
        "observation.object_points_valid": {"dtype": "float32", "shape": (N,), "names": None},
        "action": {"dtype": "float32", "shape": (15,), "names": None},
        "action_valid": {"dtype": "float32", "shape": (1,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features, root=root,
                               robot_type="aria_gen2_human", use_videos=True)

    obj_id0 = None
    for (npz, vids), d in zip(episodes, metas):
        videos = {feat: read_video(path) for feat, path in vids.items()}
        T = len(d["rgb_ts"])
        bad = {feat: len(v) for feat, v in videos.items() if len(v) != T}
        if bad:
            print(f"  skipping {npz}: npz has {T} frames, but the videos have {bad}")
            continue
        a = episode_arrays(d)
        obj_id = d["obj_id"].astype(np.int32)
        n = len(obj_id)
        if n < N:  # pad episodes with fewer points (small first-frame mask) as invalid
            a["points"] = np.concatenate([a["points"], np.zeros((T, N - n, 3), np.float32)], 1)
            a["points_valid"] = np.concatenate([a["points_valid"], np.zeros((T, N - n), np.float32)], 1)
            obj_id = np.concatenate([obj_id, np.full(N - n, -1, np.int32)])
        if obj_id0 is None:
            obj_id0 = obj_id
        elif not np.array_equal(obj_id, obj_id0):
            print(f"  warning: {npz} has a different per-object point split than the first episode")
        for t in range(T):
            ds.add_frame({
                "task": task,
                **{feat: v[t] for feat, v in videos.items()},
                "observation.state": a["state"][t],
                "observation.state_valid": a["state_valid"][t],
                "observation.object_points": a["points"][t],
                "observation.object_points_valid": a["points_valid"][t],
                "action": a["action"][t],
                "action_valid": a["action_valid"][t],
            })
        ds.save_episode()
        print(f"  episode {ds.num_episodes - 1}: {os.path.basename(os.path.dirname(npz))}  ({T} frames)")

    ds.finalize()
    with open(os.path.join(root, "meta", "objects.json"), "w") as f:
        json.dump({"object_names": names0, "point_object_id": obj_id0.tolist()}, f)
    print(f"dataset written to {root}: {ds.num_episodes} episodes, {ds.num_frames} frames at {fps} fps, task '{task}'")

    if args.push:
        ds.push_to_hub(private=not args.public)
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}"
              f"  ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
