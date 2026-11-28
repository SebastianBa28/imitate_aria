"""
Write post_process.py outputs into a LeRobot dataset, one episode per recording. Shared by log.py
(build from output/ folders) and post_process.py --append-to (append right after processing).

Features per frame (float32, AINA-style; missing values are 0 with a *_valid flag, never NaN or filled,
because LeRobot's statistics cannot handle NaN):
    observation.images.rgb          video (H,W,3)  the downscaled RGB video saved by post_process
    debug.overlay                   video (H,W,3)  the same frames with tracked points + hands drawn on
    debug.depth                     video (S,S,3)  FoundationStereo depth video (not policy inputs: debug.*)
    observation.state               (24,)  right-hand joints x xyz: 5 fingertips (thumb..pinky), wrist,
                                           index knuckle, pinky knuckle (Aria landmarks 0-4, 5, 8, 17)
    observation.state_valid         (1,)   1 = right hand tracked with a pose this frame
    observation.object_points       (N,3)  tracked object points, N = sum of n_points (post_process N_POINTS)
    observation.object_points_valid (N,)   1 = point visible with a 3D position
    action                          (24,)  the same right-hand joints at the NEXT frame (last frame repeats)
    action_valid                    (1,)   observation.state_valid of that next frame

Point slots are fixed per object: object k always owns slots [sum(n_points[:k]), sum(n_points[:k+1])).
Slots an episode could not fill (small or missing mask) stay 0 / invalid. The layout is written to
meta/objects.json: {"object_names": [...], "n_points": [...], "point_object_id": [N ints]}.
meta/sources.json lists the recording (vrs stem) of every episode, so a recording is never added twice.
All 3D data stays in the raw VIO world frame of its recording (no re-framing).
"""
import glob
import json
import os

import cv2
import numpy as np

DEFAULT_REPO_ID = "SebastianBa28/imitate-aria"
DEFAULT_ROOT = os.path.join("datasets", "imitate-aria")
RIGHT = 1  # hand axis: 0 = left, 1 = right
# Aria hand landmarks in observation.state / action, in this order
HAND_JOINTS = {"thumb_tip": 0, "index_tip": 1, "middle_tip": 2, "ring_tip": 3, "pinky_tip": 4,
               "wrist": 5, "index_knuckle": 8, "pinky_knuckle": 17}
STATE_NAMES = [f"{j}.{a}" for j in HAND_JOINTS for a in "xyz"]
STATE_DIM = len(STATE_NAMES)  # 24

VIDEOS = {  # dataset feature -> npz field naming the post_process video file
    "observation.images.rgb": "video_file",
    "debug.overlay": "overlay_file",   # tracked points + hands drawn on; debug.* so policies don't use it as input
    "debug.depth": "depth_video_file",  # FoundationStereo depth (rectified left SLAM view), turbo + shading
}


class EpisodeError(Exception):
    """A recording that cannot go into the dataset (missing files, or settings that differ from the dataset's)."""


# ----------------------------------------------------------------------------- reading one recording
def find_npz(folder):
    """The <recording>_aina.npz inside a post_process output folder."""
    hits = sorted(glob.glob(os.path.join(folder, "*_aina.npz")))
    if len(hits) != 1:
        raise EpisodeError(f"{folder}: expected one *_aina.npz, found {len(hits)}")
    return hits[0]


def episode_videos(npz_path, d):
    """{feature: video path}; raises EpisodeError if one is missing."""
    vids, missing = {}, []
    for feat, field in VIDEOS.items():
        name = str(d[field]) if field in d else ""
        path = os.path.join(os.path.dirname(npz_path), name)
        if name and os.path.isfile(path):
            vids[feat] = path
        else:
            missing.append(field)
    if missing:
        raise EpisodeError(f"{npz_path}: missing {', '.join(missing)} (re-run post_process without "
                           "--video-width 0 / --no-overlay / --no-depth-video)")
    return vids


def stem_of(npz_path):
    return os.path.basename(npz_path)[: -len("_aina.npz")]


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


def episode_fps(d):
    return int(round(float(d["fps"])))


def default_task(names):
    return f"put the {names[0]} in the {names[1]}" if len(names) >= 2 else f"move the {names[0]}"


# ----------------------------------------------------------------------------- layout
def layout_from_episode(d):
    """Dataset-wide settings taken from the first episode: fps, video sizes, objects and point slots."""
    names = [str(x) for x in d["object_names"]]
    n_points = [int(x) for x in d["n_points"]]
    if len(names) != len(n_points):
        raise EpisodeError(f"the first episode detected {names} but post_process tracks {len(n_points)} objects; "
                           "start the dataset with a recording where every object was found")
    return dict(names=names, n_points=n_points)


def point_object_id(n_points):
    return np.repeat(np.arange(len(n_points)), n_points).astype(np.int32)


def place_points(arr, obj_id, names_ep, names_ds, n_points):
    """Scatter an episode's per-point array (T,n,...) into the dataset's fixed slots (T,N,...), zeros elsewhere.
    Returns (placed, filled) where filled (N,) marks the slots that got a point."""
    N = int(sum(n_points))
    offsets = np.r_[0, np.cumsum(n_points)]
    out = np.zeros((arr.shape[0], N) + arr.shape[2:], arr.dtype)
    filled = np.zeros(N, bool)
    for k_ep, name in enumerate(names_ep):
        if name not in names_ds:
            raise EpisodeError(f"episode tracks '{name}', which is not one of the dataset's objects {names_ds}")
        k = names_ds.index(name)
        src = np.nonzero(obj_id == k_ep)[0][: n_points[k]]
        dst = np.arange(offsets[k], offsets[k] + len(src))
        out[:, dst] = arr[:, src]
        filled[dst] = True
    return out, filled


def episode_arrays(d, names_ds, n_points):
    """Per-frame arrays for one episode, NaN-free, in world coordinates, points in the dataset's slots."""
    joints = d["hand_landmarks_world"][:, RIGHT, list(HAND_JOINTS.values())].astype(np.float32)  # (T,8,3), NaN without a pose
    tips_valid = np.isfinite(joints).all((1, 2))                                # (T,)
    tips = np.where(tips_valid[:, None, None], joints, 0.0).reshape(len(joints), -1).astype(np.float32)
    names_ep = [str(x) for x in d["object_names"]]
    pts, filled = place_points(d["points_world"].astype(np.float32), d["obj_id"], names_ep, names_ds, n_points)
    pts_valid = np.isfinite(pts).all(2) & filled                                # (T,N)
    nxt = np.r_[np.arange(1, len(tips)), len(tips) - 1]                         # t+1, last frame repeats
    return dict(
        state=tips,
        state_valid=tips_valid[:, None].astype(np.float32),
        points=np.where(pts_valid[..., None], pts, 0.0).astype(np.float32),
        points_valid=pts_valid.astype(np.float32),
        action=tips[nxt],
        action_valid=tips_valid[nxt, None].astype(np.float32),
    )


def make_features(sizes, N):
    return {
        **{feat: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
           for feat, (h, w) in sizes.items()},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": STATE_NAMES},
        "observation.state_valid": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.object_points": {"dtype": "float32", "shape": (N, 3), "names": None},
        "observation.object_points_valid": {"dtype": "float32", "shape": (N,), "names": None},
        "action": {"dtype": "float32", "shape": (STATE_DIM,), "names": STATE_NAMES},
        "action_valid": {"dtype": "float32", "shape": (1,), "names": None},
    }


# ----------------------------------------------------------------------------- the dataset
class EpisodeWriter:
    """Opens a LeRobot dataset for appending (creating it from the first episode if it does not exist yet).

        with EpisodeWriter(root) as w:
            w.add(output_folder)        # one call per recording; returns False if it is already in the dataset
        w.push(private=True)            # optional, after closing
    """

    def __init__(self, root=DEFAULT_ROOT, repo_id=DEFAULT_REPO_ID, task=None):
        self.root, self.repo_id, self.task = root, repo_id, task
        self.ds = None
        self.meta_dir = os.path.join(root, "meta")
        self.sources = self._read_json("sources.json", [])
        self.objects = self._read_json("objects.json", None)
        if os.path.isfile(os.path.join(self.meta_dir, "info.json")) and self.objects is None:
            raise EpisodeError(f"{root} has no meta/objects.json; rebuild it with log.py --overwrite")
        if self.objects is not None and "n_points" not in self.objects:
            raise EpisodeError(f"{root} was built with the old point layout; rebuild it with log.py --overwrite")

    def _read_json(self, name, default):
        path = os.path.join(self.meta_dir, name)
        if not os.path.isfile(path):
            return default
        with open(path) as f:
            return json.load(f)

    def _write_json(self, name, obj):
        path = os.path.join(self.meta_dir, name)
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(path + ".tmp", path)

    def __contains__(self, stem):
        return stem in self.sources

    def _open(self, d, vids):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if self.objects is None:  # new dataset: settings come from this first episode
            lay = layout_from_episode(d)
            sizes = {feat: video_hw(path) for feat, path in vids.items()}
            features = make_features(sizes, sum(lay["n_points"]))
            self.ds = LeRobotDataset.create(repo_id=self.repo_id, fps=episode_fps(d), features=features,
                                            root=self.root, robot_type="aria_gen2_human", use_videos=True)
            self.objects = {"object_names": lay["names"], "n_points": lay["n_points"],
                            "point_object_id": point_object_id(lay["n_points"]).tolist()}
            self._write_json("objects.json", self.objects)
            print(f"  created dataset {self.root}: {episode_fps(d)} fps, objects {lay['names']}, "
                  f"{sum(lay['n_points'])} point slots")
        else:
            self.ds = LeRobotDataset.resume(repo_id=self.repo_id, root=self.root)

    def _check(self, npz_path, d, vids):
        """Raise EpisodeError if the episode does not match the dataset's fps / video sizes / objects."""
        if tuple(self.ds.features["observation.state"]["shape"]) != (STATE_DIM,):
            raise EpisodeError(f"{self.root} stores {self.ds.features['observation.state']['shape']} hand values per "
                               f"frame, this code writes {STATE_DIM} ({', '.join(HAND_JOINTS)}): rebuild the dataset")
        if self.ds.fps != episode_fps(d):
            raise EpisodeError(f"{npz_path}: {episode_fps(d)} fps, the dataset has {self.ds.fps} "
                               "(post_process --stride must be the same for all recordings)")
        for feat, path in vids.items():
            want = tuple(self.ds.features[feat]["shape"][:2])
            if video_hw(path) != want:
                raise EpisodeError(f"{npz_path}: {feat} is {video_hw(path)}, the dataset has {want} "
                                   "(--video-width / STEREO_SIZE must be the same for all recordings)")
        if [int(x) for x in d["n_points"]] != self.objects["n_points"]:
            raise EpisodeError(f"{npz_path}: n_points {d['n_points'].tolist()}, the dataset has "
                               f"{self.objects['n_points']} (N_POINTS in post_process.py changed)")

    def add(self, folder):
        """Append the recording in a post_process output folder. Returns False if it is already in the dataset."""
        npz_path = find_npz(folder)
        stem = stem_of(npz_path)
        if stem in self.sources:
            print(f"  {stem}: already in {self.root}, skipped")
            return False
        d = np.load(npz_path)
        vids = episode_videos(npz_path, d)
        if self.ds is None:
            self._open(d, vids)
        self._check(npz_path, d, vids)

        names = self.objects["object_names"]
        a = episode_arrays(d, names, self.objects["n_points"])
        T = len(d["rgb_ts"])
        videos = {feat: read_video(path) for feat, path in vids.items()}
        bad = {feat: len(v) for feat, v in videos.items() if len(v) != T}
        if bad:
            raise EpisodeError(f"{npz_path}: npz has {T} frames, but the videos have {bad}")
        task = self.task or default_task(names)
        for t in range(T):
            self.ds.add_frame({
                "task": task,
                **{feat: v[t] for feat, v in videos.items()},
                "observation.state": a["state"][t],
                "observation.state_valid": a["state_valid"][t],
                "observation.object_points": a["points"][t],
                "observation.object_points_valid": a["points_valid"][t],
                "action": a["action"][t],
                "action_valid": a["action_valid"][t],
            })
        self.ds.save_episode()
        self.sources.append(stem)
        self._write_json("sources.json", self.sources)
        print(f"  episode {len(self.sources) - 1}: {stem}  ({T} frames)  -> {self.root}")
        return True

    def close(self):
        if self.ds is not None:
            self.ds.finalize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def push(self, private=True):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(self.repo_id, root=self.root)
        ds.push_to_hub(private=private)
        print(f"pushed to https://huggingface.co/datasets/{self.repo_id}  ({'private' if private else 'public'})")


def check_hub_login():
    """Exit early, before spending time building, if there is no Hugging Face login."""
    from huggingface_hub import HfApi
    try:
        user = HfApi().whoami()["name"]
    except Exception:
        raise SystemExit("Not logged in to Hugging Face. Run once:  uv run hf auth login   (token with write access)")
    print(f"logged in to the Hub as {user}")
