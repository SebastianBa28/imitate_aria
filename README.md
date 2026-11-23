# imitate_aria

Record demonstrations with **Meta Project Aria Gen2** glasses and post-process them into 3D
object trajectories (AINA recipe: segment on the first frame, track points with CoTracker3,
lift them to 3D with FoundationStereo depth from the front SLAM cameras).

## Requirements

- Linux x86_64 with glibc 2.34+ (Ubuntu 22.04 / 24.04)
- NVIDIA GPU + driver (tested on an RTX 4060 Ti, 8 GB)
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup (once)

```bash
git clone <this repo> && cd imitate_aria
bash scripts/setup.sh
```

This creates `.venv` from `uv.lock` (identical versions for everyone, CUDA 12.8 torch) and puts
FoundationStereo + its checkpoints in `third_party/FoundationStereo` (~5 GB). Re-running it is safe.

Pair the glasses once (needs the Aria Companion App on your phone):

```bash
uv run aria_doctor
uv run aria_gen2 auth pair
```

## Record

```bash
uv run python packages/data/record_session.py --name demo23 --data-dir sample_data
```

Press ENTER to stop; the `.vrs` is downloaded into `sample_data/`.

**Objects are detected on the first frame only.** Before starting, put the cube on the table next
to the tray and look down at both from 40–60 cm. Then start, hold still ~2 s, pick the cube, put it
in the tray, hold still ~1 s, stop.

## Post-process

```bash
uv run python packages/data/post_process.py --vrs sample_data/demo23_<timestamp>.vrs
```

Useful options: `--stop N` (only the first N RGB frames, quick test), `--roi` (draw the first-frame
boxes yourself if Grounding DINO misses an object), `--prompt "..."` (repeat per object),
`--seg-frame K` (detect and segment on output frame K instead of the first, e.g. if frame 0 is badly
exposed; points are still tracked through the whole clip).
It prints, per object, the share of points with depth, the median depth and the point jitter while
still, and writes its files to `output/<recording>/` (change the base folder with `--out-dir`):

- `<recording>_aina.npz`: per frame the timestamp and glasses pose; per tracked point its 3D world
  position, depth, 2D pixel position and visibility; which object each point belongs to; the
  first-frame boxes and masks; metadata (fps, stride, ...). Missing values are NaN.
- `<recording>_rgb.mp4`: the same frames, downscaled to 640 px wide (`--video-width`, 0 = no video).
  Video frame t matches row t of every per-frame array.
- `<recording>_overlay.mp4`: the same video with the tracked points drawn on, for checking quality
  (`--no-overlay` to skip). One colour per object; filled dot = visible with a 3D position, hollow =
  visible but no depth, small red dot = hidden. Hand skeletons from the glasses' hand tracking: left
  white, right yellow (also saved in the npz as `hand_landmarks_*` / `hand_uv`).
- `<recording>_seg.jpg`: the first frame with the Grounding DINO boxes and SAM 2 masks drawn on.
  Check this first: points are only sampled inside these masks, so a bad mask means bad tracking.

The first run downloads more models: SAM 2.1-L (`sam2.1_l.pt`, into the folder you run from),
Grounding DINO (Hugging Face cache) and CoTracker3 (torch hub cache).

## Build the LeRobot dataset

```bash
uv run hf auth login                                   # once, token with write access
uv run python packages/data/log.py --push              # every output/<recording>/ becomes one episode
```

Builds `datasets/imitate-aria` locally and uploads it to `SebastianBa28/imitate-aria` (private; add
`--public` to change). Without `--push` it only builds locally; `--overwrite` rebuilds an existing local copy.
All recordings must share the same `--stride` (fps), `--video-width` and prompts. Per frame it stores the RGB
video, `observation.state` (right-hand keypoints + object centroids), `action` (right-hand keypoints at the
next frame), both hands, the object points with validity flags, and the glasses pose, all in each
recording's VIO world frame.
