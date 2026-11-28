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
uv run python packages/data/record_session.py --name demo23 --data-dir raw_data/wild
```

Press ENTER to stop; the `.vrs` is downloaded into `raw_data/wild/` as `<name>_<mon><day>.vrs` (e.g. `demo23_nov24.vrs`; `_1`, `_2`, ... for repeats on the same day).

**Objects are detected on the first frame only.** Before starting, put the cube on the table next
to the tray and look down at both from 40–60 cm. Then start, hold still ~2 s, pick the cube, put it
in the tray, hold still ~1 s, stop.

## Post-process

```bash
uv run python packages/data/post_process.py --vrs raw_data/wild/demo23_nov24.vrs
```

Useful options: `--stop N` (only the first N RGB frames, quick test),
`--start N` (start at RGB frame N; by default the script starts at the first frame with a glasses (VIO) pose, about 1 s in, and prints which frame it picked),
`--stride N` (keep every Nth frame), `--scale F` (downscale frames to save GPU memory),
`--tracker online|offline` (CoTracker3: `online` (default) uses sliding windows, so memory doesn't grow with clip length;
`offline` tracks better through occlusions but needs more memory), `--box-thr` (Grounding DINO box threshold, default 0.25),
`--device` and `--fs-path` (FoundationStereo folder, default `third_party/FoundationStereo` or `$FOUNDATION_STEREO_PATH`).
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
- `<recording>_depth.mp4`: stereo depth from the front SLAM cameras (rectified left view, 768x768) in
  turbo colours (red = near, blue = far) shaded by surface orientation so edges show, black = no depth (`--no-depth-video` to skip).
- `<recording>_seg.jpg`: the first frame with the Grounding DINO boxes and SAM 2 masks drawn on.
  Check this first: points are only sampled inside these masks, so a bad mask means bad tracking.

The first run downloads more models: SAM 2.1-L (`sam2.1_l.pt`, into the folder you run from),
Grounding DINO (Hugging Face cache) and CoTracker3 (torch hub cache).

## Build the LeRobot dataset

Two ways in, both writing to `datasets/imitate-aria` (one episode per recording):

```bash
# straight from post-processing: append, then delete output/<recording>/ (add --keep-output to keep it)
uv run python packages/data/post_process.py --vrs raw_data/wild/demo23_nov24.vrs --append-to

# from folders you already post-processed (e.g. after checking the overlay video)
uv run python packages/data/log.py --append output/demo23_nov24 [more folders...]
uv run python packages/data/log.py --overwrite          # rebuild from every output/<recording>/

uv run hf auth login                                    # once, token with write access
uv run python packages/data/log.py --push               # upload to SebastianBa28/imitate-aria (private; --public to change)
```

Without `--append-to`, `post_process.py` only writes `output/<recording>/` (debugging). `--append-to` takes an
optional folder (default `datasets/imitate-aria`), creates the dataset if it doesn't exist, and skips a recording
that is already in it (listed in `meta/sources.json`), so re-running is safe. It never pushes, and it refuses
`--video-width 0`, `--no-overlay` and `--no-depth-video` (the dataset stores all three videos).

The first episode fixes fps (`--stride`), video sizes (`--video-width`), objects (`PROMPTS`) and point counts
(`N_POINTS`); later recordings with different settings are refused. Changing one of them means rebuilding:
delete the dataset and re-run. Per frame it stores the RGB
video, `observation.state` (right hand: 5 fingertips, wrist, index and pinky knuckles, 24 values), `observation.object_points`, `action` (the same joints at
the next frame) and `*_valid` flags (missing values are 0, never filled), all in each recording's VIO world
frame. Each object owns a fixed block of point slots (`meta/objects.json`); slots a recording could not fill
are invalid. The overlay and depth videos are stored too, as `debug.overlay` and `debug.depth` (for viewing
only, not policy inputs).

## In-scene demos (fixed RGB-D cameras)

The same AINA processing for demos recorded with the fixed cameras at the robot (D435 front-left, OAK overhead),
fused into one **table frame** with a ChArUco calibration. Recording uses `packages/deploy/tools/record_cams.py`
(branch `in-scene`, pixi env); processing is here (uv env).

**In the lab, every session:**

1. Print `packages/data/assets/charuco_board.pdf` once (A4 landscape, 100 % scale), glue it flat onto something
   rigid and measure one square (nominal 36 mm).
2. Fix the cameras; don't touch them after the calibration clip.
3. Calibration clip: `pixi run -e hw record-cams --name calib --data-dir <repo>/raw_data/scene`. Board **flat on the
   table** in the workspace, seen by both cameras, still for 2 s (this defines the world frame); then lift it and
   slowly tilt/move it through both views for 10–15 s.
4. Demos: `pixi run -e hw record-cams --name cup_insc --data-dir <repo>/raw_data/scene`. Both objects visible to
   both cameras at the start; hold still ~1 s.
5. Another calibration clip at the end (shows whether a camera moved).

**Then:**

```bash
uv run python packages/data/scene_pack.py raw_data/scene/* --delete-frames   # JPEG/PNG -> color.mp4 + lossless depth.mkv
uv run python packages/data/scene_calib.py raw_data/scene/calib_nov30 [--square-mm 35.8]
uv run python packages/data/scene_process.py raw_data/scene/cup_insc_nov30 [--append-to]
uv run python packages/data/log.py --overwrite --input output/scene --repo-id SebastianBa28/imitate-aria-scene
```

- `scene_pack.py`: thousands of images become two files per camera, about 4× smaller (colour H.264 CRF 18, about 9×;
  depth FFV1, lossless, 2–6×). It checks every pack (frame count, depth bit-exact, colour PSNR) before deleting
  anything. Every script reads raw and packed recordings.
- `scene_calib.py`: writes `<clip>/calib.json` (`T_world_cam` per camera, reprojection RMS, a cross-camera
  triangulation check in mm) and `calib_check.jpg` (world axes drawn on each camera). Expect RMS below 1 px and a check
  below 3 mm. `scene_process.py` automatically uses a solved `calib*` clip from the same day (`--calib PATH` to choose one).
- `scene_process.py`: per camera, Grounding DINO + SAM 2 on the first frame, half of `N_POINTS` per camera, CoTracker3,
  lifted with that camera's own depth into the table frame; hands from WiLoR on every frame, triangulated when both
  cameras see them, otherwise single view with metric scale and depth correction. Output in `output/scene/<demo>/`: the
  same `_aina.npz` format as the Aria pipeline (plus per-camera fields), per-camera rgb/overlay/depth videos and
  `_seg.jpg`, and `_stats.json` (per object visibility and 3D %, the gap between the cameras' frame-0 object
  centroids, and per hand detection / triangulation %). Without a calibration: `--calib none --cams oak`, which
  processes one camera in its own camera frame.
- The scene dataset is separate (`datasets/imitate-aria-scene`): table frame, not the Aria VIO frame, and one set of
  videos per camera (`observation.images.<cam>`, `debug.overlay_<cam>`, `debug.depth_<cam>`).
- Not done yet: the table to robot base transform (`T_base_world` in `calib.json` is null). It needs a hand-eye
  calibration: a tag on the end effector, about 10–15 arm poses with varied rotations, seen by at least one camera.
  The demos don't need re-processing for it.
