# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Imitation-learning data pipeline: record demonstrations with Meta Project Aria Gen2 glasses, turn each
`.vrs` into 3D object-point and hand trajectories (AINA recipe, arXiv 2511.16661), then pack all
recordings into a LeRobot dataset on the Hugging Face Hub. A second pipeline (`scene_*.py`) does the same for
in-scene demos from fixed RGB-D cameras (D435 + OAK, recorded with `packages/deploy/tools/record_cams.py` on
branch `in-scene`), fused into a ChArUco-calibrated table frame. `packages/policy/il.py` and
`packages/deploy/control.py` are empty placeholders, and `packages/common/`, `packages/deploy/{deploy,launch,tests}`
hold no source yet. All working code is in `packages/data/`.

## Commands

Environment is managed by uv (Python 3.12, CUDA 12.8 torch, versions pinned in `uv.lock`). No tests,
linter or build step exist; these are standalone scripts (`package = false`). Scripts import their
siblings and the shared core package directly (`from dataset_io import ...`, `from aina.track import ...`), so run
them by path from the repo root, as below. Relative
defaults (`output/`, `datasets/`, `sam2.1_l.pt`) resolve against the cwd.

```bash
bash scripts/setup.sh                      # once: uv sync + FoundationStereo (pinned commit, patched) + checkpoint into third_party/
uv run python packages/data/record_session.py --name demo23 --data-dir raw_data/wild   # record, ENTER to stop (default dir is data/)
uv run python packages/data/post_process.py --vrs raw_data/wild/demo23_nov24.vrs   # full pipeline
uv run python packages/data/post_process.py --vrs ... --stop 60                    # quick test on first 60 frames
uv run python packages/data/post_process.py --vrs ... --append-to [DIR] [--keep-output]  # also append to datasets/imitate-aria, delete output folder
uv run python packages/data/log.py --overwrite | --append FOLDER... | --push [--public]  # build / append / upload (Hub: SebastianBa28/imitate-aria)
# in-scene (raw_data/scene/<demo>/, one folder per recording):
uv run python packages/data/scene_pack.py raw_data/scene/* [--delete-frames]     # JPEG/PNG -> color.mp4 + lossless FFV1 depth.mkv
uv run python packages/data/scene_calib.py raw_data/scene/calib_nov30             # ChArUco clip -> calib.json (T_world_cam)
uv run python packages/data/scene_process.py raw_data/scene/cup_insc_nov30 [--calib none --cams oak] [--append-to]
uv run python packages/data/log.py --overwrite --input output/scene --repo-id SebastianBa28/imitate-aria-scene
```

Needs an NVIDIA GPU (developed on an 8 GB RTX 4060 Ti — keep memory use in mind; `--tracker online`
and `--scale` exist for this). First `post_process.py` run downloads SAM 2.1-L (`sam2.1_l.pt` into the
cwd), Grounding DINO (HF cache) and CoTracker3 (torch hub).

## Architecture

**Stages and their contracts** (each stage only reads the previous one's files):
1. `record_session.py` drives the `aria_gen2` CLI → `raw_data/<name>_<mon><day>.vrs`.
2. `post_process.py` (one file, all steps in `main()`) → `output/<stem>/`:
   auto-start (first frame with a VIO pose) → Grounding DINO boxes on **frame 0 only** → SAM 2
   masks → sample `N_POINTS` per mask → CoTracker3 tracks them through the clip (no re-detection) →
   rectify the two front SLAM cameras (`RectifiedStereo`) → FoundationStereo disparity → depth →
   project into RGB (`FastProjector`, z-buffered) → each tracked point reads depth in a
   `LIFT_RADIUS` window → VIO pose puts it in world. Hand landmarks come from on-device hand tracking
   (`read_hands`), wrist frames from landmarks 5/8/17 (`wrist_poses`).
   Main output is `<stem>_aina.npz`; its full schema is in the module docstring — keep that docstring
   and the README in sync when changing outputs.
3. `dataset_io.py` (`EpisodeWriter`) turns one output folder into one LeRobot episode (state/action =
   right-hand 5 fingertips + wrist + index/pinky knuckles = 24 values, action = next frame's state; object points; `*_valid` flags), creating the
   dataset from the first episode or reopening it with `LeRobotDataset.resume`. Used by both `log.py`
   and `post_process.py --append-to`; `meta/sources.json` (vrs stems) prevents duplicates,
   `meta/objects.json` holds the fixed per-object point-slot layout (`sum(N_POINTS)` slots).

**Shared core `packages/data/aina/`** (moved out of post_process.py, behaviour unchanged): `config.py` (PROMPTS,
N_POINTS, LIFT_RADIUS), `segment.py`, `track.py`, `lift.py` (window-median lift from a per-pixel 3D buffer),
`overlay.py`, `hands.py` (Aria landmark order, `openpose_to_aria`, `wrist_poses`), `geometry.py`, `scene_io.py`
(record_cams reader, raw or packed, frame pairing by `t_capture_s`, `find_calib`), `hands_wilor.py` (WiLoR +
`place_metric`).

**In-scene pipeline:** `scene_calib.py`: world = table frame from the board lying flat during the first `--flat-s`
seconds, then a joint least-squares fit over the frames both cameras see. `scene_process.py`: per camera seg/track/lift
with the aligned sensor depth (no FoundationStereo), `N_POINTS[k] // n_cams` points per camera, concatenated; WiLoR
hands triangulated across cameras or single view (metric shape + depth). Writes the Aria npz keys (plus per-camera
fields: `point_cam`, `T_world_cam`, `hand_uv` (C,T,...), `videos` JSON) to `output/scene/<demo>/`, so `dataset_io.py`
works unchanged. Scene episodes go into a separate dataset (`datasets/imitate-aria-scene`). The robot base transform
(`T_base_world`, hand-eye) is not implemented yet. WiLoR comes from git; its pinned requirements are replaced via
`[[tool.uv.dependency-metadata]]` in pyproject (they would downgrade torch). Its MANO pickle is rewritten once
without chumpy.

**Conventions that span files:**
- Missing data is **NaN** in the npz (never zero/filled). `dataset_io.py` converts NaN → 0 plus a `*_valid`
  flag because LeRobot stats can't handle NaN.
- `T_a_b` maps b-frame points into frame a. "world" = on-device VIO frame (gravity-aligned, Z up),
  only consistent within one `session_idx`; data is never re-framed into a table/object frame.
- Row t of every per-frame array = frame t of every saved video (`_rgb`, `_overlay`, `_depth`).
- Objects tracked are set by the `PROMPTS` / `N_POINTS` constants in `aina/config.py` (not CLI
  flags). The first episode fixes prompts, `N_POINTS`, `--stride` and `--video-width` for the dataset;
  mismatching recordings are refused.
- Hand arrays have axis `(left, right)`; Aria landmark indices 0–4 are fingertips, 5 is wrist.
- `pyproject.toml` bans `opencv-python-headless` (pulled by lerobot) via `override-dependencies`;
  don't add xformers/flash-attn from FoundationStereo's environment (they downgrade torch).

`legacy/visualize_hand_trajectories.py` predates the current pipeline: it expects files from older
`extract_hand_trajectories.py` / `extract_object_trajectories.py` scripts and an AprilTag table frame,
not `_aina.npz`.

## In-scene status (as of 2026-11-28)

- **Branches:** the recorder (`record_cams.py`) stays on `in-scene` (deploy, pixi env); all processing is on
  `aina` (uv env), to be merged into `main`. The only contract between them is the recording folder format.
- **Raw data:** VRS in `raw_data/wild/`, scene demos in `raw_data/scene/` (gitignored). The 4 `cup_insc_nov27*`
  demos there are packed (2.0 GB → 522 MB; colour PSNR ~39 dB, depth bit-exact); the unpacked originals are on
  the USB (`/media/sebas/USB/imitate_aria_raw/cams`).
- **Tested:** Aria refactor bit-identical; calibration on a synthetic clip (≤0.6 mm, ≤0.05°); hand triangulation
  on synthetic data (~1 mm); single-camera `scene_process.py --calib none` on the real demos (objects and right
  hand tracked); scratch LeRobot scene dataset loads. **Not yet tested on real data:** two-camera fusion (the
  existing demos have no calibration clip).
- **Lab workflow** (README "In-scene demos"): print `packages/data/assets/charuco_board.pdf` (36 mm squares, fits
  A4; measure one, pass `--square-mm` if different) → calibration clip (board flat 2 s, then moved slowly through
  both views 10–15 s) → demos with `--data-dir raw_data/scene` → another calibration clip at the end to catch a
  camera that moved → `scene_pack.py` → `scene_calib.py` → `scene_process.py`.
- **Not built yet:** hand-eye calibration (`T_base_world`, null in `calib.json`). Tag on the end effector, seen by
  one camera is enough (camera↔camera gives the other), 10–15 varied-rotation poses, `cv2.calibrateHandEye`. The
  capture tool needs ROS + the arm, so it belongs on the deploy side; demos need no re-processing when it lands.
- **Licence:** WiLoR downloads the MANO hand model (non-commercial licence).
- **Pending:** `log.py --push` of `datasets/imitate-aria` (public repo is fine) awaits the user's go-ahead.
