# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Imitation learning from human demos recorded with Meta Project Aria Gen2 glasses, deployed on an ORCA v2 right hand + Franka Panda arm + RealSense D435 + Luxonis OAK-D Pro (second camera). Three stages, each in `packages/`:

1. **data** (`packages/data/`): record `.vrs` → post-process into 3D object point tracks + hand landmarks → build a LeRobot dataset.
2. **policy** (`packages/policy/`): training; `il.py` is an empty placeholder. No trained policy exists yet.
3. **deploy** (`packages/deploy/`): ROS 2 nodes, policy runner, manual control and the dashboard server for the robot station; the dashboard page is in `packages/dashboard/web/`.

`packages/common/imitate_common/` (numpy only, no ROS or torch) is shared by both sides: hand joint order/limits (`hand.py`), ROS topic/service names (`contract.py`), pose helpers (`transforms.py`, quaternions `(x, y, z, w)`, `T_a_b` maps b-frame points into frame a).
- **Hand IK** (`orca_kinematics.py`, `hand_ik.py`): ORCA v2 right-hand FK from `assets/orcahand_v2_right.urdf` (kinematics-only copy of rwr_system's `src/viz/models/orcahand_v2/orcahand_v2.urdf.xacro`), and per-finger Levenberg-Marquardt IK from 5 fingertip targets in the palm frame (+z along fingers, +x index→pinky, +y out of the palm) to the 17 joints. `transforms.wrist_frame` builds the same axes from Aria landmarks 5/8/17. The ORCA `wrist` joint is held at 0 (the arm takes the wrist pose). Not wired into deploy yet; that the URDF's joint signs/zeros match orca_core's is unverified.

The system was ported from rwr_system (ETH RWR course code), checked out at `~/rwr_humble_ws/src/rwr_system_imitate2`. Look there for how the hand and arm were driven before (e.g. `src/hand_control/hand_control_node.py`, `experiments/run_keyboard_teleop_franka.py`, `experiments/model_inference_node.py`).

## Two separate environments

- **Data/training: uv** (root `pyproject.toml`, `uv.lock`, Python 3.12, CUDA 12.8 torch). `[tool.uv] package = false`: the scripts are standalone, not an installed package. Setup: `bash scripts/setup.sh` (runs `uv sync`, clones FoundationStereo at a pinned commit into `third_party/`, patches its deprecated autocast calls, downloads checkpoints). Run scripts with `uv run python packages/data/<script>.py`.
  - Never add `opencv-python-headless` (overridden out because it clashes with `opencv-python`), and never add `xformers`/`flash-attn` from FoundationStereo's environment.yml (they downgrade torch).
- **Robot: pixi** (`packages/deploy/pixi.toml`, ROS 2 Jazzy from RoboStack, `linux-64` + `osx-arm64`; `pixi.lock` is committed). Installs `imitate_common` and `deploy` as editable packages. Envs `default` and `hw` (orca_core, pyrealsense2, depthai 2.32) share a solve group, so `hw`-only pins constrain `default` too: `uvicorn` is pinned to `>=0.34.2,<0.35` because `orca_core==0.5.2` requires it (without the pin, even `pixi install` fails). No torch in this env yet; a trained policy will need it.
- The station runs on this Ubuntu laptop (RTX 5060, needs CUDA ≥ 12.8 builds), not a Mac as first planned: real-time vision needs an NVIDIA GPU. `osx-arm64` is still listed in the pixi platforms.

## Commands

Data pipeline (repo root):
```bash
uv run aria_doctor && uv run aria_gen2 auth pair                                    # once: pair the glasses (needs the Aria Companion App)
uv run python packages/data/record_session.py --name demo23 --data-dir raw_data   # ENTER stops; saves raw_data/<name>_<mon><day>.vrs
uv run python packages/data/post_process.py --vrs raw_data/demo23_nov24.vrs        # add --stop 60 for a quick test
uv run python packages/data/log.py [--push] [--overwrite]                           # output/*/ -> datasets/imitate-aria (LeRobot)
```

Robot side (`cd packages/deploy`; tasks set `ROS_DOMAIN_ID` themselves, see "ROS networking"):
```bash
pixi run test                        # pytest -q tests ../common/tests (no ROS or hardware needed)
pixi run pytest -q tests/test_manual.py::test_jog_accumulates_and_stays_in_the_workspace   # one test
pixi run mock                        # everything simulated, domain 0; dashboard at http://127.0.0.1:8000
pixi run -e hw real-arm              # real Franka + RealSense, simulated hand (domain 7); append camera:=mock to also simulate the camera
pixi run -e hw real                  # real hand + RealSense + Franka (domain 7); the hand moves to neutral on startup
pixi run -e hw spike-ping arm        # arm link check: expect "OK: ~30 Hz, frame 'world', position x y z"
pixi run -e hw spike-realsense       # expect ~30 fps and USB 3.x
pixi run spike-env                   # checks the pixi env imports (no hardware)
pixi run station                     # dashboard alone (domain 0); start/stop the mock nodes from its Processes card
pixi run -e hw station-real          # dashboard alone, real profile (domain 7): start the real hand/RealSense/policy from the card
pixi run -e hw station-real-arm      # same with a simulated hand
pixi run -e hw spike-orca            # reads hand joints, moves nothing; needs config/orca/right/calibration.yaml
pixi run -e hw ros2 launch launch/robot.launch.py profile:=real calibrate:=true   # recalibrate the hand (fingers sweep full range)
```

Launch options (`launch/robot.launch.py`, run from `packages/deploy/`): `profile:=mock|real`, `hand:=mock|real`, `camera:=mock|real` (RealSense) and `oak:=mock|real|off` override the profile (the arm is never mocked in `real`), `station:=false`, `policy:=false`, `calibrate:=true`.

No linter or formatter is configured.

## Data pipeline architecture (`post_process.py`)

The AINA recipe (arXiv 2511.16661): Grounding DINO box → SAM 2 mask on the **first sampled frame only** → sample `N_POINTS` per object inside each mask → CoTracker3 tracks them through the whole clip (no re-detection) → FoundationStereo on the rectified front SLAM pair gives depth → points are lifted to 3D and put in the world frame with the VIO glasses pose.

- Because detection runs on the first frame only, recordings must start with every object in view: look down at them from 40–60 cm and hold still ~2 s before acting (README "Record"). By default the script skips ahead to the first frame where exposure has settled and a pose exists; `--start N` overrides that.
- Objects to track and point counts are `PROMPTS` / `N_POINTS` at the top of `post_process.py` (same order = `obj_id`). Other tuning constants are next to them.
- Output goes to `output/<recording>/`: `<stem>_aina.npz` plus `_rgb.mp4`, `_overlay.mp4`, `_depth.mp4`, `_seg.jpg`. Video frame t matches row t of every per-frame npz array. The module docstring documents every npz field; keep it in sync when changing outputs.
- Conventions: missing data is **NaN, never zero**. "world" is the on-device VIO frame (gravity-aligned, Z up), consistent only within one `session_idx`.
- `log.py` turns each `output/<recording>/` into one LeRobot episode: `observation.state` = right-hand 5 fingertips, `action` = fingertips at the next frame, `observation.object_points`, `*_valid` flags (missing becomes 0 there, flagged invalid, never filled). All recordings must share stride/fps, `--video-width` and `PROMPTS`. Pushes to `SebastianBa28/imitate-aria` (private by default).

## Deploy architecture (`packages/deploy/`)

- Each node is its own Python process (`python -m deploy.nodes.<x>`): `hand_node` (real ORCA via `deploy/hw/orca.py`, or a first-order-lag mock), `realsense_node`/`mock_camera`, `oak_node`/`mock_camera -p camera:=oak` (node `mock_oak`; second camera on `OAK_*` topics, size in `station.yaml` `oak:`), `mock_arm` (mock profile only), `policy_node`, and the station server (`deploy.station.server`). Which nodes a profile runs, with which parameters, is defined once in `deploy/components.py` (ROS-free), used by both the launch file and the dashboard.
- **Processes card**: the station's `deploy/station/supervisor.py` (ROS-free, `tests/test_supervisor.py`) starts/stops/restarts one profile's nodes (`POST /api/process`, `{"name": arm|hand|camera|policy|all, "action": start|stop|restart, "confirmed": bool}`; state + log lines in the WS ticks). The profile is fixed when the station starts (a ROS process can't change domain): `IMITATE_STATION_PROFILE=mock|real` (+ `IMITATE_STATION_HAND`/`_CAMERA`/`_OAK` overrides, `_OAK=off` drops the OAK and its health rows), set by the `station*` pixi tasks and parsed by `components.station_selection`. Real profile: the arm (NUC controller) is a monitored-only row (`components.monitored`, live/no data); a component with a `confirm` warning (the real hand, which moves to neutral on start) is refused with 409 + the warning until the request has `confirmed: true` (the page asks via `confirm()`); hardware nodes whose driver (`requires`: orca_core, pyrealsense2, depthai) is missing are refused, so the real profile needs the hw env. Calibration is never started from the dashboard. Each child gets its own process group (stop = SIGINT → SIGTERM → SIGKILL, 5 s apart, like the launch file) and runs through `deploy/station/child.py`, which sets `PR_SET_PDEATHSIG` so nodes die even if the station is SIGKILLed; that is why all spawns go through one long-lived spawner thread (the kernel ties it to the spawning *thread*). Crashes are shown with exit code and never auto-restarted. The mock profile is refused on domain 7; any start is refused when the component's topic is already live (started elsewhere, e.g. `pixi run mock` or `pixi run -e hw real`: shown as "external"). Not done yet: switching profile from the page (the bridge would need to restart on the new domain), a heartbeat so policy_node stops when the dashboard is gone, the Franka controller over SSH.
- All topic names come from `imitate_common.contract`; all rates, limits, workspace and paths from `config/station.yaml` (loaded by `deploy/config.py`; override the whole file with `IMITATE_DEPLOY_CONFIG=/path.yaml`). Don't hardcode either in nodes.
- Topics carry radians; `hand.py` tables are degrees with `*_RAD` arrays derived from them. Joint order is `hand.JOINT_NAMES` (17 joints, same as orca_core's `joint_ids`).
- **`policy_node` is the only node that commands the robot.** Statuses `idle | running | homing | manual | fault`; commands on `/policy/command`: `start stop reset home manual live preview`.
  - Starts in **preview**: actions go to `/policy/action_*` only. **live** also sends them to `/right/hand/command` and `/right/arm/end_effector_pose_cmd`. Mode changes only while idle.
  - Every action is clamped to the hand ROMs and `arm.workspace`, and the arm is stepped by `arm.max_step_m`/`max_step_deg` per tick.
  - Live only: an arm command is never more than `arm.max_lead_m` (2 cm) from the **measured** pose (no wind-up if the arm lags or is blocked, the rule rwr_system's arm init uses). Start/home/manual refuse when no node other than the station subscribes to the arm command topic, or the arm pose's frame isn't `world`.
  - Stale hand/arm observations (`*.stale_s`) or a policy exception → `fault`. Preserve all of these guarantees when editing.
- **Hand speed is not limited** (user's explicit choice: "this hand is pretty responsive and robust"): `hand.max_step_deg` and `manual.hand_max_step_deg` are `null` (= no limit, as in rwr_system; the motors run `current_based_position`, which limits force). Only the ROMs are enforced. In manual mode, hand targets are sent the moment they arrive (`policy_node.send_hand_now`), not on the 10 Hz tick. Setting a number brings a per-command cap back.
- **Manual control** (`policy_node` status `manual`): hand targets + arm jog (±x/y/z, 0.5–5 cm, Hold). The dashboard card for it was removed (2026-11-27, user's choice); the backend (`POST /api/manual`, `policy_node`) is still there. Targets arrive as JSON on `/policy/manual` (via `POST /api/manual`), validated/clamped in the ROS-free `deploy/manual.py` (`tests/test_manual.py`). Targets start at the measured state (no jump on entering), the arm keeps its entry orientation, the arm moves at `manual.arm_max_step_m` (2 mm/tick = 2 cm/s), and manual refuses to start if the arm is outside `arm.workspace`.
- **Policies**: subclass `deploy.policies.base.Policy` (`reset(obs)`, `act(obs) -> Action`; `needs_images = True` to get camera frames) and set `policy.plugin: module:Class` + `policy.checkpoint` in `station.yaml`. `DemoPolicy` is the default. A trained policy should run at the rate its data was recorded at (`policy.hz`, 10).
- Station (`deploy/station/`): `server.py` is one process with a ROS bridge on a background thread (`bridge.py`) and a FastAPI app: `GET /api/config`, `POST /api/command`, `POST /api/manual`, `POST /api/process`, `GET /api/processes`, WebSocket `/ws` (JSON snapshot/ticks + binary JPEG frames, first byte = channel 0 colour / 1 depth / 2 OAK colour / 3 OAK depth, `server.CHANNELS` = `app.js` `CHANNELS`). `buffers.py` and `monitor.py` are ROS-free, which is why `tests/test_station_core.py` runs without ROS; keep logic testable that way. The bridge logs refused policy commands (`"<cmd>: <reason>"`) as warnings.
- Dashboard (`packages/dashboard/web/`): static ES-module page served as-is by `server.py` (no build step, no external libraries, must work offline at the robot). `app.js` holds the series store, WebSocket client and panels (processes, policy, camera, topic health, policy metrics, events); `chart.js` is the canvas time-series chart. It reads everything from `/api/config` and the WS payloads, so new series or metrics appear without frontend changes (`METRIC_NAMES` gives nicer titles). Esc = E-stop (`all/estop`: stop policy + hand e-stop, which disconnects the hand until hand_node restarts). No login: keep `station.host: 127.0.0.1`.

## ROS networking (hard-won, keep)

- **Franka controller**: multipanda_ros2 in Docker (`multipanda-container`) on the NUC `172.16.0.254`, ROS 2 **Humble**, **CycloneDDS**, **`ROS_DOMAIN_ID=7`** (hardcoded at the top of its `run` script), `--network host`. Topics `/right/arm/end_effector_pose` (~30 Hz) and `/right/arm/end_effector_pose_cmd`, PoseStamped in frame **`world`**; world→arm offset in `franka_bringup/config/hardware_layout.yaml` on the NUC (`[0,0,0]` single arm, `[0,-0.26,0]` bimanual; check it on first runs). Start: Franka Desk (Chrome, `172.16.0.4` or `.6`) unlock brakes + activate FCI; on the NUC `cd multipanda_ros2 && ./run`, `source install/setup.bash`, `ros2 launch franka_bringup multimode_franka.launch.py robot_ip_1:=172.16.0.<4|6>`. Lab notes: Notion page "NUC and Franka controller".
- **Domains**: pixi tasks set 7 for `real`/`real-arm`/`spike-ping` and 0 for `mock`; the launch file refuses `profile:=mock` on domain 7 (the mock arm would talk to the real one). Never run anything on domain 7 without the user knowing the robot is live.
- **Fast DDS profile** (`config/fastdds.xml`, loaded via `[activation.env]`): 16 MB shared-memory segment (default 512 KB is smaller than one camera frame; images dropped from 30 Hz to ~5 Hz) + UDP restricted to `${ROBOT_NET_IP}` and loopback. `config/station_env.sh` (pixi activation script) sets `ROBOT_NET_IP` to this machine's `172.16.0.x` address (DHCP, so detected each time; `ROBOT_NET_PREFIX` overrides the subnet; fallback 127.0.0.1). Without the restriction the laptop's Wi-Fi/docker interfaces were also advertised: topics were discovered but no pose data arrived. Fast DDS (here, Jazzy) ↔ CycloneDDS (NUC, Humble) works once that's set; the type hash shows `INVALID`, which is normal across distros. This Fast DDS version supports `${ENV}` in the XML but not interface names.

## Hardware notes (as of 2026-11-26)

- **RealSense D435** must be on a USB 3 port/cable: on USB 2 it enumerated but delivered no frames in any mode. Stream is `848x480@30` (16:9 = full ~70° colour FOV; 640x480 crops to ~56°). 1280x720 gives no extra FOV, raises min depth to ~28 cm and costs bandwidth. A "wider" view in a camera app is the laptop webcam or the RealSense IR imagers (87°×58°, grayscale).
- **OAK-D Pro AF** (second camera, `oak_node`, depthai 2.x API): 1080p colour scaled on the device to 960x540 (full FOV), autofocus locked at the calibrated lens position (keeps the intrinsics valid), 640x400 stereo depth aligned to colour on the device at the output size (aligning at full 1080p capped it at ~26 fps with ~150 ms latency), extended disparity off (with subpixel it disables the median filter). Needs USB 3. Measured 30 Hz on both topics next to the D435. Only one process can open it (the recorder on branch `d435-demos` or `oak_node`); `dai.Device` needs the `DeviceInfo` from `getFirstAvailableDevice()` in this version.
- **ORCA hand**: shows up as `/dev/ttyACM0` and `/dev/ttyACM1` (same serial, `port: auto` works; user is in `dialout`). `config/orca/right/calibration.yaml` (gitignored) was copied from rwr_system's install dir (`~/rwr_humble_ws/install/hand_control/share/hand_control/orca/configs/orcahand_v2/right/`, written 2026-11-24; same `config.yaml`). hand_node refuses to start without it unless `calibrate:=true`. Motor 12 intermittently misses sync reads at rest. In one `real` run every motor stopped answering after the hand moved to neutral and the ttyACM devices re-enumerated: suspected board brown-out (hand power supply/cable); unconfirmed, `sudo dmesg -w` during startup would show it. The hand's serial bus handles ~140–160 sync reads/s; hand_node reads at 100 Hz.
- **Franka**: arm pose at ~30 Hz (`station.expected_hz` set to 30). Default/rwr start pose for the right arm `[0.45, 0.0, 0.2]`, gripper down (quat `[1, 0, 0, 0]`); `arm.home` is still `null` on purpose (Home moves at policy speed; a home pose should be confirmed at the robot). First live tests: shrink `arm.workspace` to ±5 cm around the pose `spike-ping arm` prints, jog 0.5 cm in Preview, then Live with a hand on the physical E-stop.

## Testing without disturbing a running station

- The user often runs a station on port 8000 themselves: test on an isolated copy instead, e.g. `IMITATE_DEPLOY_CONFIG=<copy of station.yaml with station.port 8001>` and `pixi run mock` (domain 0), or another `ROS_DOMAIN_ID`. Never kill processes you didn't start.
- Children started from a test station's Processes card stop with the station; check with `ps` afterwards that no `deploy.nodes` process is left.
- `ros2` puts its env's `bin/` first on `sys.path`; an env with PySide6 has an unrelated `bin/deploy.py` that shadows our package (hence the `sys.path.insert` in `robot.launch.py`).
- Start test stations in their own process group (`setsid bash -c 'echo $$ > pgid; exec <cmd>' &`; `$!` after `setsid <cmd> &` is not the PGID because setsid may fork) and stop them with `kill -INT -- -<pgid>`: signalling only `pixi` leaves the ros2 launch children running and holding port 8000.
- `pkill -f`/`pgrep -f` patterns like `deploy.nodes.mock_arm` also match your own shell's command line and kill it; anchor them to the interpreter path (`^/home/.../packages/deploy/.pixi/envs/<env>/bin/python`).
- `ros2 topic pub/echo` under pixi take several seconds to start; wait ≥ 6 s before asserting on them. `ros2 topic echo` truncates strings unless `--full-length`.
- Dashboard UI checks: headless Chrome via CDP (`google-chrome --headless=new --remote-debugging-port=...`, driven from the pixi Python's `websockets`), screenshots + `Runtime.evaluate`; override `window.confirm` for Live/manual dialogs.

## Open items / ideas (not implemented)

- **Torque handling (ported from rwr_system, recommended)**: on these Feetech motors any position command re-enables torque. rwr's hand node drops commands while torque is off and sets the goal to the current pose before re-enabling (`hold_current_pose`). Ours does neither, so after "Torque off" a slider move or policy re-enables the hand, and "Torque on" can snap it.
- hand_node should detect a lost hand connection (all motors unavailable) and report it clearly instead of repeating driver warnings.
- Optional hand smoothing: interpolate to the latest target at a fixed motor rate (e.g. 50 Hz) independent of `policy.hz`; express remaining speed limits per second instead of per tick.
- Dashboard ideas discussed: start/stop the Franka controller over SSH (`docker start multipanda-container && docker exec -d ... ros2 launch franka_bringup ...`, SSH key auth, robot IP in `station.yaml`, check `docker ps` first since the NUC is shared), a robot setup checklist panel (NUC reachable, container running, FCI/brakes from `/paco/franka_robot_state_broadcaster/robot_state`, arm pose frame/rate, hand calibrated, camera USB 3), "Recover arm" button, start-on-boot for the station. Keep unlocking brakes a deliberate human action.
- No 3D robot view on the dashboard (needs ORCA/Franka model files in `packages/deploy/assets/robot`).

## Repo notes

- Large artifacts are gitignored: `*.vrs`, `output/`, `*.npz`, `*.mp4`, model weights (`*.pt`, `*.pth`), `third_party/`, `datasets/`, `.pixi/`, `*.egg-info/`, `config/orca/*/calibration.yaml`.
- `packages/data/visualize_hand_trajectories.py` is from an older pipeline: it reads `*_hands.npz`/`*_objects.npz`/`*_table.npz` (from a removed `extract_hand_trajectories.py`, AprilTag table frame), not the current `_aina.npz`.
- First post-process run downloads SAM 2.1-L (`sam2.1_l.pt`, into the current directory), Grounding DINO (HF cache) and CoTracker3 (torch hub cache).
