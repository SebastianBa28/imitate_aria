# Robot side: ROS 2 nodes, policy runner, deployment dashboard

Separate environment from the rest of the repo: [pixi](https://pixi.sh) with ROS 2 Jazzy from RoboStack,
for the Ubuntu dev box (`linux-64`) and the Apple Silicon Mac at the robot (`osx-arm64`).

```bash
curl -fsSL https://pixi.sh/install.sh | bash      # once
cd packages/deploy
pixi run test                 # unit tests
pixi run mock                 # simulated hand + arm + camera, demo policy, dashboard at http://127.0.0.1:8000
pixi run -e hw real           # real ORCA hand + RealSense; start the Franka controller on its machine first
```
First time on the robot Mac: run the checks in [spike/README.md](spike/README.md).

## What runs

| Node | Real | Mock | Topics |
|---|---|---|---|
| `hand_node` | ORCA v2 via orca_core | first-order lag | in `/right/hand/command`, out `/right/hand/joint_states` (100 Hz), services `/right/hand/estop`, `/right/hand/set_torque` |
| camera | `realsense_node` (pyrealsense2) | `mock_camera` (video or pattern) | `/camera/color/image_raw`, `/camera/depth/image_raw` (mm, aligned), `/camera/color/camera_info` |
| arm | external Franka controller | `mock_arm` | in `/right/arm/end_effector_pose_cmd`, out `/right/arm/end_effector_pose` (frame `world`) |
| `policy_node` | plugin from `station.yaml` | `DemoPolicy` | see below |
| station | `deploy.station.server` | same | dashboard + API, port 8000 |

Names live in `packages/common/imitate_common/contract.py`; rates, limits and paths in `config/station.yaml`.

## Dashboard

Open http://127.0.0.1:8000 while the station runs. It is a static page in `packages/dashboard/web/`
(plain HTML/JS, no build step, no internet needed) served by `deploy.station.server`: camera, topic health
against `station.expected_hz`, hand joints within their ranges of motion, the arm inside its workspace,
history plots, policy metrics, the event log and the policy/hand buttons. **Esc** or the E-STOP button
stops the policy and releases the hand. To open it from another machine set `station.host: 0.0.0.0`
(there is no login).

ROS transport: `config/fastdds.xml` (set by pixi on activation) enlarges Fast DDS's shared-memory segment.
With the default one, 640x480 camera frames arrived at ~5 Hz instead of 30.

## Manual control (no policy)

The dashboard's **Manual control** card moves the robot without a trained policy: hand joint sliders and
arm jog buttons (+/-x, y, z in 0.5-5 cm steps, plus Hold). It runs through `policy_node` (status `manual`),
so the same rules apply as for a policy: Preview shows the targets without moving anything, Live sends
them; targets are clamped to the hand's ranges of motion and `arm.workspace`; and the motion is
the arm is rate-limited by the `manual:` limits in `config/station.yaml` (2 cm/s). Hand sliders go straight to
the hand with no speed limit (`hand.max_step_deg` and `manual.hand_max_step_deg` are `null`, as in rwr_system):
only the ranges of motion are enforced.
Targets start at the measured pose (nothing jumps on entering), the arm keeps its orientation, and manual
refuses to start if the arm is outside the workspace. Exit with Stop / Exit manual; Esc is the E-stop.

In Live, an arm command is never more than `arm.max_lead_m` (2 cm) ahead of the measured pose, so a
lagging or blocked arm cannot wind up commands (the rule rwr_system's arm init uses). Start/Home/Manual
in Live refuse when no arm controller subscribes to `/right/arm/end_effector_pose_cmd`, or when the
arm reports its pose in a frame other than `world`.

### First test with the real arm

What rwr_system (`~/rwr_humble_ws/src/rwr_system_imitate2`) tells us about the controller: it takes
`PoseStamped` targets in frame `world` (its keyboard teleop streams them at 20 Hz and moves the target
1 cm per key press), the right arm's start pose there is `[0.45, 0.0, 0.2]` with the gripper pointing
down (quaternion `[1, 0, 0, 0]`), and its own arm init moves at 2 mm per 10 Hz step. It has no workspace
limits of its own: `arm.workspace` here is the only box.

The controller is multipanda_ros2 in Docker on the NUC (172.16.0.254): ROS 2 Humble, CycloneDDS,
`ROS_DOMAIN_ID=7`. The pixi tasks set the domain (7 for `real`, `real-arm` and `spike-ping`, 0 for `mock`;
the launch file refuses `profile:=mock` on domain 7), and `config/station_env.sh` restricts Fast DDS to this
machine's 172.16.0.x address (`ROBOT_NET_PREFIX` to change it): with Wi-Fi and docker interfaces also up,
no pose data arrived from the controller. Fast DDS here and CycloneDDS there interoperate.

1. Start the Franka controller (Franka Desk: unlock, activate FCI; NUC: `./run`, `franka_bringup`), then
   `pixi run -e hw spike-ping arm`: `OK: ~30 Hz, frame 'world', position x y z`.
2. Check that the printed position is inside `arm.workspace` in `config/station.yaml`, and shrink the box
   to a small, clear region around it for the first runs.
3. `pixi run -e hw real-arm` (real arm and RealSense, simulated hand; append `camera:=mock` without the camera) or `pixi run -e hw real` (everything
   real), open the dashboard, confirm `/right/arm/end_effector_pose` is green.
4. Enter manual in **Preview** and jog: the green "Policy" marker moves, the arm must not.
5. Exit, switch to **Live**, enter manual, jog 0.5 cm, with a hand on the Franka's physical E-stop.

## Running a policy

The policy node starts in **preview**: actions appear on `/policy/action_*` and in the dashboard, but nothing
moves. Send `live` (only while idle) to also send them to the robot. Commands: `start`, `stop`, `reset`, `home`,
`live`, `preview` on `/policy/command` (the dashboard buttons do this). Every action, in both modes, is clamped
to the hand's ranges of motion and the arm workspace, and stepped from the previous target. A stale hand or arm
observation, or an exception in the policy, stops the run with status `fault`.

To plug in a trained policy, subclass `deploy.policies.base.Policy` and set `policy.plugin: module:Class`
(and `policy.checkpoint`) in `config/station.yaml`.

Safety: the dashboard E-stop stops the policy and releases the hand's torque. It goes through software and the
network: keep the Franka's physical E-stop and the hand's power within reach.
