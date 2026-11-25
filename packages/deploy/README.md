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
