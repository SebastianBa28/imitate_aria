"""Starts the whole robot station with one command.

    ros2 launch launch/robot.launch.py profile:=mock     # simulated hand, arm and camera (dev machine)
    ros2 launch launch/robot.launch.py profile:=real     # ORCA hand + RealSense; the Franka controller runs on its own machine
Options: station:=false (no dashboard server), policy:=false, calibrate:=true (real hand, it moves),
hand:=mock|real and camera:=mock|real override the profile (e.g. profile:=real hand:=mock tests the real
arm and camera without the ORCA hand; the arm is never mocked in the real profile).
Run from packages/deploy/ (the pixi tasks do).
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_DOMAIN_ID = "7"  # multipanda_ros2's `run` script on the NUC


def node(module, *ros_params):
    cmd = [sys.executable, "-m", module]
    if ros_params:
        cmd += ["--ros-args"] + [x for p in ros_params for x in ("-p", p)]
    return ExecuteProcess(cmd=cmd, cwd=DEPLOY_ROOT, output="screen", name=module.rsplit(".", 1)[-1],
                          sigterm_timeout="5", sigkill_timeout="5")


def setup(context):
    get = lambda name: context.launch_configurations[name].lower()  # noqa: E731
    profile = get("profile")
    if profile not in ("mock", "real"):
        raise RuntimeError(f"profile must be mock or real, got {profile!r}")
    mock = profile == "mock"
    domain = os.environ.get("ROS_DOMAIN_ID", "0")
    if mock and domain == ROBOT_DOMAIN_ID:
        raise RuntimeError(f"profile:=mock on ROS_DOMAIN_ID={domain}, the real robot's domain: the simulated "
                           "nodes would talk to the real arm. Use `pixi run mock` (domain 0).")
    pick = lambda name: (get(name) or profile) == "mock"  # noqa: E731
    for name in ("hand", "camera"):
        if get(name) not in ("", "mock", "real"):
            raise RuntimeError(f"{name} must be mock or real, got {get(name)!r}")
    nodes = [node("deploy.nodes.hand_node", f"mock:={str(pick('hand')).lower()}", f"calibrate:={get('calibrate')}")]
    nodes.append(node("deploy.nodes.mock_camera" if pick("camera") else "deploy.nodes.realsense_node"))
    if mock:
        nodes.append(node("deploy.nodes.mock_arm"))
    if get("policy") == "true":
        nodes.append(node("deploy.nodes.policy_node"))
    if get("station") == "true":
        nodes.append(ExecuteProcess(cmd=[sys.executable, "-m", "deploy.station.server"], cwd=DEPLOY_ROOT,
                                    output="screen", name="station"))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="mock", description="mock or real"),
        DeclareLaunchArgument("policy", default_value="true"),
        DeclareLaunchArgument("station", default_value="true"),
        DeclareLaunchArgument("calibrate", default_value="false", description="real hand: calibrate on startup (moves)"),
        DeclareLaunchArgument("hand", default_value="", description="mock or real; default: the profile"),
        DeclareLaunchArgument("camera", default_value="", description="mock or real; default: the profile"),
        OpaqueFunction(function=setup),
    ])
