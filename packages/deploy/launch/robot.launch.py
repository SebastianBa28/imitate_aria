"""Starts the whole robot station with one command.

    ros2 launch launch/robot.launch.py profile:=mock     # simulated hand, arm and camera (dev machine)
    ros2 launch launch/robot.launch.py profile:=real     # ORCA hand + RealSense; the Franka controller runs on its own machine
Options: station:=false (no dashboard server), policy:=false, calibrate:=true (real hand, it moves),
hand:=mock|real, camera:=mock|real (RealSense) and oak:=mock|real|off (second camera) override the profile
(e.g. profile:=real hand:=mock tests the real arm and cameras without the ORCA hand; the arm is never mocked
in the real profile).
Run from packages/deploy/ (the pixi tasks do).
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ros2 puts its bin/ dir first on sys.path; some envs have an unrelated bin/deploy.py (PySide6's tool)
sys.path.insert(0, DEPLOY_ROOT)
from deploy.components import components, domain_refusal  # noqa: E402


def process(cmd, name):
    return ExecuteProcess(cmd=cmd, cwd=DEPLOY_ROOT, output="screen", name=name,
                          sigterm_timeout="5", sigkill_timeout="5")


def setup(context):
    get = lambda name: context.launch_configurations[name].lower()  # noqa: E731
    profile = get("profile")
    refusal = domain_refusal(profile, os.environ.get("ROS_DOMAIN_ID", "0"))
    if refusal:
        raise RuntimeError(refusal)
    comps = components(profile, hand=get("hand"), camera=get("camera"), oak=get("oak"),   # raises on bad values
                       calibrate=get("calibrate") == "true", policy=get("policy") == "true")
    nodes = [process(c.argv(), c.label) for c in comps]
    if get("station") == "true":
        nodes.append(process([sys.executable, "-m", "deploy.station.server"], "station"))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="mock", description="mock or real"),
        DeclareLaunchArgument("policy", default_value="true"),
        DeclareLaunchArgument("station", default_value="true"),
        DeclareLaunchArgument("calibrate", default_value="false", description="real hand: calibrate on startup (moves)"),
        DeclareLaunchArgument("hand", default_value="", description="mock or real; default: the profile"),
        DeclareLaunchArgument("camera", default_value="", description="RealSense: mock or real; default: the profile"),
        DeclareLaunchArgument("oak", default_value="", description="OAK: mock, real or off; default: the profile"),
        OpaqueFunction(function=setup),
    ])
