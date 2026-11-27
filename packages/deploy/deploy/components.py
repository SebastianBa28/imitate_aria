"""The station's processes and how to start them, shared by launch/robot.launch.py and the dashboard's
process supervisor (deploy/station/supervisor.py) so the two can't drift apart. No ROS imports.
"""
import importlib.util
import sys
from dataclasses import dataclass

from imitate_common import contract

ROBOT_DOMAIN_ID = "7"   # multipanda_ros2's `run` script on the NUC
LIVE_AGE_S = 2.5        # a component's topic newer than this means it runs (policy state comes at 1 Hz)
# what the station's Processes card manages (set by the pixi station tasks; default: mock)
PROFILE_ENV, HAND_ENV, CAMERA_ENV = "IMITATE_STATION_PROFILE", "IMITATE_STATION_HAND", "IMITATE_STATION_CAMERA"
OAK_ENV = "IMITATE_STATION_OAK"
HAND_MOVES = ("hand_node moves the ORCA hand to its neutral pose on startup. Keep the fingers clear of "
              "objects and people.")


@dataclass(frozen=True)
class Component:
    name: str               # short key used by the dashboard: hand, camera, oak, arm, policy
    module: str             # python -m <module>
    title: str
    topic: str              # a topic it publishes: live means it is running, wherever it was started
    params: tuple = ()      # ROS parameters, "name:=value"
    requires: tuple = ()    # python modules it needs (hardware drivers, only in the pixi hw env)
    confirm: str = ""       # starting it needs this warning accepted first (real hardware that moves)
    managed: bool = True    # False: runs elsewhere (the Franka controller); the dashboard only shows it
    host_label: str = ""    # label of an unmanaged component
    node_name: str = ""     # ROS node name override (two components running the same module)

    @property
    def label(self):
        """Process name in launch output, e.g. hand_node."""
        return self.host_label or self.node_name or self.module.rsplit(".", 1)[-1]

    def missing_modules(self):
        return [m for m in self.requires if importlib.util.find_spec(m) is None]

    def argv(self, python=None):
        cmd = [python or sys.executable, "-m", self.module]
        args = [x for p in self.params for x in ("-p", p)]
        if self.node_name:
            args += ["-r", f"__node:={self.node_name}"]
        return cmd + (["--ros-args"] + args if args else [])


def components(profile, hand=None, camera=None, oak=None, calibrate=False, policy=True):
    """Processes for a profile ("mock" or "real"), in start order. hand/camera/oak ("mock"|"real"|None, oak
    also "off") override the profile; the arm is only simulated in the mock profile (the real one runs on the
    NUC)."""
    if profile not in ("mock", "real"):
        raise ValueError(f"profile must be mock or real, got {profile!r}")
    for name, value, allowed in (("hand", hand, ("mock", "real")), ("camera", camera, ("mock", "real")),
                                 ("oak", oak, ("mock", "real", "off"))):
        if value not in (None, "", *allowed):
            raise ValueError(f"{name} must be {' or '.join(allowed)}, got {value!r}")
    mock_hand = (hand or profile) == "mock"
    mock_camera = (camera or profile) == "mock"
    oak = oak or profile
    out = []
    if profile == "mock":
        out.append(Component("arm", "deploy.nodes.mock_arm", "Arm (simulated)", contract.ARM_POSE))
    out.append(Component("hand", "deploy.nodes.hand_node", "Hand (simulated)" if mock_hand else "Hand (ORCA)",
                         contract.HAND_STATE, (f"mock:={str(mock_hand).lower()}", f"calibrate:={str(calibrate).lower()}"),
                         requires=() if mock_hand else ("orca_core",), confirm="" if mock_hand else HAND_MOVES))
    out.append(Component("camera", "deploy.nodes.mock_camera" if mock_camera else "deploy.nodes.realsense_node",
                         "Camera (simulated)" if mock_camera else "Camera (RealSense)", contract.CAMERA_COLOR,
                         requires=() if mock_camera else ("pyrealsense2",)))
    if oak == "mock":
        out.append(Component("oak", "deploy.nodes.mock_camera", "OAK camera (simulated)", contract.OAK_COLOR,
                             ("camera:=oak",), node_name="mock_oak"))
    elif oak == "real":
        out.append(Component("oak", "deploy.nodes.oak_node", "OAK camera", contract.OAK_COLOR, requires=("depthai",)))
    if policy:
        out.append(Component("policy", "deploy.nodes.policy_node", "Policy node", contract.POLICY_STATE))
    return out


def monitored(profile):
    """Components the station shows but never starts: in the real profile, the Franka controller
    (multipanda_ros2 on the NUC), so the card says whether arm data arrives."""
    if profile != "real":
        return []
    return [Component("arm", "", "Arm (Franka controller)", contract.ARM_POSE, managed=False,
                      host_label="multipanda_ros2 on the NUC")]


def station_selection(env):
    """(profile, hand, camera, oak) for the station's Processes card from the environment (PROFILE_ENV etc.,
    set by the pixi station tasks). Raises ValueError on bad values."""
    get = lambda key: (env.get(key) or "").strip().lower() or None  # noqa: E731
    profile, hand, camera, oak = get(PROFILE_ENV) or "mock", get(HAND_ENV), get(CAMERA_ENV), get(OAK_ENV)
    components(profile, hand=hand, camera=camera, oak=oak)   # validates
    return profile, hand, camera, oak


def domain_refusal(profile, domain):
    """Why this profile must not run on this ROS domain, or None."""
    if profile == "mock" and str(domain) == ROBOT_DOMAIN_ID:
        return (f"profile mock on ROS_DOMAIN_ID={domain}, the real robot's domain: the simulated nodes would "
                "talk to the real arm. Use domain 0 (`pixi run mock` / `pixi run station`).")
    return None


def is_live(health_entry):
    """True if a topic health entry (deploy.station.monitor) shows recent messages."""
    age = (health_entry or {}).get("age_s")
    return age is not None and age < LIVE_AGE_S
