"""What policy_node expects from a policy. Plug one in with policy.plugin = "module:Class" in station.yaml."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    t: float                       # seconds, ROS clock
    hand_q: np.ndarray             # (17,) rad, imitate_common.hand.JOINT_NAMES order
    arm_pos: np.ndarray | None     # (3,) m in ARM_FRAME, None if the arm is not used
    arm_quat: np.ndarray | None    # (4,) x, y, z, w
    color: np.ndarray | None = None   # (H, W, 3) uint8 RGB, only if Policy.needs_images
    depth: np.ndarray | None = None   # (H, W) uint16 mm, aligned to color
    camera_k: np.ndarray | None = None  # (3, 3) colour intrinsics


@dataclass
class Action:
    hand_q: np.ndarray | None = None   # (17,) rad target; None = hold
    arm_pos: np.ndarray | None = None  # (3,) m target; None = hold
    arm_quat: np.ndarray | None = None
    info: dict = field(default_factory=dict)  # extra numbers to plot in the dashboard (name -> float)


class Policy:
    needs_images = False

    def __init__(self, checkpoint, conf):
        """checkpoint: policy.checkpoint from station.yaml (may be None). conf: the whole station config."""
        self.checkpoint = checkpoint
        self.conf = conf

    def reset(self, obs: Observation):
        """Called on start. Clear any history (action chunks, recurrent state)."""

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError
