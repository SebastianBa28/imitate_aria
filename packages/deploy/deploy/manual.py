"""Manual control targets for policy_node's "manual" status (dashboard sliders and arm jog). No ROS.

The dashboard sends JSON on POLICY_MANUAL:
  {"hand": {"index_mcp": 0.8, ...}}     absolute joint targets, rad (joints not named keep their target)
  {"arm_jog": [dx, dy, dz]}             move the arm target by this much, m (each |d| <= max_jog_m)
  {"arm_hold": true}                    arm target = current measured position (stop where it is)
  {"hand_neutral": true}                hand target = neutral pose
Targets start at the measured state, so entering manual never makes the robot jump. They are clamped
here (hand ROMs, arm workspace); policy_node then steps toward them with the manual speed limits.
"""
import numpy as np

from imitate_common import hand


class ManualTargets:
    def __init__(self, ws_lo, ws_hi, max_jog_m):
        self.ws_lo, self.ws_hi = np.asarray(ws_lo, float), np.asarray(ws_hi, float)
        self.max_jog_m = float(max_jog_m)
        self.hand = None     # (17,) rad
        self.arm_pos = None  # (3,) m, or None when the arm is not used
        self.arm_quat = None

    def inside_workspace(self, pos, tol=1e-3):
        return bool(np.all(pos >= self.ws_lo - tol) and np.all(pos <= self.ws_hi + tol))

    def start(self, hand_q, arm=None):
        """Targets = measured state. Returns None, or why manual control cannot start."""
        if arm is not None and not self.inside_workspace(arm[0]):
            p = ", ".join(f"{v:.3f}" for v in arm[0])
            return f"arm at ({p}) is outside arm.workspace in station.yaml: move it inside or widen the box"
        self.hand = hand.clamp_rad(hand_q)
        self.arm_pos = None if arm is None else np.asarray(arm[0], float).copy()
        self.arm_quat = None if arm is None else np.asarray(arm[1], float).copy()
        return None

    def apply(self, msg, measured_arm_pos=None):
        """Update targets from one dashboard message (dict). Returns None, or why it was rejected."""
        if not isinstance(msg, dict):
            return "manual message must be a JSON object"
        if "hand" in msg:
            targets = msg["hand"]
            if not isinstance(targets, dict):
                return "hand must map joint names to radians"
            q = self.hand.copy()
            for name, v in targets.items():
                if name not in hand.JOINT_NAMES:
                    return f"unknown joint {name!r}"
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    return f"bad value for {name}"
                q[hand.JOINT_NAMES.index(name)] = v
            self.hand = hand.clamp_rad(q)
        if msg.get("hand_neutral"):
            self.hand = hand.NEUTRAL_RAD.copy()
        if "arm_jog" in msg or msg.get("arm_hold"):
            if self.arm_pos is None:
                return "the arm is not used (policy.uses_arm is false)"
        if "arm_jog" in msg:
            d = msg["arm_jog"]
            if (not isinstance(d, (list, tuple)) or len(d) != 3
                    or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in d)):
                return "arm_jog must be [dx, dy, dz] in metres"
            d = np.asarray(d, float)
            if np.any(np.abs(d) > self.max_jog_m + 1e-9):
                return f"jog larger than manual.max_jog_m ({self.max_jog_m} m)"
            self.arm_pos = np.clip(self.arm_pos + d, self.ws_lo, self.ws_hi)
        if msg.get("arm_hold"):
            if measured_arm_pos is None:
                return "no arm pose to hold"
            self.arm_pos = np.clip(np.asarray(measured_arm_pos, float), self.ws_lo, self.ws_hi)
        return None

    def as_dict(self):
        return {"hand": None if self.hand is None else [round(float(x), 5) for x in self.hand],
                "arm_pos": None if self.arm_pos is None else [round(float(x), 5) for x in self.arm_pos]}
