"""Stand-in policy: slowly opens and closes the fingers and moves the arm in a small horizontal circle.

For testing the station and dashboard end to end before a trained policy exists.
"""
import time

import numpy as np

from deploy.policies.base import Action, Policy
from imitate_common import hand

CURL = [hand.JOINT_NAMES.index(j) for j in hand.JOINT_NAMES if j.endswith(("_mcp", "_pip"))]


class DemoPolicy(Policy):
    period_s = 6.0
    curl_rad = np.deg2rad(45)
    circle_m = 0.03

    def reset(self, obs):
        self.t0 = obs.t
        self.arm_start = None if obs.arm_pos is None else (obs.arm_pos.copy(), obs.arm_quat.copy())

    def act(self, obs):
        t0 = time.perf_counter()
        phase = 2 * np.pi * (obs.t - self.t0) / self.period_s
        q = hand.NEUTRAL_RAD.copy()
        q[CURL] += self.curl_rad * 0.5 * (1 - np.cos(phase))
        action = Action(hand_q=q, info={"phase": float(phase % (2 * np.pi))})
        if self.arm_start is not None:
            pos, quat = self.arm_start
            action.arm_pos = pos + self.circle_m * np.array([np.cos(phase) - 1, np.sin(phase), 0.0])
            action.arm_quat = quat
        time.sleep(max(0.0, 0.015 - (time.perf_counter() - t0)))  # pretend inference takes ~15 ms
        return action
