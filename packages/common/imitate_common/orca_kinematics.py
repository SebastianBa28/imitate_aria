"""ORCA v2 right-hand forward kinematics from its URDF (numpy only).

Joint vectors are 17 values in rad, in hand.JOINT_NAMES order. Positions are in metres.

Palm frame (the default base, origin at the wrist joint):
+z points along the fingers, +x from the index side to the pinky side, and +y out of the palm.
transforms.wrist_frame builds a human wrist frame with the same axes.
"""
from pathlib import Path

import numpy as np

from . import hand
from .urdf import axis_angle_matrix as _axis_angle_matrix
from .urdf import load_joints

URDF_PATH = Path(__file__).parent / "assets" / "orcahand_v2_right.urdf"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")  # same order as Aria landmarks 0-4


class OrcaHand:
    def __init__(self, urdf_path=URDF_PATH, base="palm"):
        joints = load_joints(urdf_path)
        self._by_child = {j.child: j for j in joints}
        self.base = base
        self.joint_index = {n: i for i, n in enumerate(hand.JOINT_NAMES)}
        # chain[f]: joints from base to the fingertip; finger_idx[f]: their indices in the 17-vector
        self.chain, self.finger_idx = {}, {}
        for f in FINGERS:
            chain = self._chain(base, f"{f}_fingertip")
            self.chain[f] = chain
            self.finger_idx[f] = np.array([self.joint_index[j.name] for j in chain if j.type == "revolute"])

    def _chain(self, base, tip):
        chain, link = [], tip
        while link != base:
            if link not in self._by_child:
                raise ValueError(f"{tip} is not below {base} in the URDF")
            j = self._by_child[link]
            if j.type == "revolute" and j.name not in self.joint_index:
                raise ValueError(f"URDF joint {j.name} is not in hand.JOINT_NAMES")
            chain.append(j)
            link = j.parent
        return chain[::-1]

    def transform(self, base, link, q):
        """T_base_link (4x4) for joint vector q (17,)."""
        T = np.eye(4)
        for j in self._chain(base, link):
            T = T @ j.origin
            if j.type == "revolute":
                R = np.eye(4)
                R[:3, :3] = _axis_angle_matrix(j.axis, q[self.joint_index[j.name]])
                T = T @ R
        return T

    def finger_fk(self, finger, qf, jacobian=False):
        """Tip position (3,) in the base frame for this finger's joints qf; with jacobian, also J (3, len(qf))."""
        T = np.eye(4)
        origins, axes, k = [], [], 0
        for j in self.chain[finger]:
            T = T @ j.origin
            if j.type == "revolute":
                origins.append(T[:3, 3].copy())
                axes.append(T[:3, :3] @ j.axis)
                R = np.eye(4)
                R[:3, :3] = _axis_angle_matrix(j.axis, qf[k])
                T = T @ R
                k += 1
        p = T[:3, 3].copy()
        if not jacobian:
            return p
        J = np.stack([np.cross(a, p - o) for a, o in zip(axes, origins)], axis=1)
        return p, J

    def fingertips(self, q):
        """(5, 3) fingertip positions in the base frame, FINGERS order."""
        q = np.asarray(q, dtype=np.float64)
        return np.stack([self.finger_fk(f, q[self.finger_idx[f]]) for f in FINGERS])
