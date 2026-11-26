"""ORCA v2 right-hand forward kinematics from its URDF (numpy only).

Joint vectors are 17 values in rad, in hand.JOINT_NAMES order. Positions are in metres.

Palm frame (the default base, origin at the wrist joint):
+z points along the fingers, +x from the index side to the pinky side, and +y out of the palm.
transforms.wrist_frame builds a human wrist frame with the same axes.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import hand

URDF_PATH = Path(__file__).parent / "assets" / "orcahand_v2_right.urdf"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")  # same order as Aria landmarks 0-4


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_matrix(axis, angle):
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


class _Joint:
    def __init__(self, el):
        self.name = el.get("name")
        self.type = el.get("type")
        self.parent = el.find("parent").get("link")
        self.child = el.find("child").get("link")
        origin = el.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        self.origin = np.eye(4)
        self.origin[:3, :3] = _rpy_matrix(*rpy)
        self.origin[:3, 3] = xyz
        axis = el.find("axis")
        a = np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.array([1.0, 0, 0])
        self.axis = a / np.linalg.norm(a)
        if self.type not in ("fixed", "revolute"):
            raise ValueError(f"joint {self.name}: unsupported type {self.type}")


class OrcaHand:
    def __init__(self, urdf_path=URDF_PATH, base="palm"):
        joints = [_Joint(el) for el in ET.parse(urdf_path).getroot().iter("joint")]
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
