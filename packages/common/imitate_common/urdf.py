"""Minimal URDF kinematics parsing (numpy only): fixed and revolute joints, origins and axes."""
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_matrix(axis, angle):
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


class Joint:
    def __init__(self, el):
        self.name = el.get("name")
        self.type = el.get("type")
        self.parent = el.find("parent").get("link")
        self.child = el.find("child").get("link")
        origin = el.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        self.origin = np.eye(4)
        self.origin[:3, :3] = rpy_matrix(*rpy)
        self.origin[:3, 3] = xyz
        axis = el.find("axis")
        a = np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.array([1.0, 0, 0])
        self.axis = a / np.linalg.norm(a)
        limit = el.find("limit")
        self.lower = float(limit.get("lower", "-inf")) if limit is not None else -np.inf
        self.upper = float(limit.get("upper", "inf")) if limit is not None else np.inf
        if self.type not in ("fixed", "revolute"):
            raise ValueError(f"joint {self.name}: unsupported type {self.type}")


def load_joints(urdf_path):
    return [Joint(el) for el in ET.parse(urdf_path).getroot().iter("joint")]


def chain(joints, base, tip):
    """Joints from link `base` down to link `tip`, in order."""
    by_child = {j.child: j for j in joints}
    out, link = [], tip
    while link != base:
        if link not in by_child:
            raise ValueError(f"{tip} is not below {base} in the URDF")
        j = by_child[link]
        out.append(j)
        link = j.parent
    return out[::-1]


def chain_json(joints):
    """A chain for the dashboard (FK in the browser): [{name, type, origin (4x4 rows), axis}]."""
    return [{"name": j.name, "type": j.type, "origin": np.round(j.origin, 9).tolist(), "axis": j.axis.tolist()}
            for j in joints]
