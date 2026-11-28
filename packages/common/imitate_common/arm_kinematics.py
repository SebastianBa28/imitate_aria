"""Franka Panda (right arm) kinematics from its URDF (numpy only).

Joint vectors are 7 values in rad, in contract.ARM_JOINT_NAMES order. Poses are in the controller's "world"
frame (contract.ARM_FRAME). The flange (right_link8) is what the controller publishes as the end-effector pose.

The dashboard does the same FK in the browser from `chain_json` (packages/dashboard/web/skeleton.js);
the IK here is only for the simulated arm (deploy/nodes/mock_arm.py), which has no joints of its own.
"""
from pathlib import Path

import numpy as np

from . import contract
from .transforms import quat_to_matrix
from .urdf import axis_angle_matrix, chain, chain_json, load_joints

URDF_PATH = Path(__file__).parent / "assets" / "panda_right.urdf"
BASE, FLANGE = "world", "right_link8"
READY_Q = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])  # franka "ready" pose


class Panda:
    def __init__(self, urdf_path=URDF_PATH):
        self.chain = chain(load_joints(urdf_path), BASE, FLANGE)
        revolute = [j for j in self.chain if j.type == "revolute"]
        if [j.name for j in revolute] != list(contract.ARM_JOINT_NAMES):
            raise ValueError(f"URDF joints {[j.name for j in revolute]} != contract.ARM_JOINT_NAMES")
        self.lower = np.array([j.lower for j in revolute])
        self.upper = np.array([j.upper for j in revolute])

    def frames(self, q):
        """T_world_x (4x4) after each joint of the chain (joint origins, then the flange last)."""
        T, out, k = np.eye(4), [], 0
        for j in self.chain:
            T = T @ j.origin
            if j.type == "revolute":
                R = np.eye(4)
                R[:3, :3] = axis_angle_matrix(j.axis, q[k])
                T = T @ R
                k += 1
            out.append(T)
        return out

    def fk(self, q):
        """T_world_flange (4x4)."""
        return self.frames(np.asarray(q, dtype=np.float64))[-1]

    def jacobian(self, q):
        """Geometric Jacobian (6, 7) of the flange: rows 0-2 linear, 3-5 angular, in world."""
        T, origins, axes, k = np.eye(4), [], [], 0
        for j in self.chain:
            T = T @ j.origin
            if j.type == "revolute":
                origins.append(T[:3, 3].copy())
                axes.append(T[:3, :3] @ j.axis)
                R = np.eye(4)
                R[:3, :3] = axis_angle_matrix(j.axis, q[k])
                T = T @ R
                k += 1
        p = T[:3, 3]
        return np.vstack([np.stack([np.cross(a, p - o) for a, o in zip(axes, origins)], axis=1),
                          np.stack(axes, axis=1)]), T

    def ik(self, position, quat, q_start=None, iters=50, tol=1e-5, damping=1e-3):
        """Joint angles reaching the flange pose (position, quat (x, y, z, w)), from q_start (default: ready).

        Damped least squares with a pull towards q_start in the null space; clamped to the joint limits.
        Returns (q, position error m). Not for real control, only for the simulated arm.
        """
        q0 = np.array(READY_Q if q_start is None else q_start, dtype=np.float64)
        q = q0.copy()
        R_goal = quat_to_matrix(quat)
        err = np.inf
        for _ in range(iters):
            J, T = self.jacobian(q)
            dp = np.asarray(position) - T[:3, 3]
            Re = R_goal @ T[:3, :3].T   # rotation still to go, in world
            dw = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            e = np.concatenate([dp, dw])
            err = np.linalg.norm(dp)
            if err < tol and np.linalg.norm(dw) < 1e-4:
                break
            JJt = J @ J.T + damping * np.eye(6)
            J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))
            dq = J_pinv @ e + (np.eye(7) - J_pinv @ J) @ (0.1 * (q0 - q))
            q = np.clip(q + dq, self.lower, self.upper)
        return q, err

    def chain_json(self):
        return chain_json(self.chain)
