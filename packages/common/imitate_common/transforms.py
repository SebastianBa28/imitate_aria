"""Pose helpers. Quaternions are (x, y, z, w) as in ROS and scipy. T_a_b maps b-frame points into a."""
import numpy as np


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_matrix(q):
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(position, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat)
    T[:3, 3] = position
    return T


def quat_angle(q0, q1):
    """Rotation angle (rad, in [0, pi]) between two orientations."""
    d = abs(float(np.dot(quat_normalize(q0), quat_normalize(q1))))
    return 2.0 * np.arccos(min(1.0, d))


def slerp(q0, q1, t):
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:  # take the short way round
        q1, d = -q1, -d
    if d > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta = np.arccos(d)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def step_pose(pos, quat, pos_target, quat_target, max_dist, max_angle):
    """Move a pose toward a target by at most max_dist (m) and max_angle (rad).

    Used so the arm never receives a far-away target in one message.
    """
    pos = np.asarray(pos, dtype=np.float64)
    delta = np.asarray(pos_target, dtype=np.float64) - pos
    dist = float(np.linalg.norm(delta))
    new_pos = pos_target if dist <= max_dist else pos + delta * (max_dist / dist)
    angle = quat_angle(quat, quat_target)
    new_quat = quat_target if angle <= max_angle else slerp(quat, quat_target, max_angle / angle)
    return np.asarray(new_pos, dtype=np.float64), quat_normalize(new_quat)
