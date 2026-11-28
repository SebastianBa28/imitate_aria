"""Small geometry helpers. T_a_b (4,4) maps b-frame points into frame a."""
import numpy as np


def as_matrix(T):
    """Sophus SE3 (or ndarray) -> (4,4) float64."""
    return np.asarray(T.to_matrix() if hasattr(T, "to_matrix") else T, dtype=np.float64)


def unit(v):
    return v / np.linalg.norm(v)


def transform(T, P):
    """Apply T (4,4) to points P (...,3); NaN stays NaN."""
    return P @ T[:3, :3].T + T[:3, 3]


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def triangulate(P_list, uv_list):
    """Linear (DLT) triangulation of one point seen by several cameras.
    P_list: (C,3,4) projection matrices (normalized/undistorted pixel coords consistent with uv_list),
    uv_list: (C,2). Returns (3,) in the frame the P matrices map from."""
    A = []
    for P, (u, v) in zip(P_list, uv_list):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]
