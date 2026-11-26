"""ORCA v2 right hand: joint order, ranges of motion and neutral pose.

Same order as orca_core's `joint_ids` (packages/deploy/config/orca/right/config.yaml) and the
joint names of the orcahand_v2 URDF. Degrees here; every ROS topic carries radians.
"""
import numpy as np

JOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip",
    "index_abd", "index_mcp", "index_pip",
    "middle_abd", "middle_mcp", "middle_pip",
    "ring_abd", "ring_mcp", "ring_pip",
    "pinky_abd", "pinky_mcp", "pinky_pip",
)
NUM_JOINTS = len(JOINT_NAMES)

# joint_roms from the orca_core config, degrees
ROM_DEG = {
    "wrist": (-65, 35),
    "thumb_cmc": (-45, 33), "thumb_abd": (-18, 55), "thumb_mcp": (-25, 100), "thumb_dip": (-15, 107),
    "index_abd": (-25, 30), "index_mcp": (-25, 100), "index_pip": (-15, 107),
    "middle_abd": (-27, 27), "middle_mcp": (-25, 100), "middle_pip": (-15, 107),
    "ring_abd": (-27, 27), "ring_mcp": (-25, 100), "ring_pip": (-15, 107),
    "pinky_abd": (-30, 30), "pinky_mcp": (-25, 100), "pinky_pip": (-15, 107),
}

NEUTRAL_DEG = {
    "wrist": -20,
    "thumb_cmc": 0, "thumb_abd": 50, "thumb_mcp": 33, "thumb_dip": 18,
    "index_abd": -14, "index_mcp": 2, "index_pip": 6,
    "middle_abd": -4, "middle_mcp": 2, "middle_pip": 4,
    "ring_abd": 10, "ring_mcp": -2, "ring_pip": 8,
    "pinky_abd": 22, "pinky_mcp": -4, "pinky_pip": -2,
}

LOWER_RAD = np.deg2rad([ROM_DEG[j][0] for j in JOINT_NAMES])
UPPER_RAD = np.deg2rad([ROM_DEG[j][1] for j in JOINT_NAMES])
NEUTRAL_RAD = np.deg2rad([NEUTRAL_DEG[j] for j in JOINT_NAMES])


def clamp_rad(q):
    """Clamp joint angles (rad, JOINT_NAMES order) to the ranges of motion. NaN stays NaN."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (NUM_JOINTS,):
        raise ValueError(f"expected {NUM_JOINTS} joint angles, got shape {q.shape}")
    return np.clip(q, LOWER_RAD, UPPER_RAD)


def reorder(names, values):
    """Return values (given in `names` order) in JOINT_NAMES order. Missing joints are NaN."""
    index = {n: i for i, n in enumerate(names)}
    out = np.full(NUM_JOINTS, np.nan)
    for k, j in enumerate(JOINT_NAMES):
        if j in index:
            out[k] = values[index[j]]
    return out


def step_limit(q_current, q_target, max_step):
    """Move from q_current toward q_target by at most max_step (rad) per joint. max_step None = no limit."""
    if max_step is None:
        return np.asarray(q_target, dtype=np.float64).copy()
    q_current = np.asarray(q_current, dtype=np.float64)
    delta = np.clip(np.asarray(q_target, dtype=np.float64) - q_current, -max_step, max_step)
    return q_current + delta
