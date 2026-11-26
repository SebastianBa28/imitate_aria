import numpy as np
import pytest

from imitate_common import hand, transforms


def test_step_limit_none_means_no_limit():
    cur, tgt = np.zeros(17), np.full(17, 1.5)
    assert np.allclose(hand.step_limit(cur, tgt, None), tgt)
    assert np.allclose(hand.step_limit(cur, tgt, 0.1), 0.1)


def test_hand_tables_cover_every_joint():
    assert hand.NUM_JOINTS == 17
    assert set(hand.ROM_DEG) == set(hand.JOINT_NAMES) == set(hand.NEUTRAL_DEG)
    assert np.all(hand.LOWER_RAD < hand.UPPER_RAD)
    assert np.all((hand.NEUTRAL_RAD >= hand.LOWER_RAD) & (hand.NEUTRAL_RAD <= hand.UPPER_RAD))


def test_clamp_keeps_nan_and_rejects_wrong_length():
    q = np.full(17, 10.0)
    q[3] = np.nan
    out = hand.clamp_rad(q)
    assert np.isnan(out[3])
    assert np.allclose(np.delete(out, 3), np.delete(hand.UPPER_RAD, 3))
    with pytest.raises(ValueError):
        hand.clamp_rad(np.zeros(16))


def test_reorder_fills_missing_with_nan():
    out = hand.reorder(["index_mcp", "wrist"], [1.0, 2.0])
    assert out[0] == 2.0 and out[hand.JOINT_NAMES.index("index_mcp")] == 1.0
    assert np.isnan(out).sum() == 15


def test_step_limit():
    assert np.allclose(hand.step_limit(np.zeros(2), [1.0, -0.01], 0.1), [0.1, -0.01])


def test_step_pose_limits_distance_and_angle():
    q_id = [0, 0, 0, 1]
    q_z90 = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    pos, quat = transforms.step_pose([0, 0, 0], q_id, [1, 0, 0], q_z90, 0.1, np.deg2rad(9))
    assert np.allclose(pos, [0.1, 0, 0])
    assert transforms.quat_angle(q_id, quat) == pytest.approx(np.deg2rad(9), abs=1e-6)
    # within limits: reaches the target exactly
    pos, quat = transforms.step_pose([0, 0, 0], q_id, [0.05, 0, 0], q_id, 0.1, 0.1)
    assert np.allclose(pos, [0.05, 0, 0]) and np.allclose(quat, q_id)


def test_quat_to_matrix_rotates_x_to_y():
    R = transforms.quat_to_matrix([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
