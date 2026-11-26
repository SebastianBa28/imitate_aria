import time

import numpy as np

from imitate_common import hand, transforms
from imitate_common.hand_ik import HandIK
from imitate_common.orca_kinematics import FINGERS, OrcaHand

MODEL = OrcaHand()


def test_urdf_joints_match_the_hand_table():
    used = sorted(int(i) for f in FINGERS for i in MODEL.finger_idx[f])
    assert used == list(range(1, hand.NUM_JOINTS))  # every joint but the wrist drives one fingertip


def test_palm_frame_convention():
    tips = MODEL.fingertips(np.zeros(hand.NUM_JOINTS))
    assert np.all(tips[1:, 2] > 0.12)                     # fingers point along +z
    assert np.all(np.diff(tips[1:, 0]) > 0)               # index -> pinky along +x
    q = np.zeros(hand.NUM_JOINTS)
    q[[6, 7]] = 1.2                                        # curl the index finger
    assert MODEL.fingertips(q)[1, 1] > 0.04                # it curls towards +y (palm side)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    for f in FINGERS:
        idx = MODEL.finger_idx[f]
        qf = rng.uniform(hand.LOWER_RAD[idx], hand.UPPER_RAD[idx])
        _, J = MODEL.finger_fk(f, qf, jacobian=True)
        eps = 1e-6
        Jn = np.stack([(MODEL.finger_fk(f, qf + eps * e) - MODEL.finger_fk(f, qf - eps * e)) / (2 * eps)
                       for e in np.eye(len(qf))], axis=1)
        assert np.allclose(J, Jn, atol=1e-6)


def test_round_trip_from_a_cold_start():
    rng = np.random.default_rng(0)
    ik = HandIK()
    errors = []
    for q in rng.uniform(hand.LOWER_RAD, hand.UPPER_RAD, size=(100, hand.NUM_JOINTS)):
        res = ik.solve(MODEL.fingertips(q))
        assert np.all((res.q >= hand.LOWER_RAD) & (res.q <= hand.UPPER_RAD))
        assert np.allclose(MODEL.fingertips(res.q), MODEL.fingertips(q), atol=res.error.max() + 1e-9)
        errors.append(res.error)
    errors = np.array(errors)
    assert np.median(errors) < 1e-4 and np.mean(errors < 1e-3) > 0.99


def test_warm_start_tracks_smooth_motion_quickly():
    rng = np.random.default_rng(2)
    ik = HandIK()
    q = q_start = hand.NEUTRAL_RAD.copy()
    times = []
    for _ in range(200):
        q = np.clip(q + rng.normal(0, 0.05, hand.NUM_JOINTS), hand.LOWER_RAD, hand.UPPER_RAD)
        t = time.perf_counter()
        res = ik.solve(MODEL.fingertips(q), q_start)
        times.append(time.perf_counter() - t)
        assert res.error.max() < 1e-3
        q_start = res.q
    assert np.median(times) < 5e-3  # well inside a 100 ms policy tick


def test_unreachable_targets_give_the_closest_pose_and_report_the_error():
    tips = MODEL.fingertips(hand.NEUTRAL_RAD) * 2.0  # a hand twice the size
    res = HandIK().solve(tips)
    assert np.all((res.q >= hand.LOWER_RAD) & (res.q <= hand.UPPER_RAD))
    assert np.all(res.error > 0.05)
    assert np.allclose(np.linalg.norm(MODEL.fingertips(res.q) - tips, axis=1), res.error)


def test_nan_target_keeps_that_finger_and_wrist_is_fixed():
    q_start = hand.NEUTRAL_RAD.copy()
    q_start[0] = 0.3
    tips = MODEL.fingertips(np.zeros(hand.NUM_JOINTS))
    tips[2] = np.nan
    res = HandIK().solve(tips, q_start)
    mid = MODEL.finger_idx["middle"]
    assert np.allclose(res.q[mid], q_start[mid]) and np.isnan(res.error[2])
    assert res.q[0] == 0.0 and np.all(res.error[[0, 1, 3, 4]] < 1e-3)


def test_wrist_frame_axes():
    wrist, index_mcp, pinky_mcp = np.array([1.0, 2, 3]), np.array([0.98, 2, 3.09]), np.array([1.05, 2.01, 3.08])
    T = transforms.wrist_frame(wrist, index_mcp, pinky_mcp)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    local = np.linalg.inv(T) @ np.array([*pinky_mcp, 1.0])
    assert np.allclose(np.linalg.inv(T) @ np.array([*wrist, 1.0]), [0, 0, 0, 1])
    assert local[0] > 0 and local[2] > 0
    try:
        transforms.wrist_frame(wrist, wrist, wrist)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
