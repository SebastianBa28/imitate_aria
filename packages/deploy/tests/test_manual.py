"""Manual control targets (no ROS)."""
import numpy as np
import pytest

from deploy.manual import ManualTargets
from imitate_common import hand

WS_LO, WS_HI = [0.25, -0.4, 0.05], [0.75, 0.4, 0.6]
ARM = (np.array([0.45, 0.0, 0.35]), np.array([1.0, 0.0, 0.0, 0.0]))


def started(arm=ARM):
    m = ManualTargets(WS_LO, WS_HI, max_jog_m=0.05)
    assert m.start(hand.NEUTRAL_RAD.copy(), arm) is None
    return m


def test_targets_start_at_the_measured_state():
    m = started()
    assert np.allclose(m.hand, hand.NEUTRAL_RAD)
    assert np.allclose(m.arm_pos, ARM[0]) and np.allclose(m.arm_quat, ARM[1])


def test_refuses_to_start_outside_the_workspace():
    m = ManualTargets(WS_LO, WS_HI, 0.05)
    why = m.start(hand.NEUTRAL_RAD.copy(), (np.array([0.1, 0.0, 0.35]), ARM[1]))
    assert why and "outside" in why
    assert m.hand is None


def test_hand_targets_are_partial_and_clamped():
    m = started()
    j = hand.JOINT_NAMES.index("index_mcp")
    assert m.apply({"hand": {"index_mcp": 10.0}}) is None
    assert m.hand[j] == pytest.approx(hand.UPPER_RAD[j])
    others = np.delete(np.arange(hand.NUM_JOINTS), j)
    assert np.allclose(m.hand[others], hand.NEUTRAL_RAD[others])


@pytest.mark.parametrize("msg", [{"hand": {"nope": 0.1}}, {"hand": {"wrist": float("nan")}}, {"hand": [1, 2]},
                                 {"arm_jog": [0.1, 0, 0]}, {"arm_jog": [0, 0]}, {"arm_jog": "up"}, [1]])
def test_bad_messages_are_rejected_and_change_nothing(msg):
    m = started()
    before = m.as_dict()
    assert m.apply(msg) is not None
    assert m.as_dict() == before


def test_jog_accumulates_and_stays_in_the_workspace():
    m = started()
    m.apply({"arm_jog": [0.0, 0.0, 0.05]})
    m.apply({"arm_jog": [0.0, 0.0, 0.05]})
    assert m.arm_pos[2] == pytest.approx(0.45)
    for _ in range(10):
        m.apply({"arm_jog": [0.0, 0.0, 0.05]})
    assert m.arm_pos[2] == pytest.approx(0.6)  # workspace top


def test_hold_and_neutral():
    m = started()
    m.apply({"arm_jog": [0.05, 0.0, 0.0], "hand": {"wrist": 0.3}})
    m.apply({"arm_hold": True, "hand_neutral": True}, measured_arm_pos=[0.46, 0.01, 0.35])
    assert np.allclose(m.arm_pos, [0.46, 0.01, 0.35])
    assert np.allclose(m.hand, hand.NEUTRAL_RAD)


def test_arm_commands_rejected_without_arm():
    m = ManualTargets(WS_LO, WS_HI, 0.05)
    assert m.start(hand.NEUTRAL_RAD.copy(), None) is None
    assert m.apply({"arm_jog": [0.01, 0, 0]}) is not None
    assert m.apply({"hand": {"wrist": 0.1}}) is None
