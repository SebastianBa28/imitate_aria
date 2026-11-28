import numpy as np

from imitate_common import contract
from imitate_common.arm_kinematics import READY_Q, Panda
from imitate_common.transforms import quat_angle, quat_to_matrix

MODEL = Panda()


def test_flange_matches_the_controllers_end_effector_pose():
    # /paco/joint_states and /right/arm/end_effector_pose read together from the real controller (2026-11-28)
    q = [0.450143647, 0.192280036, -0.372295530, -2.156250956, 0.051147827, 2.389068362, 0.137272904]
    T = MODEL.fk(q)
    assert np.linalg.norm(T[:3, 3] - [0.563367354, 0.047089444, 0.320692514]) < 1e-4
    R = quat_to_matrix([0.998525251, -0.043810280, 0.028999627, 0.013674950])
    assert np.abs(T[:3, :3] - R).max() < 1e-4


def test_chain_joints_are_the_contract_joints():
    assert [j.name for j in MODEL.chain if j.type == "revolute"] == list(contract.ARM_JOINT_NAMES)
    assert MODEL.chain[-1].child == "right_link8"   # the flange: what the controller publishes


def test_jacobian_matches_finite_differences():
    J, _ = MODEL.jacobian(READY_Q)
    eps = 1e-6
    Jn = np.stack([(MODEL.fk(READY_Q + eps * e)[:3, 3] - MODEL.fk(READY_Q - eps * e)[:3, 3]) / (2 * eps)
                   for e in np.eye(7)], axis=1)
    assert np.allclose(J[:3], Jn, atol=1e-6)


def test_ik_reaches_the_mock_start_pose():
    pos, quat = [0.45, 0.0, 0.35], [1.0, 0.0, 0.0, 0.0]   # station.yaml mock.arm_start, gripper down
    q, err = MODEL.ik(pos, quat)
    assert err < 1e-3
    assert np.all(q >= MODEL.lower) and np.all(q <= MODEL.upper)
    T = MODEL.fk(q)
    assert np.linalg.norm(T[:3, 3] - pos) < 1e-3
    assert quat_angle(quat, _matrix_to_quat(T[:3, :3])) < 1e-2


def _matrix_to_quat(R):
    # Shepperd's method, (x, y, z, w)
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q
