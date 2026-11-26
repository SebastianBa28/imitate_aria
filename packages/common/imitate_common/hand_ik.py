"""Fingertip IK for the ORCA hand (numpy only).

Each finger is solved on its own (a fingertip depends only on that finger's joints) by
Levenberg-Marquardt on |tip(q) - target|^2, clamped to the hand.py ROMs, starting from q_start (normally
last tick's solution). Small damped steps from the warm start pick one answer for the redundant thumb and
keep motion smooth. Clamping can trap a cold start against a limit (e.g. knuckle hyperextended, tip
joint fully curled), so a finger left more than `restart_tol` away is re-solved from a few fixed seeds
and the best result kept. Unreachable targets give the closest reachable pose; the leftover error per
finger is returned so it can be logged.

Targets are in the palm frame (see orca_kinematics). The ORCA `wrist` joint is not solved: it is set to
`wrist_q` (default 0) so the arm alone takes the wrist pose.
"""
from dataclasses import dataclass

import numpy as np

from . import hand
from .orca_kinematics import FINGERS, OrcaHand


@dataclass
class IKResult:
    q: np.ndarray        # (17,) rad, hand.JOINT_NAMES order
    error: np.ndarray    # (5,) m, distance from each fingertip to its target (NaN where the target was NaN)
    iters: int           # most iterations used by any finger, restarts included


class HandIK:
    def __init__(self, model=None, damping=1e-6, max_iters=30, tol=1e-4, restart_tol=2e-3, max_step=0.3,
                 wrist_q=0.0):
        """damping: Levenberg-Marquardt term (m^2 per rad^2) that keeps steps stable near singular poses.
        tol: stop a finger once its tip is closer than this (m) or its step is below 1e-6 rad.
        restart_tol: re-solve a finger from the fixed seeds if it ends further than this (m) from its target.
        max_step: largest joint change per iteration (rad); near a straight (singular) finger an uncapped step
        is huge and throws the joints into a limit corner they cannot leave."""
        self.model = model or OrcaHand()
        self.damping, self.max_iters, self.tol, self.wrist_q = damping, max_iters, tol, wrist_q
        self.restart_tol, self.max_step = restart_tol, max_step
        self.lower = hand.LOWER_RAD.copy()
        self.upper = hand.UPPER_RAD.copy()
        mid = (self.lower + self.upper) / 2
        self.seeds = [np.clip(np.zeros(hand.NUM_JOINTS), self.lower, self.upper), hand.NEUTRAL_RAD.copy(), mid]

    def solve(self, targets, q_start=None):
        """targets: (5, 3) fingertip positions in the palm frame, FINGERS order; a NaN row keeps that finger at
        q_start. q_start: (17,) warm start, default hand.NEUTRAL_RAD."""
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (len(FINGERS), 3):
            raise ValueError(f"targets must be (5, 3), got {targets.shape}")
        q_start = hand.NEUTRAL_RAD if q_start is None else np.asarray(q_start, dtype=np.float64)
        q = np.clip(np.where(np.isfinite(q_start), q_start, hand.NEUTRAL_RAD), self.lower, self.upper)
        error = np.full(len(FINGERS), np.nan)
        most = 0
        for i, f in enumerate(FINGERS):
            if not np.all(np.isfinite(targets[i])):
                continue
            idx = self.model.finger_idx[f]
            lo, hi = self.lower[idx], self.upper[idx]
            qf, err, n = self._solve_finger(f, targets[i], q[idx], lo, hi)
            for seed in self.seeds if err > self.restart_tol else ():
                qs, es, ns = self._solve_finger(f, targets[i], seed[idx], lo, hi)
                n += ns
                if es < err:
                    qf, err = qs, es
                if err <= self.restart_tol:
                    break
            q[idx] = qf
            error[i] = err
            most = max(most, n)
        q[self.model.joint_index["wrist"]] = self.wrist_q
        return IKResult(q=q, error=error, iters=most)

    def _step(self, J, e, q, lo, hi):
        """LM step that keeps joints sitting at a limit fixed when the step would push them past it,
        and re-solves the remaining joints (plain clipping would stall there)."""
        free = np.ones(len(q), dtype=bool)
        for _ in range(len(q)):
            dq = np.zeros(len(q))
            Jf = J[:, free]
            dq[free] = np.linalg.solve(Jf.T @ Jf + self.damping * np.eye(free.sum()), Jf.T @ e)
            big = np.max(np.abs(dq))
            if big > self.max_step:
                dq *= self.max_step / big
            pinned = free & (((q <= lo) & (dq < 0)) | ((q >= hi) & (dq > 0)))
            if not pinned.any():
                return dq
            free &= ~pinned
            if not free.any():
                return np.zeros(len(q))
        return dq

    def _solve_finger(self, finger, target, q0, lo, hi):
        q = q0.copy()
        for it in range(1, self.max_iters + 1):
            p, J = self.model.finger_fk(finger, q, jacobian=True)
            e = target - p
            if np.linalg.norm(e) < self.tol:
                return q, float(np.linalg.norm(e)), it - 1
            q_new = np.clip(q + self._step(J, e, q, lo, hi), lo, hi)
            step = np.max(np.abs(q_new - q))
            q = q_new
            if step < 1e-6:
                break
        p = self.model.finger_fk(finger, q)
        return q, float(np.linalg.norm(target - p)), it
