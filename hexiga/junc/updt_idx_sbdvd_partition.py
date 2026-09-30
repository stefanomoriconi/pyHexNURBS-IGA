"""
hexiga.junc.updt_idx_sbdvd_partition
====================================

Assign each
direction in ``dirTubes`` to one of two sub-space partitions (halving).

Signature::

    [idx, isValid] = updtIdxSbdvdPartition(dirTubes, QFSfcs, QFSpts)
    % dirTubes: 3xN, QFSfcs: 4x2, QFSpts: 3xn
"""

from __future__ import annotations

import numpy as np

from .is_pt_intersecting_subspace import isPtIntersectingSubSpace

__all__ = ["updtIdxSbdvdPartition"]

def updtIdxSbdvdPartition(dirTubes: np.ndarray, QFSfcs: np.ndarray, QFSpts: np.ndarray):
    """Assign each direction to one of two sub-space partitions (2-way halving).

    Returns ``(idx, isValid)`` where ``idx`` is 1-based (0 = unassigned).
    """
    dirTubes = np.asarray(dirTubes, dtype=float)
    QFSfcs = np.asarray(QFSfcs, dtype=int)
    QFSpts = np.asarray(QFSpts, dtype=float)

    if QFSfcs.shape[1] != 2:
        raise AssertionError("More than 2 possible partitions for halving!")

    n = dirTubes.shape[1]
    idx = np.zeros(n, dtype=int)
    isValid = True

    for jj in range(n):
        A = dirTubes[:, jj]
        for kk in range(QFSfcs.shape[1]):
            Ps = QFSpts[:, QFSfcs[:, kk] - 1]
            if isPtIntersectingSubSpace(A, Ps)[0]:
                idx[jj] = kk + 1
                break

    if np.any(idx == 0):
        if np.all(idx == 0):
            print(" * updtIdxSbdvdPartition: ALL directions have NOT been assigned.")
        else:
            print(" * updtIdxSbdvdPartition: At least ONE direction has not been assigned.")
        isValid = False

    if np.all(idx == 1) or np.all(idx == 2):
        print(" * updtIdxSbdvdPartition: ALL directions have been assigned to the SAME sub-group.")
        isValid = False

    return idx, isValid
