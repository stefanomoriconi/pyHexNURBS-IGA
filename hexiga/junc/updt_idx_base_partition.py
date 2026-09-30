"""
hexiga.junc.updt_idx_base_partition
===================================

Update the
cluster partition (``idx``) based on the Sub-Space geometry, optionally
correcting the QFS points (flipping opposite points / swapping quad faces)
up to 3 times when the assignment is invalid and ``flipFLAG`` is set.

Signature::

    [idx, QFSfcs, QFSpts] = updtIdxBasePartition(dirTubes, QFSfcs, QFSpts, flipFLAG)
    % dirTubes: 3xN, QFSfcs: 4xf, QFSpts: 3xn
"""

from __future__ import annotations

import numpy as np

from .is_pt_intersecting_subspace import isPtIntersectingSubSpace

__all__ = ["updtIdxBasePartition"]

def _assignIdxBasePartition(dirTubes: np.ndarray, QFSfcs: np.ndarray, QFSpts: np.ndarray):
    """Assign each direction to the first sub-space it intersects.

    Returns ``(idx, isValid)`` where ``idx`` is 1-based (0 = unassigned).
    """
    n = dirTubes.shape[1]
    idx = np.zeros(n, dtype=int)

    for jj in range(n):
        A = dirTubes[:, jj]
        for kk in range(QFSfcs.shape[1]):
            Ps = QFSpts[:, QFSfcs[:, kk] - 1]
            if isPtIntersectingSubSpace(A, Ps)[0]:
                idx[jj] = kk + 1
                break

    isValid = (
        int(np.sum(idx == 0)) == 0
        and int(np.sum(idx == 1)) > 0
        and int(np.sum(idx == 2)) > 0
        and int(np.sum(idx == 3)) > 0
    )
    return idx, isValid

def _correctQFSpts(QFSpts_IN: np.ndarray, QFSfcs_IN: np.ndarray, cycle: int):
    """Flip opposite points and swap quad-face indices for the given cycle."""
    QFSfcs_OUT = QFSfcs_IN.copy()
    QFSpts_OUT = QFSpts_IN.copy()

    if cycle == 1:
        QFSpts_OUT[:, 1] = -QFSpts_OUT[:, 1]
        QFSpts_OUT[:, 2] = -QFSpts_OUT[:, 2]
        QFSfcs_OUT[:, 0] = QFSfcs_IN[:, 2]
        QFSfcs_OUT[:, 2] = QFSfcs_IN[:, 0]
    elif cycle == 2:
        QFSpts_OUT[:, 2] = -QFSpts_OUT[:, 2]
        QFSpts_OUT[:, 3] = -QFSpts_OUT[:, 3]
        QFSfcs_OUT[:, 0] = QFSfcs_IN[:, 1]
        QFSfcs_OUT[:, 1] = QFSfcs_IN[:, 0]
    elif cycle == 3:
        QFSpts_OUT[:, 1] = -QFSpts_OUT[:, 1]
        QFSpts_OUT[:, 3] = -QFSpts_OUT[:, 3]
        QFSfcs_OUT[:, 1] = QFSfcs_IN[:, 2]
        QFSfcs_OUT[:, 2] = QFSfcs_IN[:, 1]
    else:
        raise ValueError(" * correctQFSpts: Unrecognised Case!")

    return QFSfcs_OUT, QFSpts_OUT

def updtIdxBasePartition(dirTubes: np.ndarray, QFSfcs: np.ndarray, QFSpts: np.ndarray, flipFLAG: bool):
    """Update the cluster partition based on the sub-space geometry."""
    dirTubes = np.asarray(dirTubes, dtype=float)
    QFSfcs = np.asarray(QFSfcs, dtype=int)
    QFSpts = np.asarray(QFSpts, dtype=float)

    isValid = False
    QFSpts_sel = QFSpts
    QFSfcs_sel = QFSfcs
    cycle = 0

    max_iters = 3  # safety: the 3-case correction cycles, so >3 is redundant
    guard = 0
    while not isValid and guard < max_iters:
        guard += 1
        cycle = cycle % 3 + 1

        idx, isValid = _assignIdxBasePartition(dirTubes, QFSfcs_sel, QFSpts_sel)

        if not isValid and flipFLAG:
            QFSfcs_sel, QFSpts_sel = _correctQFSpts(QFSpts, QFSfcs, cycle)

    # Robustness (N>3 generalisation): if, after the 3 correction cycles, some
    # directions were never assigned to any base face (``idx == 0``), assign
    # each straggler to the *nearest* base face (by minimum distance to that
    # face's point set).  The MATLAB original leaves such directions
    # unassigned here and later errors in ``getAdjIVX`` (min over an empty
    # distance matrix); this fallback keeps the base simplex well-defined so
    # the downstream splitting recursion can proceed.  It only fires for the
    # degenerate direction sets and is a no-op whenever ``idx`` is already
    # fully assigned (e.g. the clean 3-tube case).
    unassigned = np.where(idx == 0)[0]
    if unassigned.size:
        face_pts = [QFSpts_sel[:, QFSfcs_sel[:, kk] - 1] for kk in range(QFSfcs_sel.shape[1])]
        for jj in unassigned:
            A = dirTubes[:, jj]
            dists = [float(np.min(np.linalg.norm(A[None, :] - fp.T, axis=1))) for fp in face_pts]
            idx[jj] = int(np.argmin(dists)) + 1

    QFSfcs = QFSfcs_sel
    QFSpts = QFSpts_sel

    return idx, QFSfcs, QFSpts
