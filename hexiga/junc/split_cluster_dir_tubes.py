"""
hexiga.junc.split_cluster_dir_tubes
===================================

Split the
cluster ``dTs`` (tubular directions) into two parts and return the associate
Cis or Trans configuration of the QFS in the given sub-space.

Signature::

    [M, FlagCIS, idx, fatalErr] = splitClusterDirTubes(dTs, QFSpt)
    % dTs: 3xN, QFSpt: 3x4, idx: 1xN, FlagCIS: logical
"""

from __future__ import annotations

import numpy as np

from .get_centroid import getCentroid
from .get_cis_trans_qfs import getCisTransQFSs
from .get_cis_trans_diagonals import getCisTransDiagonals
from .updt_idx_sbdvd_partition import updtIdxSbdvdPartition
from .get_adj_ivx import getAdjIVX

__all__ = ["splitClusterDirTubes"]

def _adjCisTransM(dTs: np.ndarray, idxCisTrans: np.ndarray, iniM: np.ndarray, VX: np.ndarray):
    """Adjust the centroid ``M`` between the two sub-groups."""
    dTAcis = dTs[:, idxCisTrans == 1]
    dTBcis = dTs[:, idxCisTrans == 2]
    return getAdjIVX(dTAcis, dTBcis, iniM, VX)

def _fallbackSplit(dTs: np.ndarray, QFSpt: np.ndarray, iniM: np.ndarray):
    """Degenerate-case recovery for ``splitClusterDirTubes``.

    When NEITHER the Cis nor the Trans partition captures every direction
    (the MATLAB gate ``"Both Cis and Trans partitions are NOT Valid! [BUG]!"``),
    fall back to a deterministic 2-way split of the directions so the recursion
    can still terminate with one direction per face (a consistent quadrilateral
    interface per branch).

    The split places each direction on the sub-face whose original quad corner
    (``A = QFSpt[:,1]`` or ``B = QFSpt[:,3]``) is nearest -- matching the Cis
    face layout produced by ``mapCisTransQFSs`` -- and guarantees both
    sub-groups are non-empty (``half = max(1, n//2)``) so ``sdvBaseQForkSimplex``
    makes strict progress and cannot loop.

    This is the one sanctioned divergence from MATLAB: rather than
    ``fatalErr``-skip (which leaves directions unassigned and breaks
    ``makeQFSs2nrbHexa``), we always return a valid split.
    """
    n = dTs.shape[1]
    a = QFSpt[:, 1]
    b = QFSpt[:, 3]
    # Higher score => nearer corner A (face 1 in the Cis layout).
    score = (np.linalg.norm(dTs - b[:, None], axis=0)
             - np.linalg.norm(dTs - a[:, None], axis=0))
    order = np.argsort(score)
    half = max(1, n // 2)
    idx = np.zeros(n, dtype=int)
    idx[order[half:]] = 1   # larger half -> face 1 (A-side)
    idx[order[:half]] = 2   # smaller half -> face 2 (B-side)
    return iniM, True, idx

def splitClusterDirTubes(dTs: np.ndarray, QFSpt: np.ndarray):
    """Split a cluster of directions into two parts (Cis/Trans).

    Returns ``(M, FlagCIS, idx, fatalErr)``.
    """
    dTs = np.asarray(dTs, dtype=float)
    QFSpt = np.asarray(QFSpt, dtype=float)

    if dTs.shape[0] != 3:
        raise AssertionError("Wrong Coordinate size for input dTs!")
    if QFSpt.shape != (3, 4):
        raise AssertionError("Wrong size for input QFSpt!")

    fatalErr = False

    if dTs.shape[1] > 1:  # Splitting in 2 sub-groups
        iniM = getCentroid(dTs)
        cQFSfcs, tQFSfcs, QFSpts = getCisTransQFSs(QFSpt, iniM)

        VXcis = getCisTransDiagonals(QFSpt)[0]
        VXtrans = getCisTransDiagonals(QFSpt)[1]

        idxCis, isCisValid = updtIdxSbdvdPartition(dTs, cQFSfcs, QFSpts)
        idxTrans, isTransValid = updtIdxSbdvdPartition(dTs, tQFSfcs, QFSpts)

        if isCisValid and isTransValid:
            Mcis, cisDmin = _adjCisTransM(dTs, idxCis, iniM, VXcis)
            Mtrans, transDmin = _adjCisTransM(dTs, idxTrans, iniM, VXtrans)

            if cisDmin >= transDmin:  # Return Cis-QSF
                adjQFSpts = QFSpts.copy()
                adjQFSpts[:, 3] = Mcis
                idxCisAdj, isValidAdj = updtIdxSbdvdPartition(dTs, cQFSfcs, adjQFSpts)
                if isValidAdj:
                    M, idx, FlagCIS = Mcis, idxCisAdj, True
                else:
                    M, idx, FlagCIS = iniM, idxCis, True
            else:  # Return Trans-QSF
                adjQFSpts = QFSpts.copy()
                adjQFSpts[:, 3] = Mtrans
                idxTransAdj, isValidAdj = updtIdxSbdvdPartition(dTs, tQFSfcs, adjQFSpts)
                if isValidAdj:
                    M, idx, FlagCIS = Mtrans, idxTransAdj, False
                else:
                    M, idx, FlagCIS = iniM, idxTrans, False

        elif isCisValid and not isTransValid:
            Mcis, _ = _adjCisTransM(dTs, idxCis, iniM, VXcis)
            adjQFSpts = QFSpts.copy()
            adjQFSpts[:, 3] = Mcis
            idxCisAdj, isValidAdj = updtIdxSbdvdPartition(dTs, cQFSfcs, adjQFSpts)
            if isValidAdj:
                M, idx, FlagCIS = Mcis, idxCisAdj, True
            else:
                M, idx, FlagCIS = iniM, idxCis, True

        elif not isCisValid and isTransValid:
            Mtrans, _ = _adjCisTransM(dTs, idxTrans, iniM, VXtrans)
            adjQFSpts = QFSpts.copy()
            adjQFSpts[:, 3] = Mtrans
            idxTransAdj, isValidAdj = updtIdxSbdvdPartition(dTs, tQFSfcs, adjQFSpts)
            if isValidAdj:
                M, idx, FlagCIS = Mtrans, idxTransAdj, False
            else:
                M, idx, FlagCIS = iniM, idxTrans, False

        else:
            # Both Cis and Trans partitions are NOT valid.
            # MATLAB raises this gate and skips (leaving directions unassigned).
            # Sanctioned divergence: recover with a deterministic valid split so
            # the recursion still terminates with one direction per face.
            print("splitClusterDirTubes: Both Cis and Trans partitions are NOT Valid! "
                  "-> recovering with a deterministic split.")
            M, FlagCIS, idx = _fallbackSplit(dTs, QFSpt, iniM)

    else:  # No splitting
        print("Individual direction dTs. No further splitting!")
        M = np.empty((3, 0))
        idx = np.array([1], dtype=int)
        FlagCIS = True

    return M, FlagCIS, idx, fatalErr
