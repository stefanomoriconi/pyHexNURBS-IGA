"""
hexiga.junc.get_base_qfork_simplex
==================================

Determine the
Base Quadrilateral Forking Simplex for N 3-directions (bifurcation, N==3, or
N-junction, N>3).

Signature::

    [QFSfcs, QFSpts, idx, lblCIS] = getBaseQForkSimplex(dirTubes)
    % dirTubes: 3xN, QFSfcs: 4x3, QFSpts: 3x5, idx: 1xN
"""

from __future__ import annotations

import numpy as np

from .sort_dir3_tubes import sortDir3Tubes
from .clstr_dirs import clstrDirs
from .get_concave import getConcaVe
from .orthog import orthog
from .get_ivx import getIVX
from .get_adj_ivx import getAdjIVX
from .updt_idx_base_partition import updtIdxBasePartition

__all__ = ["getBaseQForkSimplex"]

def _computeBaseQForkSimplex(dir3Tubes: np.ndarray):
    """Compute the initial Quad-based Forking Simplex from 3 main directions."""
    A = dir3Tubes[:, 0]
    B = dir3Tubes[:, 1]
    C = dir3Tubes[:, 2]

    flipFLAG = False

    V = getConcaVe(A, B, C)   # concaVe
    X = -V                    # conveX

    oA = orthog(A, V)
    oB = orthog(B, V)
    oC = orthog(C, V)

    iAB, flipFLAG = getIVX(A, B, oC, V, flipFLAG)
    iBC, flipFLAG = getIVX(B, C, oA, V, flipFLAG)
    iCA, _ = getIVX(C, A, oB, V, flipFLAG)

    QFSpts = np.column_stack([V, iAB, iBC, iCA, X])  # Interleaving directions

    # QFSfcs: 1-based face labels (3 x 4)
    QFSfcs = np.array([[1, 4, 5, 2],    # face of A
                       [1, 2, 5, 3],    # face of B
                       [1, 3, 5, 4]]).T  # face of C

    return QFSfcs, QFSpts, flipFLAG

def _adjustBaseIVXs(dirTubes: np.ndarray, idx: np.ndarray, QFSpts: np.ndarray) -> np.ndarray:
    """Adjust the coordinates of the Interleaving-sides (IVXs) points in QFSpts."""
    dTA = dirTubes[:, idx == 1]
    dTB = dirTubes[:, idx == 2]
    dTC = dirTubes[:, idx == 3]

    V = QFSpts[:, 1 - 1]
    ABref = QFSpts[:, 2 - 1]
    BCref = QFSpts[:, 3 - 1]
    CAref = QFSpts[:, 4 - 1]

    adjAB = getAdjIVX(dTA, dTB, ABref, V)[0]
    adjBC = getAdjIVX(dTB, dTC, BCref, V)[0]
    adjCA = getAdjIVX(dTC, dTA, CAref, V)[0]

    QFSpts = QFSpts.copy()
    QFSpts[:, 2 - 1] = adjAB
    QFSpts[:, 3 - 1] = adjBC
    QFSpts[:, 4 - 1] = adjCA
    return QFSpts

def getBaseQForkSimplex(dirTubes: np.ndarray):
    """Determine the Base Quadrilateral Forking Simplex for N 3D-directions.

    Returns ``(QFSfcs, QFSpts, idx, lblCIS)``.
    """
    dirTubes = np.asarray(dirTubes, dtype=float)

    if not (dirTubes.shape[0] == 3 and dirTubes.shape[1] > 2):
        raise AssertionError("getBaseQForkSimplex: dirTubes must be 3xN with N>2.")

    maxNumDirs = 3
    lblCIS = np.ones(dirTubes.shape[1], dtype=bool)

    if dirTubes.shape[1] == maxNumDirs:  # Bifurcation
        dir3Tubes = sortDir3Tubes(dirTubes)
        QFSfcs, QFSpts, flipFLAG = _computeBaseQForkSimplex(dir3Tubes)
        idx, QFSfcs, QFSpts = updtIdxBasePartition(dirTubes, QFSfcs, QFSpts, flipFLAG)

    else:  # N-junction
        dir3Tubes = clstrDirs(dirTubes, maxNumDirs)
        QFSfcs, QFSpts, flipFLAG = _computeBaseQForkSimplex(dir3Tubes)
        idx, QFSfcs, QFSpts = updtIdxBasePartition(dirTubes, QFSfcs, QFSpts, flipFLAG)
        QFSpts = _adjustBaseIVXs(dirTubes, idx, QFSpts)

    if (
        int(np.sum(idx == 0)) > 0
        or int(np.sum(idx == 1)) == 0
        or int(np.sum(idx == 2)) == 0
        or int(np.sum(idx == 3)) == 0
    ):
        # MATLAB: visualiseIndividualQFSs(dir3Tubes, QFSfcs, QFSpts, idx)
        print("BUG here!")

    return QFSfcs, QFSpts, idx, lblCIS
