"""
hexiga.junc.get_adj_ivx
=======================

Compute the adjusted
coordinates of ONE interleaving-side (IVX) from two sets of tube directions,
picking the closest pair to a reference direction.

Signature::

    [iAB, Dmin] = getAdjIVX(dTA, dTB, ABref, VX)   % dTA:3xn, dTB:3xm, ABref:3x1, VX:3x1
"""

from __future__ import annotations

import numpy as np

from .get_centroid import getCentroid
from .orthog import orthog
from .proj import proj

__all__ = ["getAdjIVX"]

def _pdist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise euclidean distances between rows of ``A`` (n x d) and ``B`` (m x d)."""
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

def getAdjIVX(dTA: np.ndarray, dTB: np.ndarray, ABref: np.ndarray, VX: np.ndarray):
    """Compute the adjusted IVX (3 x 1) and the minimal pairwise distance.

    ``dTA`` (3 x n) and ``dTB`` (3 x m) are orthogonalised w.r.t. ``VX``; the
    closest pair defines the midpoint, signed wrt ``ABref``.
    """
    dTA = np.asarray(dTA, dtype=float).copy()
    dTB = np.asarray(dTB, dtype=float).copy()
    ABref = np.asarray(ABref, dtype=float)
    VX = np.asarray(VX, dtype=float)

    for jA in range(dTA.shape[1]):
        dTA[:, jA] = orthog(dTA[:, jA], VX)
    for jB in range(dTB.shape[1]):
        dTB[:, jB] = orthog(dTB[:, jB], VX)

    # Robustness (N>3 generalisation): for N>3 the base partition
    # (``updtIdxBasePartition``) can, for some direction sets, leave one of
    # the three base direction groups empty.  The MATLAB original then hits
    # ``min(D(:))`` on an empty distance matrix and errors.  Instead we fall
    # back to the reference direction (orthogonalised to VX) so the base
    # simplex is still well-defined and the downstream splitting recursion
    # can proceed.
    if dTA.shape[1] == 0 or dTB.shape[1] == 0:
        iAB = orthog(np.asarray(ABref, dtype=float).copy(), VX)
        return iAB, float("inf")

    D = _pdist2(dTA.T, dTB.T)
    Dmin = float(np.min(D))
    idxA, idxB = np.unravel_index(np.argmin(D), D.shape)  # find(D==Dmin,1,'first')

    AB = getCentroid(np.column_stack([dTA[:, idxA], dTB[:, idxB]]))
    AB = AB * np.sign(proj(AB, ABref))

    iAB = orthog(AB, VX)
    return iAB, Dmin
