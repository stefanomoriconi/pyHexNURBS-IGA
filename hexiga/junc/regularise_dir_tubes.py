"""
hexiga.junc.regularise_dir_tubes
================================

Spread a set of
tube directions apart using an inverse-distance repulsion, re-normalising each
iteration.

Signature::

    dirTubesReg = regulariseDirTubes(dirTubes)   % 3 x N
"""

from __future__ import annotations

import numpy as np

from .uvect import uvect

__all__ = ["regulariseDirTubes"]

def _pdist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise euclidean distances between rows of ``A`` (n x d) and ``B`` (m x d)."""
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

def regulariseDirTubes(dirTubes: np.ndarray) -> np.ndarray:
    """Spread ``dirTubes`` (3 x N) apart via inverse-distance repulsion."""
    dirTubes = np.asarray(dirTubes, dtype=float)
    n = dirTubes.shape[1]
    dirTubesReg = dirTubes.copy()
    itr = 50

    eye = np.eye(n)
    for _ in range(itr):
        D = 1.0 / _pdist2(dirTubes.T, dirTubes.T)
        D[np.diag_indices(n)] = 1.0
        # MATLAB: dirTubesReg * 0.01 * D  ->  matrix product (3xN) * (NxN)
        dirTubesReg = dirTubesReg - (0.01 * dirTubesReg @ D)
        dirTubesReg = uvect(dirTubesReg)
    return dirTubesReg
