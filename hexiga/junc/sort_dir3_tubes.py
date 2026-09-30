"""
hexiga.junc.sort_dir3_tubes
===========================

Permute the 3 tube
directions so that the lower-triangle pairwise distances (``d12, d13, d23``)
are in ascending order.

Signature::

    Sdir3Tubes = sortDir3Tubes(dir3Tubes)   % 3 x 3
"""

from __future__ import annotations

import numpy as np

__all__ = ["sortDir3Tubes"]

def _pdist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise euclidean distances between rows of ``A`` (n x d) and ``B`` (m x d)."""
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

def _lowerTriNonzeros(D: np.ndarray) -> np.ndarray:
    """Strict lower-triangle entries in MATLAB (column-major) order.

    MATLAB ``nonzeros(tril(D))`` traverses column by column and drops the
    zero diagonal, yielding ``(D[1,0], D[2,0], D[2,1], ...)`` in 0-based terms.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    out = []
    for j in range(n):
        for i in range(j + 1, n):
            out.append(D[i, j])
    return np.asarray(out, dtype=float)

def sortDir3Tubes(dir3Tubes: np.ndarray) -> np.ndarray:
    """Permute the 3 tubes so the lower-triangle distances are ascending."""
    d3T = np.asarray(dir3Tubes, dtype=float).copy()

    D3lin = _lowerTriNonzeros(_pdist2(d3T.T, d3T.T))
    SD3lin = np.sort(D3lin)

    if not np.array_equal(D3lin, SD3lin):
        sorted_flag = False
        cycle = 0
        # The op sequence (shift, shift, flip) enumerates all 6 column
        # permutations within 6 steps, so this always terminates.
        max_cycles = 12
        while (not sorted_flag) and (cycle < max_cycles):
            cycle = cycle + 1
            if cycle % 3 == 0:
                d3T = np.flip(d3T, axis=1)  # MATLAB flip(d3T, 2)
            else:
                d3T = np.roll(d3T, 1, axis=1)  # MATLAB circshift(d3T, 1, 2)

            D3lin = _lowerTriNonzeros(_pdist2(d3T.T, d3T.T))
            if np.array_equal(D3lin, SD3lin):
                sorted_flag = True

    return d3T
