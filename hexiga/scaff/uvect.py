"""
hexiga.scaff.uvect
==================

Normalise each column of a
matrix ``A`` to unit length.

Signature: ``uA = uvect(A)``.

Notes
-----

* MATLAB: ``uA = A ./ sqrt(sum(A.^2, 1));`` -- per-column (axis 1)
  Euclidean norm.
* For a single 3-vector ``A`` of shape ``(3,1)`` the result is a
  ``3``-vector of shape ``(3,1)``.
* Zero-length columns divide by zero and yield ``NaN`` -- the same as in
  MATLAB (no guard is added, preserving the ground-truth behaviour).
"""

from __future__ import annotations

import numpy as np

__all__ = ["uvect"]

def uvect(A: np.ndarray) -> np.ndarray:
    """Normalise each column of ``A`` to unit length.

    Parameters
    ----------
    A
        A 2-D array of shape ``(n, m)`` (each column a vector to be
        normalised).

    Returns
    -------
    np.ndarray
        Array of the same shape as ``A`` with each column scaled to unit
        length.
    """
    A = np.asarray(A, dtype=float)
    denom = np.sqrt(np.sum(A ** 2, axis=0))
    return A / denom
