"""
hexiga.junc.uvect
=================

Normalise a vector (or each
column of a matrix) to unit length.

Signature::

    uA = uvect(A)   % G
"""

from __future__ import annotations

import numpy as np

__all__ = ["uvect"]

def uvect(A: np.ndarray) -> np.ndarray:
    """Normalise each column of ``A`` to unit length."""
    A = np.asarray(A, dtype=float)
    denom = np.sqrt(np.sum(A ** 2, axis=0))
    # A zero column yields NaN in MATLAB too; match it while suppressing the warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return A / denom
