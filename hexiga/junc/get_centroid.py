"""
hexiga.junc.get_centroid
========================

Average a set of
unit-vectors and return the mean unit-vector.

Signature::

    M = getCentroid(uVects)   % uVects: 3 x N, M: 3 x 1
"""

from __future__ import annotations

import numpy as np

from .uvect import uvect

__all__ = ["getCentroid"]

def getCentroid(uVects: np.ndarray) -> np.ndarray:
    """Average unit-vectors ``uVects`` (3 x N) and return the mean unit-vector (3 x 1)."""
    uVects = np.asarray(uVects, dtype=float)
    M = np.mean(uVects, axis=1)
    M = uvect(M)
    return M
