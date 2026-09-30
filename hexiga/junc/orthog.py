"""
hexiga.junc.orthog
==================

Orthogonalise vector ``A``
with respect to vector ``B``, then normalise.

Signature::

    oA = orthog(A, B)   % G
"""

from __future__ import annotations

import numpy as np

from .proj import proj
from .uvect import uvect

__all__ = ["orthog"]

def orthog(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonalise vector ``A`` w.r.t. vector ``B``, then normalise."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    oA = A - proj(A, B) * B
    return uvect(oA)
