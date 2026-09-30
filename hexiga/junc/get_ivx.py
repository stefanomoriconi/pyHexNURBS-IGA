"""
hexiga.junc.get_ivx
===================

Retrieve the
interleaving-side point (IVX) for vectors A and B, wrt V.

Signature::

    [iAB, flip] = getIVX(A, B, oC, V, flip)   % A,B,oC,V: 3x1, flip: 1x1 logical
"""

from __future__ import annotations

import numpy as np

from .det_sign import detSign
from .get_centroid import getCentroid
from .orthog import orthog

__all__ = ["getIVX"]

def getIVX(A: np.ndarray, B: np.ndarray, oC: np.ndarray, V: np.ndarray, flip: bool):
    """Retrieve the interleaving-side point (IVX) for vectors A and B, wrt V."""
    A = np.asarray(A, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)
    oC = np.asarray(oC, dtype=float).reshape(3)
    V = np.asarray(V, dtype=float).reshape(3)

    # average
    AB = getCentroid(np.column_stack([A, B]))
    # get the orthogonal projection of AB w.r.t. V
    iAB = orthog(AB, V)
    # check sign
    m, flip = detSign(iAB, oC, flip)
    iAB = m * iAB
    return iAB, flip
