"""
hexiga.junc.get_concave
=======================

Retrieve the concave
direction ``V`` as a unit-vector from three direction vectors.

Signature::

    V = getConcaVe(A, B, C)   % each 3 x 1, V: 3 x 1
"""

from __future__ import annotations

import numpy as np

from .get_centroid import getCentroid

__all__ = ["getConcaVe"]

def getConcaVe(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Retrieve the concave direction ``V`` (3 x 1) from directions A, B, C (each 3 x 1)."""
    A = np.asarray(A, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)
    C = np.asarray(C, dtype=float).reshape(3)
    # [cross(A,B), cross(B,C), cross(C,A)] -> 3 x 3 column layout
    V = getCentroid(np.column_stack([np.cross(A, B), np.cross(B, C), np.cross(C, A)]))
    return V
