"""
hexiga.junc.proj
================

Scalar projection (dot
product) of vector ``A`` on vector ``B``.

Signature::

    a = proj(A, B)   % G
"""

from __future__ import annotations

import numpy as np

__all__ = ["proj"]

def proj(A: np.ndarray, B: np.ndarray) -> float:
    """Scalar projection of vector ``A`` on vector ``B`` (dot product)."""
    return float(np.dot(np.asarray(A, dtype=float), np.asarray(B, dtype=float)))
