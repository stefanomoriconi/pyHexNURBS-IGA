"""
hexiga.junc.is_pt_in_triangle
=============================

Test whether point
``P`` lies strictly inside triangle ``T`` (barycentric coordinates).

Signature::

    in = isPtInTriangle(P, T)   % P: 3x1, T: 3x3
"""

from __future__ import annotations

import numpy as np

__all__ = ["isPtInTriangle"]

def isPtInTriangle(P: np.ndarray, T: np.ndarray) -> bool:
    """Test whether point ``P`` (3 x 1) is strictly inside triangle ``T`` (3 x 3)."""
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)

    # Compute vectors
    v0 = T[:, 2] - T[:, 0]
    v1 = T[:, 1] - T[:, 0]
    v2 = P - T[:, 0]

    # Compute dot products
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))

    # Compute barycentric coordinates. MATLAB `1/(denom)` yields Inf/NaN (not an
    # error) for a degenerate triangle; match that so the comparison simply fails.
    with np.errstate(divide="ignore", invalid="ignore"):
        invDenom = np.float64(1.0) / np.float64(dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is in triangle (strict interior); NaN/Inf comparisons are False
    return bool((u > 0) and (v > 0) and (u + v < 1))
