"""
hexiga.junc.get_intersecting_point_for2_segments
================================================

Compute the intersection point of two (co-planar) segments ``[p1,p2]`` and
``[p3,p4]`` via the standard cross-ratio (``dmnop``) formulation.

Signature::

    ip = getIntersectingPointFor2Segments(p1, p2, p3, p4)

Notes
-----

* ``getDmnop(m,n,o,p)`` = dot product of ``(m-n)`` and ``(o-p)`` (per
  coordinate sum over x,y,z).
* The returned ``ip`` is a ``(3,)`` point.
"""

from __future__ import annotations

import numpy as np

__all__ = ["getIntersectingPointFor2Segments"]

def _getDmnop(m, n, o, p) -> float:
    m = np.asarray(m, dtype=float).ravel()
    n = np.asarray(n, dtype=float).ravel()
    o = np.asarray(o, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    return float(np.dot(m - n, o - p))

def getIntersectingPointFor2Segments(p1, p2, p3, p4) -> np.ndarray:
    """Intersection of segment ``[p1,p2]`` with segment ``[p3,p4]``."""
    p1 = np.asarray(p1, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()
    p3 = np.asarray(p3, dtype=float).ravel()
    p4 = np.asarray(p4, dtype=float).ravel()

    D1343 = _getDmnop(p1, p3, p4, p3)
    D4321 = _getDmnop(p4, p3, p2, p1)
    D1321 = _getDmnop(p1, p3, p2, p1)
    D4343 = _getDmnop(p4, p3, p4, p3)
    D2121 = _getDmnop(p2, p1, p2, p1)

    # MATLAB-faithful division: for a degenerate (parallel-diagonal) quad the
    # denominator is zero and MATLAB yields NaN/Inf (NOT an exception).  Let
    # NumPy emit IEEE Inf/NaN exactly like MATLAB instead of raising.
    with np.errstate(divide="ignore", invalid="ignore"):
        mua = (np.float64(D1343 * D4321) - np.float64(D1321 * D4343)) / (
            np.float64(D2121 * D4343) - np.float64(D4321 * D4321)
        )

    return p1 + mua * (p2 - p1)
