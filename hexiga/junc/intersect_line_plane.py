"""
hexiga.junc.intersect_line_plane
================================

Compute the
intersection point between a line and a plane in 3D. The line is defined by
its start point ``la`` and end point ``lb``; the plane by 3 points
``p1, p2, p3``. It is assumed the intersection exists and is real.

Signature::

    ip = intersectLinePlane(la, lb, p1, p2, p3)   % all 3x1
"""

from __future__ import annotations

import numpy as np

__all__ = ["intersectLinePlane"]

def intersectLinePlane(
    la: np.ndarray, lb: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    """Intersection point between line ``la``-``lb`` and plane ``p1,p2,p3`` (each 3 x 1)."""
    la = np.asarray(la, dtype=float)
    lb = np.asarray(lb, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    lab = lb - la
    normal = np.cross(p2 - p1, p3 - p1)

    t_num = float(np.dot(normal, la - p1))
    t_den = float(np.dot(-lab, normal))

    # MATLAB `t_num / t_den` yields Inf/NaN (not an error) on a degenerate
    # (parallel/co-planar) configuration; match that with numpy scalar division
    # (suppressing the warning) so downstream `isPtInTriangle` simply rejects it.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.float64(t_num) / np.float64(t_den)

    ip = la + lab * t
    return ip
