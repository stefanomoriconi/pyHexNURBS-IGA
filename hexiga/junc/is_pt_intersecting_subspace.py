"""
hexiga.junc.is_pt_intersecting_subspace
=======================================

Determine whether a point ``A`` lies strictly inside the subspace defined by a
(planar/non-planar) polygonal surface with ordered vertices ``Ps``.

Signature::

    [inPt, inTris] = isPtIntersectingSubSpace(A, Ps)   % A: 3x1, Ps: 3xp, p>=3
"""

from __future__ import annotations

import numpy as np

from .intersect_line_plane import intersectLinePlane
from .is_pt_in_triangle import isPtInTriangle
from .proj import proj
from .sbdv_poly2_tris import sbdvPoly2Tris

__all__ = ["isPtIntersectingSubSpace"]

def isPtIntersectingSubSpace(A: np.ndarray, Ps: np.ndarray):
    """Determine whether point ``A`` (3 x 1) is strictly inside the subspace of ``Ps`` (3 x p)."""
    A = np.asarray(A, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    assert Ps.shape[1] >= 3, "Polygon not defined! Less than 3 vertices!"

    inTris = np.zeros(Ps.shape[1], dtype=bool)
    O = np.zeros(3)

    Tris = sbdvPoly2Tris(Ps)[0]  # MATLAB: Tris = sbdvPoly2Tris(Ps) takes 1st output

    for pp in range(inTris.size):
        T = Tris[:, :, pp]
        iA = intersectLinePlane(O, A, T[:, 0], T[:, 1], T[:, 2])
        inTris[pp] = bool(isPtInTriangle(iA, T) and (proj(iA, A) > 0))

    inPt = bool(np.any(inTris))
    return inPt, inTris
