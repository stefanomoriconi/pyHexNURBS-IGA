"""
hexiga.junc.sbdv_poly2_tris
===========================

Triangulate a (quad)
polygonal surface defined by ordered vertices ``Ps`` about its diagonal
midpoint ``C``, returning the triangle set, face normals, and convexity flag.

Signature::

    [Ts, nFs, isPolyConvex] = sbdvPoly2Tris(Ps)   % Ps: 3xp, p>2
"""

from __future__ import annotations

import numpy as np

from .proj import proj
from .uvect import uvect

__all__ = ["sbdvPoly2Tris"]

def _getNewQuadDiagMidPoints(Ps: np.ndarray) -> np.ndarray:
    """Port of the MATLAB ``getNewQuadDiagMidPoints`` local function.

    NB: the diagonal between point 2 and point 4 MUST always be the one
    connecting the new partition point ``M``.
    """
    # MATLAB: mean([Ps(:,4), Ps(:,2)], 2) -> 3 x 2 -> mean over columns -> (3,)
    C = np.mean(np.column_stack([Ps[:, 3], Ps[:, 1]]), axis=1)
    return C

def sbdvPoly2Tris(Ps: np.ndarray):
    """Triangulate a (quad) polygon ``Ps`` (3 x p, p>2) about its diagonal midpoint.

    Returns ``(Ts, nFs, isPolyConvex)`` where ``Ts`` is (3, 3, p) of triangle
    vertex blocks, ``nFs`` (3, p) face normals, and ``isPolyConvex`` a bool.
    """
    Ps = np.asarray(Ps, dtype=float)
    assert Ps.shape[1] > 2, "Input Ps is not a Polygon!"

    p = Ps.shape[1]
    Ts = np.zeros((3, 3, p))
    nFs = np.zeros((3, p))
    isPolyConvex = False

    C = _getNewQuadDiagMidPoints(Ps)

    padPs = np.hstack([Ps, Ps[:, 0:1]])

    for jj in range(p):
        lA = padPs[:, jj] - C
        lB = padPs[:, jj + 1] - C
        nFs[:, jj] = uvect(np.cross(lA, lB))
        Ts[:, :, jj] = np.column_stack([C, padPs[:, jj], padPs[:, jj + 1]])

    for jj in range(p):
        for kk in range(p):
            if jj != kk:
                CHKsign = proj(nFs[:, jj], nFs[:, kk]) < 0
                isPolyConvex = isPolyConvex or CHKsign

    return Ts, nFs, isPolyConvex
