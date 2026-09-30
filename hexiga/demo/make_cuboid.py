"""
hexiga.demo.make_cuboid
=======================

Build a
*linear* hexahedral NURBS from eight corner points.

Signature::

    C = makeCuboid(A0,B0,C0,D0,A1,B1,C1,D1)

Corner layout (verbatim from the MATLAB comment block)::

        A1 ----------- B1
        /|             /|
       /              / |
      D1 ----------- C1  |
      |              |   |
      |   |          |   |
      |   |          |   |
      |              |   |
      |   A0 ------- | - B0
      |  /           |  /
      | /            | /
      D0 ----------- C0

All eight inputs are 3-vectors (or 4-vectors in homogeneous form); the
resulting NURBS is degree 1 in every direction with clamped knot vectors
``[0 0 1 1]``.  The weight row is 1 (implicit) because the inputs are
Cartesian.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.make import nrbmak

__all__ = ["makeCuboid"]

def makeCuboid(
    A0: Sequence[float],
    B0: Sequence[float],
    C0: Sequence[float],
    D0: Sequence[float],
    A1: Sequence[float],
    B1: Sequence[float],
    C1: Sequence[float],
    D1: Sequence[float],
) -> Nrb:
    """Linear hexahedral NURBS from eight corner points (see module doc)."""
    knots = [np.array([0.0, 0.0, 1.0, 1.0]) for _ in range(3)]
    # MATLAB (verbatim)::
    #
    #     coefs = cat( 4 , cat( 3 , cat( 2 , A0 , B0 ) , ...
    #                              cat( 2 , D0 , C0 ) ) , ...
    #                      cat( 3 , cat( 2 , A1 , B1 ) , ...
    #                              cat( 2 , D1 , C1 ) ) );
    #
    # MATLAB's ``cat(2, X, Y)`` on 3-vectors (shape (3,1)) concatenates along
    # dimension 2, so each corner is a *column* and the intermediate
    # ``cat(2, A0, B0)`` has shape (3, 2) with A0 in column 1 and B0 in
    # column 2.  The final ``coefs`` therefore has layout
    # ``coefs[k, 1, 1, 1] == A0(k)`` (row-major: points first, coordinate
    # last).  We build that natural layout as (2, 2, 2, 3) and transpose to
    # the (3, 2, 2, 2) layout that ``nrbmak`` expects (coord first).
    A0v = np.asarray(A0, dtype=float).ravel()
    B0v = np.asarray(B0, dtype=float).ravel()
    C0v = np.asarray(C0, dtype=float).ravel()
    D0v = np.asarray(D0, dtype=float).ravel()
    A1v = np.asarray(A1, dtype=float).ravel()
    B1v = np.asarray(B1, dtype=float).ravel()
    C1v = np.asarray(C1, dtype=float).ravel()
    D1v = np.asarray(D1, dtype=float).ravel()
    # (2, 2) bottom face:  [[A0, B0], [D0, C0]]  (each entry a 3-vector)
    bottom = np.stack([np.stack([A0v, B0v], axis=0),
                       np.stack([D0v, C0v], axis=0)], axis=0)  # (2, 2, 3)
    top    = np.stack([np.stack([A1v, B1v], axis=0),
                       np.stack([D1v, C1v], axis=0)], axis=0)  # (2, 2, 3)
    # points[a, i, j, k]: a = level (A0/A1), i = row (AB/DC), j = col (A/B, C/D).
    # MATLAB's final layout is coefs[k, j, i, a] (see cat order above), i.e.
    # knot dir 1 <-> A0/B0 column, dir 2 <-> AB/DC row, dir 3 <-> level.
    points = np.stack([bottom, top], axis=0)  # (2, 2, 2, 3)
    coefs  = np.transpose(points, (3, 2, 1, 0))  # (3, 2, 2, 2)
    return nrbmak(coefs, knots)
