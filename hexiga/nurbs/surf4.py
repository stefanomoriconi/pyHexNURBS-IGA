"""
hexiga.nurbs.surf4
==================

Construct a *bilinear* NURBS surface from its four corner points.

Signature: ``srf = nrb4surf(p11, p12, p21, p22)``.

Notes
-----

* The corner layout::

          ^ V direction
          |
          ----------------
          |p21        p22|
          |              |
          |p11        p12|
          -------------------> U direction

  i.e. ``u`` is the first (u/lhs-rhs) axis and ``v`` the second
  (v/bottom-top) axis:  ``p11 @ (u1, v1)`` (bottom-left),
  ``p12 @ (u2, v1)`` (bottom-right), ``p21 @ (u1, v2)`` (top-left),
  ``p22 @ (u2, v2)`` (top-right).
* Each input is a 3x1 column of Cartesian coordinates.  The control-point
  array is ``(3, 2, 2)`` with the weight row (row index 3) appended via
  ``nrbmak``'s homogeneous promotion, giving a ``(4, 2, 2)`` bilinear
  surface (degree ``(1, 1)``) with clamped knots ``[0 0 1 1]`` in each
  direction.
* ``nargin ~= 4`` in MATLAB is an error; the Python signature requires
  exactly four positional arguments.
"""

from __future__ import annotations

import numpy as np

from .nrb import Nrb
from .make import nrbmak

__all__ = ["nrb4surf"]

def _col(p: np.ndarray) -> np.ndarray:
    """Coerce a corner point to a ``(3,)`` column vector."""
    return np.asarray(p, dtype=float).reshape(3)

def nrb4surf(
    p11: np.ndarray,
    p12: np.ndarray,
    p21: np.ndarray,
    p22: np.ndarray,
) -> Nrb:
    """Construct a bilinear NURBS surface from its four corners.

    Parameters
    ----------
    p11, p12, p21, p22
        Cartesian corner points (3-vectors), laid out as documented in the
        module docstring (u = first axis, v = second axis).

    Returns
    -------
    Nrb
        A bilinear (degree ``(1, 1)``) surface NURBS in homogeneous
        coordinates (``dim == 4``).
    """
    p11 = _col(p11)
    p12 = _col(p12)
    p21 = _col(p21)
    p22 = _col(p22)

    # MATLAB: coefs = cat(1, zeros(3,2,2), ones(1,2,2));  -- but only the
    # first 3 rows (the Cartesian coords) are stored here; nrbmak promotes
    # to homogeneous by appending the weight row.
    coefs = np.empty((3, 2, 2), dtype=float)
    coefs[:, 0, 0] = p11  # (u1, v1) bottom-left
    coefs[:, 1, 0] = p12  # (u2, v1) bottom-right
    coefs[:, 0, 1] = p21  # (u1, v2) top-left
    coefs[:, 1, 1] = p22  # (u2, v2) top-right

    knots = [[0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 1.0, 1.0]]

    return nrbmak(coefs, knots)
