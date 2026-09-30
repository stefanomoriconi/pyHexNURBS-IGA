"""
hexiga.nurbs.tform
===================

Faithful Python port of MATLAB's ``nrbtform.m``: apply a homogeneous
affine transformation matrix (4x4) to a NURBS curve, surface or volume.

MATLAB algorithm (verbatim from ``nrbtform.m``)::

    % curve
    coefs = tmat * coefs;
    % surface
    coefs = reshape (tmat * reshape (coefs, dim, nu*nv), [dim nu nv]);
    % volume
    coefs = reshape (tmat * reshape (coefs, dim, nu*nv*nw), [dim nu nv nw]);
    tcurv = nrbmak (coefs, curv.knots);

Because the transformation matrix is applied only to the 4x4 homogeneous
control-point block, the weights (last row) are transformed in exactly the
same way as the Cartesian coordinates -- i.e. the resulting NURBS is again
in homogeneous form, with the same knot vectors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .nrb import Nrb
from .make import nrbmak

__all__ = ["nrbtform"]

def nrbtform(
    nurbs: Nrb,
    tmat: Optional[np.ndarray] = None,
) -> Nrb:
    """Apply the homogeneous transformation ``tmat`` to ``nurbs``.

    Parameters
    ----------
    nurbs
        A NURBS curve, surface or volume (in homogeneous form).
    tmat
        A 4x4 homogeneous transformation matrix, applied as ``tmat @ coefs``.

    Returns
    -------
    Nrb
        A new NURBS with the transformed control points.

    Raises
    ------
    ValueError
        If ``tmat`` is ``None`` or not 4x4.
    """
    if tmat is None:
        raise ValueError("Not enough input arguments!")
    t = np.asarray(tmat, dtype=float)
    if t.shape != (4, 4):
        raise ValueError(f"tmat must be a 4x4 matrix, got shape {t.shape}")

    coefs = np.asarray(nurbs.coefs, dtype=float)
    d = coefs.shape[0]
    data = coefs.shape[1:]
    flat = coefs.reshape(d, -1)
    tcoefs = t @ flat
    tcoefs = tcoefs.reshape((d,) + data)

    # For a curve the knots are a single 1-D array; for a surface/volume a
    # tuple of 1-D arrays.  Pass them through untouched.
    if nurbs.ndim == 1:
        knots = nurbs.knots[0]
    else:
        knots = list(nurbs.knots)
    return nrbmak(tcoefs, knots)
