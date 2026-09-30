"""
hexiga.nurbs.interp
====================

B-Spline curve interpolation through a set of points.

Signature::

    [crv, u] = bspinterpcrv (Q, p, method)

    INPUT:
      Q      - points to be interpolated, shape (3, N) (x; y; z rows).
      p      - degree of the interpolating curve.
      method - parametrization method:
               'equally_spaced' | 'chord_length' | 'centripetal' (default)

    OUTPUT:
      crv    - the B-Spline curve (Nrb).
      u      - parametric points corresponding to the interpolation points.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._kernels import basisfun, findspan
from .make import nrbmak

__all__ = ["bspinterpcrv"]

def bspinterpcrv(
    Q: np.ndarray,
    p: int,
    method: str = "centripetal",
) -> Tuple["Nrb", np.ndarray]:  # noqa: F821
    """B-Spline interpolation of a 3-D curve.

    The knot sequence and linear system are built so that the resulting
    control points reproduce ``Q`` at the interpolated parameters.
    """
    if method is None or method == "":
        method = "centripetal"

    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != 3:
        raise ValueError(f"Q must be a (3, N) array, got shape {Q.shape}")
    N = Q.shape[1]
    p = int(p)

    if method.lower() == "equally_spaced":
        u = np.linspace(0.0, 1.0, N)
    elif method.lower() == "chord_length":
        d = float(np.sum(np.sqrt(np.sum(np.diff(Q.T) ** 2, axis=1))))
        u = np.zeros(N)
        if N > 1:
            u[1:] = np.cumsum(
                np.sqrt(np.sum(np.diff(Q, axis=1) ** 2, axis=0))
            ) / d
        u[-1] = 1.0
    elif method.lower() == "centripetal":
        d = float(
            np.sum(np.sqrt(np.sqrt(np.sum(np.diff(Q.T) ** 2, axis=1))))
        )
        u = np.zeros(N)
        if N > 1:
            u[1:] = np.cumsum(
                np.sqrt(
                    np.sqrt(np.sum(np.diff(Q, axis=1) ** 2, axis=0))
                )
            ) / d
        u[-1] = 1.0
    else:
        raise ValueError(
            f"bspinterpcrv: unrecognized parametrization method '{method}'."
        )

    knts = np.zeros(N + p + 1)
    # MATLAB: for jj = 2 : n-p, knts(jj+p) = 1/p * sum (u(jj:jj+p-1));
    for jj in range(1, N - p):  # jj in [1, N-p-1]  (0-based)
        knts[jj + p] = (1.0 / p) * np.sum(u[jj : jj + p])
    knts[-(p + 1) :] = 1.0

    A = np.zeros((N, N))
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    # MATLAB uses findspan (0-based n = N-1, knot vector knts[0..N+p]).
    n_matlab = N - 1
    for ii in range(1, N - 1):
        s = findspan(n_matlab, p, float(u[ii]), knts)
        # MATLAB: A(ii, span-p+1 : span+1) => Python 0-based: span-p .. span
        A[ii, s - p : s + 1] = basisfun(s, float(u[ii]), p, knts)

    x = np.linalg.solve(A, Q[0, :])
    y = np.linalg.solve(A, Q[1, :])
    z = np.linalg.solve(A, Q[2, :])
    pnts = np.vstack([x, y, z, np.ones(N)])

    crv = nrbmak(pnts, knts)
    return crv, u
