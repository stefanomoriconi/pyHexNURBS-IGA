"""
hexiga.scaff.get_uniform_knot_vect
==================================

Build a clamped
*uniform* knot vector for a direction with ``dim`` control points and
degree ``deg``.

Signature: ``kntVect = getUniformKnotVect(dim, deg)``.

Notes (literal port, quirks preserved)
--------------------------------------

* ``deg >= dim`` is an error in MATLAB (``assert(deg < dim, ...)``); the
  port raises :class:`ValueError`.
* The MATLAB branch structure for the ``deg > 1`` case is::

        if mod(dim,2) > 0
            if mod(deg,2) > 0
                kntVect = [ zeros(1,deg), linspace(0,1,dim-2), ones(1,deg) ];
            else
                kntVect = [ zeros(1,deg), linspace(0,1,dim-1), ones(1,deg) ];
            end
        else
            if mod(deg,2) > 0
                kntVect = [ zeros(1,deg), linspace(0,1,dim-2), ones(1,deg) ];
            else
                kntVect = [ zeros(1,deg), linspace(0,1,dim-1), ones(1,deg) ];
            end
        end

  The outer ``if mod(dim,2)>0`` test is therefore *dead* -- both branches
  return the same knot vector.  The port preserves the literal two-level
  structure so that any future divergence in the MATLAB file propagates
  cleanly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["getUniformKnotVect"]

def getUniformKnotVect(dim: int, deg: int) -> np.ndarray:
    """Clamped, uniform knot vector for ``dim`` control points, degree ``deg``.

    Parameters
    ----------
    dim
        Number of control points in the direction (``>= 2``).
    deg
        Spline degree (``>= 1`` and ``< dim``).

    Returns
    -------
    np.ndarray
        1-D float array of length ``dim + deg``.

    Raises
    ------
    ValueError
        If ``deg >= dim`` (mirroring the MATLAB ``assert``).
    """
    dim = int(dim)
    deg = int(deg)
    if not (deg < dim):
        raise ValueError(
            "B-spline Degree is greater than the Number of Control-Points!"
        )

    if deg == 1:
        kntVect = np.concatenate([[0.0], np.linspace(0.0, 1.0, dim), [1.0]])
    else:
        if dim % 2 > 0:
            if deg % 2 > 0:
                kntVect = np.concatenate(
                    [np.zeros(deg, dtype=float),
                     np.linspace(0.0, 1.0, dim - 2),
                     np.ones(deg, dtype=float)]
                )
            else:
                kntVect = np.concatenate(
                    [np.zeros(deg, dtype=float),
                     np.linspace(0.0, 1.0, dim - 1),
                     np.ones(deg, dtype=float)]
                )
        else:
            if deg % 2 > 0:
                kntVect = np.concatenate(
                    [np.zeros(deg, dtype=float),
                     np.linspace(0.0, 1.0, dim - 2),
                     np.ones(deg, dtype=float)]
                )
            else:
                kntVect = np.concatenate(
                    [np.zeros(deg, dtype=float),
                     np.linspace(0.0, 1.0, dim - 1),
                     np.ones(deg, dtype=float)]
                )
    return kntVect
