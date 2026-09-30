"""
hexiga.nurbs.eval
=================

``nrbeval`` -- evaluate a NURBS at parametric points.

Port of ``nrbeval.m``.  Works for curves, surfaces and volumes, and accepts
either

* a *grid* description -- a sequence of ``ndim`` 1-D arrays, one per
  parametric direction (e.g. ``[u, v]`` for a surface); or
* a *scattered* description -- a single ``(ndim, N)`` (or ``(N, ndim)``)
  array of arbitrary parametric points.

Returns Cartesian coordinates, shape ``(3, ...)``:

* curve:    ``(3, N)``  (or ``(3,)`` for a single scalar parameter)
* surface:  grid ``(3, nu, nv)``  or scattered ``(3, *shape)``
* volume:   grid ``(3, nu, nv, nw)`` or scattered ``(3, *shape)``

Implementation note
-------------------
The MATLAB permute/reshape cascade is replaced by a mathematically identical
"evaluate one parametric axis at a time" helper built directly on the
univariate ``bspeval`` kernel.  Grid evaluation is fully vectorised per axis;
scattered evaluation loops over points (the parameters differ per point, so
they cannot be factored).
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from .nrb import Nrb
from ._kernels import bspeval

__all__ = ["nrbeval"]

def _eval_axis(C, axis, d, k, tvals):
    """Evaluate B-spline ``axis`` (0-based over data dims) at all ``tvals``.

    ``C`` has shape ``(4, *sizes)``; the result has shape
    ``(4, *sizes_without_axis, len(tvals))`` with the evaluated axis appended
    at the last data position.
    """
    nd = C.shape[1:]
    perm = _axis_perm(len(nd), axis)
    Cp = np.transpose(C, perm)
    lead_shape = Cp.shape[:-1]
    last = Cp.shape[-1]
    C2 = Cp.reshape((-1, last))
    val = bspeval(d, C2, k, tvals)  # (prod(lead_shape), len(tvals))
    return val.reshape(lead_shape + (tvals.size,))

def _axis_perm(ndim: int, axis: int):
    """Permutation moving data-axis ``axis`` (0-based) to the last position.

    Returns a list of full-array axis indices (row axis 0 first) suitable for
    :func:`numpy.transpose`.
    """
    if ndim == 1:
        return [0, 1]
    return [0] + [a + 1 for a in range(ndim) if a != axis] + [axis + 1]

def _eval_point_axis(C, axis, d, k, t):
    """Evaluate B-spline ``axis`` at a single scalar ``t``; axis is removed."""
    nd = C.shape[1:]
    perm = _axis_perm(len(nd), axis)
    Cp = np.transpose(C, perm)
    lead_shape = Cp.shape[:-1]
    last = Cp.shape[-1]
    C2 = Cp.reshape((-1, last))
    val = bspeval(d, C2, k, np.array([t]))[:, 0]  # (prod(lead_shape),)
    return val.reshape(lead_shape)

def _parse_tt(tt, ndim):
    """Classify the parametric input as ('grid', list-of-arrays) or
    ('scattered', (ndim, N) array)."""
    # A sequence of exactly ndim 1-D arrays is a grid.
    if isinstance(tt, (list, tuple)) and len(tt) == ndim and all(
        np.asarray(x).ndim == 1 for x in tt
    ):
        return "grid", [np.asarray(a, dtype=float).ravel() for a in tt]

    arr = np.asarray(tt, dtype=float)

    # MATLAB convention: for a curve, a bare 1-D parameter vector t is a grid.
    if ndim == 1 and arr.ndim == 1:
        return "grid", [arr.astype(float).ravel()]

    # Otherwise interpret as a scattered 2-D array.
    if arr.ndim == 2:
        if arr.shape[0] == ndim:
            return "scattered", arr
        if arr.shape[1] == ndim:
            return "scattered", arr.T
    raise ValueError(
        "tt must be a list of ndim 1-D arrays (grid) or a 2-D array "
        f"(ndim, N) of scattered points (ndim={ndim})"
    )

def _squeeze_curve(
    cp: np.ndarray,
    cw: Optional[np.ndarray],
    kind: str,
    ndim: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Collapse a 1-D curve grid ``(3, 1)`` to ``(3,)`` (MATLAB convention)."""
    if ndim == 1 and kind == "grid" and cp.ndim == 2 and cp.shape[1] == 1:
        cp = cp[:, 0]
        if cw is not None:
            cw = np.asarray(cw).reshape(-1)[0]
    return cp, cw

def nrbeval(
    nrb: Nrb, tt: Union[np.ndarray, Sequence], homogeneous: bool = True
):
    """Evaluate ``nrb`` at the parametric points given by ``tt``.

    Parameters
    ----------
    nrb
        A curve, surface or volume NURBS.
    tt
        A grid (sequence of ``ndim`` 1-D arrays) or a scattered ``(ndim, N)``
        array of parametric points.
    homogeneous
        *Default True* -- return the homogeneous two-output form
        ``(cp, cw)``: ``cp`` is the un-normalised part (rows 0..2) with shape
        ``(3, ...)`` and ``cw`` is the evaluated weight with shape ``(...)``.
        This mirrors MATLAB's ``[cp, cw] = nrbeval(...)`` two-output form,
        which is how the entire MATLAB dependency stack (GeoPDEs / NURBS
        toolbox / demo toolkit) calls evaluation -- they are *always* in
        homogeneous form.

        If False, return the single Cartesian array ``cp / cw`` of shape
        ``(3, ...)``.  Use this only at the final, user-facing boundary.

    See the module docstring for output shapes.
    """
    ndim = nrb.ndim
    if ndim not in (1, 2, 3):
        raise ValueError(f"unsupported ndim={ndim}")
    order = nrb.order
    knots = nrb.knots
    coefs = nrb.coefs

    kind, parsed = _parse_tt(tt, ndim)

    if kind == "grid":
        grids = parsed  # list of ndim 1-D arrays
        C = coefs
        for axis in range(ndim - 1, -1, -1):
            C = _eval_axis(C, axis, order[axis] - 1, knots[axis], grids[axis])
        # Data axes are now in reversed parametric order; restore (u, v, w).
        if ndim > 1:
            perm = [0] + list(range(1, 1 + ndim))[::-1]
            C = np.transpose(C, perm)
    else:
        pts = parsed  # (ndim, N)
        N = pts.shape[1]
        out = np.empty((4, N), dtype=float)
        for i in range(N):
            C = coefs
            for axis in range(ndim - 1, -1, -1):
                C = _eval_point_axis(
                    C, axis, order[axis] - 1, knots[axis], float(pts[axis, i])
                )
            out[:, i] = C
        C = out

    if homogeneous:
        # MATLAB ``[cp, cw] = nrbeval(...)``: cp un-normalised, cw the weight.
        cp = C[:3]
        cw = C[3]
        cp, cw = _squeeze_curve(cp, cw, kind, ndim)
        return cp, cw

    C = C[:3] / C[3]
    return _squeeze_curve(C, None, kind, ndim)[0]
