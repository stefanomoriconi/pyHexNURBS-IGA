"""
hexiga.nurbs.deriv
==================

Faithful Python port of MATLAB's ``nrbderiv.m``: given a NURBS object and a
list of parametric directions, return one NURBS per direction whose control
points are the first partial derivatives (in the homogeneous sense) with
respect to the chosen parametric variable.

The derivative is produced *without* normalising by the weight -- i.e. the
resulting ``dnurbs`` stores the raw homogeneous derivative.  Division by the
weight happens later, inside :func:`hexiga.nurbs.deval.nrbdeval` (exactly the
way MATLAB's ``nrbdeval`` does it).

Algorithm:

* u-derivative:
    dcoefs = permute(coefs, [1 3 4 2]);  # u -> last
    dcoefs = reshape(dcoefs, 4*num2*num3, num1);
    dcoefs, kn1 = bspderiv(degree1, dcoefs, knots{1});
    dcoefs = permute(reshape(dcoefs, [4 num2 num3 n]), [1 4 2 3]);
    dnurbs{1} = nrbmak(dcoefs, [kn1, knots{2}, knots{3}]);

* v- and w-derivatives: analogous, moving the target axis to last, reshaping,
  applying :func:`hexiga.nurbs._backend.bspderiv`, and permuting back.

The Python port uses :func:`numpy.moveaxis` (semantically identical to MATLAB's
``permute``) plus the NumPy ``bspderiv`` kernel from :mod:`hexiga.nurbs._kernels`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

from .nrb import Nrb
from .make import nrbmak
from ._backend import bspderiv

__all__ = ["nrbderiv"]

def _derivative_axis(
    nrb: Nrb, axis: int
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Tuple[int, ...]]:
    """Return ``(dcoefs, dknots, dorder)`` for the first derivative along
    ``axis`` (0-based) of ``nrb``.

    The control points ``dcoefs`` are in homogeneous coordinates with shape
    ``(4, *number_of_dnurb)`` where one axis is consumed (its size drops by 1,
    or it disappears if it was size 1).  The knot vector for that axis is
    shortened by 1.
    """
    ndim = nrb.ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} out of range for ndim={ndim}")
    d = nrb.degree[axis]
    if d == 0:
        raise ValueError(
            "derivative of a degree-0 B-spline is zero; "
            "use nrbdegelev or a higher-degree input"
        )
    order = nrb.order
    knots = nrb.knots

    # Move the target axis to last, flatten leading dims.
    # coefs has shape (4, n0, n1, ..., n{ndim-1}); axis a -> moveaxis to last.
    Cp = np.moveaxis(nrb.coefs, axis + 1, -1)  # (4, ..., n_axis)
    lead = Cp.shape[:-1]
    C2 = Cp.reshape((-1, Cp.shape[-1]))  # (4*prod(lead), n_axis)
    dC, dK = bspderiv(d, C2, knots[axis])
    # Reconstruct the per-axis size: n_axis - 1.
    new_last = dC.shape[-1]
    lead_count = C2.shape[0] // (nrb.coefs.shape[0])  # prod(lead)
    # New leading shape (4, *lead) without the dropped axis.
    new_lead = list(nrb.coefs.shape[1:])
    del new_lead[axis]
    dC_full = dC.reshape((nrb.coefs.shape[0], *new_lead, new_last))
    dC_full = np.moveaxis(dC_full, -1, axis + 1)  # move back to its slot

    # Knots: replace axis with dK; all other knot vectors unchanged.
    dknots = list(knots)
    dknots[axis] = dK
    # Order: axis order drops by 1.
    dorder = list(order)
    dorder[axis] = order[axis] - 1

    return dC_full, tuple(dknots), tuple(dorder)

def nrbderiv(
    nrb: Nrb, idir: Union[int, Sequence[int], None] = None
) -> List[Nrb]:
    """Return the list of NURBS objects representing the first partial
    derivatives of ``nrb`` with respect to each requested parametric
    direction.

    Parameters
    ----------
    nrb
        A curve, surface or volume NURBS.
    idir
        Optional list of 1-based (MATLAB convention) directions to
        differentiate.  Defaults to all directions.  Accepted values are 1, 2,
        3 (or a subset thereof) -- matching MATLAB's ``nrbderiv(nurbs, dir)``
        convention.

    Returns
    -------
    list of Nrb
        One :class:`Nrb` per requested direction, in the same order as
        ``idir``.  Each stores the homogeneous derivative; use
        :func:`hexiga.nurbs.deval.nrbdeval` to divide by the weight and obtain
        the Cartesian derivative.
    """
    ndim = nrb.ndim
    if idir is None:
        dirs = list(range(1, ndim + 1))
    elif isinstance(idir, int):
        dirs = [idir]
    else:
        dirs = [int(x) for x in idir]
    for d in dirs:
        if not (1 <= d <= ndim):
            raise ValueError(
                f"direction {d} out of range for ndim={ndim} (1-based)"
            )

    out: List[Nrb] = []
    for d in dirs:
        axis = d - 1
        dC, dknots, dorder = _derivative_axis(nrb, axis)
        # The differentiated axis loses one control point; all others keep
        # their count.
        dnumber = tuple(
            nrb.number[i] - 1 if i == axis else nrb.number[i]
            for i in range(ndim)
        )
        dn = Nrb(
            number=dnumber,
            order=dorder,
            knots=dknots,
            coefs=dC,
        )
        out.append(dn)
    return out
