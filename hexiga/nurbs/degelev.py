"""
hexiga.nurbs.degelev
====================

Faithful Python port of MATLAB's ``nrbdegelev.m``: degree elevation of a
NURBS curve, surface or volume by a given number of times in each
parametric direction.

The algorithm is verbatim from ``nrbdegelev.m`` (volume case shown):

    % u-direction:
    dcoefs = permute(coefs, [1 3 4 2]);        % u -> last
    dcoefs = reshape(dcoefs, 4*num2*num3, num1);
    dcoefs, dknots{1} = bspdegelev(degree1, dcoefs, knots{1}, times{1});
    dcoefs = permute(reshape(dcoefs, [4 num2 num3 num1+times{1}]), [1 4 2 3]);
    % ... repeat for v and w directions ...
    dnurbs = nrbmak(dcoefs, dknots);

The Python port uses :func:`numpy.moveaxis` (semantically identical to
MATLAB's ``permute``) plus the NumPy ``bspdegelev`` kernel from
:mod:`hexiga.nurbs._kernels`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

from .nrb import Nrb
from ._backend import bspdegelev

__all__ = ["nrbdegelev"]

def nrbdegelev(nrb: Nrb, ntimes: Union[int, Sequence[int]]) -> Nrb:
    """Elevate the degree of ``nrb`` by ``ntimes`` in each parametric
    direction.

    Parameters
    ----------
    nrb
        A curve, surface or volume NURBS.
    ntimes
        Either a single int (applied to every direction) or a sequence of
        length ``ndim`` with per-direction elevation counts.

    Returns
    -------
    Nrb
        The degree-elevated NURBS.
    """
    ndim = nrb.ndim
    if isinstance(ntimes, int):
        times = [ntimes] * ndim
    else:
        times = [int(t) for t in ntimes]
        if len(times) != ndim:
            raise ValueError(
                f"len(ntimes)={len(times)} must match ndim={ndim}"
            )
    if any(t < 0 for t in times):
        raise ValueError(f"ntimes must be >= 0, got {times}")

    coefs = nrb.coefs
    knots = list(nrb.knots)
    number = list(nrb.number)
    order = list(nrb.order)

    for axis in range(ndim):
        t = times[axis]
        if t == 0:
            continue
        d = nrb.degree[axis]
        if d == 0:
            # Degree-0 B-spline (a constant) cannot be elevated without
            # changing the meaning; MATLAB would just no-op here.
            continue
        # Move the target axis to last, flatten the leading dims.
        # Cp.shape = (other axes in current order..., n_axis).
        Cp = np.moveaxis(coefs, axis + 1, -1)
        lead_shape = Cp.shape[:-1]  # current shape of the non-target axes
        C2 = Cp.reshape((-1, Cp.shape[-1]))
        dC, dK = bspdegelev(d, C2, knots[axis], t)
        new_last = dC.shape[-1]  # n_axis + t (degree-elevated CP count)
        # dC is (prod(lead_shape), new_last); restore the per-axis layout.
        # Use the CURRENT lead_shape (not nrb.coefs.shape), because earlier
        # axes have already changed coefs.shape in this sequential loop.
        dC_full = dC.reshape((*lead_shape, new_last))
        coefs = np.moveaxis(dC_full, -1, axis + 1)
        knots[axis] = dK
        number[axis] = new_last
        order[axis] = d + 1 + t  # degree elevated by t

    return Nrb(
        number=tuple(number),
        order=tuple(order),
        knots=tuple(knots),
        coefs=coefs,
    )
