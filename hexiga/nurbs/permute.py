"""
hexiga.nurbs.permute
=====================

Faithful Python port of MATLAB's ``nrbpermute.m``: rearrange the parametric
directions of a NURBS surface or volume (transposition of axes).

MATLAB algorithm (verbatim from ``nrbpermute.m``)::

    if (~iscell (vol.knots))
      error ('A NURBS curve cannot be rearranged.');
    end
    tvol = nrbmak (permute (vol.coefs, [1, ord+1]), {vol.knots{ord}});

``ord`` is a 1-based permutation of ``1:ndim`` (e.g. ``[1 3 2]``).  The
control-point array is transposed so that the row axis stays first and the
data axes are reordered by ``ord``; the knot vectors are reordered in the
same way.  Curves have a single parametric direction, so they cannot be
rearranged and an error is raised (exactly like MATLAB).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .nrb import Nrb
from .make import nrbmak

__all__ = ["nrbpermute"]

def nrbpermute(nurbs: Nrb, ord: Sequence[int]) -> Nrb:
    """Rearrange the parametric directions of a surface/volume NURBS.

    Parameters
    ----------
    nurbs
        A NURBS surface or volume (a curve cannot be rearranged).
    ord
        A 1-based permutation list of length ``ndim`` giving the new order
        of the parametric directions (MATLAB convention), e.g. ``[1, 3, 2]``.

    Returns
    -------
    Nrb
        A new NURBS with the transposed axes and correspondingly reordered
        knot vectors.

    Raises
    ------
    ValueError
        If ``nurbs`` is a curve (1 parametric direction) or ``ord`` is not a
        valid permutation of ``1:ndim``.
    """
    ndim = nurbs.ndim
    if ndim == 1:
        raise ValueError("A NURBS curve cannot be rearranged.")

    ord_list = [int(o) for o in ord]
    if len(ord_list) != ndim:
        raise ValueError(
            f"ord must have {ndim} entries, got {len(ord_list)}"
        )
    if sorted(ord_list) != list(range(1, ndim + 1)):
        raise ValueError(f"ord must be a permutation of 1:{ndim}, got {ord_list}")

    # permute(vol.coefs, [1, ord])  ->  row axis first, then data axes in
    # the order given by ord (ord is 1-based over the data axes; in numpy
    # the data axes are 1..ndim, so the axes list is [0] + ord verbatim).
    axis_perm = [0] + list(ord_list)
    coefs = np.transpose(np.asarray(nurbs.coefs, dtype=float), axis_perm)

    # {vol.knots{ord}}  ->  reorder the knot vectors by ord.
    knots = [np.asarray(nurbs.knots[o - 1], dtype=float) for o in ord_list]

    return nrbmak(coefs, knots)
