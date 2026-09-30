"""
hexiga.nurbs.kntins
===================

Inserts new knots into a NURBS curve, surface or volume.

Algorithm:

* Curve -- the single knot vector::

    [coefs, knots] = bspkntins(degree, coefs, knots, iknots);

* Surface -- insert along v first, then u (axis order immaterial; axes are
  independent)::

    coefs = reshape(coefs, 4*num1, num2);
    [coefs, knots{2}] = bspkntins(degree(2), coefs, knots{2}, iknots{2});
    coefs = reshape(coefs, [4 num1 newnum2]);
    coefs = permute(coefs, [1 3 2]);
    coefs = reshape(coefs, 4*num2, newnum3');
    [coefs, knots{1}] = bspkntins(degree(1), coefs, knots{1}, iknots{1});
    coefs = permute(coefs, [1 3 2]);

* Volume -- along w, then v, then u, moving the target axis to last,
  reshaping to a 2-D ``(4*prod(others), n_axis)`` matrix, calling
  :func:`hexiga.nurbs._backend.bspkntins`, and permuting back.

An *empty* knot list for a direction leaves that axis untouched.  Knots are
validated to lie inside ``[min(knots), max(knots)]`` of the corresponding
axis; violating this raises an error exactly as in MATLAB.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

from .nrb import Nrb
from ._backend import bspkntins as _bspkntins

__all__ = ["nrbkntins"]

def _kntins_axis(
    coefs: np.ndarray,
    knot_axis: np.ndarray,
    degree: int,
    axis: int,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert knots ``u`` into ``knot_axis`` with control points ``coefs``
    (shape ``(4, *number)``), operating on parametric ``axis`` (0-based).
    Returns ``(new_coefs, new_knot_axis)``."""
    Cp = np.moveaxis(coefs, axis + 1, -1)  # (4, ..., n_axis)
    C2 = Cp.reshape((-1, Cp.shape[-1]))  # (4*prod(lead), n_axis)
    Cn, kn = _bspkntins(degree, C2, knot_axis, np.asarray(u, dtype=float))
    new_lead = list(coefs.shape[1:])
    del new_lead[axis]
    Cn_full = Cn.reshape((coefs.shape[0], *new_lead, Cn.shape[-1]))
    Cn_full = np.moveaxis(Cn_full, -1, axis + 1)
    return np.asarray(Cn_full, dtype=float), np.asarray(kn, dtype=float)

def nrbkntins(
    nrb: Nrb,
    iknots: Union[np.ndarray, Sequence],
) -> Nrb:
    """Insert knots into a NURBS curve, surface or volume.

    Parameters
    ----------
    nrb
        A NURBS curve (1-D), surface (2-D) or volume (3-D).
    iknots
        * Curve: a 1-D array of knot values (non-decreasing), possibly with
          repeats for multiplicity.
        * Surface/volume: a sequence of ``ndim`` entries, one per parametric
          direction (1-based order, i.e. entry ``i`` corresponds to direction
          ``i+1``).  An *empty* entry (``[]`` or an empty array) leaves that
          direction unchanged.

    Returns
    -------
    Nrb
        The knot-inserted NURBS (same :class:`~hexiga.nurbs.nrb.Nrb` type).
    """
    if nrb.ndim == 1:
        # ---- curve ------------------------------------------------------
        if isinstance(iknots, (list, tuple)):
            raise ValueError(
                "for a curve, iknots must be a single 1-D array of knots"
            )
        u = np.asarray(iknots, dtype=float)
        if u.size == 0:
            return nrb
        if np.any(u > nrb.knots[0].max()) or np.any(u < nrb.knots[0].min()):
            raise ValueError(
                "Trying to insert a knot outside the interval of definition"
            )
        new_coefs, kn = _kntins_axis(
            nrb.coefs, nrb.knots[0], nrb.degree[0], 0, u
        )
        new_knots = (kn,)
        new_number = (int(new_coefs.shape[1]),)
    else:
        # ---- surface / volume ------------------------------------------
        if not isinstance(iknots, (list, tuple)):
            raise TypeError(
                "for a surface/volume, iknots must be a sequence of "
                f"{nrb.ndim} arrays (one per direction)"
            )
        if len(iknots) != nrb.ndim:
            raise ValueError(
                f"iknots must have {nrb.ndim} entries, got {len(iknots)}"
            )
        # Validate range per axis (MATLAB: cellfun fmax/fmin check).
        for i in range(nrb.ndim):
            u = np.asarray(iknots[i], dtype=float)
            if u.size == 0:
                continue
            if np.any(u > nrb.knots[i].max()) or np.any(
                u < nrb.knots[i].min()
            ):
                raise ValueError(
                    "Trying to insert a knot outside the interval of "
                    "definition"
                )
        coefs = nrb.coefs
        knots = list(nrb.knots)
        number = list(nrb.number)
        # MATLAB processes axes in reverse (w, v, u); the axes are
        # independent so the forward order below is equivalent.
        for i in range(nrb.ndim):
            u = np.asarray(iknots[i], dtype=float)
            if u.size == 0:
                continue
            coefs, kn = _kntins_axis(
                coefs, knots[i], nrb.degree[i], i, u
            )
            knots[i] = kn
            number[i] = int(coefs.shape[i + 1])
        new_coefs = coefs
        new_knots = tuple(knots)
        new_number = tuple(number)

    return Nrb(
        number=new_number,
        order=nrb.order,
        knots=new_knots,
        coefs=new_coefs,
    )
