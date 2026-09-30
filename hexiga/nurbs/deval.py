"""
hexiga.nurbs.deval
==================

Evaluates the first partial derivatives of a NURBS at parametric points,
using the rational (weight-corrected) formula.  Optionally evaluates
second derivatives (Hessian) when the caller supplies the second-
derivative NURBS objects.

Setup (all cases)::

    [cp, cw] = nrbeval(nurbs, tt);
    temp  = cw;                # the weight
    pnt   = cp ./ temp;

Curve (1-D) -- first derivative::

    [cup, cuw] = nrbeval(dnurbs, tt);
    temp1 = cuw;
    jac   = (cup - temp1 .* pnt) ./ temp;

Curve -- second derivative (Hessian)::

    [cuup, cuuw] = nrbeval(dnurbs2, tt);
    temp2 = cuuw;
    hess  = (cuup - (2*cup.*temp1 + cp.*temp2) ./ temp
             + 2*cp.*temp1.^2 ./ temp.^2) ./ temp;

Surface (2-D) -- first derivatives::

    [cup, cuw] = nrbeval(dnurbs{1}, tt);  tempu = cuw;
    [cvp, cvw] = nrbeval(dnurbs{2}, tt);  tempv = cvw;
    jac{1} = (cup - tempu .* pnt) ./ temp;
    jac{2} = (cvp - tempv .* pnt) ./ temp;

Surface -- second derivatives (Hessian):

* Diagonal (u,u)::

    [cuup, cuuw] = nrbeval(dnurbs2{1,1}, tt);  tempuu = cuuw;
    hess{1,1} = (cuup - (2*cup.*tempu + cp.*tempuu)./temp
                 + 2*cp.*tempu.^2./temp.^2)./temp;

* Off-diagonal (u,v) [== (v,u) by symmetry]::

    [cuvp, cuvw] = nrbeval(dnurbs2{1,2}, tt);  tempuv = cuvw;
    hess{1,2} = (cuvp - (cup.*tempv + cvp.*tempu + cp.*tempuv)./temp
                 + 2*cp.*tempu.*tempv./temp.^2)./temp;

* Diagonal (v,v) is analogous.

Volume (3-D) is the direct generalisation to six entries (3 diagonals + 3
off-diagonals).

Python conventions
------------------

* ``pnt``   -- ``(3, ...)`` Cartesian points, shape of ``...`` matches
  :func:`hexiga.nurbs.eval.nrbeval`.
* ``jac``   -- a tuple of length ``ndim`` of ``(3, ...)`` arrays (one per
  parametric direction), matching MATLAB's ``jac{1..ndim}`` cell.
* ``hess``  -- only returned when ``dnurbs2`` is supplied.  For a curve
  (``ndim==1``) it is a single ``(3, ...)`` array.  For a surface/volume it
  is a 2-D list ``hess[i][j]`` (``i, j in 0..ndim-1``) of ``(3, ...)``
  arrays, mirroring MATLAB's ``hess{i,j}`` cell.  The diagonal and
  off-diagonal entries use the exact formulas above.

The homogeneous two-output form of :func:`nrbeval` (``homogeneous=True``)
is used exactly as MATLAB's ``[cp, cw] = nrbeval(...)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .nrb import Nrb
from .eval import nrbeval

__all__ = ["nrbdeval"]

def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division with broadcasting, matching MATLAB's ``./``.

    Where ``b == 0`` the result is 0 (avoids NaNs / warnings).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    np.divide(a, b, out=out, where=b != 0)
    return out

def _eval_homo(nrb: Nrb, tt):
    """Homogeneous eval of ``nrb`` at ``tt``, returning ``(cp, cw)``."""
    c, w = nrbeval(nrb, tt, homogeneous=True)
    return np.asarray(c, dtype=float), np.asarray(w, dtype=float)

def nrbdeval(
    nrb: Nrb,
    dnurbs: List[Nrb],
    dnurbs2: Optional[List[List[Nrb]]] = None,
    tt: Union[Sequence, np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[Union[np.ndarray, List[List[np.ndarray]]]]]:
    """Evaluate the first partial derivatives (and, optionally, second) of
    ``nrb`` at ``tt``.

    Parameters
    ----------
    nrb
        A curve, surface or volume NURBS (homogeneous, ``dim=4``).
    dnurbs
        List of :class:`Nrb` objects of first partial derivatives, in the
        same order as the requested directions (length ``ndim``).
    dnurbs2
        Optional: for a curve, a length-1 list (single Nrb); for a
        surface/volume, a 2-D list ``dnurbs2[i][j]`` of second partial
        derivatives ``d^2 p / d t_i d t_j`` (length ``ndim`` in each
        direction).  When supplied, the Hessian is computed and returned.
    tt
        Parametric points, same convention as :func:`nrbeval`.

    Returns
    -------
    pnt : np.ndarray
        ``(3, ...)`` Cartesian points.
    jac : list of np.ndarray
        ``ndim`` arrays, each ``(3, ...)``.
    hess : optional
        Only present when ``dnurbs2`` is supplied.  For ``ndim==1`` a
        single ``(3, ...)`` array.  For ``ndim >= 2`` a 2-D list
        ``hess[i][j]`` of ``(3, ...)`` arrays (symmetric).
    """
    if tt is None:
        raise ValueError("tt must be provided (list of 1-D arrays or (ndim, N))")
    ndim = nrb.ndim

    # Setup (verbatim from MATLAB).
    cp, cw = _eval_homo(nrb, tt)
    pnt = _div(cp, cw)

    # First derivatives (rational formula, verbatim from MATLAB).
    jac: List[np.ndarray] = []
    first_homo: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(ndim):
        c, w = _eval_homo(dnurbs[i], tt)
        first_homo.append((c, w))
        jac.append(_div(c - w * pnt, cw))

    # Hessian (only when dnurbs2 is supplied).
    hess: Optional[Union[np.ndarray, List[List[np.ndarray]]]] = None
    if dnurbs2 is not None:
        # Normalise dnurbs2 into a 2-D list h2[i][j] of Nrb.
        if ndim == 1:
            h2 = [[dnurbs2[0]]]
        else:
            h2 = [list(row) for row in dnurbs2]
            if len(h2) != ndim or any(len(r) != ndim for r in h2):
                raise ValueError(
                    "dnurbs2 must be a 2-D list of shape "
                    f"({ndim},{ndim}) for ndim={ndim}"
                )

        # Pre-evaluate all h2[i][j] in homogeneous form.
        h2_homo: List[List[Tuple[np.ndarray, np.ndarray]]] = [
            [_eval_homo(h2[i][j], tt) for j in range(ndim)]
            for i in range(ndim)
        ]

        if ndim == 1:
            # hess = (cuup - (2*cup.*temp1 + cp.*temp2)./temp
            #         + 2*cp.*temp1.^2./temp.^2)./temp
            cuup, temp2 = h2_homo[0][0]
            cup, temp1 = first_homo[0]
            hess = _div(
                cuup
                - (2.0 * cup * temp1 + cp * temp2) / cw
                + 2.0 * cp * temp1 * temp1 / (cw * cw),
                cw,
            )
        else:
            hess_list: List[List[np.ndarray]] = [
                [None] * ndim for _ in range(ndim)
            ]
            for i in range(ndim):
                for j in range(ndim):
                    if hess_list[i][j] is not None:
                        continue
                    if i == j:
                        # hess{ii,ii} = (cuup - (2*cup.*tempu + cp.*tempuu)./temp
                        #                + 2*cp.*tempu.^2./temp.^2)./temp
                        cuup, tempuu = h2_homo[i][i]
                        cup, tempu = first_homo[i]
                        val = _div(
                            cuup
                            - (2.0 * cup * tempu + cp * tempuu) / cw
                            + 2.0 * cp * tempu * tempu / (cw * cw),
                            cw,
                        )
                    else:
                        # hess{i,j} = (cuvp - (cup.*tempv + cvp.*tempu + cp.*tempuv)./temp
                        #              + 2*cp.*tempu.*tempv./temp.^2)./temp
                        cuvp, tempuv = h2_homo[i][j]
                        cup, tempu = first_homo[i]
                        cvp, tempv = first_homo[j]
                        val = _div(
                            cuvp
                            - (cup * tempv + cvp * tempu + cp * tempuv) / cw
                            + 2.0 * cp * tempu * tempv / (cw * cw),
                            cw,
                        )
                    hess_list[i][j] = val
                    hess_list[j][i] = val
            hess = hess_list

    return pnt, jac, hess
