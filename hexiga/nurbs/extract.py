"""
hexiga.nurbs.extract
====================

Faithful Python port of MATLAB's ``nrbextract.m`` (Rafael Vazquez): extract
the boundary entities of a NURBS surface or volume.

* Curve  -- two degenerate NURBS "curves", one per boundary control point
  (index 0 and -1), each sharing the parent's knot vector.
* Surface -- four boundary *curves*, ordered:

    1: U = 0      (``coefs[:, 0, :]`` with the **v** knot vector)
    2: U = 1      (``coefs[:, -1, :]`` with the **v** knot vector)
    3: V = 0      (``coefs[:, :, 0]`` with the **u** knot vector)
    4: V = 1      (``coefs[:, :, -1]`` with the **u** knot vector)

* Volume -- six boundary *surfaces*, ordered (``ind = 1:3`` in the MATLAB
  source, ``inds = setdiff(1:3, ind)``):

    1: U = 0      (``coefs[:, 0, :, :]`` with knot vectors (v, w))
    2: U = 1      (``coefs[:, -1, :, :]`` with knot vectors (v, w))
    3: V = 0      (``coefs[:, :, 0, :]`` with knot vectors (u, w))
    4: V = 1      (``coefs[:, :, -1, :]`` with knot vectors (u, w))
    5: W = 0      (``coefs[:, :, :, 0]`` with knot vectors (u, v))
    6: W = 1      (``coefs[:, :, :, -1]`` with knot vectors (u, v))

* Curve -- two degenerate entries, one per boundary control point, each
  keeping the parent's **full** knot vector (as in MATLAB).

The extracted entities must have *open* knot vectors on every axis of the
parent; otherwise an error is raised (as in MATLAB).  A ``sides`` argument
(1-based, as in MATLAB) selects a subset of the extracted entities.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .nrb import Nrb
from .make import nrbmak

__all__ = ["nrbextract"]

def _is_open_knots(k: np.ndarray, order: int) -> bool:
    return (
        k[0] == k[order - 1] and k[-1] == k[-(order - 1) - 1]
    )

def nrbextract(
    nrb: Nrb,
    sides: Optional[Sequence[int]] = None,
) -> List[Nrb]:
    """Extract the boundary curves / surfaces of a NURBS entity.

    Parameters
    ----------
    nrb
        A NURBS curve, surface or volume with open knot vectors.
    sides
        Optional 1-based list of side indices to return (MATLAB convention).
        Defaults to all ``2*ndim`` sides.

    Returns
    -------
    list of Nrb
        * For a curve: a list of 2 degenerate NURBS (one per boundary CP).
        * For a surface: a list of 4 boundary curves.
        * For a volume: a list of 6 boundary surfaces.
    """
    if sides is None:
        sides = list(range(1, 2 * nrb.ndim + 1))
    else:
        sides = [int(s) for s in sides]
        if any(s < 1 or s > 2 * nrb.ndim for s in sides):
            raise ValueError(
                f"sides out of range for ndim={nrb.ndim} (1-based, "
                f"valid 1..{2 * nrb.ndim})"
            )

    # ---- curve ----------------------------------------------------------
    # MATLAB: crvs(1).knots = srf.knots(1); crvs(1).coefs = srf.coefs(:,1);
    # crvs(2).knots = srf.knots(end); crvs(2).coefs = srf.coefs(:,end);
    # i.e. the full parent knot vector is kept for both boundary entries.
    if nrb.ndim == 1:
        crvs = [
            Nrb(
                number=(1,),
                order=(nrb.order[0],),
                knots=(nrb.knots[0].copy(),),
                coefs=nrb.coefs[:, 0:1].copy(),
            ),
            Nrb(
                number=(1,),
                order=(nrb.order[0],),
                knots=(nrb.knots[0].copy(),),
                coefs=nrb.coefs[:, -1:].copy(),
            ),
        ]
        return [crvs[s - 1] for s in sides]

    # ---- surface / volume ----------------------------------------------
    for i in range(nrb.ndim):
        if not _is_open_knots(nrb.knots[i], nrb.order[i]):
            raise ValueError(
                "nrbextract: only working for open knot vectors"
            )

    if nrb.ndim == 2:
        # u is axis 0, v is axis 1.
        u_k, v_k = nrb.knots
        crvs: List[Nrb] = []
        # sides 1,2 : U = 0, U = 1 (varying v -> use v knot vector)
        crvs.append(nrbmak(nrb.coefs[:, 0, :].copy(), v_k))
        crvs.append(nrbmak(nrb.coefs[:, -1, :].copy(), v_k))
        # sides 3,4 : V = 0, V = 1 (varying u -> use u knot vector)
        crvs.append(nrbmak(nrb.coefs[:, :, 0].copy(), u_k))
        crvs.append(nrbmak(nrb.coefs[:, :, -1].copy(), u_k))
    elif nrb.ndim == 3:
        # u axis 0, v axis 1, w axis 2.
        # MATLAB emission order (ind = 1:3): U-pair first, then V-pair,
        # then W-pair; remaining knot vectors in ascending index order.
        u_k, v_k, w_k = nrb.knots
        crvs = []
        # U = 0, U = 1 (varying v, w)
        crvs.append(nrbmak(nrb.coefs[:, 0, :, :].copy(), (v_k, w_k)))
        crvs.append(nrbmak(nrb.coefs[:, -1, :, :].copy(), (v_k, w_k)))
        # V = 0, V = 1 (varying u, w)
        crvs.append(nrbmak(nrb.coefs[:, :, 0, :].copy(), (u_k, w_k)))
        crvs.append(nrbmak(nrb.coefs[:, :, -1, :].copy(), (u_k, w_k)))
        # W = 0, W = 1 (varying u, v)
        crvs.append(nrbmak(nrb.coefs[:, :, :, 0].copy(), (u_k, v_k)))
        crvs.append(nrbmak(nrb.coefs[:, :, :, -1].copy(), (u_k, v_k)))
    else:  # pragma: no cover - ndim validated by Nrb
        raise ValueError("The entity is not a surface nor a volume")

    return [crvs[s - 1] for s in sides]
