"""Port of ``pushbackCtrlPtsTopologyToMultiPatchHexa.m``.

Reassembles a list of NURBS patches from the smoothed CPT topology: for each
patch, recovers its 4-row homogeneous coefficient grid (rows x,y,z,w) from the
CPT entries that reference the patch, then wraps it in a new ``Nrb`` using the
stored patch attributes (form/dim/number/knots/order).
"""
from __future__ import annotations

import numpy as np

from ..nurbs.nrb import Nrb
from .cpt import CPT, CPTAttr, ind2sub

def _recover_ctrl_pts(mpCPT, patchID: int, UVWize: tuple):
    """Port of ``recoverCtrlPts``."""
    number = tuple(int(x) for x in UVWize)
    coefs = np.zeros((4,) + number, dtype=float)
    for pt in range(1, len(mpCPT) + 1):
        c = mpCPT[pt - 1]
        if patchID in c.ptcIDs:
            idx = np.asarray(c.ptcIDs) == patchID
            ptIdx3_sel = np.asarray(c.ptIdx3)[idx]
            for lin in ptIdx3_sel:
                U, V, W = ind2sub(number, int(lin))
                coefs[0, U - 1, V - 1, W - 1] = c.pt3D[0]
                coefs[1, U - 1, V - 1, W - 1] = c.pt3D[1]
                coefs[2, U - 1, V - 1, W - 1] = c.pt3D[2]
                coefs[3, U - 1, V - 1, W - 1] = c.ptw
    return coefs

def _recover_hexa(selCPTatrb: CPTAttr, coefs: np.ndarray) -> Nrb:
    """Port of ``recoverHexa``."""
    return Nrb(
        number=tuple(int(x) for x in selCPTatrb.number),
        order=tuple(int(x) for x in selCPTatrb.order),
        knots=tuple(np.asarray(k, dtype=float) for k in selCPTatrb.knots),
        coefs=coefs,
    )

def pushback_ctrl_pts_topology_to_mp(mpCPT, mpCPTatrb):
    """Port of ``pushbackCtrlPtsTopologyToMultiPatchHexa``.

    Returns a list of ``Nrb`` (one per patch).  Returns ``[]`` when either
    input is empty.
    """
    mpHexa: list = []
    if len(mpCPT) == 0 or len(mpCPTatrb) == 0:
        return mpHexa
    for mp in range(len(mpCPTatrb)):
        coefs = _recover_ctrl_pts(mpCPT, mp + 1, tuple(mpCPTatrb[mp].number))
        Hexa = _recover_hexa(mpCPTatrb[mp], coefs)
        if mp == 0:
            mpHexa = [Hexa]
        else:
            mpHexa.append(Hexa)
    return mpHexa
