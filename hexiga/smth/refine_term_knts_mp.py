"""Port of ``refineTermKntsMPHexa.m``.

Refines the *terminal* knot intervals (``[0, minKnt]`` and
``[maxKnt, 1]``) of each patch in each enabled direction, producing
additional control points that the subsequent smoothing cycle can act on.

Note
----
This port assumes the **canonical demo-toolkit convention** that each patch's
knot vector is normalised to span ``[0, 1]`` in every parametric direction.
The terminal knot values ``mean([0, minKnt])`` / ``mean([1, maxKnt])`` and the
``knts > 0`` / ``knts < 1`` guards below only make sense under that
normalisation. (General NURBS knots may span any range; the toolkit keeps
``[0, 1]`` for stability.)
"""
from __future__ import annotations

import numpy as np

from ..nurbs.kntins import nrbkntins
from ..nurbs.nrb import Nrb

def _get_inputs(pairs):
    OPTs: dict = {}
    it = iter(pairs)
    for key in it:
        OPTs[str(key).upper()] = next(it)
    OPTs.setdefault("UREF", True)
    OPTs.setdefault("VREF", True)
    OPTs.setdefault("WREF", True)
    OPTs.setdefault("USIDE", -1)
    OPTs.setdefault("VSIDE", -1)
    OPTs.setdefault("WSIDE", -1)
    return OPTs

def _get_mp_hexa_refining_knts(mpHexa: list, OPTs: dict):
    """Port of ``getMPHexaRefiningKnts``.  Returns per-patch list of
    (uKntsRef, vKntsRef, wKntsRef) where each is a numpy array of scalar knot
    values (possibly empty) to be inserted on that axis."""
    n = len(mpHexa)
    uKntsRef = [np.zeros(0) for _ in range(n)]
    vKntsRef = [np.zeros(0) for _ in range(n)]
    wKntsRef = [np.zeros(0) for _ in range(n)]

    def refine(side, dir_idx, hh, knts):
        """Port of the inner switch in MATLAB's ``getMPHexaRefiningKnts``."""
        knts = np.asarray(knts, dtype=float)
        if knts.size == 0:
            return np.zeros(0)
        if isinstance(side, (list, tuple, np.ndarray)) and len(np.atleast_1d(side)) > 1:
            side = np.asarray(side)[hh]
        side = int(side)
        if side == 0:
            # first knot strictly greater than 0
            m = knts > 0
            if not np.any(m):
                return np.zeros(0)
            minKnt = knts[np.argmax(m)]
            return np.array([0.5 * (0.0 + minKnt)])
        elif side == 1:
            m = knts < 1
            if not np.any(m):
                return np.zeros(0)
            # "last" knot strictly less than 1
            idx = np.where(m)[0]
            maxKnt = knts[idx[-1]]
            return np.array([0.5 * (1.0 + maxKnt)])
        else:
            m0 = knts > 0
            m1 = knts < 1
            if not (np.any(m0) and np.any(m1)):
                return np.zeros(0)
            minKnt = knts[np.argmax(m0)]
            maxKnt = knts[np.where(m1)[0][-1]]
            return np.unique(np.array([0.5 * (0.0 + minKnt), 0.5 * (1.0 + maxKnt)]))

    for hh in range(n):
        H = mpHexa[hh]
        if bool(OPTs["UREF"]):
            uKntsRef[hh] = refine(OPTs["USIDE"], 0, hh, H.knots[0])
        if bool(OPTs["VREF"]) and len(H.knots) > 1:
            vKntsRef[hh] = refine(OPTs["VSIDE"], 1, hh, H.knots[1])
        if bool(OPTs["WREF"]) and len(H.knots) > 2:
            wKntsRef[hh] = refine(OPTs["WSIDE"], 2, hh, H.knots[2])

    return uKntsRef, vKntsRef, wKntsRef

def refine_term_knts_mp(mpHexa: list, *pairs) -> list:
    """Port of ``refineTermKntsMPHexa``.

    ``mpHexa`` is a list of ``Nrb``.  Returns a new list with the terminal
    knot intervals refined according to ``OPTs``."""
    OPTs = _get_inputs(pairs)
    mpHexaRef = [H for H in mpHexa]
    uKntsRef, vKntsRef, wKntsRef = _get_mp_hexa_refining_knts(mpHexa, OPTs)
    for hh in range(len(mpHexa)):
        ndim = len(mpHexa[hh].knots)
        # MATLAB passes the cell {uKntsRef, vKntsRef, wKntsRef}; nrbkntins
        # takes a single 1-D array for a curve and a per-axis list otherwise.
        if ndim == 1:
            iknots = uKntsRef[hh]
        elif ndim == 2:
            iknots = [uKntsRef[hh], vKntsRef[hh]]
        else:
            iknots = [uKntsRef[hh], vKntsRef[hh], wKntsRef[hh]]
        mpHexaRef[hh] = nrbkntins(mpHexa[hh], iknots)
    return mpHexaRef
