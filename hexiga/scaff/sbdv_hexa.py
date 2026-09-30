"""
hexiga.scaff.sbdv_hexa
======================

Subdivide *once* a solid
NURBS hexahedron by inserting interior knots (midpoints for an ODD topology,
third-points for an EVEN topology) into each of its three knot vectors.

Signature: ``sHexa = sbdvHexa(Hexa, ODD_Flag)``.

Notes
-----

* ``Hexa`` must be a 3-D solid (``len(Hexa.number) == 3``), else an
  :class:`AssertionError` is raised (mirroring the MATLAB ``assert``).
* ``ODD_Flag``:

  * non-empty ``True``  -- force an ODD control-point topology in every
    direction (midpoint splitting).
  * non-empty ``False`` -- force an EVEN topology (third-point splitting).
  * empty / ``None``    -- keep the per-direction topology of the input:
    ODD if that direction's control-point count is odd, EVEN otherwise.

* The result is obtained via :func:`hexiga.nurbs.kntins.nrbkntins` with the
  three splitting-knot lists.
* The splitting-knot helpers are ported verbatim from the MATLAB local
  functions ``getSplittingKnotsODD`` / ``getSplittingKnotsEVN``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.kntins import nrbkntins

__all__ = ["sbdvHexa"]

def _get_splitting_knots_odd(kntVect: np.ndarray) -> np.ndarray:
    """Port of ``getSplittingKnotsODD``: midpoint of each unique span."""
    unq = np.sort(np.unique(kntVect))
    d = np.diff(unq)
    return unq[:-1] + d / 2

def _get_splitting_knots_evn(kntVect: np.ndarray) -> np.ndarray:
    """Port of ``getSplittingKnotsEVN``: third-points of each unique span."""
    unq = np.sort(np.unique(kntVect))
    d = np.diff(unq)
    knts = np.concatenate([unq[:-1] + d / 3, unq[:-1] + 2 * (d / 3)])
    return np.sort(knts)

def sbdvHexa(Hexa: Nrb, ODD_Flag: Optional[bool] = None) -> Nrb:
    """Subdivide a solid NURBS hexahedron once (see module docstring).

    Parameters
    ----------
    Hexa
        A 3-D solid NURBS patch.
    ODD_Flag
        ``None`` (empty) to keep the input per-direction topology, or a
        boolean to force ODD (``True``) / EVEN (``False``) everywhere.

    Returns
    -------
    Nrb
        The once-subdivided solid NURBS patch.
    """
    if len(Hexa.number) != 3:
        raise AssertionError("Input Hexa is not a Solid NURBS patch!")

    UkntVect = np.asarray(Hexa.knots[0])
    VkntVect = np.asarray(Hexa.knots[1])
    WkntVect = np.asarray(Hexa.knots[2])

    if ODD_Flag is not None:
        # FORCING subdiv. with either ODD or EVEN ctrl-pts.
        if ODD_Flag:
            spltUknts = _get_splitting_knots_odd(UkntVect)
            spltVknts = _get_splitting_knots_odd(VkntVect)
            spltWknts = _get_splitting_knots_odd(WkntVect)
        else:
            spltUknts = _get_splitting_knots_evn(UkntVect)
            spltVknts = _get_splitting_knots_evn(VkntVect)
            spltWknts = _get_splitting_knots_evn(WkntVect)
    else:
        # AUTOMATIC subdiv. keeping the input ctrl-pts topology.
        if Hexa.number[0] % 2 > 0:
            spltUknts = _get_splitting_knots_odd(UkntVect)
        else:
            spltUknts = _get_splitting_knots_evn(UkntVect)

        if Hexa.number[1] % 2 > 0:
            spltVknts = _get_splitting_knots_odd(VkntVect)
        else:
            spltVknts = _get_splitting_knots_evn(VkntVect)

        if Hexa.number[2] % 2 > 0:
            spltWknts = _get_splitting_knots_odd(WkntVect)
        else:
            spltWknts = _get_splitting_knots_evn(WkntVect)

    return nrbkntins(Hexa, [spltUknts, spltVknts, spltWknts])
