"""
hexiga.scaff.mp_merge
=====================

Merge several
*conforming, adjacent* hexahedral NURBS patches along one parametric
dimension (u=1, v=2, or w=3) into a single patch.

Signature: ``Hexa = mpMerge(mpHexa, dim)``.

Algorithm
--------------------

* ``mergeCoefs``: for every patch except the last, drop the shared face
  along ``dim`` (the last index along that axis) and concatenate along that
  axis; the last patch is kept whole.
* ``mergeKnots``: start from the first patch's ``dim`` knot vector; for each
  subsequent patch, assert the *other two* knot vectors are identical
  (conformity), trim the running vector to ``1:end-1``, append the incoming
  patch's ``dim`` knot vector (shifted by ``hh-1``) from the first index at
  which it equals the current maximum onward; finally normalise by
  ``max(knots)``.  The other two knot vectors are the first patch's.
* The whole thing is wrapped in MATLAB's ``try/catch``: on any error it
  prints ``'Some Error Occurred while Merging mpHexa!'`` and returns empty
  (``[]``).  The port mirrors that: on any exception it prints the same
  line and returns ``None``.

Note: the MATLAB port is a straight transliteration; the per-direction
branches (u/v/w) are structurally identical and are written out explicitly
to preserve the literal source.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.make import nrbmak

__all__ = ["mpMerge"]

def _merge_coefs(mpHexa: Sequence[Nrb], dim: int) -> np.ndarray:
    """Port of the ``mergeCoefs`` subfunction (dim is 1/2/3, MATLAB-style)."""
    coefsMerge = np.empty((0,), dtype=float)
    n = len(mpHexa)
    for hh in range(n):
        c = mpHexa[hh].coefs
        if hh < n - 1:
            if dim == 1:   # U-dir
                part = c[:, 0:-1, :, :]
            elif dim == 2:  # V-dir
                part = c[:, :, 0:-1, :]
            elif dim == 3:  # W-dir
                part = c[:, :, :, 0:-1]
            else:
                print("Unrecognised Parametric Dimension!")
                part = np.empty((0,))
        else:
            part = c
        if coefsMerge.size == 0:
            coefsMerge = np.asarray(part, dtype=float)
        else:
            if dim == 1:
                coefsMerge = np.concatenate([coefsMerge, part], axis=1)
            elif dim == 2:
                coefsMerge = np.concatenate([coefsMerge, part], axis=2)
            elif dim == 3:
                coefsMerge = np.concatenate([coefsMerge, part], axis=3)
            else:
                print("Unrecognised Parametric Dimension!")
    return coefsMerge

def _knots_conforming(a: Nrb, b: Nrb, i: int, j: int, msg: str) -> None:
    """Mirror a MATLAB ``assert(isequal(...) && isequal(...), msg)``."""
    if not (
        np.array_equal(np.asarray(a.knots[i], float), np.asarray(b.knots[i], float))
        and np.array_equal(np.asarray(a.knots[j], float), np.asarray(b.knots[j], float))
    ):
        raise ValueError(msg)

def _merge_knots(mpHexa: Sequence[Nrb], dim: int) -> List[np.ndarray]:
    """Port of the ``mergeKnots`` subfunction (dim is 1/2/3, MATLAB-style).

    The three per-direction branches are written out explicitly to preserve
    the literal MATLAB source (each asserts conformity on the *other two*
    directions and places the merged knot vector at the dim-th position).
    """
    n = len(mpHexa)
    if dim == 1:  # U-dir
        knots = np.asarray(mpHexa[0].knots[0], float).copy()
        for hh in range(1, n):
            _knots_conforming(
                mpHexa[hh - 1], mpHexa[hh], 1, 2,
                "input mpHexa are not conforming on (V,W)!",
            )
            knots = knots[0:-1]
            appendknots = np.asarray(mpHexa[hh].knots[0], float) + (hh - 1 + 1)
            pos = np.where(appendknots == np.max(knots))[0][-1]
            knots = np.concatenate([knots, appendknots[pos + 1:]])
        knots = knots / np.max(knots)
        return [
            knots,
            np.asarray(mpHexa[0].knots[1], float),
            np.asarray(mpHexa[0].knots[2], float),
        ]
    elif dim == 2:  # V-dir
        knots = np.asarray(mpHexa[0].knots[1], float).copy()
        for hh in range(1, n):
            _knots_conforming(
                mpHexa[hh - 1], mpHexa[hh], 0, 2,
                "input mpHexa are not conforming on (U,W)!",
            )
            knots = knots[0:-1]
            appendknots = np.asarray(mpHexa[hh].knots[1], float) + (hh - 1 + 1)
            pos = np.where(appendknots == np.max(knots))[0][-1]
            knots = np.concatenate([knots, appendknots[pos + 1:]])
        knots = knots / np.max(knots)
        return [
            np.asarray(mpHexa[0].knots[0], float),
            knots,
            np.asarray(mpHexa[0].knots[2], float),
        ]
    elif dim == 3:  # W-dir
        knots = np.asarray(mpHexa[0].knots[2], float).copy()
        for hh in range(1, n):
            _knots_conforming(
                mpHexa[hh - 1], mpHexa[hh], 0, 1,
                "input mpHexa are not conforming on (U,V)!",
            )
            knots = knots[0:-1]
            appendknots = np.asarray(mpHexa[hh].knots[2], float) + (hh - 1 + 1)
            pos = np.where(appendknots == np.max(knots))[0][-1]
            knots = np.concatenate([knots, appendknots[pos + 1:]])
        knots = knots / np.max(knots)
        return [
            np.asarray(mpHexa[0].knots[0], float),
            np.asarray(mpHexa[0].knots[1], float),
            knots,
        ]
    print("Unrecognised Parametric Dimension!")
    return []

def mpMerge(mpHexa: Sequence[Nrb], dim: int) -> Optional[Nrb]:
    """Merge conforming adjacent patches along dimension ``dim`` (1/2/3).

    Parameters
    ----------
    mpHexa
        Sequence of NURBS volumes (homogeneous form) ordered for merging.
    dim
        1 (u), 2 (v) or 3 (w).

    Returns
    -------
    Nrb or None
        The merged NURBS volume, or ``None`` on any internal error (matching
        MATLAB's ``try/catch`` which prints and returns ``[]``).
    """
    try:
        coefsMerge = _merge_coefs(mpHexa, dim)
        knotsMerge = _merge_knots(mpHexa, dim)
        return nrbmak(coefsMerge, knotsMerge)
    except Exception:
        print("Some Error Occurred while Merging mpHexa!")
        return None
