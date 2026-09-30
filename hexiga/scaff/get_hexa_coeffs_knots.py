"""
hexiga.scaff.get_hexa_coeffs_knots
==================================

Assemble the
homogeneous control-point array and clamped knot vector of a single *linear*
hexahedral patch from its eight corners, given in the canonical
``(iA, iB, iC, iD, eA, eB, eC, eD)`` order.

Signature::

    [Hcfs, Hknt] = getHexaCoeffsKnots(iA,iB,iC,iD, eA,eB,eC,eD)

Notes
-----

* Each input is a 3x1 homogeneous-or-Cartesian point vector.  The control
  points are stacked as::

        Hcfs = cat(4, cat(3, cat(2,iA,iB), cat(2,iC,iD)),
                         cat(3, cat(2,eA,eB), cat(2,eC,eD)));

  i.e. a ``(3, 2, 2, 2)`` array indexed ``Hcfs(coord, u, v, w)`` with
  ``u`` = A->B, ``v`` = {AB,CD}, ``w`` = {lower, upper} face:
  ``w=1`` bottom face ``[iA iB iC iD]``, ``w=2`` top face ``[eA eB eC eD]``.
* The knots are the clamped linear triple ``[0 0 1 1]`` repeated for all
  three directions.
* The port returns the *Cartesian* 3x2x2x2 control points (row 3 is the
  weight) and appends the weight row on the caller side via ``nrbmak``'s
  homogeneous promotion -- but to match the MATLAB return exactly, the
  function returns the raw 3-row ``Hcfs`` and the ``Hknt`` list.
"""

from __future__ import annotations

import numpy as np

__all__ = ["getHexaCoeffsKnots"]

def _col(p: np.ndarray) -> np.ndarray:
    """Coerce a point to a ``(3,)`` column vector."""
    return np.asarray(p, dtype=float).reshape(3)

def getHexaCoeffsKnots(
    iA: np.ndarray,
    iB: np.ndarray,
    iC: np.ndarray,
    iD: np.ndarray,
    eA: np.ndarray,
    eB: np.ndarray,
    eC: np.ndarray,
    eD: np.ndarray,
):
    """Build the linear-hex control points and clamped knot vectors.

    Parameters
    ----------
    iA, iB, iC, iD
        Lower-face corners, each a 3-vector (3x1).
    eA, eB, eC, eD
        Upper-face corners, each a 3-vector (3x1).

    Returns
    -------
    (Hcfs, Hknt)
        ``Hcfs`` is a ``(3, 2, 2, 2)`` float array (``coord, u, v, w``) and
        ``Hknt`` is the list ``[[0,0,1,1], [0,0,1,1], [0,0,1,1]]``.
    """
    iA = _col(iA); iB = _col(iB); iC = _col(iC); iD = _col(iD)
    eA = _col(eA); eB = _col(eB); eC = _col(eC); eD = _col(eD)

    # Hcfs[coord, u, v, w]
    Hcfs = np.empty((3, 2, 2, 2), dtype=float)

    # w = 0 : lower face.   v = 0 : [iA, iB]   ;   v = 1 : [iC, iD]
    Hcfs[:, 0, 0, 0] = iA
    Hcfs[:, 1, 0, 0] = iB
    Hcfs[:, 0, 1, 0] = iC
    Hcfs[:, 1, 1, 0] = iD

    # w = 1 : upper face.   v = 0 : [eA, eB]   ;   v = 1 : [eC, eD]
    Hcfs[:, 0, 0, 1] = eA
    Hcfs[:, 1, 0, 1] = eB
    Hcfs[:, 0, 1, 1] = eC
    Hcfs[:, 1, 1, 1] = eD

    Hknt = [[0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]]

    return Hcfs, Hknt
