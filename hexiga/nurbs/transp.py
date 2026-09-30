"""
hexiga.nurbs.transp
===================

Transposes the parametric directions of a NURBS surface, i.e. swap its
two parametric directions.

Algorithm::

    tsrf = nrbmak(permute(srf.coefs, [1 3 2]), fliplr(srf.knots));

Raises the equivalent errors for curves ("A NURBS curve cannot be
transposed") and volumes ("The transposition of NURBS volumes has not
been implemented.").
"""

from __future__ import annotations

import numpy as np

from .nrb import Nrb

__all__ = ["nrbtransp"]

def nrbtransp(nrb: Nrb) -> Nrb:
    """Transpose a NURBS surface (swap the u and v parametric directions).

    Parameters
    ----------
    nrb
        A NURBS **surface** (2-D).  Curves and volumes raise
        ``ValueError`` (matching MATLAB's behaviour).

    Returns
    -------
    Nrb
        The transposed surface: ``coefs[i, a, b]`` becomes
        ``coefs[i, b, a]`` and the two knot vectors are swapped.
    """
    if nrb.ndim == 1:
        raise ValueError("A NURBS curve cannot be transposed")
    if nrb.ndim > 2:
        raise ValueError(
            "The transposition of NURBS volumes has not been implemented."
        )
    coefs = np.transpose(nrb.coefs, (0, 2, 1)).copy()
    knots = (nrb.knots[1].copy(), nrb.knots[0].copy())
    return Nrb(
        number=(nrb.number[1], nrb.number[0]),
        order=(nrb.order[1], nrb.order[0]),
        knots=knots,
        coefs=coefs,
    )
