"""
hexiga.nurbs.reverse
====================

Reverses the parametric direction(s) of a NURBS curve, surface or volume.

Algorithm::

    % surface / volume
    for ii = idir
      nrb.knots{ii} = sort (nrb.knots{ii}(end) - nrb.knots{ii});
      nrb.coefs = flip (nrb.coefs, ii+1);
    end

    % curve
    nrb.knots = sort (nrb.knots(end) - nrb.knots);
    nrb.coefs = fliplr (nrb.coefs);

The public API uses 1-based direction numbers, exactly like MATLAB's
``nrbreverse(nrb, idir)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .nrb import Nrb

__all__ = ["nrbreverse"]

def nrbreverse(
    nrb: Nrb,
    idir: Union[int, Sequence[int], None] = None,
) -> Nrb:
    """Reverse the evaluation direction(s) of a NURBS entity.

    Parameters
    ----------
    nrb
        A NURBS curve, surface or volume.
    idir
        Optional 1-based direction(s) to reverse (MATLAB convention).
        Defaults to all directions.

    Returns
    -------
    Nrb
        The reversed NURBS (a new object; ``nrb`` is unchanged).
    """
    if nrb.ndim == 1:
        knots = (np.sort(nrb.knots[0][-1] - nrb.knots[0]),)
        coefs = np.flip(nrb.coefs, axis=1)
    else:
        if idir is None:
            dirs = list(range(1, nrb.ndim + 1))
        elif isinstance(idir, int):
            dirs = [idir]
        else:
            dirs = [int(x) for x in idir]
        for d in dirs:
            if not (1 <= d <= nrb.ndim):
                raise ValueError(
                    f"direction {d} out of range for ndim={nrb.ndim} (1-based)"
                )
        knots = list(nrb.knots)
        coefs = nrb.coefs
        for d in dirs:
            i = d - 1
            knots[i] = np.sort(knots[i][-1] - knots[i])
            coefs = np.flip(coefs, axis=i + 1)
        knots = tuple(knots)

    return Nrb(
        number=nrb.number,
        order=nrb.order,
        knots=knots,
        coefs=np.asarray(coefs, dtype=float),
    )
