"""
hexiga.scaff.make_iso_hexa
==========================

Build a *linear*
hexahedral NURBS solid whose control points are the Cartesian coordinates
of the tensor-product grid ``linspace(-dims(i)/maxDim, +dims(i)/maxDim,
dims(i))`` per direction, with the clamped-uniform knot vectors produced by
:func:`hexiga.scaff.get_uniform_knot_vect.getUniformKnotVect`.

Signature: ``isoH = makeIsoHexa(dims, deg)``.

Notes
-----

* ``dims`` must be a length-3 vector (else MATLAB ``assert``s).
* ``deg`` may be a scalar (broadcast to 3) or a length-3 vector.
* The control points are the homogeneous coordinates of the grid nodes
  (``coefs = [xI(:); yI(:); zI(:); 1s]``), so the resulting NURBS is
  *linear* (degree 1 in every direction) -- the higher-degree versions of
  the demo geometries are produced by a subsequent
  :func:`hexiga.nurbs.degelev.nrbdegelev` call.
* ``ndgrid`` in MATLAB produces arrays indexed (x, y, z); the port uses
  :func:`numpy.meshgrid` with ``indexing='ij'`` to mirror that ordering
  exactly.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.make import nrbmak
from .get_uniform_knot_vect import getUniformKnotVect

__all__ = ["makeIsoHexa"]

def makeIsoHexa(
    dims: Sequence[int],
    deg: Union[int, Sequence[int]],
) -> Nrb:
    """Linear hexahedral NURBS solid with ``dims`` control points per axis.

    Parameters
    ----------
    dims
        A length-3 sequence of positive ints: number of control points along
        (x, y, z).
    deg
        Spline degree, either a scalar (applied to all three axes) or a
        length-3 sequence.

    Returns
    -------
    Nrb
        A volume NURBS in homogeneous coordinates with ``dim == 4``.

    Raises
    ------
    ValueError
        If ``dims`` is not length 3.
    """
    dims = [int(d) for d in dims]
    if len(dims) != 3:
        raise ValueError(
            'Input dimensions "dims" do NOT represent a 3D Solid Hexahedron!'
        )
    if np.isscalar(deg) or (hasattr(deg, "__len__") is False):
        deg_v: Tuple[int, int, int] = (int(deg),) * 3
    else:
        deg_v = tuple(int(d) for d in deg)
        if len(deg_v) != 3:
            raise ValueError("deg must be a scalar or a length-3 sequence")

    minIsoVal = -1.0
    maxIsoVal = 1.0
    maxDim = float(max(dims))

    # MATLAB: ndgrid(linspace(...), linspace(...), linspace(...));
    # numpy: meshgrid with indexing='ij' gives the identical axis ordering.
    xIso = np.linspace(minIsoVal * dims[0] / maxDim,
                       maxIsoVal * dims[0] / maxDim,
                       dims[0])
    yIso = np.linspace(minIsoVal * dims[1] / maxDim,
                       maxIsoVal * dims[1] / maxDim,
                       dims[1])
    zIso = np.linspace(minIsoVal * dims[2] / maxDim,
                       maxIsoVal * dims[2] / maxDim,
                       dims[2])
    [xIso, yIso, zIso] = np.meshgrid(xIso, yIso, zIso, indexing="ij")

    # MATLAB: coefs = ones([3,dims]);  coefs(1,:) = xIso(:); ...
    # The weight row is appended by nrbmak's homogeneous promotion.
    coefs = np.ones((3,) + tuple(dims), dtype=float)
    coefs[0] = xIso.reshape(tuple(dims))
    coefs[1] = yIso.reshape(tuple(dims))
    coefs[2] = zIso.reshape(tuple(dims))

    UkntVctr = getUniformKnotVect(dims[0], deg_v[0])
    VkntVctr = getUniformKnotVect(dims[1], deg_v[1])
    WkntVctr = getUniformKnotVect(dims[2], deg_v[2])

    return nrbmak(coefs, [UkntVctr, VkntVctr, WkntVctr])
