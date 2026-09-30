"""
hexiga.nurbs.make
=================

``nrbmak`` -- build an :class:`~hexiga.nurbs.nrb.Nrb` from control points and
knot vectors.

Port of ``nrbmak.m``.  Handles curves, surfaces and volumes; promotes
Cartesian control points (dim 2 or 3) to homogeneous coordinates (dim 4);
sorts the knot vectors; and derives the spline order from the knot-vector
length and the control-point count (``order = knot_len - number``).

The optional ``normalize`` flag maps each direction's knot span ``[k[order-1],
k[-(order-1)-1]]`` (0-based) onto ``[0, 1]``.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from ._backend import backend_name  # noqa: F401  (re-export surface)
from .nrb import Nrb

__all__ = ["nrbmak"]

_HOMOGENEOUS = 4

def _normalize_knots(k: np.ndarray, order: int) -> np.ndarray:
    a = k[order - 1]
    b = k[-(order - 1) - 1]
    return (k - a) / (b - a)

def nrbmak(
    coefs: np.ndarray,
    knots: Union[np.ndarray, Sequence[np.ndarray]],
    normalize: bool = False,
) -> Nrb:
    """Construct an NURBS curve, surface or volume.

    Parameters
    ----------
    coefs
        Control points.  Shape ``(dim, n1)`` for a curve,
        ``(dim, n1, n2)`` for a surface, ``(dim, n1, n2, n3)`` for a volume.
        ``dim`` may be 2, 3 or 4 (4 means homogeneous input is given).
    knots
        A single 1-D array for a curve, or a sequence of 1-D arrays (one per
        parametric direction) for a surface/volume.
    normalize
        If True, rescale each knot vector so its clamped span maps to [0, 1].

    Returns
    -------
    Nrb
    """
    coefs = np.asarray(coefs, dtype=float)
    if coefs.ndim < 2:
        raise ValueError(
            f"coefs must have shape (dim, n1[, n2, n3]), got {coefs.shape}"
        )
    dim = coefs.shape[0]
    if dim not in (2, 3, 4):
        raise ValueError(f"unsupported spatial dimension {dim}; use 2, 3 or 4")

    # Determine number of parametric directions.
    if isinstance(knots, np.ndarray) and knots.ndim == 1:
        ndim = 1
        k_list = [knots]
    else:
        try:
            k_list = list(knots)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("knots must be an array or a sequence of arrays") from exc
        # A length-1 sequence is a single 1-D knot vector (a curve), not a
        # surface with one direction.  MATLAB's ``{knots}`` cell has one
        # element for a curve; mirror that here.
        if len(k_list) == 1 and np.asarray(k_list[0]).ndim == 1:
            ndim = 1
        else:
            ndim = len(k_list)
            if ndim not in (2, 3):
                raise ValueError(f"unsupported number of knot vectors: {ndim}")

    # Number of control points along each direction = coefs.shape[1:1+ndim].
    number = tuple(int(v) for v in coefs.shape[1 : 1 + ndim])
    if len(number) != ndim:
        raise ValueError(
            f"coefs has {len(coefs.shape)-1} data dims but {ndim} knot vectors"
        )

    # Promote to homogeneous coordinates of shape (4, *number).
    if dim < _HOMOGENEOUS:
        hom = np.zeros((_HOMOGENEOUS,) + number, dtype=float)
        hom[_HOMOGENEOUS - 1] = 1.0  # weight row
        # coefs has shape (dim, n1[, n2, n3]); broadcast into first dim slots.
        # Because coefs.shape[1:] may carry extra trailing singleton dims for
        # a curve given as (dim, n, 1), slice precisely.
        hom[:dim] = coefs
    else:
        hom = coefs

    # Build per-direction knots + order.
    out_knots = []
    out_order = []
    for i in range(ndim):
        k = np.asarray(k_list[i], dtype=float)
        n_i = number[i]
        o_i = k.size - n_i
        if o_i < 1:
            raise ValueError(
                f"direction {i}: knot vector too short (order {o_i} < 1)"
            )
        if n_i < o_i:
            raise ValueError(
                f"direction {i}: not enough control points "
                f"(number {n_i} < order {o_i})"
            )
        ks = np.sort(k)
        if normalize:
            ks = _normalize_knots(ks, o_i)
        out_knots.append(ks)
        out_order.append(int(o_i))

    return Nrb(
        number=number,
        order=tuple(out_order),
        knots=tuple(out_knots),
        coefs=hom,
    )
