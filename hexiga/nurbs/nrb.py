"""
hexiga.nurbs.nrb
================

The NURBS data structure.

A NumPy-backed ``'B-NURBS'`` structure: curves, surfaces and volumes are
all represented uniformly as homogeneous control points with per-direction
knot vectors.

Storage contract:

* ``form``   is always the string ``'B-NURBS'``.
* ``dim``    is always ``4``: control points are stored in homogeneous
  coordinates, with row index ``3`` holding the weight.
* ``number`` is the control-point COUNT per parametric direction, stored as a
  tuple of Python ints of length ``ndim`` (1 for a curve, 2 for a surface,
  3 for a volume).
* ``order``  is the spline ORDER (= degree + 1) per parametric direction, a
  tuple of ints of the same length as ``number``.
* ``knots``  is a tuple of 1-D ``np.ndarray`` (float), one clamped knot
  vector per direction.  Each has length ``number[i] + order[i]``.
* ``coefs``  is a float ``np.ndarray`` of shape ``(4, *number)`` -- the first
  axis is homogeneous (x, y, z, w).

The number of parametric directions (0/1/2/3 -> curve/surface/volume) is
derived from ``len(number)``; a "curve" is a 1-D parametric NURBS.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

__all__ = ["Nrb"]

_HOMOGENEOUS = 4

def _as_tuple_ints(x, name: str) -> Tuple[int, ...]:
    x = tuple(int(v) for v in x)
    if len(x) == 0:
        raise ValueError(f"{name} must have at least one entry")
    if any(v <= 0 for v in x):
        raise ValueError(f"{name} entries must be positive, got {x}")
    return x

@dataclass
class Nrb:
    """A NURBS curve, surface or volume stored in homogeneous coordinates.

    See the module docstring for the exact storage contract.  Instances are
    intentionally value-like (they carry a handful of numpy arrays); use
    :meth:`copy` for a deep copy and :meth:`__eq__`-free comparisons (use
    :func:`hexiga.nurbs.api.compare_sides` style checks instead of relying on
    array equality).
    """

    number: Tuple[int, ...] = field(default=())
    order: Tuple[int, ...] = field(default=())
    knots: Tuple[np.ndarray, ...] = field(default=())
    coefs: np.ndarray = field(default=None)  # shape (4, *number)

    # -- fixed structural fields -------------------------------------------
    form: str = field(default="B-NURBS", init=False)
    dim: int = field(default=_HOMOGENEOUS, init=False)

    def __post_init__(self) -> None:
        self.form = "B-NURBS"
        self.dim = _HOMOGENEOUS
        self.number = _as_tuple_ints(self.number, "number")
        self.order = _as_tuple_ints(self.order, "order")
        ndim = len(self.number)
        if len(self.order) != ndim:
            raise ValueError(
                f"len(order)={len(self.order)} must match len(number)={ndim}"
            )
        if any(o < 1 for o in self.order):
            raise ValueError(f"order must be >= 1 (degree >= 0), got {self.order}")
        # A single-control-point direction (e.g. the degenerate boundary
        # curves produced by nrbextract, which keep the parent's full knot
        # vector and order) is exempt from the n >= o rule, matching
        # MATLAB's permissive struct handling.
        if any(n < o for n, o in zip(self.number, self.order) if n > 1):
            raise ValueError(
                "number must be >= order per direction (unless number==1): "
                f"number={self.number}, order={self.order}"
            )

        knots = self.knots
        if not isinstance(knots, (tuple, list)):
            raise TypeError("knots must be a tuple/list of arrays")
        if len(knots) != ndim:
            raise ValueError(
                f"len(knots)={len(knots)} must match len(number)={ndim}"
            )
        knots = tuple(np.asarray(k, dtype=float) for k in knots)
        for i, (k, n, o) in enumerate(zip(knots, self.number, self.order)):
            if k.ndim != 1:
                raise ValueError(f"knots[{i}] must be 1-D, got ndim={k.ndim}")
            # A single-control-point direction (e.g. the degenerate boundary
            # curves produced by nrbextract, which keep the parent's full
            # knot vector) is exempt from the knot-length rule, matching
            # MATLAB's permissive struct handling.
            if n > 1 and k.size != n + o:
                raise ValueError(
                    f"knots[{i}] length {k.size} != number+order = {n}+{o}"
                )
        self.knots = knots

        coefs = np.asarray(self.coefs, dtype=float)
        if coefs.shape[0] != _HOMOGENEOUS:
            raise ValueError(
                f"coefs first axis must be { _HOMOGENEOUS} (homogeneous), "
                f"got {coefs.shape[0]}"
            )
        if coefs.shape[1:] != tuple(self.number):
            raise ValueError(
                f"coefs shape {coefs.shape} does not match "
                f"(4, {tuple(self.number)})"
            )
        self.coefs = coefs

    # -- derived properties -------------------------------------------------
    @property
    def ndim(self) -> int:
        """Number of parametric directions (1 curve, 2 surface, 3 volume)."""
        return len(self.number)

    @property
    def degree(self) -> Tuple[int, ...]:
        """Spline degree per direction (``order - 1``)."""
        return tuple(o - 1 for o in self.order)

    def copy(self) -> "Nrb":
        """Return a deep copy (arrays included)."""
        return copy.deepcopy(self)

    # -- coordinate helpers -------------------------------------------------
    @property
    def weight(self) -> np.ndarray:
        """The weight array (``coefs[3]``)."""
        return self.coefs[3]

    def cart(self) -> np.ndarray:
        """Return the Cartesian control points ``coefs[:3] / coefs[3]``.

        Shape is ``(3, *number)``.  Division is performed element-wise; the
        caller is responsible for ensuring non-zero weights.
        """
        w = self.weight
        return self.coefs[:3] / w

    def __repr__(self) -> str:  # pragma: no cover - repr only
        return (
            f"Nrb(ndim={self.ndim}, number={self.number}, "
            f"order={self.order}, coefs={self.coefs.shape})"
        )
