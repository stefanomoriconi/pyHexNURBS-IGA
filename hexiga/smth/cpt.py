"""
hexiga.smth.cpt
===============

Control-point topology (CPT) data structures and small helpers shared by the
``mpTurboSmooth`` dependency tree.

This is a faithful, literal port of the MATLAB structs used by
``smth_utils/multipatch``:

* :class:`CPT`       -- one entry of the ``mpCPT`` struct array (a single
  aggregated control point shared by one or more patches).
* :class:`CPTAttr`   -- one entry of the ``mpCPTatrb`` struct array (per-patch
  NURBS attributes: ``form, dim, number, knots, order``).
* :class:`Cuboid`    -- one entry of the ``mpCuboid`` struct array (the
  per-patch cuboid grid of shell values, boundary/frame flags and solve state).

MATLAB uses 1-based linear indices throughout the CPT (``ptIdx3``) and the
cuboid grid.  We preserve that convention in the *stored* fields (so that the
output matches the MATLAB ground truth exactly) and only convert to 0-based at
the ``numpy`` boundary via the helpers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["CPT", "CPTAttr", "Cuboid", "sub2ind", "ind2sub"]

def sub2ind(shape, *indices) -> int:
    """MATLAB ``sub2ind`` (1-based, column-major) for arbitrary rank.

    ``indices`` are the 1-based subscripts (``ndim`` of them).  Returns the
    1-based linear index.
    """
    shape = tuple(int(s) for s in shape)
    idx0 = tuple(int(i) - 1 for i in indices)
    return int(np.ravel_multi_index(idx0, shape, order="F")) + 1

def ind2sub(shape, linear: int) -> Tuple[int, ...]:
    """MATLAB ``ind2sub`` (1-based, column-major).

    ``linear`` is a 1-based linear index.  Returns the 1-based subscripts.
    """
    shape = tuple(int(s) for s in shape)
    idx0 = np.unravel_index(int(linear) - 1, shape, order="F")
    return tuple(int(i) + 1 for i in idx0)

@dataclass
class CPT:
    """One aggregated control point (entry of ``mpCPT``).

    Fields mirror the MATLAB struct exactly (see ``aggregateMultiPatchHexa.m``
    and ``appendSmthAtrbs_new.m``).  ``ptIdx3`` and ``ptIdx3R`` are stored
    1-based; ``pt3D`` is a ``(3,)`` Cartesian vector; ``ptw`` is the scalar
    (normalised) weight; ``dimShlR`` is ``(N,3)`` and ``ptIdx3R`` is ``(N,)``
    where ``N = len(ptcIDs)`` (per-patch redundant shell bookkeeping).
    """

    # -- geometry ----------------------------------------------------------
    pt3D: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ptw: float = 1.0

    # -- multi-patch identity ---------------------------------------------
    ptcIDs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    ptIdx3: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # -- shell / frame attributes -----------------------------------------
    ptShl: float = 0.0
    isShlB: bool = False
    ptFrm: float = 0.0

    # -- per-patch redundant shell bookkeeping (length == len(ptcIDs)) -----
    dimShlR: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=int)
    )
    ptIdx3R: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # -- smoothing (set by appendSmthAtrbs / smthCtrlPtsTopologyAttribs) --
    nnIDs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    nnLbls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    isSmth: bool = True

    # -- helpers -----------------------------------------------------------
    @property
    def n_patches(self) -> int:
        return int(np.size(self.ptcIDs))

@dataclass
class CPTAttr:
    """Per-patch NURBS attributes (entry of ``mpCPTatrb``)."""

    form: str = "B-NURBS"
    dim: int = 4
    number: Tuple[int, ...] = ()
    knots: Tuple[np.ndarray, ...] = ()
    order: Tuple[int, ...] = ()

@dataclass
class Cuboid:
    """Per-patch cuboid grid (entry of ``mpCuboid``).

    * ``Grid``          -- 3-D ``float`` array, shape ``== Hexa.number``;
      initially filled with NaN, resolved to integer shell values (0 = medial
      locus, ``>0`` = shell distance from the medial locus) by
      :func:`hexiga.smth.cuboid.compute_cuboid_grid_dist`.
    * ``ShellBoundary`` -- 3-D ``bool`` array, shape ``== number`` (per-cell
      shell-boundary flag).
    * ``Frame``         -- 3-D ``int`` array, shape ``== number`` (0=face,
      1=edge, 2=corner).
    * ``Solved`` / ``Ready`` -- scalar solve / propagation flags.
    """

    Grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ShellBoundary: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    Frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    Solved: bool = False
    Ready: bool = False
